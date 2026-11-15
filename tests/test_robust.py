import copy

import numpy as np
import pytest
from builders import inputs, neutral_w0
from hypothesis import given
from hypothesis import strategies as st

from robust_mvo.optimizer import Constraints, CostModel, Inputs, Solution, constraint_violations
from robust_mvo.optimizer import solve as real_solve
from robust_mvo.risk_model import FactorRiskModel
from robust_mvo.robust import Config, precheck, robust_optimize, sanitize

TOL = 1e-6
BOOK = Constraints(max_position=0.05, max_trade=0.03, max_gross=1.0)
COSTS = CostModel(linear=5e-4, impact=2e-3)


def fake(behaviour: dict[str, str]):
    """A solve_fn that fails on demand for some solvers and calls the real one otherwise.

    behaviour maps solver -> "raise" | "error" | "inaccurate_bad" | "inaccurate_ok"
    | "nan". "inaccurate_ok" returns the real solution relabelled as
    optimal_inaccurate, so the D2 acceptance rule is tested without depending
    on when a real solver happens to stop early (which differs by platform).
    """

    def solve_fn(inp, solver, **kw):
        mode = behaviour.get(solver)
        if mode == "raise":
            raise RuntimeError("solver crashed")
        if mode == "error":
            return Solution(None, "error", solver, None, 0.0, "SolverError('boom')")
        if mode == "inaccurate_bad":
            w = np.full(inp.n_assets, 0.5)  # violates caps and neutrality
            return Solution(w, "optimal_inaccurate", solver, 0.0, 0.0)
        if mode == "inaccurate_ok":
            sol = real_solve(inp, solver, **kw)
            return Solution(sol.w, "optimal_inaccurate", solver, sol.objective, sol.solve_time)
        if mode == "nan":
            return Solution(np.full(inp.n_assets, np.nan), "optimal", solver, 0.0, 0.0)
        return real_solve(inp, solver, **kw)

    return solve_fn


def check_invariants(inp: Inputs, w: np.ndarray, rep) -> None:
    imposed = sanitize(inp)[0] if np.isfinite(inp.risk.factor_cov).all() else inp
    assert rep.violations == constraint_violations(imposed, w)
    if rep.level <= 1:
        assert max(rep.violations.values(), default=0.0) <= TOL
    elif rep.level == 2:
        hard = {g: v for g, v in rep.violations.items() if g not in rep.relaxed}
        assert max(hard.values(), default=0.0) <= TOL
    else:
        np.testing.assert_array_equal(w, inp.w0)
    for i in rep.frozen:
        assert w[i] == pytest.approx(inp.w0[i], abs=TOL)


# --- Level 0 and 1 -----------------------------------------------------------------


def test_happy_path_is_level_zero():
    inp = inputs(20, w0=neutral_w0(20, 0, 0.03), costs=COSTS, constraints=BOOK)
    w, rep = robust_optimize(inp)
    assert rep.level == 0
    assert rep.reasons == []
    assert [a.solver for a in rep.attempts] == ["CLARABEL"]
    check_invariants(inp, w, rep)


@pytest.mark.parametrize("mode", ["raise", "error", "inaccurate_bad", "nan"])
def test_primary_failure_falls_back_to_scs(mode):
    inp = inputs(20, w0=neutral_w0(20, 0, 0.03), costs=COSTS, constraints=BOOK)
    w, rep = robust_optimize(inp, solve_fn=fake({"CLARABEL": mode}))
    assert rep.level == 1
    assert rep.reasons[0].startswith("level 0 -> 1: CLARABEL")
    assert rep.attempts[0].accepted is False
    assert rep.attempts[-1].solver == "SCS" and rep.attempts[-1].accepted
    check_invariants(inp, w, rep)


def test_osqp_is_skipped_on_the_three_halves_form_and_used_on_a_qp():
    w0 = neutral_w0(20, 0, 0.03)
    inp = inputs(20, w0=w0, costs=COSTS, constraints=BOOK)
    _, rep = robust_optimize(inp, solve_fn=fake({"CLARABEL": "error", "SCS": "error"}))
    assert "OSQP" not in [a.solver for a in rep.attempts]
    assert "skipped ['OSQP'] (not a QP)" in rep.reasons[0]
    assert rep.level == 3  # every solver failed, the elastic level failed too

    qp = inputs(
        20, w0=w0, constraints=BOOK, costs=CostModel(linear=5e-4, impact=2e-2, impact_power=2.0)
    )
    w, rep = robust_optimize(qp, solve_fn=fake({"CLARABEL": "error", "SCS": "error"}))
    assert rep.level == 1
    assert rep.attempts[-1].solver == "OSQP" and rep.attempts[-1].accepted
    check_invariants(qp, w, rep)


def test_inaccurate_accepted_by_default_and_rejected_when_configured():
    inp = inputs(20, w0=neutral_w0(20, 0, 0.03), costs=COSTS, constraints=BOOK)
    relabel = fake({"CLARABEL": "inaccurate_ok"})
    _, rep = robust_optimize(inp, solve_fn=relabel)
    assert rep.level == 0 and rep.inaccurate
    w, rep = robust_optimize(inp, Config(accept_inaccurate=False), solve_fn=relabel)
    assert rep.level == 1
    assert rep.attempts[0].status == "optimal_inaccurate" and not rep.attempts[0].accepted
    check_invariants(inp, w, rep)


# --- Infeasible constraints -> level 2 ---------------------------------------------


def unreachable_neutrality(n: int = 10) -> Inputs:
    # w0 is net long 0.4 and each asset may trade only 0.02, so the book can
    # move at most 0.2 toward neutral: dollar neutrality is unreachable.
    w0 = np.full(n, 0.04)
    return inputs(
        n,
        w0=w0,
        costs=COSTS,
        constraints=Constraints(max_position=0.05, max_trade=0.02, max_gross=1.0),
    )


def test_precheck_catches_unreachable_neutrality_and_relaxes_liquidity():
    inp = unreachable_neutrality()
    w, rep = robust_optimize(inp)
    assert rep.level == 2
    assert "pre-check: dollar neutrality unreachable" in rep.reasons[0]
    assert rep.relaxed == ("max_trade",)
    assert rep.violations["max_trade"] > 0
    assert rep.violations["dollar_neutral"] <= TOL
    check_invariants(inp, w, rep)


def test_solver_detected_infeasibility_skips_level_one():
    # Each interval passes the pre-check on its own; together they do not:
    # asset 0 must hold >= 0.1, neutrality then needs >= 0.1 short elsewhere,
    # so gross >= 0.2 > 0.15.
    n = 3
    w0 = np.array([0.15, 0.0, 0.0])
    inp = inputs(
        n, w0=w0, constraints=Constraints(max_trade=np.array([0.05, 0.3, 0.3]), max_gross=0.15)
    )
    assert precheck(inp, np.zeros(n, dtype=bool)) is None
    w, rep = robust_optimize(inp)
    assert rep.reasons[0].startswith("level 0 -> 2: CLARABEL reports infeasible")
    assert [a.level for a in rep.attempts][:2] == [0, 2]
    assert rep.level == 2
    assert rep.relaxed == ("max_trade",)
    check_invariants(inp, w, rep)


def test_relaxation_escalates_to_position_caps():
    # Asset 0 has no alpha, so it is frozen at +0.1. Neutrality then needs 0.1
    # of shorts, but the other assets have zero position caps: relaxing the
    # trade cap alone is not enough, and the chain must relax position caps too.
    n = 4
    w0 = np.array([0.1, 0.0, 0.0, 0.0])
    base = inputs(n, w0=w0)
    alpha = base.alpha.copy()
    alpha[0] = np.nan
    cons = Constraints(max_position=np.array([0.1, 0.0, 0.0, 0.0]), max_trade=0.01)
    inp = Inputs(alpha, base.risk, w0, gamma=base.gamma, constraints=cons)
    w, rep = robust_optimize(inp)
    assert rep.level == 2
    assert rep.relaxed == ("max_trade", "max_position")
    stages = [a.relaxed for a in rep.attempts if a.level == 2]
    assert stages[0] == ("max_trade",) and stages[-1] == ("max_trade", "max_position")
    assert rep.violations["max_position"] > 0.01
    assert w[1:].sum() == pytest.approx(-0.1, abs=TOL)
    check_invariants(inp, w, rep)


def test_elastic_returns_original_optimum_when_feasible():
    # Exact penalty: if the problem was feasible, relaxing it changes nothing.
    from robust_mvo.optimizer import solve

    inp = inputs(
        20,
        w0=neutral_w0(20, 0, 0.03),
        constraints=BOOK,
        costs=CostModel(linear=5e-4, impact=2e-2, impact_power=2.0),
    )
    a = solve(inp)
    b = solve(inp, elastic={"max_trade": 1.0, "max_position": 1.0, "max_gross": 1.0})
    np.testing.assert_allclose(a.w, b.w, atol=1e-6)


# --- Level 3 -------------------------------------------------------------------------


def test_everything_frozen_holds_w0():
    inp = inputs(5, w0=neutral_w0(5, 1, 0.02))
    inp = Inputs(np.full(5, np.nan), inp.risk, inp.w0, gamma=inp.gamma)
    w, rep = robust_optimize(inp)
    assert rep.level == 3
    assert "every asset is frozen" in rep.reasons[-1]
    check_invariants(inp, w, rep)


def test_every_solver_failing_everywhere_holds_w0():
    inp = inputs(20, w0=neutral_w0(20, 0, 0.03), costs=COSTS, constraints=BOOK)
    w, rep = robust_optimize(inp, solve_fn=fake({"CLARABEL": "raise", "SCS": "error"}))
    assert rep.level == 3
    assert [r.split(":")[0] for r in rep.reasons] == [
        "level 0 -> 1",
        "level 1 -> 2",
        "level 2 -> 3",
    ]
    assert rep.reasons[-1].endswith("relaxed problem also failed")
    check_invariants(inp, w, rep)


def test_nonfinite_factor_covariance_holds_w0():
    # FactorRiskModel rejects this at construction; the chain still guards
    # against a model mutated afterwards.
    inp = inputs(5, w0=neutral_w0(5, 2, 0.02))
    rm = copy.copy(inp.risk)
    f = rm.factor_cov.copy()
    f[0, 0] = np.inf
    object.__setattr__(rm, "factor_cov", f)
    w, rep = robust_optimize(Inputs(inp.alpha, rm, inp.w0))
    assert rep.level == 3 and "factor covariance not finite" in rep.reasons[0]
    np.testing.assert_array_equal(w, inp.w0)


# --- Bad inputs (D5) -----------------------------------------------------------------


@pytest.mark.parametrize("field", ["alpha", "exposures", "idio_var", "max_trade"])
def test_asset_with_nan_input_is_frozen_and_reported(field):
    n = 12
    w0 = neutral_w0(n, 3, 0.03)
    base = inputs(n, w0=w0, costs=COSTS, constraints=BOOK)
    alpha = base.alpha.copy()
    b = base.risk.exposures.copy()
    d = base.risk.idio_var.copy()
    trade = np.full(n, 0.03)
    if field == "alpha":
        alpha[4] = np.nan
    elif field == "exposures":
        b[4, 1] = np.nan
    elif field == "idio_var":
        d[4] = np.inf
    else:
        trade[4] = np.nan  # e.g. ADV unknown
    cons = Constraints(max_position=0.05, max_trade=trade, max_gross=1.0)
    inp = Inputs(
        alpha,
        FactorRiskModel(b, base.risk.factor_cov, d),
        w0,
        gamma=base.gamma,
        costs=COSTS,
        constraints=cons,
    )
    w, rep = robust_optimize(inp)
    assert list(rep.frozen) == [4]
    assert w[4] == pytest.approx(w0[4], abs=TOL)
    assert rep.level == 0
    assert np.abs(w - w0).max() > 1e-4  # the rest of the book still trades


def test_nonfinite_w0_raises():
    inp = inputs(5)
    w0 = np.zeros(5)
    w0[0] = np.nan
    with pytest.raises(ValueError, match="w0"):
        robust_optimize(Inputs(inp.alpha, inp.risk, w0))


def test_indefinite_factor_covariance_is_repaired_and_reported():
    f = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]) * 1e-4
    rng = np.random.default_rng(0)
    rm = FactorRiskModel(rng.normal(size=(15, 3)), f, np.full(15, 2e-4))
    inp = Inputs(rng.normal(0, 5e-4, 15), rm, np.zeros(15), gamma=10.0, constraints=BOOK)
    w, rep = robust_optimize(inp)
    assert rep.factor_cov_repaired == 1
    assert rep.level == 0
    check_invariants(inp, w, rep)


# --- Property test: the chain never raises and its report is honest ---------------------


@given(
    n=st.integers(2, 15),
    seed=st.integers(0, 5000),
    net=st.floats(-0.3, 0.3),
    pos_cap=st.floats(0.0, 0.2),
    trade_cap=st.floats(0.0, 0.1),
    gross=st.floats(0.0, 2.0),
    n_nan=st.integers(0, 3),
    power=st.sampled_from([1.5, 2.0]),
)
def test_chain_invariants_on_arbitrary_inputs(
    n, seed, net, pos_cap, trade_cap, gross, n_nan, power
):
    rng = np.random.default_rng(seed)
    w0 = rng.normal(0, 0.05, n) + net / n
    base = inputs(
        n,
        2,
        seed,
        w0=w0,
        costs=CostModel(linear=3e-4, impact=1e-3, impact_power=power),
        constraints=Constraints(max_position=pos_cap, max_trade=trade_cap, max_gross=gross),
    )
    alpha = base.alpha.copy()
    alpha[rng.choice(n, size=min(n_nan, n), replace=False)] = np.nan
    inp = Inputs(
        alpha, base.risk, w0, gamma=base.gamma, costs=base.costs, constraints=base.constraints
    )
    w, rep = robust_optimize(inp)
    assert 0 <= rep.level <= 3
    assert bool(rep.reasons) == (rep.level > 0)
    assert all(r.startswith("level ") for r in rep.reasons)
    check_invariants(inp, w, rep)


def test_frozen_asset_over_its_cap_is_reported_not_fixed():
    # Found by the property test: asset 0 has no alpha and sits above a cap it
    # cannot trade down to. The solve is still level 0; the breach is listed.
    n = 6
    w0 = neutral_w0(n, 0, 0.03)
    w0[0], w0[1] = 0.08, w0[1] - (0.08 - w0[0])
    base = inputs(n, w0=w0)
    alpha = base.alpha.copy()
    alpha[0] = np.nan
    inp = Inputs(
        alpha,
        base.risk,
        w0,
        gamma=base.gamma,
        constraints=Constraints(max_position=0.05, max_trade=0.2),
    )
    w, rep = robust_optimize(inp)
    assert rep.level == 0
    assert w[0] == pytest.approx(0.08, abs=TOL)
    assert "max_position" in rep.frozen_cap_breaches[0]
    assert rep.violations["max_position"] <= TOL  # against the caps actually imposed
    assert constraint_violations(inp, w)["max_position"] == pytest.approx(0.03, abs=TOL)
