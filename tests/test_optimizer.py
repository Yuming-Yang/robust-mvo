import numpy as np
import pytest
from builders import inputs, neutral_w0, risk_model
from hypothesis import given
from hypothesis import strategies as st

from robust_mvo.optimizer import (
    Constraints,
    CostModel,
    Inputs,
    adv_cap,
    constraint_violations,
    cost_terms,
    risk_terms,
    solve,
    square_root_impact,
)
from robust_mvo.risk_model import FactorRiskModel

TOL = 1e-6  # D7: constraint tolerance in weight units
# D2: "optimal_inaccurate" is accepted when the post-check passes. CLARABEL
# reports it on the 3/2-power form when many trades sit exactly at zero.
OK = ("optimal", "optimal_inaccurate")


def closed_form(sigma: np.ndarray, a: np.ndarray, gamma: float, neutral: bool) -> np.ndarray:
    """maximize a'w - gamma w'Sigma w  [s.t. 1'w = 0]; see DERIVATIONS.md 2.2."""
    inv_a = np.linalg.solve(sigma, a)
    if not neutral:
        return inv_a / (2 * gamma)
    inv_1 = np.linalg.solve(sigma, np.ones_like(a))
    eta = inv_a.sum() / inv_1.sum()
    return (inv_a - eta * inv_1) / (2 * gamma)


# --- Closed form ---------------------------------------------------------------


@pytest.mark.parametrize("neutral", [False, True])
@pytest.mark.parametrize("full_cov", [False, True])
def test_matches_closed_form(neutral, full_cov):
    inp = inputs(25, 3, seed=4, constraints=Constraints(dollar_neutral=neutral))
    sigma = inp.risk.covariance()
    if full_cov:
        inp = Inputs(
            inp.alpha,
            FactorRiskModel.from_covariance(sigma),
            inp.w0,
            gamma=inp.gamma,
            constraints=inp.constraints,
        )
    sol = solve(inp)
    assert sol.status == "optimal"
    np.testing.assert_allclose(sol.w, closed_form(sigma, inp.alpha, inp.gamma, neutral), atol=TOL)


def test_factor_form_risk_equals_dense_quadratic_form():
    inp = inputs(30, 4, seed=1)
    w = np.random.default_rng(0).normal(size=30)
    factor, idio = risk_terms(inp, w)
    assert factor + idio == pytest.approx(w @ inp.risk.covariance() @ w, rel=1e-12)


# --- Comparative statics (DERIVATIONS.md 2.5) ------------------------------------

BOOK = Constraints(max_position=0.05, max_gross=1.0)
COSTS = CostModel(linear=5e-4, impact=2e-3, impact_power=1.5)


def test_risk_does_not_increase_with_gamma():
    w0 = neutral_w0(20, 0, 0.03)
    risks = []
    for g in (2.0, 5.0, 10.0, 20.0, 40.0):
        inp = inputs(20, gamma=g, w0=w0, costs=COSTS, constraints=BOOK)
        sol = solve(inp)
        assert sol.status in OK
        risks.append(sum(risk_terms(inp, sol.w)))
    assert all(b <= a + 1e-10 for a, b in zip(risks, risks[1:], strict=False))
    assert risks[-1] < risks[0]


@pytest.mark.parametrize("term", ["linear", "impact", "turnover"])
def test_cost_term_does_not_increase_with_its_coefficient(term):
    w0 = neutral_w0(20, 1, 0.03)
    values = []
    for scale in (0.0, 0.5, 1.0, 2.0, 4.0):
        kw = {"linear": 5e-4, "impact": 2e-3}
        tp = 0.0
        if term == "turnover":
            tp = 5e-4 * scale
        else:
            kw[term] *= scale
        inp = inputs(
            20, gamma=10.0, w0=w0, costs=CostModel(**kw), constraints=BOOK, turnover_penalty=tp
        )
        sol = solve(inp)
        assert sol.status in OK
        # Compare the *unscaled* term, i.e. the quantity the coefficient multiplies.
        raw = cost_terms(inp, sol.w)[term]
        values.append(raw / scale if scale else np.inf)
    finite = values[1:]
    assert all(b <= a * (1 + 1e-6) + 1e-12 for a, b in zip(finite, finite[1:], strict=False))


# --- L1 turnover and linear cost (D3, DERIVATIONS.md 2.4) --------------------------


def test_l1_turnover_merges_with_linear_cost():
    w0 = neutral_w0(15, 2, 0.04)
    a = solve(
        inputs(15, w0=w0, constraints=BOOK, turnover_penalty=3e-4, costs=CostModel(linear=2e-4))
    )
    b = solve(inputs(15, w0=w0, constraints=BOOK, costs=CostModel(linear=5e-4)))
    np.testing.assert_allclose(a.w, b.w, atol=TOL)


def test_large_linear_cost_gives_no_trade():
    # If c_i exceeds |gradient of the smooth part at w0| for every i, w0 is optimal.
    inp0 = inputs(15, seed=3, constraints=Constraints(dollar_neutral=False))
    w0 = neutral_w0(15, 3, 0.02)
    sigma = inp0.risk.covariance()
    grad = inp0.alpha - 2 * inp0.gamma * sigma @ w0
    c = np.abs(grad) * 1.5
    sol = solve(
        inputs(
            15,
            seed=3,
            w0=w0,
            costs=CostModel(linear=c),
            constraints=Constraints(dollar_neutral=False),
        )
    )
    np.testing.assert_allclose(sol.w, w0, atol=TOL)
    # Just below the threshold for one asset, that asset trades.
    c2 = c.copy()
    i = int(np.argmax(np.abs(grad)))
    c2[i] = np.abs(grad[i]) * 0.5
    sol2 = solve(
        inputs(
            15,
            seed=3,
            w0=w0,
            costs=CostModel(linear=c2),
            constraints=Constraints(dollar_neutral=False),
        )
    )
    assert abs(sol2.w[i] - w0[i]) > 1e-4


# --- Solvers ---------------------------------------------------------------------


def test_osqp_agrees_on_the_qp_form():
    inp = inputs(
        20,
        w0=neutral_w0(20, 5, 0.03),
        constraints=BOOK,
        costs=CostModel(linear=5e-4, impact=2e-2, impact_power=2.0),
    )
    assert inp.is_qp()
    a, b = solve(inp, "CLARABEL"), solve(inp, "OSQP")
    assert a.status == b.status == "optimal"
    np.testing.assert_allclose(a.w, b.w, atol=1e-4)


def test_osqp_cannot_take_the_three_halves_form_and_solve_does_not_raise():
    inp = inputs(10, costs=COSTS, constraints=BOOK)
    assert not inp.is_qp()
    sol = solve(inp, "OSQP")
    assert sol.status == "error"
    assert sol.w is None
    assert "SolverError" in sol.error


def test_scs_is_close_on_the_three_halves_form():
    inp = inputs(20, w0=neutral_w0(20, 6, 0.03), costs=COSTS, constraints=BOOK)
    a, b = solve(inp, "CLARABEL"), solve(inp, "SCS")
    assert a.status == "optimal"
    assert b.status in OK
    np.testing.assert_allclose(a.w, b.w, atol=1e-3)


# --- Property test: constraints hold on feasible inputs (D7 tolerance) ------------


@given(
    n=st.integers(3, 30),
    k=st.integers(1, 4),
    seed=st.integers(0, 5000),
    pos_cap=st.floats(0.02, 0.2),
    trade_frac=st.floats(0.1, 2.0),
    adv_frac=st.floats(0.5, 3.0),
    gross=st.floats(0.5, 3.0),
    power=st.sampled_from([1.5, 2.0]),
    gamma=st.floats(0.5, 50.0),
)
def test_constraints_hold_on_feasible_inputs(
    n, k, seed, pos_cap, trade_frac, adv_frac, gross, power, gamma
):
    # w0 is chosen to satisfy every constraint, so w = w0 is feasible.
    w0 = neutral_w0(n, seed, 0.9 * pos_cap * min(1.0, adv_frac))
    w0 *= min(1.0, 0.9 * gross / np.abs(w0).sum())
    cons = Constraints(
        max_position=pos_cap,
        max_trade=pos_cap * trade_frac,
        max_position_adv=pos_cap * adv_frac,
        max_gross=gross,
    )
    costs = CostModel(linear=3e-4, impact=1e-3, impact_power=power)
    inp = inputs(n, k, seed, gamma=gamma, w0=w0, costs=costs, constraints=cons)
    sol = solve(inp)
    assert sol.status in OK
    viol = constraint_violations(inp, sol.w)
    assert set(viol) == {
        "dollar_neutral",
        "max_position",
        "max_trade",
        "max_position_adv",
        "max_gross",
    }
    assert max(viol.values()) <= TOL, viol


# --- Inputs and helpers ------------------------------------------------------------


def test_input_validation():
    rm = risk_model(5)
    with pytest.raises(ValueError, match="shape"):
        Inputs(np.zeros(4), rm, np.zeros(5))
    with pytest.raises(ValueError, match="gamma"):
        Inputs(np.zeros(5), rm, np.zeros(5), gamma=0.0)
    with pytest.raises(ValueError, match="impact_power"):
        CostModel(impact_power=1.6)


def test_helpers():
    k = square_root_impact(np.array([0.02, 0.02]), np.array([1.0, 4.0]), coef=0.5)
    np.testing.assert_allclose(k, [0.01, 0.005])
    np.testing.assert_allclose(adv_cap(np.array([1.0, 0.2]), 0.1), [0.1, 0.02])
    with pytest.raises(ValueError):
        adv_cap(np.ones(2), 0.0)


def test_violations_measure_the_right_thing():
    inp = inputs(
        4,
        w0=np.zeros(4),
        constraints=Constraints(
            max_position=0.1, max_trade=0.05, max_position_adv=0.2, max_gross=0.3
        ),
    )
    w = np.array([0.15, -0.05, 0.0, 0.0])
    v = constraint_violations(inp, w)
    assert v["dollar_neutral"] == pytest.approx(0.1)
    assert v["max_position"] == pytest.approx(0.05)
    assert v["max_trade"] == pytest.approx(0.10)
    assert v["max_position_adv"] == 0.0
    assert v["max_gross"] == 0.0


def test_inaccurate_status_is_reported_not_raised():
    # The case found while writing these tests: CLARABEL returns AlmostSolved.
    inp = inputs(20, gamma=10.0, w0=neutral_w0(20, 0, 0.03), costs=COSTS, constraints=BOOK)
    sol = solve(inp)
    assert sol.status == "optimal_inaccurate"
    assert any("inaccurate" in m for m in sol.warnings)
    assert max(constraint_violations(inp, sol.w).values()) <= TOL
