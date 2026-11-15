"""The degradation chain: one call that always returns weights and a report.

    level 0  full problem, primary solver
    level 1  same problem, fallback solvers (OSQP only if the problem is a QP)
    level 2  elastic problem: caps become penalized slacks, relaxed in the
             order liquidity -> position caps -> gross; dollar neutrality is
             never relaxed (D2)
    level 3  hold the previous weights w0 (no trade)

A solve is accepted only if the solver returned weights, its status is
"optimal" (or "optimal_inaccurate" when ``accept_inaccurate`` is on), and the
post-check finds every hard constraint satisfied within ``tol``. A proved
infeasibility skips level 1, since another solver cannot fix it.

Bad market data never raises. Assets with non-finite alpha, exposures,
idiosyncratic variance or caps are frozen at w0 (D5) and listed in the
report. Only programmer errors raise: wrong shapes, or a non-finite w0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from robust_mvo.optimizer import (
    Constraints,
    Inputs,
    Solution,
    constraint_violations,
    solve,
)
from robust_mvo.risk_model import FactorRiskModel

SolveFn = Callable[..., Solution]

# D2 relaxation order. Each stage adds groups to the set relaxed before it.
RELAX_STAGES: tuple[tuple[str, ...], ...] = (
    ("max_trade", "max_position_adv"),
    ("max_position",),
    ("max_gross",),
)
INFEASIBLE = ("infeasible", "infeasible_inaccurate")


@dataclass(frozen=True)
class Config:
    solvers: tuple[str, ...] = ("CLARABEL", "SCS", "OSQP")
    tol: float = 1e-6  # D7
    accept_inaccurate: bool = True  # D2
    relax_penalty: float = 1.0  # per unit of weight violation, in return units
    precheck: bool = True


@dataclass(frozen=True)
class Attempt:
    level: int
    solver: str
    status: str
    accepted: bool
    note: str = ""
    solve_time: float = 0.0
    relaxed: tuple[str, ...] = ()


@dataclass
class SolveReport:
    level: int = 0
    reasons: list[str] = field(default_factory=list)  # one per step down
    attempts: list[Attempt] = field(default_factory=list)
    frozen: dict[int, str] = field(default_factory=dict)  # asset -> why
    factor_cov_repaired: int = 0  # eigenvalues of F raised by psd_floor
    relaxed: tuple[str, ...] = ()
    # Measured on the returned w against the constraints actually imposed, i.e.
    # after frozen assets' caps were widened to admit their w0 (D5).
    violations: dict[str, float] = field(default_factory=dict)
    # Frozen assets whose w0 already broke an original per-asset cap: reported,
    # not fixed, since a frozen asset cannot trade.
    frozen_cap_breaches: dict[int, str] = field(default_factory=dict)
    inaccurate: bool = False

    def step_down(self, to_level: int, reason: str) -> None:
        self.reasons.append(f"level {self.level} -> {to_level}: {reason}")
        self.level = to_level


# --- input sanitation (D5) ---------------------------------------------------------


def _cap_vector(cap, n: int) -> np.ndarray | None:
    if cap is None:
        return None
    return np.broadcast_to(np.asarray(cap, dtype=float), (n,)).copy()


def sanitize(inp: Inputs) -> tuple[Inputs, dict[int, str], dict[int, str]]:
    """Freeze assets with unusable data and replace their values with harmless ones.

    A frozen asset keeps w_i = w0_i, so its alpha and caps no longer matter;
    they are replaced so the solver never sees NaN. Its risk still counts when
    the risk data is finite. If exposures or idiosyncratic variance are
    unusable, they are set to zero and the report says so, because the
    asset's risk contribution is then unknown, not zero.
    """
    n = inp.n_assets
    w0 = inp.w0
    if not np.isfinite(w0).all():
        raise ValueError("w0 contains NaN or inf; the current book must be known")
    why: dict[int, list[str]] = {}

    def flag(mask: np.ndarray, reason: str) -> None:
        for i in np.flatnonzero(mask):
            why.setdefault(int(i), []).append(reason)

    alpha = inp.alpha.copy()
    bad = ~np.isfinite(alpha)
    flag(bad, "alpha not finite")
    alpha[bad] = 0.0

    b = inp.risk.exposures.copy()
    d = inp.risk.idio_var.copy()
    bad_b = ~np.isfinite(b).all(axis=1)
    bad_d = ~np.isfinite(d) | (d < 0)
    flag(bad_b, "exposures not finite (risk contribution treated as zero)")
    flag(bad_d, "idiosyncratic variance invalid (treated as zero)")
    b[bad_b] = 0.0
    d[bad_d] = 0.0

    cs = inp.constraints
    caps = {}
    for group in ("max_position", "max_trade", "max_position_adv"):
        v = _cap_vector(getattr(cs, group), n)
        if v is not None:
            bad_v = ~np.isfinite(v) | (v < 0)
            flag(bad_v, f"{group} not finite or negative")
            # Set the cap to a value the frozen w0_i satisfies.
            v[bad_v] = 0.0 if group == "max_trade" else np.abs(w0[bad_v])
        caps[group] = v
    frozen = {i: "; ".join(r) for i, r in sorted(why.items())}
    breaches: dict[int, str] = {}
    if frozen:
        idx = list(frozen)
        for group, v in caps.items():
            if v is not None and group != "max_trade":
                over = [i for i in idx if abs(w0[i]) > v[i]]
                for i in over:
                    msg = f"|w0|={abs(w0[i]):.4g} > {group} {v[i]:.4g}"
                    breaches[i] = f"{breaches[i]}; {msg}" if i in breaches else msg
                v[idx] = np.maximum(v[idx], np.abs(w0[idx]))
    new_cs = Constraints(
        dollar_neutral=cs.dollar_neutral,
        max_position=caps["max_position"],
        max_trade=caps["max_trade"],
        max_position_adv=caps["max_position_adv"],
        max_gross=cs.max_gross,
    )
    risk = FactorRiskModel(b, inp.risk.factor_cov, d)
    return replace(inp, alpha=alpha, risk=risk, constraints=new_cs), frozen, breaches


# --- cheap feasibility pre-check -------------------------------------------------


def precheck(inp: Inputs, frozen: np.ndarray) -> str | None:
    """Necessary conditions for feasibility, from per-asset intervals.

    Each asset can reach  [lo_i, hi_i] = [w0_i - tau_i, w0_i + tau_i]
    intersected with its position caps (a point if frozen). Returns a reason
    string if an interval is empty, if 0 is outside [sum lo, sum hi] under
    dollar neutrality, or if the smallest reachable gross exceeds the cap.
    Passing the pre-check does not guarantee feasibility; the solver decides.
    """
    n = inp.n_assets
    cs = inp.constraints
    w0 = inp.w0
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    tau = _cap_vector(cs.max_trade, n)
    if tau is not None:
        lo, hi = w0 - tau, w0 + tau
    for group in ("max_position", "max_position_adv"):
        u = _cap_vector(getattr(cs, group), n)
        if u is not None:
            lo, hi = np.maximum(lo, -u), np.minimum(hi, u)
    lo = np.where(frozen, w0, lo)
    hi = np.where(frozen, w0, hi)
    empty = np.flatnonzero(lo > hi + 1e-12)
    if empty.size:
        return f"assets {empty.tolist()[:5]} cannot reach their caps within the trade limit"
    if cs.dollar_neutral and not (lo.sum() <= 1e-12 and hi.sum() >= -1e-12):
        return (
            f"dollar neutrality unreachable: reachable net exposure is "
            f"[{lo.sum():.4g}, {hi.sum():.4g}]"
        )
    if cs.max_gross is not None:
        min_gross = np.sum(np.maximum(0.0, np.maximum(lo, -hi)))
        if min_gross > cs.max_gross + 1e-12:
            return f"smallest reachable gross {min_gross:.4g} exceeds cap {cs.max_gross:.4g}"
    return None


# --- the chain ---------------------------------------------------------------------


def _hard_violation(viol: dict[str, float], relaxed: tuple[str, ...]) -> float:
    return max((v for g, v in viol.items() if g not in relaxed), default=0.0)


def _try(
    inp: Inputs,
    solver: str,
    level: int,
    report: SolveReport,
    cfg: Config,
    solve_fn: SolveFn,
    relaxed: tuple[str, ...] = (),
    **kwargs,
) -> tuple[np.ndarray | None, str]:
    """One solve plus the acceptance test. Returns (weights or None, status)."""
    try:
        sol = solve_fn(inp, solver, **kwargs)
    except Exception as exc:  # a fallback chain must survive any solver failure
        report.attempts.append(Attempt(level, solver, "error", False, repr(exc), 0.0, relaxed))
        return None, "error"
    status = sol.status
    note = sol.error or ""
    ok_status = status == "optimal" or (status == "optimal_inaccurate" and cfg.accept_inaccurate)
    w = sol.w
    if ok_status and (w is None or not np.isfinite(w).all()):
        ok_status, note = False, "solver returned no finite weights"
    if ok_status:
        worst = _hard_violation(constraint_violations(inp, w), relaxed)
        if worst > cfg.tol:
            ok_status, note = False, f"post-check: violation {worst:.2e} > tol {cfg.tol:.0e}"
    report.attempts.append(Attempt(level, solver, status, ok_status, note, sol.solve_time, relaxed))
    if ok_status:
        report.inaccurate = status == "optimal_inaccurate"
        return w, status
    return None, status


def robust_optimize(
    inp: Inputs, config: Config | None = None, *, solve_fn: SolveFn = solve
) -> tuple[np.ndarray, SolveReport]:
    """Run the degradation chain. Always returns (weights, report)."""
    cfg = config or Config()
    report = SolveReport()

    imposed = inp  # replaced by the sanitized inputs once they exist

    def finish(w: np.ndarray) -> tuple[np.ndarray, SolveReport]:
        report.violations = constraint_violations(imposed, w)
        return w, report

    def hold(reason: str) -> tuple[np.ndarray, SolveReport]:
        report.step_down(3, reason)
        return finish(inp.w0.copy())

    if not np.isfinite(inp.w0).all():
        raise ValueError("w0 contains NaN or inf; the current book must be known")
    if not np.isfinite(inp.risk.factor_cov).all():
        return hold("factor covariance not finite")
    clean, frozen_reasons, report.frozen_cap_breaches = sanitize(inp)
    imposed = clean
    report.frozen = frozen_reasons
    frozen = np.zeros(inp.n_assets, dtype=bool)
    frozen[list(frozen_reasons)] = True

    if frozen.all():
        return hold("every asset is frozen")
    chol, report.factor_cov_repaired = clean.risk.factor_cholesky()
    kw = {"chol_f": chol, "frozen": frozen}

    solvers = [s for s in cfg.solvers if s != "OSQP" or clean.is_qp()]
    skipped = [s for s in cfg.solvers if s not in solvers]

    reason = precheck(clean, frozen) if cfg.precheck else None
    if reason is not None:
        report.step_down(2, f"pre-check: {reason}")
    else:
        # Levels 0 and 1.
        for i, solver in enumerate(solvers):
            w, status = _try(clean, solver, report.level, report, cfg, solve_fn, **kw)
            if w is not None:
                return finish(w)
            if status in INFEASIBLE:
                report.step_down(2, f"{solver} reports {status}; other solvers cannot fix that")
                break
            if i + 1 < len(solvers):
                if report.level == 0:
                    note = f"; skipped {skipped} (not a QP)" if skipped else ""
                    report.step_down(1, f"{solver} returned {status}{note}")
            else:
                report.step_down(2, f"all solvers failed (last: {solver} {status})")

    # Level 2: relax caps in D2 order.
    relaxed: tuple[str, ...] = ()
    tried: tuple[str, ...] = ()
    for stage in RELAX_STAGES:
        relaxed = relaxed + stage
        active = tuple(g for g in relaxed if getattr(clean.constraints, g) is not None)
        if not active or active == tried:
            continue  # this stage adds no constraint that is actually present
        tried = active
        elastic = dict.fromkeys(active, cfg.relax_penalty)
        for solver in solvers:
            w, _ = _try(
                clean, solver, 2, report, cfg, solve_fn, relaxed=active, elastic=elastic, **kw
            )
            if w is not None:
                report.relaxed = active
                return finish(w)
    return hold("relaxed problem also failed")
