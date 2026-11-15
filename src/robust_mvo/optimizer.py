"""Single-period mean-variance problem for a long-short book, built in cvxpy.

    maximize    alpha'w
              - gamma * ( ||L_F' B' w||^2 + ||sqrt(D) * w||^2 )     factor + idio risk
              - lambda_to * ||z||_1                                    turnover penalty
              - sum_i c_i |z_i|                                        linear cost
              - sum_i k_i |z_i|^p          (p = 1.5 or 2)              impact cost
    subject to  1'w = 0                                                dollar neutral
                |w_i| <= max_position_i                                position caps
                |z_i| <= max_trade_i                                   liquidity: trade size
                |w_i| <= max_position_adv_i                            liquidity: position size
                ||w||_1 <= max_gross                                   gross leverage
    with z = w - w0.

Weights are fractions of NAV. Every constraint is optional. The algebra for
each term is in DERIVATIONS.md, part 2.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Literal

import cvxpy as cp
import numpy as np

from robust_mvo.risk_model import FactorRiskModel

ArrayLike = float | np.ndarray


@dataclass(frozen=True)
class CostModel:
    """Separable transaction costs per asset, in units of NAV.

    cost_i(z) = linear_i * |z| + impact_i * |z| ** impact_power

    D1: the default power is 1.5 (square-root impact law); 2.0 (linear impact,
    Almgren-Chriss) keeps the whole problem a QP. Use ``square_root_impact``
    to build ``impact`` from volatility and liquidity.
    """

    linear: ArrayLike = 0.0
    impact: ArrayLike = 0.0
    impact_power: Literal[1.5, 2.0] = 1.5

    def __post_init__(self) -> None:
        if self.impact_power not in (1.5, 2.0):
            raise ValueError("impact_power must be 1.5 or 2.0")


def square_root_impact(sigma: np.ndarray, adv_frac: np.ndarray, coef: float = 1.0) -> np.ndarray:
    """Coefficient k_i = coef * sigma_i / sqrt(adv_frac_i) for the 3/2-power term.

    With impact per unit traded ~ coef * sigma * sqrt(|z| / adv_frac), the cost
    of trading |z| is |z| times that, i.e. k_i * |z|^{3/2}. ``adv_frac`` is
    average daily volume as a fraction of NAV, and ``sigma`` is daily volatility.
    """
    return coef * np.asarray(sigma, dtype=float) / np.sqrt(np.asarray(adv_frac, dtype=float))


def adv_cap(adv_frac: np.ndarray, participation: float) -> np.ndarray:
    """Per-asset cap participation * ADV / NAV, used for either liquidity limit (D4)."""
    if participation <= 0:
        raise ValueError("participation must be positive")
    return participation * np.asarray(adv_frac, dtype=float)


@dataclass(frozen=True)
class Constraints:
    dollar_neutral: bool = True
    max_position: ArrayLike | None = None
    max_trade: ArrayLike | None = None
    max_position_adv: ArrayLike | None = None
    max_gross: float | None = None


@dataclass(frozen=True)
class Inputs:
    alpha: np.ndarray
    risk: FactorRiskModel
    w0: np.ndarray
    gamma: float = 1.0
    turnover_penalty: float = 0.0
    costs: CostModel = field(default_factory=CostModel)
    constraints: Constraints = field(default_factory=Constraints)

    def __post_init__(self) -> None:
        a = np.asarray(self.alpha, dtype=float)
        w0 = np.asarray(self.w0, dtype=float)
        n = self.risk.n_assets
        if a.shape != (n,) or w0.shape != (n,):
            raise ValueError(f"alpha and w0 must have shape ({n},), got {a.shape} and {w0.shape}")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.turnover_penalty < 0:
            raise ValueError("turnover_penalty must be non-negative")
        object.__setattr__(self, "alpha", a)
        object.__setattr__(self, "w0", w0)

    @property
    def n_assets(self) -> int:
        return self.risk.n_assets

    def is_qp(self) -> bool:
        """True if every term is linear or quadratic (so OSQP is a valid solver)."""
        has_impact = np.any(np.asarray(self.costs.impact) != 0)
        return not has_impact or self.costs.impact_power == 2.0


@dataclass(frozen=True)
class Solution:
    w: np.ndarray | None
    status: str
    solver: str
    objective: float | None
    solve_time: float
    error: str | None = None
    warnings: tuple[str, ...] = ()


def _vec(x: ArrayLike, n: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(x, dtype=float), (n,)).copy()


def build_problem(
    inp: Inputs, *, chol_f: np.ndarray | None = None
) -> tuple[cp.Problem, cp.Variable]:
    """Build the cvxpy problem. ``chol_f`` is the lower Cholesky factor of F
    (computed from the risk model if not given). Returns (problem, w)."""
    n = inp.n_assets
    if chol_f is None:
        chol_f, _ = inp.risk.factor_cholesky()
    w = cp.Variable(n, name="w")
    z = w - inp.w0
    b = inp.risk.exposures
    f = b.T @ w  # K-vector of factor exposures; the N x N matrix is never formed
    idio_sd = np.sqrt(np.maximum(inp.risk.idio_var, 0.0))

    risk = cp.sum_squares(chol_f.T @ f) + cp.sum_squares(cp.multiply(idio_sd, w))
    obj = inp.alpha @ w - inp.gamma * risk
    if inp.turnover_penalty:
        obj -= inp.turnover_penalty * cp.norm1(z)
    c = _vec(inp.costs.linear, n)
    if np.any(c):
        obj -= c @ cp.abs(z)
    k = _vec(inp.costs.impact, n)
    if np.any(k):
        if inp.costs.impact_power == 2.0:
            obj -= k @ cp.square(z)
        else:
            obj -= k @ cp.power(cp.abs(z), 1.5)

    cs = inp.constraints
    cons: list[cp.Constraint] = []
    if cs.dollar_neutral:
        cons.append(cp.sum(w) == 0)
    for group, expr in (("max_position", w), ("max_trade", z), ("max_position_adv", w)):
        cap = getattr(cs, group)
        if cap is not None:
            cons.append(cp.abs(expr) <= _vec(cap, n))
    if cs.max_gross is not None:
        cons.append(cp.norm1(w) <= cs.max_gross)

    # Rescale the objective so its terms are O(1). Daily variances are ~1e-4, so
    # unscaled objective values sit near the solvers' absolute tolerances and the
    # returned weights lose accuracy (up to 3.5e-5 in DECISIONS.md, above the
    # 1e-6 tolerance; DERIVATIONS.md 2.6). A positive constant does not change
    # the argmax.
    return cp.Problem(cp.Maximize(objective_scale(inp) * obj), cons), w


def objective_scale(inp: Inputs) -> float:
    """1 / (average asset variance), computed in O(N K^2) without forming Sigma."""
    b = inp.risk.exposures
    diag = np.sum((b @ inp.risk.factor_cov) * b, axis=1) + inp.risk.idio_var
    avg = float(np.mean(diag[np.isfinite(diag)])) if np.isfinite(diag).any() else 1.0
    return 1.0 / avg if avg > 0 else 1.0


def solve(inp: Inputs, solver: str = "CLARABEL", **build_kwargs) -> Solution:
    """Solve once with one solver.

    Never raises on solver failure: errors become status "error", and solver
    warnings (e.g. cvxpy's "Solution may be inaccurate") are captured in
    ``Solution.warnings`` instead of being emitted. Whether an inaccurate
    solution is acceptable is decided by the caller (D2).
    """
    problem, w = build_problem(inp, **build_kwargs)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            problem.solve(solver=solver)
        except (cp.SolverError, ValueError, ArithmeticError) as exc:
            return Solution(
                None,
                "error",
                solver,
                None,
                time.perf_counter() - t0,
                repr(exc),
                tuple(str(c.message) for c in caught),
            )
    elapsed = time.perf_counter() - t0
    value = None if w.value is None else np.asarray(w.value, dtype=float)
    obj = None if problem.value is None else float(problem.value)
    return Solution(
        value,
        str(problem.status),
        solver,
        obj,
        elapsed,
        warnings=tuple(str(c.message) for c in caught),
    )


def risk_terms(inp: Inputs, w: np.ndarray) -> tuple[float, float]:
    """(factor variance, idiosyncratic variance) of w, computed without cvxpy."""
    f = inp.risk.exposures.T @ w
    return float(f @ inp.risk.factor_cov @ f), float(np.sum(inp.risk.idio_var * w * w))


def cost_terms(inp: Inputs, w: np.ndarray) -> dict[str, float]:
    """Value of each penalty term of the objective at w."""
    n = inp.n_assets
    z = np.abs(w - inp.w0)
    return {
        "turnover": float(inp.turnover_penalty * z.sum()),
        "linear": float(_vec(inp.costs.linear, n) @ z),
        "impact": float(_vec(inp.costs.impact, n) @ z**inp.costs.impact_power),
    }


def constraint_violations(inp: Inputs, w: np.ndarray) -> dict[str, float]:
    """Largest violation of each active constraint group at w (0.0 if satisfied)."""
    n = inp.n_assets
    cs = inp.constraints
    z = w - inp.w0
    out: dict[str, float] = {}
    if cs.dollar_neutral:
        out["dollar_neutral"] = float(abs(w.sum()))
    for group, expr in (("max_position", w), ("max_trade", z), ("max_position_adv", w)):
        cap = getattr(cs, group)
        if cap is not None:
            out[group] = float(max(0.0, np.max(np.abs(expr) - _vec(cap, n))))
    if cs.max_gross is not None:
        out["max_gross"] = float(max(0.0, np.abs(w).sum() - cs.max_gross))
    return out
