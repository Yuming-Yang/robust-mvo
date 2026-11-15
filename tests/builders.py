"""Small, seeded input builders shared by the optimizer tests."""

from __future__ import annotations

import numpy as np

from robust_mvo.optimizer import Constraints, CostModel, Inputs
from robust_mvo.risk_model import FactorRiskModel
from robust_mvo.synthetic import make_factor_market


def risk_model(n: int, k: int = 3, seed: int = 0) -> FactorRiskModel:
    m = make_factor_market(n, k, seed=seed)
    return FactorRiskModel(m.exposures, m.factor_cov, m.idio_var)


def alpha(n: int, seed: int = 0, scale: float = 5e-4) -> np.ndarray:
    return np.random.default_rng(seed + 1000).normal(0.0, scale, n)


def neutral_w0(n: int, seed: int, cap: float) -> np.ndarray:
    """A dollar-neutral starting book with every |w0_i| <= cap."""
    w = np.random.default_rng(seed + 2000).normal(0.0, 1.0, n)
    w -= w.mean()
    return w * (cap / np.max(np.abs(w)))


def inputs(
    n: int = 20,
    k: int = 3,
    seed: int = 0,
    *,
    gamma: float = 20.0,
    w0: np.ndarray | None = None,
    costs: CostModel | None = None,
    constraints: Constraints | None = None,
    turnover_penalty: float = 0.0,
) -> Inputs:
    return Inputs(
        alpha=alpha(n, seed),
        risk=risk_model(n, k, seed),
        w0=np.zeros(n) if w0 is None else w0,
        gamma=gamma,
        turnover_penalty=turnover_penalty,
        costs=costs or CostModel(),
        constraints=constraints or Constraints(),
    )
