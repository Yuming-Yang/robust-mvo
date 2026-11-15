"""Seeded synthetic markets with a known true covariance.

Returns follow a linear factor model  r = B f + e  with f ~ N(0, F) and
e ~ N(0, diag(D)), so the true covariance is  Sigma = B F B' + diag(D).
Knowing Sigma exactly is what lets the estimators and the optimizer be
scored against ground truth instead of against each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorMarket:
    """A factor-structured market. All quantities are per period (e.g. daily)."""

    exposures: np.ndarray  # B, shape (N, K)
    factor_cov: np.ndarray  # F, shape (K, K)
    idio_var: np.ndarray  # D, shape (N,)

    @property
    def n_assets(self) -> int:
        return self.exposures.shape[0]

    @property
    def n_factors(self) -> int:
        return self.exposures.shape[1]

    @property
    def covariance(self) -> np.ndarray:
        """The true covariance  B F B' + diag(D)."""
        b = self.exposures
        return b @ self.factor_cov @ b.T + np.diag(self.idio_var)


def make_factor_market(
    n_assets: int,
    n_factors: int = 3,
    *,
    seed: int,
    market_vol: float = 0.01,
    style_vol: float = 0.005,
    idio_vol_range: tuple[float, float] = (0.01, 0.025),
) -> FactorMarket:
    """Build a market with one strong 'market' factor and weaker style factors.

    The first factor has exposures near 1 for every asset, which produces one
    large eigenvalue well above the noise band, as in real equity correlation
    matrices. The remaining factors have exposures centred on zero.
    """
    if n_assets < 1 or n_factors < 1:
        raise ValueError("n_assets and n_factors must be positive")
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 1.0, size=(n_assets, n_factors))
    b[:, 0] = rng.normal(1.0, 0.25, size=n_assets)
    vols = np.full(n_factors, style_vol)
    vols[0] = market_vol
    # Mildly correlated factors, so F is a full matrix and not just diagonal.
    a = rng.normal(0.0, 0.3, size=(n_factors, n_factors))
    corr = a @ a.T + np.eye(n_factors)
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    factor_cov = corr * np.outer(vols, vols)
    idio_vol = rng.uniform(*idio_vol_range, size=n_assets)
    return FactorMarket(exposures=b, factor_cov=factor_cov, idio_var=idio_vol**2)


def simulate_returns(market: FactorMarket, n_obs: int, *, seed: int) -> np.ndarray:
    """Draw an (n_obs, N) panel of Gaussian returns with mean zero."""
    if n_obs < 1:
        raise ValueError("n_obs must be positive")
    rng = np.random.default_rng(seed)
    chol_f = np.linalg.cholesky(market.factor_cov)
    f = rng.standard_normal((n_obs, market.n_factors)) @ chol_f.T
    e = rng.standard_normal((n_obs, market.n_assets)) * np.sqrt(market.idio_var)
    return f @ market.exposures.T + e
