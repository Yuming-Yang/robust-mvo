"""Factor risk model  Sigma = B F B' + diag(D).

The optimizer never forms Sigma. It needs a factor L_F with F = L_F L_F',
so that  w' B F B' w = ||L_F' (B' w)||^2  (DERIVATIONS.md, part 2). That
factorization fails if F is not positive definite, which is why this module
validates inputs and repairs F explicitly, and records when it did.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robust_mvo.covariance import psd_floor


@dataclass(frozen=True)
class FactorRiskModel:
    exposures: np.ndarray  # B, (N, K)
    factor_cov: np.ndarray  # F, (K, K)
    idio_var: np.ndarray  # D, (N,)

    def __post_init__(self) -> None:
        b = np.asarray(self.exposures, dtype=float)
        f = np.asarray(self.factor_cov, dtype=float)
        d = np.asarray(self.idio_var, dtype=float)
        if b.ndim != 2:
            raise ValueError(f"exposures must be 2-D (N, K), got shape {b.shape}")
        n, k = b.shape
        if f.shape != (k, k):
            raise ValueError(f"factor_cov must be ({k}, {k}), got {f.shape}")
        if d.shape != (n,):
            raise ValueError(f"idio_var must be ({n},), got {d.shape}")
        if not np.isfinite(f).all():
            raise ValueError("factor_cov contains NaN or inf")
        object.__setattr__(self, "exposures", b)
        object.__setattr__(self, "factor_cov", f)
        object.__setattr__(self, "idio_var", d)

    @classmethod
    def from_covariance(cls, sigma: np.ndarray) -> FactorRiskModel:
        """Full-covariance mode: B = I, F = Sigma, D = 0 (so K = N)."""
        s = np.asarray(sigma, dtype=float)
        n = s.shape[0]
        return cls(exposures=np.eye(n), factor_cov=s, idio_var=np.zeros(n))

    @property
    def n_assets(self) -> int:
        return self.exposures.shape[0]

    @property
    def n_factors(self) -> int:
        return self.exposures.shape[1]

    def covariance(self) -> np.ndarray:
        """Dense Sigma. For tests and diagnostics only; the optimizer avoids it."""
        b = self.exposures
        return b @ self.factor_cov @ b.T + np.diag(self.idio_var)

    def factor_cholesky(self) -> tuple[np.ndarray, int]:
        """Lower-triangular L with F ~= L L', plus the number of repaired eigenvalues.

        If F is already positive definite, L is its exact Cholesky factor and
        the repair count is 0. Otherwise F is passed through psd_floor first.
        """
        try:
            return np.linalg.cholesky(self.factor_cov), 0
        except np.linalg.LinAlgError:
            repaired, n_fixed = psd_floor(self.factor_cov)
            return np.linalg.cholesky(repaired), n_fixed

    def bad_assets(self) -> np.ndarray:
        """Boolean mask of assets with non-finite exposures or idiosyncratic variance,
        or negative idiosyncratic variance."""
        bad_b = ~np.isfinite(self.exposures).all(axis=1)
        bad_d = ~np.isfinite(self.idio_var) | (self.idio_var < 0)
        return bad_b | bad_d
