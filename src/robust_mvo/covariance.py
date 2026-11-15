"""Covariance estimation from a return panel, with two stabilizers.

Every estimator takes a (T, N) panel of returns, demeans it, and uses the
maximum-likelihood normalization 1/T (the same as scikit-learn), so all three
estimators are built from the same sample covariance S.

* ``sample``      : S itself. Singular when N >= T; kept as the baseline.
* ``ledoit_wolf`` : linear shrinkage toward a scaled identity
                    (Ledoit & Wolf, 2004).
* ``clip``        : eigenvalue clipping of the correlation matrix at the
                    Marchenko-Pastur edge (Laloux et al., 1999).

``prepare_returns`` handles missing data before estimation, and ``psd_floor``
is a numerical repair for matrices that should be PSD but are not. The
derivations are in DERIVATIONS.md, part 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Method = Literal["sample", "ledoit_wolf", "clip"]


@dataclass(frozen=True)
class CleanReturns:
    """A complete return panel plus a record of what was removed to get it."""

    returns: np.ndarray  # (T', N'), no NaN
    assets: np.ndarray  # column indices of the input that were kept
    dropped_assets: dict[int, str]  # column index -> reason
    n_rows_dropped: int


@dataclass(frozen=True)
class CovEstimate:
    matrix: np.ndarray
    method: str
    assets: np.ndarray  # columns of the original panel this matrix covers
    dropped_assets: dict[int, str] = field(default_factory=dict)
    diagnostics: dict[str, float | int] = field(default_factory=dict)


def condition_number(m: np.ndarray) -> float:
    """Ratio of largest to smallest eigenvalue; inf if the matrix is singular."""
    eig = np.linalg.eigvalsh(m)
    if eig[0] <= 0:
        return float("inf")
    return float(eig[-1] / eig[0])


def prepare_returns(returns: np.ndarray, *, max_missing: float = 0.1) -> CleanReturns:
    """Turn a panel with gaps into a complete panel, recording every removal.

    1. Drop assets whose fraction of missing (NaN or inf) rows exceeds
       ``max_missing``.
    2. Drop every remaining row that still has a gap (listwise deletion).
    3. Drop assets with zero variance over the remaining rows, since their
       correlation is undefined.

    Pairwise-complete covariance is deliberately not offered: each entry is
    estimated on a different subset of rows, so the matrix can be indefinite
    (see tests/test_covariance.py::test_pairwise_complete_can_be_indefinite).
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"returns must be 2-D (T, N), got shape {x.shape}")
    if not 0.0 <= max_missing < 1.0:
        raise ValueError("max_missing must be in [0, 1)")
    bad = ~np.isfinite(x)
    dropped: dict[int, str] = {}
    miss_frac = bad.mean(axis=0) if x.shape[0] else np.ones(x.shape[1])
    keep = miss_frac <= max_missing
    for j in np.flatnonzero(~keep):
        dropped[int(j)] = f"missing fraction {miss_frac[j]:.2f} > {max_missing:.2f}"
    rows = ~bad[:, keep].any(axis=1)
    x_kept = x[np.ix_(rows, keep)]
    cols = np.flatnonzero(keep)
    if x_kept.shape[0] >= 1:
        flat = np.ptp(x_kept, axis=0) == 0
        for j in cols[flat]:
            dropped[int(j)] = "zero variance"
        x_kept = x_kept[:, ~flat]
        cols = cols[~flat]
    return CleanReturns(
        returns=x_kept,
        assets=cols,
        dropped_assets=dropped,
        n_rows_dropped=int((~rows).sum()),
    )


def _check_panel(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"returns must be 2-D (T, N), got shape {x.shape}")
    t, n = x.shape
    if t < 2 or n < 1:
        raise ValueError(f"need at least 2 observations and 1 asset, got T={t}, N={n}")
    if not np.isfinite(x).all():
        raise ValueError("returns contain NaN or inf; call prepare_returns first")
    return x


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """S = X_c' X_c / T, with X_c the demeaned panel."""
    x = _check_panel(returns)
    xc = x - x.mean(axis=0)
    return xc.T @ xc / x.shape[0]


def ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage toward  mu * I.

    Returns (Sigma_LW, delta) with  Sigma_LW = delta * mu * I + (1 - delta) * S.
    Uses the paper's normalized Frobenius norm ||A||^2 = tr(A A') / N.

        m      = tr(S) / N
        d^2    = ||S - m I||^2
        bbar^2 = (1 / T^2) sum_t ||x_t x_t' - S||^2
               = (1 / (T N)) * ( (1/T) sum_t ||x_t||^4 - ||S||_F^2 )
        b^2    = min(bbar^2, d^2)
        delta  = b^2 / d^2
    """
    x = _check_panel(returns)
    t, n = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / t
    m = np.trace(s) / n
    d2 = (np.sum(s * s) - 2 * m * np.trace(s) + n * m * m) / n
    row_sq = np.sum(xc * xc, axis=1)  # ||x_t||^2
    bbar2 = (np.mean(row_sq**2) - np.sum(s * s)) / (t * n)
    b2 = min(bbar2, d2)
    # d2 == 0 means S is already a multiple of I, so any delta gives S back.
    delta = 0.0 if d2 <= 0 else float(b2 / d2)
    sigma = delta * m * np.eye(n) + (1.0 - delta) * s
    return sigma, delta


def mp_upper_edge(n_assets: int, n_obs: int) -> float:
    """Marchenko-Pastur upper edge (1 + sqrt(N/T))^2 for a unit-variance correlation."""
    return (1.0 + np.sqrt(n_assets / n_obs)) ** 2


def clip_eigenvalues(returns: np.ndarray) -> tuple[np.ndarray, int]:
    """Eigenvalue clipping of the sample correlation matrix (Laloux et al., 1999).

    Eigenvalues of the correlation matrix at or below the Marchenko-Pastur
    upper edge are treated as noise and replaced by their mean, which keeps
    the trace equal to N. The cleaned matrix is rescaled to unit diagonal and
    turned back into a covariance with the sample volatilities.

    Returns (Sigma_clip, n_clipped).
    """
    x = _check_panel(returns)
    t, n = x.shape
    s = sample_covariance(x)
    vol = np.sqrt(np.diag(s))
    if np.any(vol <= 0):
        raise ValueError("an asset has zero variance; call prepare_returns first")
    corr = s / np.outer(vol, vol)
    eig, vec = np.linalg.eigh(corr)
    noise = eig <= mp_upper_edge(n, t)
    n_clipped = int(noise.sum())
    if n_clipped:
        # Every zero eigenvalue is below the edge (which is >= 1), so the noise
        # mean is (N - sum of kept eigenvalues) / n_clipped > 0 and the result
        # is positive definite.
        eig = eig.copy()
        eig[noise] = eig[noise].mean()
    cleaned = (vec * eig) @ vec.T
    d = np.sqrt(np.diag(cleaned))
    cleaned = cleaned / np.outer(d, d)
    cleaned = (cleaned + cleaned.T) / 2
    return cleaned * np.outer(vol, vol), n_clipped


def psd_floor(m: np.ndarray, *, rel_floor: float = 1e-10) -> tuple[np.ndarray, int]:
    """Symmetrize and raise eigenvalues to at least rel_floor * max eigenvalue.

    This is a numerical repair, not an estimator: it fixes round-off and
    small indefiniteness without pretending to remove noise. Returns the
    repaired matrix and the number of eigenvalues that were raised; a matrix
    that needs no repair is returned unchanged (after symmetrization).
    """
    a = np.asarray(m, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("matrix contains NaN or inf")
    a = (a + a.T) / 2
    eig, vec = np.linalg.eigh(a)
    top = max(float(eig[-1]), 0.0)
    floor = rel_floor * top if top > 0 else rel_floor
    low = eig < floor
    if not low.any():
        return a, 0
    eig = np.where(low, floor, eig)
    out = (vec * eig) @ vec.T
    return (out + out.T) / 2, int(low.sum())


def estimate_covariance(
    returns: np.ndarray, method: Method = "ledoit_wolf", *, max_missing: float = 0.1
) -> CovEstimate:
    """Clean the panel, estimate, and report diagnostics before and after."""
    clean = prepare_returns(returns, max_missing=max_missing)
    x = clean.returns
    s = sample_covariance(x)
    diag: dict[str, float | int] = {
        "n_obs": x.shape[0],
        "n_assets": x.shape[1],
        "n_rows_dropped": clean.n_rows_dropped,
        "n_assets_dropped": len(clean.dropped_assets),
        "cond_sample": condition_number(s),
    }
    if method == "sample":
        out = s
    elif method == "ledoit_wolf":
        out, delta = ledoit_wolf(x)
        diag["shrinkage"] = delta
    elif method == "clip":
        out, n_clipped = clip_eigenvalues(x)
        diag["n_clipped"] = n_clipped
        diag["mp_edge"] = mp_upper_edge(x.shape[1], x.shape[0])
    else:
        raise ValueError(f"unknown method {method!r}")
    diag["cond"] = condition_number(out)
    diag["min_eig"] = float(np.linalg.eigvalsh(out)[0])
    return CovEstimate(
        matrix=out,
        method=method,
        assets=clean.assets,
        dropped_assets=clean.dropped_assets,
        diagnostics=diag,
    )
