import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.covariance import LedoitWolf

from robust_mvo.covariance import (
    clip_eigenvalues,
    condition_number,
    estimate_covariance,
    ledoit_wolf,
    mp_upper_edge,
    prepare_returns,
    psd_floor,
    sample_covariance,
)
from robust_mvo.synthetic import make_factor_market, simulate_returns


def panel(n_assets: int, n_obs: int, seed: int = 0) -> np.ndarray:
    market = make_factor_market(n_assets, 3, seed=seed)
    return simulate_returns(market, n_obs, seed=seed + 1)


def is_pd(m: np.ndarray) -> bool:
    return bool(np.allclose(m, m.T) and np.linalg.eigvalsh(m)[0] > 0)


# --- Ledoit-Wolf ---------------------------------------------------------------


@pytest.mark.parametrize(("n", "t"), [(5, 200), (30, 60), (50, 50), (80, 40)])
def test_ledoit_wolf_matches_sklearn(n, t):
    x = panel(n, t)
    ours, delta = ledoit_wolf(x)
    ref = LedoitWolf().fit(x)
    assert delta == pytest.approx(ref.shrinkage_, abs=1e-10)
    np.testing.assert_allclose(ours, ref.covariance_, atol=1e-10, rtol=0)


def test_ledoit_wolf_is_pd_when_sample_is_singular():
    x = panel(80, 40)
    assert condition_number(sample_covariance(x)) == float("inf")
    sigma, delta = ledoit_wolf(x)
    assert 0 < delta <= 1
    assert is_pd(sigma)


def test_ledoit_wolf_shrinks_less_with_more_data():
    deltas = [ledoit_wolf(panel(40, t))[1] for t in (40, 400, 4000)]
    assert deltas[0] > deltas[1] > deltas[2]


def test_ledoit_wolf_on_scaled_identity_sample_is_a_no_op():
    # Columns orthogonal with equal norms -> S is exactly a multiple of I.
    x = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    sigma, delta = ledoit_wolf(x)
    np.testing.assert_allclose(sigma, np.eye(2))
    assert delta == 0.0


# --- Eigenvalue clipping -------------------------------------------------------


@pytest.mark.parametrize(("n", "t"), [(20, 200), (50, 60), (60, 50), (100, 40)])
def test_clipping_keeps_unit_diagonal_correlation_and_sample_vols(n, t):
    x = panel(n, t)
    sigma, n_clipped = clip_eigenvalues(x)
    s = sample_covariance(x)
    np.testing.assert_allclose(np.diag(sigma), np.diag(s), rtol=1e-12)
    vol = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(vol, vol)
    assert np.trace(corr) == pytest.approx(n, rel=1e-12)
    assert is_pd(sigma)
    assert 0 < n_clipped < n


def test_clipping_keeps_the_market_factor():
    # The market factor's eigenvalue is far above the noise band and must survive.
    x = panel(50, 500)
    s = sample_covariance(x)
    vol = np.sqrt(np.diag(s))
    top_before = np.linalg.eigvalsh(s / np.outer(vol, vol))[-1]
    sigma, _ = clip_eigenvalues(x)
    vol = np.sqrt(np.diag(sigma))
    top_after = np.linalg.eigvalsh(sigma / np.outer(vol, vol))[-1]
    assert top_before > mp_upper_edge(50, 500)
    # The final rescale to unit diagonal moves eigenvalues slightly, so not exact.
    assert top_after == pytest.approx(top_before, rel=0.05)


def test_pure_noise_is_clipped_to_identity_correlation():
    # Independent assets: every eigenvalue is noise, so the cleaned correlation is I.
    rng = np.random.default_rng(3)
    x = rng.standard_normal((100, 30))
    sigma, n_clipped = clip_eigenvalues(x)
    assert n_clipped == 30
    vol = np.sqrt(np.diag(sigma))
    np.testing.assert_allclose(sigma / np.outer(vol, vol), np.eye(30), atol=1e-12)


def test_mp_edge_values():
    assert mp_upper_edge(100, 100) == pytest.approx(4.0)
    assert mp_upper_edge(25, 100) == pytest.approx(2.25)


# --- Property test: both stabilizers always return a PD matrix -------------------


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(2, 40),
    t=st.integers(3, 80),
    seed=st.integers(0, 10_000),
    method=st.sampled_from(["ledoit_wolf", "clip"]),
)
def test_stabilized_estimate_is_symmetric_pd(n, t, seed, method):
    x = panel(n, t, seed)
    est = estimate_covariance(x, method)
    assert est.matrix.shape == (n, n)
    assert np.array_equal(est.matrix, est.matrix.T) or np.allclose(est.matrix, est.matrix.T)
    assert est.diagnostics["min_eig"] > 0
    assert np.isfinite(est.diagnostics["cond"])


# --- Missing data ----------------------------------------------------------------


def test_prepare_returns_drops_sparse_assets_then_rows():
    x = panel(6, 50)
    x[:20, 1] = np.nan  # 40% missing -> asset dropped
    x[[3, 7], 4] = np.nan  # 4% missing -> asset kept, rows 3 and 7 dropped
    x[10, 5] = np.inf  # treated like NaN
    clean = prepare_returns(x, max_missing=0.1)
    assert list(clean.assets) == [0, 2, 3, 4, 5]
    assert set(clean.dropped_assets) == {1}
    assert "missing fraction 0.40" in clean.dropped_assets[1]
    assert clean.n_rows_dropped == 3
    assert clean.returns.shape == (47, 5)
    assert np.isfinite(clean.returns).all()


def test_prepare_returns_drops_constant_assets():
    x = panel(4, 30)
    x[:, 2] = 0.001
    clean = prepare_returns(x)
    assert list(clean.assets) == [0, 1, 3]
    assert clean.dropped_assets == {2: "zero variance"}


def test_estimate_reports_what_was_dropped():
    x = panel(6, 50)
    x[:30, 0] = np.nan
    est = estimate_covariance(x, "clip")
    assert list(est.assets) == [1, 2, 3, 4, 5]
    assert 0 in est.dropped_assets
    assert est.diagnostics["n_assets_dropped"] == 1


def test_estimators_reject_nan_directly():
    x = panel(3, 10)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="prepare_returns"):
        ledoit_wolf(x)


def test_pairwise_complete_can_be_indefinite():
    # Why prepare_returns uses listwise deletion: estimating each entry on a
    # different subset of rows gives a matrix that is not a covariance matrix.
    # Here a and b agree, b and c agree, but a and c disagree, each on the rows
    # where only that pair is observed.
    nan = np.nan
    x = np.array(
        [
            [1.0, 1.0, nan],
            [-1.0, -1.0, nan],
            [nan, 1.0, 1.0],
            [nan, -1.0, -1.0],
            [1.0, nan, -1.0],
            [-1.0, nan, 1.0],
        ]
    )
    n = x.shape[1]
    pairwise = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            ok = np.isfinite(x[:, i]) & np.isfinite(x[:, j])
            pairwise[i, j] = np.mean(x[ok, i] * x[ok, j])
    assert np.linalg.eigvalsh(pairwise)[0] < 0
    # Listwise deletion leaves no complete rows here, which is an honest failure.
    assert prepare_returns(x, max_missing=0.5).returns.shape[0] == 0


# --- PSD repair ------------------------------------------------------------------


def test_psd_floor_repairs_indefinite_matrix():
    m = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    assert np.linalg.eigvalsh(m)[0] < 0
    fixed, n_fixed = psd_floor(m)
    assert n_fixed == 1
    assert is_pd(fixed)


def test_psd_floor_leaves_pd_matrix_alone():
    m = sample_covariance(panel(5, 100))
    fixed, n_fixed = psd_floor(m)
    assert n_fixed == 0
    np.testing.assert_array_equal(fixed, m)
