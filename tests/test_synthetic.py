import numpy as np

from robust_mvo.synthetic import make_factor_market, simulate_returns


def test_synthetic_market_is_deterministic_and_pd():
    a = make_factor_market(20, 3, seed=7)
    b = make_factor_market(20, 3, seed=7)
    np.testing.assert_array_equal(a.covariance, b.covariance)
    assert np.linalg.eigvalsh(a.covariance)[0] > 0


def test_simulated_returns_converge_to_true_covariance():
    market = make_factor_market(10, 2, seed=1)
    t = 200_000
    x = simulate_returns(market, t, seed=2)
    sigma = market.covariance
    # Gaussian sampling s.e. of S_ij is sqrt((S_ii S_jj + S_ij^2) / T); allow 5 s.e.
    var = np.diag(sigma)
    se = np.sqrt((np.outer(var, var) + sigma**2) / t)
    assert np.all(np.abs(np.cov(x, rowvar=False) - sigma) < 5 * se)
