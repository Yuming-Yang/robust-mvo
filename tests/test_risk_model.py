import numpy as np
import pytest

from robust_mvo.risk_model import FactorRiskModel
from robust_mvo.synthetic import make_factor_market


def test_risk_model_covariance_matches_market():
    m = make_factor_market(15, 4, seed=0)
    rm = FactorRiskModel(m.exposures, m.factor_cov, m.idio_var)
    np.testing.assert_allclose(rm.covariance(), m.covariance)


def test_factor_cholesky_exact_when_pd():
    m = make_factor_market(15, 4, seed=0)
    rm = FactorRiskModel(m.exposures, m.factor_cov, m.idio_var)
    chol, n_fixed = rm.factor_cholesky()
    assert n_fixed == 0
    np.testing.assert_allclose(chol @ chol.T, m.factor_cov)


def test_factor_cholesky_repairs_indefinite_f():
    f = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    rm = FactorRiskModel(np.ones((4, 3)), f, np.full(4, 0.01))
    chol, n_fixed = rm.factor_cholesky()
    assert n_fixed == 1
    assert np.all(np.isfinite(chol))


def test_full_covariance_mode():
    sigma = make_factor_market(6, 2, seed=3).covariance
    rm = FactorRiskModel.from_covariance(sigma)
    assert rm.n_factors == 6
    np.testing.assert_allclose(rm.covariance(), sigma)


def test_bad_assets_mask():
    b = np.ones((4, 2))
    b[1, 0] = np.nan
    d = np.array([0.01, 0.01, np.inf, -0.01])
    rm = FactorRiskModel(b, np.eye(2), d)
    assert list(rm.bad_assets()) == [False, True, True, True]


@pytest.mark.parametrize(
    ("b", "f", "d", "msg"),
    [
        (np.ones(3), np.eye(1), np.ones(3), "2-D"),
        (np.ones((3, 2)), np.eye(3), np.ones(3), "factor_cov"),
        (np.ones((3, 2)), np.eye(2), np.ones(4), "idio_var"),
        (np.ones((3, 2)), np.array([[1.0, np.nan], [np.nan, 1.0]]), np.ones(3), "NaN"),
    ],
)
def test_shape_and_value_validation(b, f, d, msg):
    with pytest.raises(ValueError, match=msg):
        FactorRiskModel(b, f, d)
