import cvxpy as cp

import robust_mvo


def test_package_imports():
    assert robust_mvo.__version__


def test_default_solvers_are_installed():
    # The degradation chain relies on these three being present without extras.
    installed = set(cp.installed_solvers())
    assert {"CLARABEL", "SCS", "OSQP"} <= installed
