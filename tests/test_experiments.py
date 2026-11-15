import math

from robust_mvo.experiments import METHODS, run, summarize


def test_experiment_runs_and_scores_are_sane():
    rows = run(n_assets=20, ratios=(0.5, 1.5), n_seeds=2)
    assert len(rows) == 2 * 2 * len(METHODS)
    for r in rows:
        assert r.frob_err > 0
        assert r.efficiency <= 1 + 1e-6  # nothing beats the true optimum
    summary = summarize(rows)
    assert {(s["q"], s["method"]) for s in summary} == {(q, m) for q in (0.5, 1.5) for m in METHODS}


def test_singular_sample_covariance_is_repaired_not_fatal():
    # N > T: the sample estimate is singular, the PSD guard repairs it, and the
    # chain still returns weights. This is a numerical success that is a
    # statistical failure, which the notebook shows.
    rows = [r for r in run(n_assets=30, ratios=(1.5,), n_seeds=2) if r.method == "sample"]
    for r in rows:
        assert math.isinf(r.cond)
        assert r.repaired > 0
        assert r.level == 0
        assert r.efficiency < 0
