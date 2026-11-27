"""Reproduce the objective-scaling measurement recorded in DECISIONS.md (milestone 2).

For three seeded QP-form problems, compare the weights from a default CLARABEL
solve with and without objective scaling against a reference solved with
CLARABEL tolerances of 1e-14.

    PYTHONPATH=tests uv run python scripts/objective_scaling_check.py
"""

import warnings

import numpy as np
from builders import inputs, neutral_w0

import robust_mvo.optimizer as opt

warnings.simplefilter("ignore")
TIGHT = {"tol_gap_abs": 1e-14, "tol_gap_rel": 1e-14, "tol_feas": 1e-14, "max_iter": 500}
BOOK = opt.Constraints(max_position=0.05, max_gross=1.0)
scaled = opt.objective_scale

for seed in range(3):
    inp = inputs(
        30,
        seed=seed,
        w0=neutral_w0(30, seed, 0.03),
        costs=opt.CostModel(linear=5e-4, impact=2e-2, impact_power=2.0),
        constraints=BOOK,
    )
    row = {}
    for label, scale in (("unscaled", lambda _: 1.0), ("scaled", scaled)):
        opt.objective_scale = scale
        p, w = opt.build_problem(inp)
        p.solve(solver="CLARABEL")
        default = w.value.copy()
        p, w = opt.build_problem(inp)
        p.solve(solver="CLARABEL", **TIGHT)
        row[label] = np.abs(default - w.value).max()
    opt.objective_scale = scaled
    print(
        f"seed {seed}: max |w - w_ref| unscaled {row['unscaled']:.1e}, scaled {row['scaled']:.1e}"
    )
