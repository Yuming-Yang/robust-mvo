"""A two-minute tour: one normal rebalance, then one that cannot be done as asked.

uv run python scripts/demo.py
"""

import numpy as np

from robust_mvo.covariance import estimate_covariance
from robust_mvo.optimizer import Constraints, CostModel, Inputs, adv_cap, square_root_impact
from robust_mvo.risk_model import FactorRiskModel
from robust_mvo.robust import robust_optimize
from robust_mvo.synthetic import make_factor_market, simulate_returns

n, t = 50, 60  # N close to T on purpose
market = make_factor_market(n, 3, seed=0)
returns = simulate_returns(market, t, seed=1)
returns[:30, 7] = np.nan  # an asset with half its history missing

est = estimate_covariance(returns, "clip")
print(f"estimate: {est.method}, kept {len(est.assets)} of {n} assets, dropped {est.dropped_assets}")
print(
    f"condition number: sample {est.diagnostics['cond_sample']:.3g} -> clipped "
    f"{est.diagnostics['cond']:.3g}"
)

# Optimize on the assets the estimate covers.
m = len(est.assets)
rng = np.random.default_rng(2)
alpha = rng.normal(0, 5e-4, m)
alpha[3] = np.nan  # no signal for one name today
adv = rng.uniform(0.2, 2.0, m)  # daily volume as a fraction of NAV
vol = np.sqrt(np.diag(est.matrix))
inp = Inputs(
    alpha=alpha,
    risk=FactorRiskModel.from_covariance(est.matrix),
    w0=np.zeros(m),
    gamma=20.0,
    costs=CostModel(linear=5e-4, impact=square_root_impact(vol, adv, coef=0.1)),
    constraints=Constraints(max_position=0.04, max_trade=adv_cap(adv, 0.05), max_gross=1.0),
)


def show(title, w, rep):
    print(f"\n{title}")
    print(f"  level {rep.level}, gross {np.abs(w).sum():.3f}, net {w.sum():+.2e}")
    for r in rep.reasons:
        print(f"  reason: {r}")
    print(f"  frozen: {rep.frozen}")
    if rep.relaxed:
        print(
            f"  relaxed {rep.relaxed}: "
            + ", ".join(f"{g} by {rep.violations[g]:.4f}" for g in rep.relaxed)
        )


show("normal rebalance from cash", *robust_optimize(inp))

# Start from a net-long book that the trade limits cannot neutralize in one day.
w0 = np.full(m, 0.02)
show(
    "net-long start, tight trade limits",
    *robust_optimize(
        Inputs(
            alpha,
            inp.risk,
            w0,
            gamma=20.0,
            costs=inp.costs,
            constraints=Constraints(max_position=0.04, max_trade=0.01, max_gross=1.5),
        )
    ),
)
