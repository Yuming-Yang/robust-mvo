"""The N ~ T experiment behind the notebook and the README numbers.

A fixed factor market with known Sigma and a fixed, known alpha. For each
ratio q = N/T and each seed, simulate T returns, estimate Sigma three ways,
solve the same dollar-neutral mean-variance problem with each estimate
(through ``robust_optimize``), and score the weights against the truth.

Only the covariance estimate differs between methods, so differences in the
scores are caused by estimation error alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robust_mvo.covariance import condition_number, estimate_covariance
from robust_mvo.optimizer import Constraints, Inputs
from robust_mvo.risk_model import FactorRiskModel
from robust_mvo.robust import robust_optimize
from robust_mvo.synthetic import make_factor_market, simulate_returns

METHODS = ("sample", "ledoit_wolf", "clip")


@dataclass(frozen=True)
class Row:
    q: float
    seed: int
    method: str
    cond: float  # condition number of the estimate (inf if singular)
    frob_err: float  # ||Sigma_hat - Sigma||_F / ||Sigma||_F
    efficiency: float  # true utility of w_hat / true utility of w*  (1 is perfect)
    risk_ratio: float  # predicted variance / true variance of w_hat (< 1: underestimated)
    level: int  # degradation level reached by robust_optimize
    repaired: int  # eigenvalues of the estimate raised by the PSD guard


def run(
    n_assets: int = 100,
    ratios: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5),
    n_seeds: int = 20,
    gamma: float = 20.0,
    market_seed: int = 0,
    market_vol: float = 0.01,
    style_vol: float = 0.005,
) -> list[Row]:
    """Run the experiment. ``market_vol`` and ``style_vol`` set the factor
    strengths; the defaults give one strong market factor (see synthetic.py)."""
    market = make_factor_market(
        n_assets, 3, seed=market_seed, market_vol=market_vol, style_vol=style_vol
    )
    sigma = market.covariance
    alpha = np.random.default_rng(market_seed + 1).normal(0.0, 5e-4, n_assets)
    cons = Constraints(dollar_neutral=True)
    w0 = np.zeros(n_assets)

    def utility(w: np.ndarray) -> float:
        return float(alpha @ w - gamma * w @ sigma @ w)

    truth = Inputs(alpha, FactorRiskModel.from_covariance(sigma), w0, gamma=gamma, constraints=cons)
    w_star, _ = robust_optimize(truth)
    u_star = utility(w_star)

    rows: list[Row] = []
    for q in ratios:
        t = max(2, round(n_assets / q))
        for seed in range(n_seeds):
            x = simulate_returns(market, t, seed=10_000 * seed + t)
            for method in METHODS:
                est = estimate_covariance(x, method).matrix
                inp = Inputs(
                    alpha, FactorRiskModel.from_covariance(est), w0, gamma=gamma, constraints=cons
                )
                w, rep = robust_optimize(inp)
                rows.append(
                    Row(
                        q=q,
                        seed=seed,
                        method=method,
                        cond=condition_number(est),
                        frob_err=float(np.linalg.norm(est - sigma) / np.linalg.norm(sigma)),
                        efficiency=utility(w) / u_star,
                        risk_ratio=float((w @ est @ w) / (w @ sigma @ w)),
                        level=rep.level,
                        repaired=rep.factor_cov_repaired,
                    )
                )
    return rows


def summarize(rows: list[Row]) -> list[dict[str, float | str]]:
    """Median of each metric per (q, method)."""
    out = []
    keys = sorted({(r.q, r.method) for r in rows}, key=lambda k: (k[0], METHODS.index(k[1])))
    for q, method in keys:
        sel = [r for r in rows if r.q == q and r.method == method]
        out.append(
            {
                "q": q,
                "method": method,
                "cond": float(np.median([r.cond for r in sel])),
                "frob_err": float(np.median([r.frob_err for r in sel])),
                "efficiency": float(np.median([r.efficiency for r in sel])),
                "risk_ratio": float(np.median([r.risk_ratio for r in sel])),
                "max_level": float(max(r.level for r in sel)),
                "repaired": float(np.median([r.repaired for r in sel])),
            }
        )
    return out
