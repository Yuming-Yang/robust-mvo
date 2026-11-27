# robust-mvo

Mean-variance optimization for one rebalance of a long-short book, built so that every way it can fail, from a near-singular covariance and contradictory constraints to a failing solver or missing data, has a defined, tested behaviour and a recorded reason.

```python
w, report = robust_optimize(inputs)
report.level  # 0 primary solver, 1 fallback solver, 2 caps relaxed, 3 held previous book
report.reasons  # one line per step down, e.g. "level 0 -> 2: pre-check: dollar neutrality unreachable ..."
report.frozen  # assets held because their data was unusable, and why
```

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Yuming-Yang/robust-mvo && cd robust-mvo
uv sync
uv run python scripts/demo.py
```

The demo estimates a covariance with N = 50 and T = 60, drops an asset with half its history missing, freezes an asset with no alpha, and runs two rebalances. The second starts from a net-long book whose trade limits cannot reach dollar neutrality in one step:

```
net-long start, tight trade limits
  level 2, gross 0.526, net +1.50e-15
  reason: level 0 -> 2: pre-check: dollar neutrality unreachable: reachable net exposure is [0.5, 1.46]
  frozen: {3: 'alpha not finite'}
  relaxed ('max_trade',): max_trade by 0.0443
```

## Results: covariance stabilization when N is close to T

The same dollar-neutral problem is solved with three covariance estimates on simulated data where the true covariance and alpha are known. Efficiency is the true utility of the chosen portfolio divided by that of the true optimum (1 is perfect; below 0 is worse than holding nothing). Risk ratio is predicted variance divided by true variance of the chosen portfolio. N = 100, 20 seeds per ratio, medians.

Strong market factor:

| N/T | method | efficiency | risk ratio | condition no. |
| --- | --- | --- | --- | --- |
| 0.25 | sample | 0.34 | 0.57 | 198 |
| 0.25 | Ledoit–Wolf | 0.62 | 0.67 | 139 |
| 0.25 | clipping | 0.96 | 1.05 | 92 |
| 0.9 | sample | −705 | 0.01 | 1.2e4 |
| 0.9 | Ledoit–Wolf | −0.52 | 0.43 | 119 |
| 0.9 | clipping | 0.95 | 0.99 | 93 |
| 1.5 | sample | −2.4e16 | 4.9e−17 | inf |
| 1.5 | Ledoit–Wolf | −0.38 | 0.44 | 74 |
| 1.5 | clipping | 0.92 | 0.93 | 97 |

Weak factors: over N/T from 0.1 to 1.5, Ledoit–Wolf efficiency stays between 0.82 and 0.93 and clipping between 0.94 and 1.00. The sample covariance is below 0 from N/T = 0.5 on in both regimes.

Reproduce: `uv run python scripts/n_close_to_t.py` (prints both full tables), or open [`notebooks/n_close_to_t.ipynb`](notebooks/n_close_to_t.ipynb), which CI executes.

What the numbers show:

- The sample covariance fails well before N = T. At N ≥ T it is singular; the PSD guard repairs it and the solver reports success, but the weights are meaningless. The degradation chain guards against numerical failure, not against a bad estimate.
- Ledoit–Wolf with a scaled-identity target has lower Frobenius error than the sample covariance from N/T = 0.25 on (Frobenius loss is what it minimizes), yet with a strong market factor its portfolios are worse than holding nothing from N/T ≈ 0.5. Shrinking toward μI pulls down the market eigenvalue and pulls individual variances toward their mean.
- Clipping wins here, but see Limitations: the simulated data have exactly the structure clipping assumes.

## What is in the library

| Module | Contents |
| --- | --- |
| `covariance.py` | Sample, Ledoit–Wolf (2004) and Marchenko–Pastur eigenvalue clipping (Laloux et al. 1999); listwise missing-data handling; a PSD repair; diagnostics. |
| `risk_model.py` | `FactorRiskModel(B, F, D)`; a Cholesky of `F` that repairs and reports a non-PD input. |
| `optimizer.py` | The cvxpy problem: alpha, factor plus idiosyncratic risk, L1 turnover, linear cost, 3/2-power or quadratic impact; dollar neutral, position, trade-size and position-size liquidity, and gross caps. |
| `robust.py` | The degradation chain and `SolveReport`. |
| `synthetic.py`, `experiments.py` | Seeded factor markets with known covariance; the N ≈ T experiment. |

Every formula is derived by hand in [`DERIVATIONS.md`](DERIVATIONS.md): the Ledoit–Wolf oracle, the eigenvalue spread at N ≈ T, the KKT closed form, why impact exponents of 3/2 and 2 arise, the no-trade region, a convexity-free comparative-statics proof, and the exact-penalty argument behind the relaxation.

### The degradation chain

| Level | What happens | Moves on when |
| --- | --- | --- |
| 0 | Full problem, CLARABEL | solver error, unacceptable status, non-finite weights, or a post-check violation above 1e-6 |
| 1 | SCS, then OSQP if the problem is a QP | all fail; a proved infeasibility skips this level |
| 2 | Caps become penalized slacks, relaxed liquidity → position → gross; dollar neutrality never | still fails |
| 3 | Hold the previous weights | — |

An `O(N)` interval pre-check catches the common infeasible cases before any solver runs. Assets with non-finite alpha, exposures, idiosyncratic variance or caps are frozen at their current weight and listed.

## Design trade-offs

- **3/2-power impact by default, quadratic as an option.** The square-root impact law gives a 3/2-power cost, which needs a power cone, so OSQP (QP only) cannot solve it. The quadratic option keeps a QP path.
- **L1 turnover penalty, kept as its own parameter.** With a linear cost already in the objective, an L1 turnover penalty is the same term with a larger coefficient (tested). It stays separate because a turnover budget and a spread estimate are set for different reasons.
- **`optimal_inaccurate` accepted when the post-check passes.** CLARABEL often stops at "AlmostSolved" on the 3/2 form when many trades are exactly zero. Rejecting those would send good solves down the chain.
- **Objective scaled by 1 / average variance.** Unscaled daily-variance objectives sit near solver tolerances. On three QP test problems, scaling cut the worst weight error from 3.5e-5 to 4.3e-7 (`PYTHONPATH=tests uv run python scripts/objective_scaling_check.py`; the script reuses the test builders).
- **L1 slack in the relaxation.** It is an exact penalty: a feasible problem's solution is unchanged. The cost is that violations concentrate on a few assets.
- **Listwise deletion, not pairwise-complete covariance.** Pairwise estimates can be indefinite (a test shows a 3-asset example).
- **Hold the previous book as the last resort.** It trades nothing, so it cannot breach a liquidity limit; any cap it already breaks is reported, not fixed.

Dropped alternatives and the findings behind these choices are in [`DECISIONS.md`](DECISIONS.md); the scope is in [`SPEC.md`](SPEC.md).

## Limitations

- The N ≈ T comparison uses a factor model plus i.i.d. Gaussian noise, the "spiked" structure that eigenvalue clipping assumes. Real returns, heavy tails, or Ledoit–Wolf with a constant-correlation or single-index target could change the ranking. None of these were run.
- Ledoit–Wolf is implemented only with the scaled-identity target, and clipping uses noise variance 1 rather than the refinement that removes the market mode first.
- Single period only: no multi-period trading, execution scheduling or alpha decay.
- The impact coefficients are inputs; nothing here calibrates them to real trades.
- Every solver failure in the tests is injected. The chain has not been exercised against solvers failing on their own on real data.
- Constraint tolerance is 1e-6 in weight units. A solution SCS accepts may be less accurate than a CLARABEL one inside that tolerance.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest                          # unit and property tests
uv run pytest --nbmake notebooks       # executes the notebook
uv run pytest --hypothesis-profile=stress   # 500 examples per property test
```

## License

MIT. See [LICENSE](LICENSE).
