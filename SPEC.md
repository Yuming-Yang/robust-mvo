# SPEC: Robust MVO

Status: v1 scope and design decisions D1–D7 fixed on 2026-11-15 (see Decisions at the end). Implemented; see the README for results.

## Problem

Mean-variance optimization is easy to write down and easy to break. With a sample covariance estimated from about as many days as there are assets, the matrix is close to singular. The optimizer then leans hard on its smallest, noisiest eigenvalues and returns extreme, unstable weights. Constraints that look reasonable on their own can contradict each other given yesterday's holdings. Solvers can also fail, stall or return "optimal_inaccurate". Most open-source examples skip all of this: they show the happy path and leave the failure modes to the user.

This project is a small library for one rebalance step of a long-short portfolio. Its focus is numerical stability and explicit handling of failure. Every way the optimization can go wrong has a defined, tested behaviour and a recorded reason. The goal is a small set of features that hold up under bad input, not a long feature list.

## Approach

### Single-period problem

Inputs at a rebalance: expected returns `alpha` (N), a risk model, current weights `w0` (N), cost parameters, and constraint limits. Weights are fractions of NAV. The decision variable is the target weight vector `w`, with trade `z = w - w0`.

```
maximize    alpha' w
          - gamma * ( w' B F B' w + w' D w )          # factor risk + idiosyncratic risk
          - lambda_to * TurnoverPenalty(z)            # see Decision D3
          - sum_i TC_i(z_i)                           # see Decision D1
subject to  1' w = 0                                  # dollar neutral
            |w_i| <= u_i                              # per-name position cap
            |z_i| <= rho * ADV_i / NAV                # liquidity cap on trade size (see D4)
            ||w||_1 <= L                              # gross leverage cap (optional)
```

- **Risk model.** `B` is N×K exposures, `F` is the K×K factor covariance, `D` is diagonal idiosyncratic variance. The solver never forms the N×N matrix. It adds an auxiliary variable `f = B'w` and writes risk as `||L_F' f||² + ||sqrt(D) ∘ w||²`, where `L_F` is the Cholesky factor of `F`. Cost per solve scales with N·K, not N². A full-covariance mode (`Σ` given directly, K = N) is kept for the N ≈ T comparison and for tests against closed forms.
- **Why Cholesky matters here.** If `F` is not positive definite, the Cholesky step fails. This is one of the concrete places where covariance stabilization is needed, not just a nice-to-have.
- **Transaction costs.** Separable per asset. The candidate forms (linear spread, quadratic impact, 3/2-power impact) are all convex and DCP-representable in cvxpy. The choice is Decision D1.
- **Constraints.** Dollar neutrality, per-name caps and a liquidity cap, as listed in the project checklist. The gross leverage cap is an addition, since dollar-neutral books are almost always run with one. It can be switched off.

### Covariance stabilization (two methods)

Both work on a T×N return panel and return a symmetric positive definite matrix along with diagnostics: condition number, minimum eigenvalue, shrinkage intensity or number of eigenvalues clipped. They can be used for the full `Σ` and for the factor covariance `F` estimated from factor returns.

1. **Ledoit–Wolf linear shrinkage.** `Σ_LW = δ·μI + (1−δ)·S`, with target `μI` (μ = average sample variance) and the shrinkage intensity `δ` estimated by the closed-form formula in Ledoit & Wolf (2004). This is implemented from the formula, and `sklearn.covariance.LedoitWolf` (same estimator) is used as a test oracle only. The derivation of `δ` goes in `DERIVATIONS.md`.
2. **Eigenvalue clipping (random-matrix theory).** Work on the correlation matrix. Eigenvalues below the Marchenko–Pastur upper edge `λ+ = σ²(1 + sqrt(N/T))²` are treated as noise and replaced by their average, which preserves the trace. The result is rescaled to unit diagonal and converted back to covariance. This follows Laloux, Cizeau, Bouchaud & Potters (1999); the review by Bun, Bouchaud & Potters (2017) covers the variants. The simpler "floor at ε" repair is kept separately as a numerical PSD guard, not as a statistical estimator.

### Failure handling: an explicit degradation chain

Each solve returns `(w, SolveReport)`. The report records the level reached, the reason for every step down, solver statuses and timings, constraint violations measured on the returned `w`, and input diagnostics. The library never raises on bad market data; it raises only on programmer errors such as mismatched shapes.

Default order (see Decision D2):

| Level | What happens | Trigger to move to the next level |
| --- | --- | --- |
| 0 | Full problem, primary solver (CLARABEL) | `SolverError`, status not `optimal` (whether `optimal_inaccurate` is accepted is part of D2), or post-check finds a violated constraint |
| 1 | Same problem, fallback solvers in order (SCS; OSQP only when the problem is a QP, see note below) | All solvers fail, or the problem is proved `infeasible` (switching solvers does not fix infeasibility, so this case skips straight to level 2) |
| 2 | Relaxed problem: hard constraints that can be softened become penalized slack ("elastic" formulation), and the report lists which constraints were relaxed and by how much | Still fails |
| 3 | Keep previous holdings `w = w0` (no trade) | Final level, always returns |

Note: in a local test (cvxpy 1.9.3), OSQP raised `SolverError` on a problem with a 3/2-power cost term, while CLARABEL and SCS solved it. So OSQP is a valid fallback only when costs are linear or quadratic. This fact is one input to Decision D1.

### Ill-conditioned and invalid inputs

| Input | Behaviour |
| --- | --- |
| N ≈ T or N > T (singular sample covariance) | Estimation still returns a PD matrix through either stabilizer; diagnostics report the condition number before and after. The raw sample covariance is rejected by the PD check before reaching the solver. |
| Infeasible constraints. With symmetric caps, `w = 0` always satisfies dollar neutrality, position caps and gross leverage, so infeasibility comes from the trade caps: e.g. `w0` is far from dollar neutral, or above a newly tightened position cap, and the liquidity cap forbids trading far enough to fix it | Level 0 reports `infeasible`; the chain skips level 1 and goes to level 2 with the reason recorded. A cheap pre-check catches the obvious cases before calling a solver. |
| NaN or inf in `alpha`, `B`, `D` or `ADV` for an asset | The asset is frozen at `w0_i` (no trade) and listed in the report. See Decision D5. |
| NaN in the return panel | Assets with a missing fraction above a threshold are dropped from estimation; the remaining rows with gaps are dropped (listwise). Pairwise-complete covariance is deliberately not used, because it can be indefinite. A test shows this happening. |
| `w0` itself violates the new constraints (e.g. a cap was tightened) | If the trade caps allow fixing it, level 0 does so. If not, it is the infeasible case above: level 2 relaxes, and if that fails, level 3 returns `w0` and flags the violation. |
| Factor covariance `F` not PD | Repaired by the PSD guard, and the repair is recorded. |

## Scope

- Single-period, single-rebalance optimizer for a long-short equity-style book.
- Factor risk model as input (B, F, D); helpers to estimate F or Σ from returns with the two stabilizers.
- The degradation chain and report described above.
- Synthetic data generators with a known true covariance (factor structure plus noise), so estimation error can be measured against ground truth.

## Non-goals

- Multi-period optimization, execution scheduling, intraday impact models.
- Estimating `alpha`. The library takes it as given.
- Fundamental factor model construction (industry and style factors from company data). Exposures are an input.
- Nonlinear shrinkage (Ledoit & Wolf 2020) and other estimators beyond the two named. They can be listed in the README as natural extensions.
- Real market data. Everything is synthetic and seeded. The library can be pointed at real data by the user, but no results in the README depend on it.
- Any commercial solver (MOSEK, Gurobi). Only solvers installed by default with cvxpy.
- Backtesting. A walk-forward demo would duplicate project 10; the N ≈ T notebook measures out-of-sample risk on simulated data instead.

## Acceptance criteria

- [x] **Property tests (hypothesis).** For randomly generated feasible inputs: when the report says level 0 or 1, every constraint holds on the returned `w` within a stated tolerance (proposed 1e-6 in weight units); at level 2 the reported violations match the violations measured on `w`; at level 3, `w == w0` exactly.
- [x] **Comparative-statics tests.** Raising `gamma` never increases ex-ante risk; raising a cost coefficient never increases that cost term's value (both follow from optimality of a convex problem in a single scaled term, proved in `DERIVATIONS.md`).
- [x] **Closed-form checks.** With no costs and only dollar neutrality, the solver matches the Lagrangian closed form `w* = (1/2γ) Σ⁻¹ (alpha − η1)`, with η chosen so that `1'w* = 0`, to 1e-6.
- [x] **Estimator checks.** The Ledoit–Wolf implementation matches `sklearn.covariance.LedoitWolf` to 1e-10; eigenvalue clipping preserves trace and unit diagonal, and its output is PD with a bounded condition number.
- [x] **Ill-conditioned input tests.** One test per row of the table above, each asserting the exact level reached, the recorded reason, and the returned weights.
- [x] **Fault injection.** A fake solver that raises, or one that returns `optimal_inaccurate`, moves the chain to the expected next level. CI does not depend on a real solver happening to fail.
- [x] **N ≈ T notebook.** For N/T from about 0.1 to 1.5 on simulated factor-structured returns, compare sample, Ledoit–Wolf and clipped covariance on: condition number, distance to the true Σ, and the true risk of the resulting minimum-variance portfolio relative to the true optimum. The notebook runs in CI (via `nbmake`) and every number it shows comes from a fixed seed.
- [x] **`DERIVATIONS.md`.** Hand-derivable steps for every term: the factor-risk reformulation, the Ledoit–Wolf intensity, the Marchenko–Pastur edge, the closed-form solution and its KKT conditions, convexity and the conic form of each cost term, the L1-to-linear reformulation, the elastic relaxation, and the comparative-statics argument.
- [x] `uv run pytest` and `ruff` clean in GitHub Actions; no paid APIs; README runs from a fresh clone.
- [x] README: one-sentence problem, a runnable demo, results with numbers (from the notebook), design trade-offs, Limitations.

## Milestones

Stage 2 (skeleton, CI, private repo) happens first and is not counted here.

1. **Risk model and covariance stabilization.** Synthetic factor-return generator with known Σ; sample, Ledoit–Wolf and clipped estimators; PSD guard; NaN handling for the return panel; estimator tests. `DERIVATIONS.md` part 1 (Ledoit–Wolf, Marchenko–Pastur).
2. **Optimizer core.** cvxpy formulation of the objective and constraints with the factor-risk reformulation; closed-form and comparative-statics tests; property tests for constraint satisfaction on feasible inputs. `DERIVATIONS.md` part 2 (KKT, cost-term convexity, L1 reformulation).
3. **Failure handling.** Input validation, the degradation chain, `SolveReport`, fault-injected solvers, one test per ill-conditioned case. `DERIVATIONS.md` part 3 (elastic relaxation).
4. **N ≈ T notebook and docs.** The comparison notebook, README with its numbers, `DECISIONS.md`.

## Dependencies (checked 2026-11-15)

| Package | Version on PyPI | License | Notes |
| --- | --- | --- | --- |
| cvxpy | 1.9.3 (released 2026-09-19) | Apache-2.0 | Requires Python ≥ 3.11. Installs OSQP, CLARABEL, SCS and HiGHS by default. |
| clarabel | 0.11.1 | Apache-2.0 | Interior point; handles QP, SOCP, exponential and power cones. Default solver for the power-cone problem in the local test. |
| osqp | 1.1.3 | Apache-2.0 | ADMM; QP only. cvxpy's default for pure QPs. |
| scs | 3.3.2 | MIT | First-order conic solver; lower accuracy, very robust. |
| highspy | 1.16.0 | MIT | LP/MILP/QP; installed by cvxpy but not useful here beyond LP sub-cases. |
| ecos | 2.0.14 | **GPLv3** | Optional extra in cvxpy (no longer default). Not used, to keep the repo's license unconstrained. |
| numpy / scipy | 2.5.4 / 1.18.1 | BSD | |
| scikit-learn | 1.9.1 | BSD-3 | Test-only dependency (Ledoit–Wolf oracle). |
| hypothesis | 6.170.15 | MPL-2.0 | Test-only. |
| nbmake | 1.5.5 | Apache-2.0 | Test-only, runs the notebook in CI. |

Local check in a throwaway venv: `cvxpy.installed_solvers()` returned `['CLARABEL', 'SCS', 'SCIPY', 'HIGHS', 'OSQP']`; a small infeasible problem returned status `infeasible` as expected.

## Cost

Zero. No paid APIs, no cloud compute, no paid data; everything runs on a laptop in seconds to minutes.

## Related work

- **cvxportfolio** (Boyd et al.; 1.5.1, GPLv3): multi-period trading and backtesting on cvxpy; its accompanying monograph is the main public reference for the cost model. Because of GPLv3, this project reads the paper, not the code.
- **PyPortfolioOpt** (MIT), **skfolio** (BSD-3), **Riskfolio-Lib** (BSD-3): broad portfolio-optimization libraries with many estimators and objectives. This project is narrower. Its contribution is the tested failure handling and the derivation document, not more features.

## References (public literature only)

Bibliographic details marked ✓ were checked online on 2026-11-15; the rest are from memory and will be checked against the papers when `DERIVATIONS.md` cites them.

- ✓ Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis* 88(2), 365–411.
- Ledoit, O. & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. *Journal of Empirical Finance* 10(5), 603–621. (Single-index target; alternative if D6 picks a different target.)
- ✓ Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M. (1999). Noise dressing of financial correlation matrices. *Physical Review Letters* 83(7), 1467–1470.
- Bun, J., Bouchaud, J.-P. & Potters, M. (2017). Cleaning large correlation matrices: tools from random matrix theory. *Physics Reports* 666, 1–109.
- Marchenko, V. A. & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. *Math. USSR-Sbornik* 1(4), 457–483.
- ✓ Boyd, S., Busseti, E., Diamond, S., Kahn, R., Koh, K., Nystrup, P. & Speth, J. (2017). Multi-period trading via convex optimization. *Foundations and Trends in Optimization* 3(1), 1–76. (Section on transaction costs uses a linear term plus a 3/2-power impact term scaled by volatility and volume; the exact form will be checked against the PDF.)
- Almgren, R. & Chriss, N. (2000). Optimal execution of portfolio transactions. *Journal of Risk* 3(2), 5–39. (Linear temporary impact, so cost is quadratic in trade size.)
- ✓ Almgren, R., Thum, C., Hauptmann, E. & Li, H. (2005). Direct estimation of equity market impact. *Risk* 18, 57–62. (Temporary impact roughly a 3/5 power of trade rate, so cost scales with about the 1.6 power.)
- Gârleanu, N. & Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. *Journal of Finance* 68(6), 2309–2340. (Quadratic costs; smooth partial adjustment toward an aim portfolio.)
- Lobo, M. S., Fazel, M. & Boyd, S. (2007). Portfolio optimization with linear and fixed transaction costs. *Annals of Operations Research* 152, 341–365.
- Brodie, J., Daubechies, I., De Mol, C., Giannone, D. & Loris, I. (2009). Sparse and stable Markowitz portfolios. *PNAS* 106(30), 12267–12272. (L1 penalty as a stabilizer.)
- Michaud, R. (1989). The Markowitz optimization enigma: is "optimized" optimal? *Financial Analysts Journal* 45(1), 31–42. (Error maximization; motivation.)
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*, 2nd ed. McGraw-Hill. (Factor risk models; square-root impact rule of thumb.)
- Higham, N. J. (2002). Computing the nearest correlation matrix. *IMA Journal of Numerical Analysis* 22(3), 329–343. (Alternative to the ε-floor PSD repair; not planned for v1.)

## Design decisions: options considered

Each item lists the options, the public basis and the lean at draft time. The outcome is summarized under Decisions below.

**D1. Transaction cost form.** Options:
(a) linear only, `c_i|z_i|` (half-spread plus fees); LP-representable.
(b) linear + quadratic impact, `c_i|z_i| + k_i z_i²` (Almgren–Chriss linear impact; Gârleanu–Pedersen). Whole problem stays a QP, so OSQP remains a valid fallback.
(c) linear + 3/2-power impact, `c_i|z_i| + b_i σ_i |z_i|^{3/2} / sqrt(ADV_i/NAV)` (square-root law; Boyd et al. 2017). Needs a power cone, so OSQP drops out of the fallback list.
(d) a general exponent `p ∈ [1.5, 2]` as a parameter (covers Almgren et al. 2005's ~1.6).
Lean: (c) as default with (b) selectable. (c) is the form most practitioners cite and it shows the cone formulation in the derivations; (b) keeps a QP path so the OSQP fallback is exercised. (d) adds a knob without adding much to explain.

**D2. Degradation order and acceptance rules.** The checklist order is solver → relax → hold. Sub-questions:
- Should `optimal_inaccurate` be accepted at level 0 (with a warning) if the post-check finds no violation above tolerance? Lean: yes, accept and flag; rejecting it outright sends many fine solves from SCS down the chain.
- Which constraints may be relaxed at level 2, and in what order? Lean: liquidity cap first, then position caps, then gross leverage; dollar neutrality is never relaxed (it is the book's mandate, and breaking it changes the risk profile entirely). If the elastic problem still fails, go to level 3.
- Is "hold `w0`" the right last resort, or should it be "scale `w0` toward zero" or "flatten to cash"? Lean: hold `w0`, since it trades nothing and so cannot breach liquidity; but it may leave a book that violates new caps, which is reported, not fixed.

**D3. Turnover penalty: L1 or L2.** L1 (`λ‖z‖₁`) creates a no-trade region and sparse trades (Lobo, Fazel & Boyd 2007; Brodie et al. 2009). But with linear costs in the objective, an L1 turnover penalty is mathematically the same term with a larger coefficient. L2 (`λ‖z‖²`) gives smooth partial adjustment toward the target (Gârleanu & Pedersen 2013) and improves conditioning, but every name trades a little.
Lean: L1, and say plainly in the docs that it merges with the linear cost; it stays a separate named parameter because "turnover budget" and "spread cost" are set by different people for different reasons. An interviewer is likely to ask about this equivalence.

**D4. What "liquidity cap" bounds.** (a) trade size per rebalance, `|z_i| ≤ ρ·ADV_i/NAV` (participation limit), or (b) position size, `|w_i| ≤ κ·ADV_i/NAV` (days-to-liquidate limit), or both. Lean: both, with (a) on by default; (a) is what makes infeasibility from `w0` possible, which the failure tests need.

**D5. Assets with NaN inputs.** Freeze at `w0_i` (lean: no information means no trade, and it keeps dollar neutrality reachable), or force to zero (forces a trade on a name the system knows nothing about, possibly against a liquidity cap), or reject the whole solve.

**D6. Ledoit–Wolf target.** Scaled identity (Ledoit & Wolf 2004; matches sklearn, so an exact oracle exists) or constant correlation / single index (Ledoit & Wolf 2003, 2004 JPM), which fits equity returns better. Lean: scaled identity for v1, because the oracle makes the test exact; mention the others under Limitations.

**D7. Default solver tolerance for the property tests.** Proposed 1e-6 in weight units (0.01 bp of NAV). Tighter needs CLARABEL-only settings; looser hides real bugs. Lean: 1e-6, with the tolerance written in the README.

## Decisions (2026-11-15)

Every item went with the lean above:

1. **D1.** Default cost is linear plus 3/2-power impact; linear plus quadratic is selectable and keeps a QP path for OSQP. No general-exponent parameter.
2. **D2.** `optimal_inaccurate` is accepted at level 0 or 1 when the post-check finds no violation above tolerance, and the report flags it. Level 2 relaxes the liquidity cap first, then position caps, then gross leverage; dollar neutrality is never relaxed. Level 3 holds `w0`, and any cap it violates is reported, not fixed.
3. **D3.** Turnover penalty is L1, kept as its own parameter; the docs state that it merges with the linear cost term.
4. **D4.** Both liquidity caps are implemented: the trade-size cap (on by default) and the position-size cap (off by default).
5. **D5.** Assets with NaN or inf inputs are frozen at `w0_i` and listed in the report.
6. **D6.** Ledoit–Wolf uses the scaled-identity target; other targets go under Limitations.
7. **D7.** Constraint tolerance for property tests is 1e-6 in weight units, stated in the README.
