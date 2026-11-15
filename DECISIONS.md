# Decisions

Design choices made while building, with the alternatives that were dropped. The scope decisions D1–D7 are in `SPEC.md`.

## 2026-11-15 — Milestone 2

**Scale the objective by 1 / average variance.** Found while writing the closed-form and no-trade tests. Daily-variance-sized objectives (~1e-4) sit close to the solvers' absolute gap tolerance, and the returned weights were less accurate than the 1e-6 constraint tolerance (D7). On three seeded QP-form problems (N = 30, linear plus quadratic costs, position and gross caps; `scripts/objective_scaling_check.py`), the maximum weight error against a reference solved with CLARABEL tolerances of 1e-14 was 3.5e-5, 2.4e-8 and 4.9e-7 unscaled, versus 4.3e-7, 2.4e-10 and 5.1e-9 scaled. On the 3/2-power form the comparison was inconclusive, because the tight-tolerance reference also stopped at "AlmostSolved". Alternative considered: tightening solver tolerances. Rejected, because tolerances are solver-specific, and scaling fixes the cause for every solver in the fallback chain.

**`solve()` captures solver warnings instead of letting them propagate.** CLARABEL returns `optimal_inaccurate` on many 3/2-power problems (DERIVATIONS.md 2.7). cvxpy emits a `UserWarning` in that case; the test suite turns warnings into errors. The warning is now recorded on `Solution.warnings`, and the decision to accept the solution moves to the post-check, as D2 specifies.

**Test-only finding: a test built a full-covariance problem and silently dropped the constraints.** The closed-form test reconstructed `Inputs` without passing `constraints`, so it compared a dollar-neutral solve with the unconstrained closed form. Fixed in the test. It is worth noting because `Inputs` defaults to `dollar_neutral=True`, which makes this mistake easy to make.

## 2026-11-15 — Milestone 3

**Report two kinds of violation for frozen assets.** Found by the chain's property test: an asset with NaN alpha was frozen at a `w0` above its position cap, so a level-0 solve appeared to break the cap. The choices were (a) force the frozen asset to the cap, which is a trade the system has no information to make and which D5 rules out; (b) report the violation as if the optimizer caused it; or (c) widen the frozen asset's cap to `|w0_i|`, measure `report.violations` against the constraints actually imposed, and list the original breach in `report.frozen_cap_breaches`. (c) was chosen because it keeps "level 0 or 1 means every imposed constraint holds" true and still surfaces the breach.

**Catch every exception from the solver call.** `solve()` already turns cvxpy errors into a status, but the chain also wraps `solve_fn` in a broad `except Exception`. A fallback chain whose purpose is to survive solver failures should not die on an unexpected exception type from a solver binding. Programmer errors (shapes, non-finite `w0`) are raised before the chain starts, so the broad catch cannot hide them.

**Check `F` before sanitizing.** `FactorRiskModel` rejects a non-finite `F` at construction, but the chain checks again before rebuilding the model, so a model mutated after construction leads to level 3 with a reason, not an exception.

**L1 slack in the elastic problem, not L2.** An L1 penalty is exact for a large enough weight: a feasible problem's optimum is returned unchanged (DERIVATIONS.md 3.1). An L2 penalty would always take some slack. The cost is that violations concentrate on a few assets instead of spreading.

**A pre-check that is necessary but not sufficient.** It gives a readable reason and skips solver calls in the common infeasible cases; the solver remains the authority. Considered: an exact LP feasibility solve as the pre-check. Rejected, since it is one more solver call that the chain's first solve already performs.
