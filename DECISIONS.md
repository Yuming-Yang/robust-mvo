# Decisions

Design choices made while building, with the alternatives that were dropped. The scope decisions D1–D7 are in `SPEC.md`.

## 2026-11-15 — Milestone 2

**Scale the objective by 1 / average variance.** Found while writing the closed-form and no-trade tests. Daily-variance-sized objectives (~1e-4) sit close to the solvers' absolute gap tolerance, and the returned weights were less accurate than the 1e-6 constraint tolerance (D7). On three seeded QP-form problems (N = 30, linear plus quadratic costs, position and gross caps; `scripts/objective_scaling_check.py`), the maximum weight error against a reference solved with CLARABEL tolerances of 1e-14 was 3.5e-5, 2.4e-8 and 4.9e-7 unscaled, versus 4.3e-7, 2.4e-10 and 5.1e-9 scaled. On the 3/2-power form the comparison was inconclusive, because the tight-tolerance reference also stopped at "AlmostSolved". Alternative considered: tightening solver tolerances. Rejected, because tolerances are solver-specific, and scaling fixes the cause for every solver in the fallback chain.

**`solve()` captures solver warnings instead of letting them propagate.** CLARABEL returns `optimal_inaccurate` on many 3/2-power problems (DERIVATIONS.md 2.7). cvxpy emits a `UserWarning` in that case; the test suite turns warnings into errors. The warning is now recorded on `Solution.warnings`, and the decision to accept the solution moves to the post-check, as D2 specifies.

**Test-only finding: a test built a full-covariance problem and silently dropped the constraints.** The closed-form test reconstructed `Inputs` without passing `constraints`, so it compared a dollar-neutral solve with the unconstrained closed form. Fixed in the test. It is worth noting because `Inputs` defaults to `dollar_neutral=True`, which makes this mistake easy to make.
