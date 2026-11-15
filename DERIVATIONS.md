# Derivations

Every formula the code relies on, derived in enough detail to redo by hand. Each section says which file implements it and what is standard literature versus a choice made in this project.

Notation: `T` observations, `N` assets, `q = N/T`. `x_t ∈ R^N` is the demeaned return vector at time `t`, `X` is the `T × N` panel. `S = X'X / T` is the sample covariance (normalized by `T`, not `T−1`, to match scikit-learn). `Σ` is the true covariance. For Ledoit–Wolf the norm is the *normalized* Frobenius norm `‖A‖² = tr(AA')/N` with inner product `⟨A, B⟩ = tr(AB')/N`, so `‖I‖ = 1`.

## Part 1. Covariance estimation (`src/robust_mvo/covariance.py`)

### 1.1 Why the sample covariance is a bad input to an optimizer

**Rank.** `S = X'X/T` has rank at most `T − 1` after demeaning. If `N ≥ T` it is singular, and `Σ⁻¹` does not exist.

**The spectrum is too spread out, even when `S` is invertible.** Let `λ_min(·)` and `λ_max(·)` be the extreme eigenvalues.

- `λ_min(A) = min_{‖u‖=1} u'Au` is a minimum of linear functions of `A`, so it is concave in `A`. By Jensen, `E[λ_min(S)] ≤ λ_min(E S) = λ_min(Σ)`.
- `λ_max` is a maximum of linear functions, so it is convex: `E[λ_max(S)] ≥ λ_max(Σ)`.

So the smallest sample eigenvalues are biased down and the largest are biased up. This needs no distributional assumption.

**How much spread, in the simplest case.** Take `Σ = I` and i.i.d. standard normal entries, known mean (no demeaning). Then the average eigenvalue is `tr(S)/N`, with expectation 1. For the second moment:

- diagonal: `S_ii = (1/T) Σ_t x_ti²`, so `E[S_ii²] = Var(S_ii) + 1 = 2/T + 1`;
- off-diagonal: `S_ij = (1/T) Σ_t x_ti x_tj`, mean 0, so `E[S_ij²] = 1/T`.

`E[tr(S²)]/N = (1/N)[N(1 + 2/T) + N(N−1)/T] = 1 + (N+1)/T ≈ 1 + q`.

Since `tr(S²)/N` is the average squared eigenvalue, the eigenvalues have mean 1 and variance about `q`. All true eigenvalues are 1, yet the sample ones spread with standard deviation `≈ √q`. At `N = T` the spread is as large as the mean.

**Why the optimizer cares.** The unconstrained mean-variance solution is `w ∝ Σ⁻¹α = Σ_k (v_k'α / λ_k) v_k`. Each eigen-direction gets weight inversely proportional to its eigenvalue. The directions with the smallest sample eigenvalues get the largest weights, and those are exactly the eigenvalues biased down the most. The optimizer bets most heavily on the estimation error ("error maximization", Michaud 1989). Both stabilizers below work by pulling the spectrum back together.

### 1.2 Ledoit–Wolf linear shrinkage (Ledoit & Wolf, 2004)

*Standard literature.* Look for the best estimator of the form `Σ* = ρ₁ I + ρ₂ S` under expected squared normalized Frobenius loss.

Define four scalars:

- `μ = ⟨Σ, I⟩ = tr(Σ)/N`, the average true variance;
- `α² = ‖Σ − μI‖²`, how far `Σ` is from a scaled identity;
- `β² = E‖S − Σ‖²`, the estimation error of `S`;
- `δ² = E‖S − μI‖²`.

**Step 1: `δ² = α² + β²`.** Write `S − μI = (S − Σ) + (Σ − μI)`. The cross term `E⟨S − Σ, Σ − μI⟩` is zero because `E S = Σ` and `Σ − μI` is not random.

**Step 2: split the loss.** For fixed `ρ₁, ρ₂`,

```
E‖ρ₁I + ρ₂S − Σ‖² = ‖ρ₁I − (1−ρ₂)Σ‖² + ρ₂² β².
```

This holds because `ρ₁I + ρ₂S − Σ = [ρ₁I − (1−ρ₂)Σ] + ρ₂(S − Σ)`, and the cross term again has expectation zero.

**Step 3: optimize `ρ₁`.** Write `Σ = μI + (Σ − μI)`. Since `⟨I, Σ − μI⟩ = μ − μ = 0` and `‖I‖ = 1`,

```
‖ρ₁I − (1−ρ₂)Σ‖² = (ρ₁ − (1−ρ₂)μ)² + (1−ρ₂)² α².
```

The first term is zero at `ρ₁ = (1−ρ₂)μ`.

**Step 4: optimize `ρ₂`.** Minimize `(1−ρ₂)²α² + ρ₂²β²`. Setting the derivative to zero gives `ρ₂ = α²/(α² + β²) = α²/δ²`.

**Result.**

```
Σ* = (β²/δ²) μ I + (α²/δ²) S,   expected loss = α²β²/δ².
```

The shrinkage intensity is `β²/δ²`, the share of `S`'s distance from the target that is due to noise. More noise (large `β²`) means more shrinkage. A true covariance far from the target (large `α²`) means less.

**Step 5: plug-in estimates.** `Σ` is unknown, so each scalar is replaced by a consistent estimator as `N, T → ∞` together:

- `m = tr(S)/N` for `μ`;
- `d² = ‖S − mI‖²` for `δ²`;
- `b̄² = (1/T²) Σ_t ‖x_t x_t' − S‖²` for `β²`. Reason: `S` is the average of the `T` matrices `x_t x_t'`, so its variance is `1/T` times the variance of one term, and the variance of one term is estimated by the average of `‖x_t x_t' − S‖²` over `t`;
- `b² = min(b̄², d²)`, so the intensity cannot exceed 1;
- `a² = d² − b²`.

The estimator is `Σ_LW = (b²/d²) m I + (a²/d²) S`. The code calls the intensity `delta = b²/d²`.

**Computing `b̄²` in O(TN) extra work.** Expanding with `Σ_t x_t x_t' = T S`:

```
Σ_t ‖x_t x_t' − S‖²_F = Σ_t ‖x_t‖⁴ − 2⟨Σ_t x_t x_t', S⟩_F + T‖S‖²_F = Σ_t ‖x_t‖⁴ − T‖S‖²_F
```

(`F` is the unnormalized Frobenius norm and `‖x_t x_t'‖²_F = ‖x_t‖⁴`). After the `1/N` normalization and the `1/T²` factor,

```
b̄² = ( (1/T) Σ_t ‖x_t‖⁴ − ‖S‖²_F ) / (T N).
```

This is the line in `ledoit_wolf()`. The tests check it against `sklearn.covariance.LedoitWolf` to 1e-10.

**What it does to the spectrum.** `Σ_LW` has the same eigenvectors as `S`, and each eigenvalue becomes `δm + (1−δ)λ_k`. The whole spectrum is pulled linearly toward its mean. The smallest eigenvalue is at least `δm > 0`, so the result is positive definite whenever `δ > 0`, even when `N > T`.

**Project choice (D6).** The target is the scaled identity, which keeps an exact test oracle. Its weakness is that equity correlations are mostly positive, so a target with zero correlation is far from the truth. Constant-correlation or single-index targets (Ledoit & Wolf 2003, 2004) fit equities better and are listed as an extension.

### 1.3 Eigenvalue clipping (Laloux, Cizeau, Bouchaud & Potters, 1999)

*Standard literature.* For a `T × N` matrix of i.i.d. unit-variance noise, as `N, T → ∞` with `q = N/T ≤ 1` fixed, the eigenvalues of the sample correlation matrix fill the Marchenko–Pastur interval

```
[ (1 − √q)², (1 + √q)² ].
```

Two checks with section 1.1. The mean of the distribution is 1 and its variance is `q`. The width of the interval grows like `4√q`, which matches the `√q` standard deviation found by hand. The edge formula itself is a random-matrix-theory result, not derived here (Marchenko & Pastur 1967; see Bun, Bouchaud & Potters 2017, §3, for a readable account). For `q > 1` the upper edge is the same, and `N − T` eigenvalues are exactly zero.

**The procedure (`clip_eigenvalues`).**

1. Convert to correlation: `C = V^{-1/2} S V^{-1/2}`, with `V = diag(S)`. Working in correlation units makes the noise variance 1, so the edge is simply `λ₊ = (1 + √q)²`.
2. Eigen-decompose `C = Σ_k λ_k v_k v_k'`.
3. Treat every `λ_k ≤ λ₊` as noise. Replace all of them with their mean `λ̄`.
4. Rebuild `C̃ = Σ_k λ̃_k v_k v_k'`, rescale it to unit diagonal, and return `V^{1/2} C̃ V^{1/2}`.

**The trace is preserved.** Replacing a group of numbers with their mean keeps their sum, so `tr(C̃) = tr(C) = N` before the rescaling. The rescaling then forces each diagonal entry to 1, which keeps the trace at `N`.

**The result is positive definite, even for N > T.** Every eigenvalue that survives is above `λ₊ ≥ 1`. Every zero eigenvalue is below `λ₊`, so it gets clipped. If `k` eigenvalues are kept with sum `K < N`, the clipped ones share the remaining `N − K > 0` equally. So every eigenvalue of `C̃` is positive. Rescaling by a positive diagonal matrix keeps it PD.

**Linear shrinkage versus clipping.**

| | Ledoit–Wolf | Clipping |
| --- | --- | --- |
| Large eigenvalues (real factors) | pulled toward the mean by a factor `(1−δ)` | kept |
| Small eigenvalues (noise) | raised, but still ordered | flattened to one value |
| Tuning | none (δ is estimated) | none (edge from `q`) |
| Assumption | target structure | noise is i.i.d., with variance 1 after standardizing |
| Exact test oracle | scikit-learn | none; tested by properties (trace, PD, unit diagonal, market factor kept) |

**Project choices.** Variance is taken from `S` unchanged. Only the correlation structure is cleaned, as in the original paper. The edge uses noise variance 1 rather than the refinement `σ² = 1 − λ_max/N`, which removes the market mode's share of the variance first. That refinement would lower the edge and clip slightly fewer eigenvalues; it is noted as an extension.

### 1.4 Missing data: listwise, not pairwise

*Project choice.* `prepare_returns` drops assets that are missing more than `max_missing` of their rows, then drops every row that still has a gap, then drops zero-variance assets.

The tempting alternative is pairwise-complete estimation: each `S_ij` uses the rows where both `i` and `j` are observed. Each entry is then an inner product over a *different* set of rows. The matrix is no longer `X'X/T` for any single `X`, so it need not be PSD.

Example (the test `test_pairwise_complete_can_be_indefinite`): on their shared rows, `a` and `b` move together, `b` and `c` move together, and `a` and `c` move opposite. Pairwise correlations are `+1, +1, −1`. No three random variables can have these correlations, and the matrix has a negative eigenvalue. A negative eigenvalue means some portfolio appears to have negative variance, and an optimizer will find it.

Listwise deletion is always PSD but discards data. The `max_missing` threshold controls the trade-off by sacrificing a sparse asset rather than many rows.

### 1.5 PSD repair (`psd_floor`)

*Project choice.* A numerical guard, not an estimator. Symmetrize, eigen-decompose, raise every eigenvalue below `ε · λ_max` (default `ε = 1e-10`) to that floor, and rebuild. It is applied where a matrix must be PD for an algebraic step (the Cholesky factor of `F` in `risk_model.py`), and the number of raised eigenvalues is reported. The nearest-correlation-matrix algorithm (Higham 2002) is the principled alternative for correlation inputs and is not used in v1.

### 1.6 Cost

| Step | Cost |
| --- | --- |
| `S` | `O(T N²)` |
| Ledoit–Wolf intensity | `O(T N)` on top of `S` |
| Clipping (one symmetric eigen-decomposition) | `O(N³)` |
| `psd_floor` | `O(N³)` |

For the optimizer, the factor form keeps per-solve work at `O(N K)` (Part 2). The `O(N³)` steps happen once per estimation, not once per solve.

## Part 2. The optimization problem (`src/robust_mvo/optimizer.py`)

All weights are fractions of NAV, `z = w − w0` is the trade, and the problem is

```
maximize    α'w − γ (w'BFB'w + w'Dw) − λ‖z‖₁ − Σ_i c_i|z_i| − Σ_i k_i|z_i|^p
subject to  1'w = 0,  |w_i| ≤ u_i,  |z_i| ≤ τ_i,  |w_i| ≤ v_i,  ‖w‖₁ ≤ L.
```

`p ∈ {3/2, 2}` (D1). Every constraint is optional.

### 2.1 Factor-form risk

*Standard practice* (e.g. Grinold & Kahn; Boyd et al. 2017). With `Σ = BFB' + D`, `D` diagonal and `F = L_F L_F'` (Cholesky),

```
w'Σw = w'B L_F L_F' B'w + Σ_i D_i w_i² = ‖L_F' (B'w)‖² + ‖√D ∘ w‖².
```

The solver sees the `K`-vector `f = B'w` and two sums of squares. The problem data has about `NK + K² + N` nonzeros instead of the `N²` of a dense `Σ`, so per-solve work scales with `NK`. This needs `F` to be positive definite, which is why `risk_model.py` repairs `F` before the Cholesky step and records the repair.

In full-covariance mode (`B = I`, `F = Σ`, `D = 0`) the same code solves the dense problem. The tests use it to check the factor form against the dense quadratic form.

### 2.2 Closed form with dollar neutrality (the KKT check)

*Standard.* With no costs and no constraints other than `1'w = 0`, the problem is `max α'w − γw'Σw` subject to `1'w = 0`. The objective is strictly concave when `Σ` is PD, so the KKT point is the unique optimum.

Lagrangian: `ℒ(w, η) = α'w − γw'Σw − η·1'w`.

Stationarity: `α − 2γΣw − η1 = 0`, so `w = Σ⁻¹(α − η1) / (2γ)`.

Feasibility: `1'w = 0` gives `1'Σ⁻¹α = η·1'Σ⁻¹1`, so `η = 1'Σ⁻¹α / 1'Σ⁻¹1`.

Reading: `η` is the part of alpha shared by every asset, the "market" bet that dollar neutrality forbids. It is removed before inverting `Σ`. Without the constraint, `η = 0` and `w = Σ⁻¹α/(2γ)`. Both forms are tested against the solver in `test_matches_closed_form`, in factor and full-covariance mode.

Linking back to 1.1: `Σ⁻¹` is where small, noisy eigenvalues blow up the weights. Everything in Part 1 exists to make this inverse trustworthy.

### 2.3 Transaction cost terms

*Standard literature, with the exponent choice made in D1.* Model the price impact of trading `|z|` (as a fraction of NAV) as `I(z) = η σ (|z| / ADV)^β`, with `σ` daily volatility and `ADV` daily volume as a fraction of NAV. The cost of the trade is size times impact:

```
cost(z) = |z| · I(z) = η σ |z|^{1+β} / ADV^β.
```

| Impact model | β | Cost exponent `p = 1 + β` | Source |
| --- | --- | --- | --- |
| Linear temporary impact | 1 | 2 | Almgren & Chriss (2000) |
| Square-root law | 1/2 | 3/2 | Grinold & Kahn; Boyd et al. (2017) |
| Empirical fit | ≈ 3/5 | ≈ 1.6 | Almgren, Thum, Hauptmann & Li (2005) |

`square_root_impact()` computes `k_i = η σ_i / √ADV_i` for `p = 3/2`. The half-spread and fees are the linear term `c_i|z_i|`.

**Convexity.** For `p ≥ 1`, `t ↦ t^p` is convex and nondecreasing on `t ≥ 0`, and `|z|` is convex. A convex nondecreasing function of a convex function is convex, so `|z|^p` is convex. Subtracting it from a concave objective keeps the objective concave. This is the rule cvxpy's DCP check applies to `power(abs(z), 1.5)`.

**Conic form, and why OSQP drops out.** `t ≥ |z|^{3/2}` is the same as `|z| ≤ t^{2/3} · 1^{1/3}`. That is membership of `(t, 1, z)` in the 3-D power cone `{(x, y, z) : x^α y^{1−α} ≥ |z|, x, y ≥ 0}` with `α = 2/3`. CLARABEL and SCS handle power cones. OSQP only takes quadratic objectives with linear constraints, so it cannot represent the 3/2 term. The test `test_osqp_cannot_take_the_three_halves_form...` confirms this. With `p = 2` the term is `k z²` and the whole problem stays a QP, so OSQP is a valid fallback (`Inputs.is_qp()`).

### 2.4 L1 terms, and the no-trade region

*Standard (Lobo, Fazel & Boyd 2007), with the D3 consequence spelled out.*

**LP form.** `|z_i|` is handled by an epigraph variable: `|z_i| ≤ s_i` is the pair of linear constraints `−s_i ≤ z_i ≤ s_i`. Caps `|w_i| ≤ u_i` and `‖w‖₁ ≤ L` are linear in the same way.

**L1 turnover merges with linear cost.** `λ‖z‖₁ + Σ_i c_i|z_i| = Σ_i (λ + c_i)|z_i|`. A turnover penalty `λ` plus spread `c` is exactly the same problem as spread `c + λ` (`test_l1_turnover_merges_with_linear_cost`). The library keeps `λ` as a separate parameter because a turnover budget and a spread estimate are set by different people for different reasons. Mathematically they are one term.

**No-trade region.** Let `g(w)` be the smooth concave part (alpha minus risk), and drop the constraints. The objective `g(w) − Σ_i c_i|w_i − w0_i|` is concave, so `w` is optimal iff `0` is in its superdifferential. At `w = w0`, the subdifferential of `|w_i − w0_i|` is the whole interval `[−1, 1]`. So:

```
w0 is optimal  ⇔  |∂g/∂w_i (w0)| ≤ c_i  for every i,   with ∇g(w0) = α − 2γΣw0.
```

If the marginal benefit of trading an asset is smaller than its marginal cost, that asset does not trade at all. This produces exact zeros, which is what "sparse trades" means. An L2 penalty `λz²` has derivative `0` at `z = 0`, so it never produces an exact no-trade: every asset trades a little. `test_large_linear_cost_gives_no_trade` sets `c = 1.5·|∇g(w0)|` and checks `w = w0`. It then lowers one `c_i` below its threshold and checks that this asset trades.

### 2.5 Comparative statics without convexity

*Standard argument.* Let `x(t)` minimize `G(x) + t·H(x)` over any fixed set. Take `t₁ < t₂` with optimizers `x₁, x₂`. Optimality of each gives:

```
G(x₁) + t₁H(x₁) ≤ G(x₂) + t₁H(x₂)
G(x₂) + t₂H(x₂) ≤ G(x₁) + t₂H(x₁)
```

Adding the two and cancelling `G` gives `(t₂ − t₁)(H(x₂) − H(x₁)) ≤ 0`, so `H(x₂) ≤ H(x₁)`. The penalized quantity never increases when its weight increases. No convexity, differentiability or uniqueness is needed.

Applied here, written as minimizing the negative objective:

- `t = γ`, `H` = ex-ante variance: risk does not increase with `γ`;
- `t = c` (scaled), `H = Σ|z_i|`: traded volume does not increase with the linear cost;
- `t = k` (scaled), `H = Σ|z_i|^p`: same for impact;
- `t = λ`, `H = ‖z‖₁`: same for turnover.

The tests check these on grids of coefficients. Their tolerance accounts for the solver returning an approximate optimizer.

### 2.6 Objective scaling (project choice)

Positive scaling does not change the argmax: `argmax s·φ(w) = argmax φ(w)` for any `s > 0`. The code multiplies the objective by `s = 1 / (average asset variance)`, computed as `mean_i (diag(BFB')_i + D_i)` in `O(NK²)`.

Why: daily variances are about `1e-4`, so the unscaled objective is about `1e-4` too. Interior-point solvers stop on gap criteria that include an absolute part (CLARABEL's `tol_gap_abs` defaults to `1e-8`). Near the optimum the objective is roughly quadratic with curvature `2γΣ`. An objective error `ε` then allows a weight error of about `√(2ε / (2γλ_min(Σ)))`. With `ε = 1e-8` this can reach `1e-4` to `1e-3`, far above the 1e-6 tolerance in D7. On three QP-form test problems (`scripts/objective_scaling_check.py`), scaling reduced the maximum weight error against a tight-tolerance reference from up to `3.5e-5` to at most `4.3e-7` (recorded in DECISIONS.md).

### 2.7 Inaccurate solutions on the 3/2 form (observed)

On the 3/2-power form, CLARABEL often stops with "AlmostSolved", which cvxpy reports as `optimal_inaccurate`. A likely cause is that many trades sit exactly at `z_i = 0` because of the no-trade region (2.4), which is the tip of the power cone, where interior-point steps get short. The relative gap at stop is around `1e-8`, and the constraint post-check passes. Following D2, `solve()` records the status and the warning instead of raising. Acceptance is decided by the post-check (Part 3).
