# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/) and semantic versioning.

## [2.6.0] - 2026-10-26

### Added

- [z-rule] `z_rule="stratified"`, the default of `multiunit_views` and `joint_two_units` (spec section 4A): the shared factor is partitioned into strata of probability, `z_nodes` (default 32) of equal width over the lower three quarters and a geometric ladder of ratio √2 down to 2^-24 in the upper tail, 76 strata in all (`meta.z_strata`). Each unit's conditional law is averaged over the stratum exactly, through the bivariate normal cdf below loading one and through the marginal restricted to the stratum at loading one, so every unit's marginal and every view mean is exact at any `z_nodes`; the units are independent given the stratum, and the ladder holds that error on the event total's exceedance within 7e-3 at the 1e-2 to 1e-5 levels (`benchmarks/z-convergence-2.6.0.md`). The node rules stay selectable: midpoint at 32 was off by 10% to 100% in those tails, Gauss-Hermite at 32 is exact to 1e-9 where every loading is well below one and off by 0.15 in the marginal of a unit at loading one.
- The bivariate normal cdf in the engine (`normal::bivariate_cdf`, Genz 2004) and in the reference (Owen's T), agreeing within 5e-15; `exact_two_unit_joint` in the reference, the two-unit joint with no quadrature over the factor, the convergence study's reference in place of the midpoint rule at 4001 nodes.
- `benchmarks/z_convergence.py` rewritten with tail metrics (the event total's exceedance at the 1e-2 to 1e-5 levels), the three Homer events with their two-point rows, a lone unit at loadings 0.5 to 0.99, and units at loading one beside a partner; spec test 28 implemented on both backends.

### Changed

- The default factor rule moves from 32 midpoint nodes (and 1001 for the joint) to the stratified rule, so every multi-unit grid changes: the `multiunit_small` golden is regenerated. Equal-probability strata, the plan's first form of the rule, were measured and rejected: with the whole upper tail of a total inside one stratum the exceedance at 1e-4 and 1e-5 was short by 50% to 99% even at 128 strata.
- Cost: on the 0.91M unit-row benchmark portfolio the default takes 8.6 s against 2.5 s under midpoint 32 and 2.0 s under Gauss-Hermite 32 (76 strata against 32 nodes, plus the bivariate normal per edge and stratum); `z_nodes=8` (52 strata) takes 6.6 s with the same tails.

## [2.5.0] - 2026-10-26

### Changed

- [bernoulli] The multi-unit route treats an inadmissible unit row as the one-row route has since 1.13.0: the two-point law on 0 and M with `P(X = M) = mu / M`, given the shared factor the mass `1 - Phi((Phi^{-1}(1 - p) - w z) / sqrt(1 - w^2))` at M (a step in z at loading one), Burke's loading kept (spec section 4A). `variance_policy="bernoulli"` is the default of `multiunit_views` and `joint_two_units`, reported under `variance_bernoulli`; the rolled row of the roll-up diagnostic classifies the same way. Rows the clamp used to smear across the interior now sit on 0 and M, so a portfolio with inadmissible rows changes numbers; the goldens hold, since the fixture has none.
- `variance_policy="clamp"` on the multi-unit route is deprecated: still accepted, with a `DeprecationWarning`, until 3.0.0.
- Spec test 26 (both backends): the two-point conditional law, its marginal, and the three events of Homer and Li 2017, Table 11 against the multi-unit article's own construction within 1e-12 per cell at 64 Gauss-Hermite nodes.

## [2.4.3] - 2026-10-26

### Changed

- [tidy] Four pieces of redundant work removed, each bitwise neutral on the outputs and measured on the benchmark portfolios (`benchmarks/report-2.4.2.md`): the survival grid walks the curve's intervals with a cursor and writes the bucket differences as it goes, instead of a binary search per edge and a stored value vector (the large-exposure portfolio at 2^22 buckets, 1.20 s to 1.15 s); the tail interpolant takes the logarithm of the distance from its caller once, not twice; the midpoint rule's normal scores are computed once per quantization and shared by every event on the Cornish-Fisher route (a million large-parameter events single-threaded, 0.43 s to 0.11 s); the mass and first moment of each output grid are taken in one pass, and the secondary moment is not taken twice. Mixture and small-alpha portfolios 2% to 3% faster.
- `beta_midpoint_quantiles_scored` and `MidpointScores` on the Rust kernel; `beta_midpoint_quantiles` is unchanged and still computes the scores itself.

## [2.4.2] - 2026-10-26

### Changed

- [schedule] The quantiles method's default chunk is a rule of the input and the grid, `clamp(max(ceil(rows / 64), ceil(n / 64)), 4096, 65536)` rows, in place of the fixed 65,536, which gave 262,144 rows four chunks and left most of a 32-thread machine idle (peer review, finding P1). On that input at 65,536 buckets the quantiles method takes 0.14 s against 0.65 s before; on a million rows 0.48 s against 0.76 s; from 2^22 buckets the chunk is the old one and nothing changes (`benchmarks/report-2.4.2.md`). The rule depends on the input and the configuration, never on the thread count, so results stay bitwise reproducible across threads; against the fixed chunk a grid differs by summation order only, within 1e-14 of the total rate.
- A pass runs its chunks in waves of `min(2 * threads, 1 GiB / partial size)` (`Config::wave_memory_bytes`), so the partials in flight are bounded by memory as well as by threads: at 2^24 buckets a quantiles partial is 256 MiB and 64 in flight were 16 GiB. The grid-length vectors of a wave's partials are folded in parallel over disjoint blocks, each element in chunk order, so the fold no longer serializes behind the kernel; the result is bitwise the serial fold's. One thread pool serves a call (three were built, one per pass), and an empty pass returns before any work is scheduled.
- `benchmarks/run_schedule.py`, the scheduling sweep (time and peak memory by row count, thread count, chunk rule and grid length); the README's claim that grid length is irrelevant to throughput is qualified: deposits are local, but each chunk's two partial grids are initialized and folded at the grid's length.

## [2.4.1] - 2026-10-26

### Fixed

- [ratios] The admissibility test `sigma^2 >= mu (M - mu)` and the shape parameters are formed in ratios of the row's fields on both backends, so that no product of two monetary amounts appears. Scaled by 1e160 the admissible row (M, mu, sigma) = (1, 0.4, 0.1) was a Bernoulli row and scaled by 1e-200 an atom, every field finite; both are now the beta the unit-scale row is (peer review, finding I3; spec test 10 extended to those scales). The same test on the multi-unit route. Grids at ordinary scales move by rounding only (the goldens hold at 1e-9).
- The Cornish-Fisher cumulants' standard deviation is formed in ratios: the product form's denominator overflowed from `alpha + beta = 1e103`, inside the quantiles method's stated 1e150 range, and returned 0 for asymmetric shapes there.
- Documented, not changed: an atom in a windowed grid's top half-bucket goes to the top node and its displacement is not overflow (spec section 6, README); below the switch point the incomplete beta's survival is a complement with absolute error of order epsilon, which loses relative precision only for alpha below about 1e-12, where the event's whole contribution is below the 1e-12 floor (spec section 5A).

## [2.4.0] - 2026-10-26

### Added

- [mesh] `nodes_per_sigma` (default 12), the survival method's accuracy knob, on `quantize_arrays`, `quantize_csv` and `exceedance_curve`. A bell-shaped event (both beta parameters above 1) narrower than that many base intervals asks for that many nodes per standard deviation around its mean and is carried on a second curve over a fine mesh; the base curve and the small-alpha events never see the fine nodes. The contract (spec test 33): within 1e-4 relative of each such event's survival where it is at least 1e-4, 1e-3 where at least 1e-8, 3e-3 where at least 1e-12, measured over alpha from 1.5 to 1e4 and coefficients of variation from 1e-3 to 0.2 (`benchmarks/report-2.4.0.md`, which also tabulates 8, 16 and 20). Before, the shared cubic was off by 1e-3 to 1e-2 in such an event's tail at the thousand-year band and by 22.8% on the review's Beta(1800, 1e6) case (peer review, finding A3); the review's single case was a sample of a regime, not an edge case.
- `Meta.fine_nodes` and `Meta.requesting_events` (survival method), and the same on `CurveMeta`; `benchmarks/run_accuracy.py`, the accuracy sweep behind the contract.

### Changed

- The narrow route requires both parameters at least 2 (was above 1): its quadrature was off by up to 5e-4 for a parameter between 1 and 2, where the tail ladder is within 1e-5; such events stay on a curve. The plateau below an event's lower skip is one difference per event, not a visit per node.
- The survival goldens are regenerated: the medium golden moves by at most 6.9e-7 of the total rate (522 of its 4,909 contributing rows are requesting events) and the small golden by 1.4e-5 (two requesting rows); on the medium fixture's beta rows the tail sums are now within 2.0e-5 of the exact exceedance at every bucket edge (4.9e-5 at 2.3.6).
- Cost: on the mixture benchmark portfolio (10.6% of events requesting) the survival method takes 1.8 times as long at the default (1.90 s against 1.05 s for a million rows); on the small-alpha and large-exposure portfolios it is 3% to 7% faster. `nodes_per_sigma=1` turns the fine mesh off and restores the 2.3.6 behavior and speed.

## [2.3.6] - 2026-10-26

### Fixed

- [tails] The endpoint tail's power law below its last ladder node is used only where the endpoint expansion's first correction `c t`, with `c = (alpha - 1) beta / (beta + 1)`, is at most 1e-6 there; otherwise the one bucket edge that can lie below the node is evaluated exactly. The pure power law's relative error is `c t`, 0.17 on Beta(1e6, 1/2) at an edge 5e-7 below the exposed value, where the grid put 0.054 of the event's rate above the edge against an exact 0.683 (peer review, finding A2). Both backends.
- Where `c > 1` and `c t` lies between 1e-2 and 1e2 the ladder steps by 2^(1/4) instead of sqrt(2): the cubic in log distance was off by 3e-4 at beta = 5 there, 2e-5 now. Small-alpha events (`c < 1`) are untouched.
- The survival goldens are regenerated: the fix moves the medium golden by up to 3e-6 of the total rate (the top bucket below the exposed value of its bell-shaped events, where the power law was extrapolated from a quarter bucket away) and the small golden by 4e-8; on the medium fixture's beta rows the tail sums are now within 4.9e-5 of the exact exceedance at every bucket edge. The quantiles goldens' metadata gain the `narrow_events` key they lacked, and the multi-unit golden moves by 4e-16 relative (rounding of the regeneration).

## [2.3.5] - 2026-10-25

### Fixed

- [overflow] The survival grid's reported excess moment above a windowed grid (`mean_above_grid["secondary"]`) is now the integral of the interpolated curve above the top edge plus the tails' partial expectations, instead of a sum over the per-event tail records. An event whose endpoint survival underflows in double precision has no tail record, so its share of the moment was lost; on a real ELT, where most events have a small mean against a large exposed value, that was most of them (peer review, finding I1: Beta(2, 1000) windowed at 0.00195 reported 0 against an exact 5.80e-4, which at an exposed value of 1e6 is 580 per unit rate). The reference backend already summed every event exactly and is unchanged.
- Grids are unchanged; only the reported moment moves, and only when the grid is a window. The engine's figure is now within the interpolation tolerance of the reference's (1e-4 relative on the medium fixture windowed at a third of its largest exposure; test 18).

## [2.3.4] - 2026-10-25

### Fixed

- [atoms] A narrow event with both parameters at or above 1e16 takes the standardized normal law in closed form (own `erf`) instead of a step at its mean; the step, applied above `alpha + beta = 1e15` since 2.3.0, rested on a wrong resolution argument and put a symmetric event whose mean sits on a bucket edge wholly into one bucket where its `sigma = 0` twin split half and half (peer review, finding A1). The quadrature kernel stays below 1e16; the two agree to a few units in 1e-6 at the handover. `NARROW_STEP_PARAMETER_SUM` is retired; `NARROW_NORMAL_FROM` replaces it in both backends.
- A beta row whose standard deviation is below 2^-48 of its mean (sixteen ulps of the loss at the mean) is an atom at the mean, deposited by the linear rule like its `sigma = 0` twin, under the new diagnostic category `atom_by_resolution` (spec section 4). This also restores the overflow moment the step lost (the review's I1 step case: rate 1, moment 0; now 0.15 exact).
- The reference backend takes the normal law from 1e16 as well, where SciPy's incomplete beta is itself off by 1e-4 at 1e24, and guards the isolated NaN SciPy returns at the mean when both parameters approach 1e16.
- Grids unchanged at default settings on every fixture; no fixture holds a row in either regime.

## [2.3.3] - 2026-10-25

### Fixed

- [kernel] The engine's incomplete beta reports a continued fraction that has not converged instead of returning its last iterate: the probabilities become NaN and the survival method raises a typed error naming the row. Before, a symmetric beta with `alpha = beta = 1e9` at 200,000 nodes per decade returned bucket weights of -0.45 and 1.45 (peer review, finding I2).
- The survival method keeps an event with `min(alpha, beta)` at or above 2e5 out of the shared curve whatever the node density (the narrow route, whose quadrature does not use the fraction), so the fraction is never asked to work where it fails; before, the routing depended on the density alone. Both backends; spec section 5A and test 30.
- The fraction's iteration cap rises from 400 to 1,000: at the threshold a lopsided event needed 443, so the regime the engine calls valid was not fully covered. No cost where the fraction converges fast, which is everywhere one parameter is small.
- The quantiles method's robust inverse walk accepts a best-so-far point when it exhausts its iterations only if the point's CDF residual is within 1e-8 (the fraction's own rounding near the threshold); otherwise it fails typed. Grids are unchanged at default settings on every fixture.

## [2.3.2] - 2026-10-25

### Fixed

- [large-parameters] The quantiles method (and the multi-unit beta units) at large parameters: from `min(alpha, beta) = 2e5` every beta quantile comes from the Cornish-Fisher expansion in the beta's cumulants (Abramowitz and Stegun 26.2.49) instead of inverting the engine's incomplete beta, whose continued fraction stops converging there and whose front factor overflows beyond `alpha + beta` of about 5e18. Before, such rows got wrong points silently (a third of the rate at dyadic fractions of M) or failed with "beta quantile evaluation failed"; which of the two depended on rounding, which is why 2.3.1 failed on rows 2.3.0 accepted. Within 4e-9 standard deviations of SciPy on both sides of the threshold (spec section 5, test 24); goldens unchanged; cheaper than the walk it replaces.
- The reference backend takes the same route: SciPy's `beta.ppf` returns non-monotone points hundreds of standard deviations off at `alpha = 5e20`, so it was no oracle there either.
- Downstream: 32-point results change for rows with `min(alpha, beta)` above about 5e5, a correction of 2.3.0's points; the survival method is unaffected, such events being narrow.

## [2.3.1] - 2026-10-24

### Fixed

- [ln-beta] The engine's incomplete beta at a large parameter: the constant `ln B(alpha, beta)` is now the difference of two Stirling series for the large parameter (at or above 100) plus `ln Gamma` of the small one, instead of three `ln Gamma` values whose cancellation cost seven digits at beta = 1e8. On real events with alpha below 2 and beta from 1e7 to 8e7 the survival at the nodes moves from 3e-6 relative off SciPy to 1e-13 below the continued fraction's switch point and the formulation's eps times beta (1e-8 at beta = 8e7) above it (spec section 5A, test 23). Portfolio grids change in rounding where such events sit; the goldens are unchanged. No cost: the constant is computed once per event as before.
- The continued fraction itself was not at fault: with one parameter small it converges in tens of iterations at any beta. The 2.3.0 note on its failure above 1e7 applies when both parameters are large, the narrow events' regime.
- Found on the way: SciPy's `betaincc` is off by 5e-9 against mpmath at an integer alpha of 2 or more with beta = 8e7 (6e-13 at beta = 1e4), so the reference backend carries that there; test 23 uses mpmath as its oracle and `mpmath` joins the dev dependencies.

## [2.3.0] - 2026-10-24

### Changed

- [narrow-events] A beta whose standard deviation is below one node interval at its mean (2.3% of the loss at the default density) is kept out of the shared curve, which could not follow its step and was off by up to the event's rate across one interval, and is walked along the bucket edges of its bulk (a few standard deviations each side of the mean, to the rate floor) with its survival by quadrature of its density. `exceedance(x)` is exact for such events to about 1e-12; the survival grid changes near their means (spec section 5A, test 22). Events with a parameter at or below 1, or whose bulk does not reach the floor within 40 standard deviations on either side, take the regular route.
- Found on the way: the engine's incomplete beta (a continued fraction of 400 iterations) is good only to about 1e-10 near the mean at `alpha + beta` of 1e4 and fails above 1e7, where SciPy is right. The narrow events are the one place the engine evaluates such a beta inside its bulk, and they no longer use it; the regular routes still do for the rare events the narrow criterion rejects.
- The survival goldens `small_survival` and `medium_survival` are regenerated (the medium fixture has four narrow events; its grid moves by up to 2.7e-6 of rate in buckets near their means, the total unchanged).

### Added

- `narrow_events` in `Meta` (survival method; None under quantiles) and in `CurveMeta`: the count of events treated this way.

## [2.2.0] - 2026-10-24

### Changed

- [edge-cost] The survival grid's tail pass balances across threads: each chunk's histogram covers only the bucket span its tails touch instead of the whole grid, and the survival method's default `chunk_rows` is 256 (was 4,096), so long tails spread over many cheap chunks. Results stay bitwise identical across thread counts at a fixed `chunk_rows`; at the new default they differ in rounding from 2.1.0 for inputs over 256 rows (the fold order). On the large-exposed-value benchmark portfolio the survival grid falls from 5.1 s to 1.4 s, 1.5 times the quantiles method at K = 32 (16 threads scaled 4.5 times before, 13 times now).
- Not done, measured unnecessary: the plan's merge of the edges with the nodes for the curve part (the fixed cost of a 2**22 grid is 0.1 s) and a cubic in `x` for the tails.

## [2.1.0] - 2026-10-24

### Changed

- [tail-cut] The survival grid walks each event's tail along the bucket edges only while the event's exceedance rate exceeds 1e-20 of the total beta rate (`TAIL_RATE_FLOOR`), and only up to the top edge of the grid; the remainder is lumped into the next bucket. The rate stays exact and the tail sums move by at most N times that fraction of the total rate (spec section 5A, test 21). On a 100,000-row portfolio with endpoints to 1e9 at `h = 25` and 2**22 buckets the survival grid falls from 240.7 s to 5.1 s (269 to 5.7 times the quantiles method at K = 32).
- `ExceedanceCurve.exceedance` is unchanged and exact; the grid now departs from it by the bound above where the cut applies. Grids with a window short of the largest endpoint change in rounding only (the overflow is one deposit per tail instead of one per edge).
- The tail interpolant is built once per event with its limited slopes fixed per interval and walked with a cursor instead of a search per edge.

### Added

- `benchmarks/run_end_to_end.py --survival-sweep --large-m --rows 100000`: the large-exposed-value portfolio at `h = 25` and 2**22 buckets, recorded before and after the cut in `benchmarks/results.json`.
- Reference only: `SurvivalCurve.bucket(..., tail_rate_floor=0.0)` walks every tail to its endpoint, for test 21.

## [2.0.1] - 2026-10-24

### Fixed

- [oep-pml] `ExceedanceCurve.oep_pml` returned the largest atom's loss at every return period whose answer lies above every atom (the clipped atom search). It now returns the bisection answer there; an atom whose step jumps the target is still returned as before (`dev/done/plan-survival-tail-cost.md`, finding 3).

## [2.0.0] - 2026-10-24

Major: the default `secondary` output changes, and the clamp policy leaves the one-row route.

### Changed

- [default] `method="survival"` is the default of `quantize_arrays`, `quantize_csv`, `quantize_dataframe`, `quantize_into`, `quantize_to_npy` and the reference, in Rust (`Config::new`) and in the bindings. The K-point rule stays as `method="quantiles"`; its tail cut (spec section 5) loses tens of percent of the expected loss on small-alpha events, which the survival method does not (`benchmarks/report-1.16.0.md`).
- The `clamp` variance policy and its `clamp_factor` keyword are removed from the one-row route (`quantize_*`, `exceedance_curve`); `variance_policy="clamp"` raises, in Rust (`Error::ClampPolicyRemoved`) and in Python. The multi-unit route keeps `clamp` with `clamp_factor` and its `sigma_clamped` category, as ruled on 2026-10-24.
- Goldens: `small` and `medium` now record `method: quantiles`; `small_survival` and `medium_survival` are the same fixtures under the default. The acceptance tests of the K-point rule (4, 5, 7, 9, the dyadic case of 10) select it; test 11's clamp case checks the refusal; test 14 runs all four goldens on both backends and the Rust golden test too.
- README, the module docstring and the project description describe the secondary grid by the survival method.
- `reproductions/Homer2017a.qmd`: the sentence on the default variance policy describes the Bernoulli treatment (the chapter's numbers were computed with the clamp of 1.12.0 and say so).

## [1.17.0] - 2026-10-24

### Added

- [curve] `eltq.exceedance_curve(rate, max_loss, mean_loss, sd_indep, sd_corr, *, min_loss, ...)` builds the portfolio exceedance curve once, from `min_loss` up, on either backend. The `ExceedanceCurve` it returns exposes the nodes (`loss`) with the exact exceedance rate of the beta events at them (`exceedance_rate`), the atoms (`atom_loss`, `atom_rate`), a `CurveMeta`, `exceedance(x)` (atoms included; NaN below `min_loss`), `grid(bucket_size, grid_length=None, *, normalize=False, overflow="top_bucket")` returning the secondary grid and a `GridMeta` for any `bucket_size >= 2 min_loss`, and `oep_pml(return_periods)`, the occurrence exceedance loss under Poisson occurrence by bisection, an atom's loss where that atom's step jumps the target.
- The curve is now the one implementation of the survival method: `quantize(method="survival")` builds it at `min_loss = h / 2` and takes its grid, so the two agree bitwise, and the separate survival pass of 1.15.0 is gone.

### Changed

- The curve keeps each event's tail parameters (about 56 bytes per beta row), so the survival method's peak memory grows by that; the Rust engine's `Classified` rows and `classify_rows` are public for callers that build curves from their own inputs.

## [1.16.0] - 2026-10-24

### Changed

- [benchmark] The survival method's default node density is 100 nodes per decade (was 150): the smallest density whose bucketed tail sums match the exact incomplete beta at every bucket edge within 1e-4 on every benchmark portfolio (50 misses on the J-shaped ones). `benchmarks/report-1.16.0.md` has the measurements.
- Speed gate met: on the million-row mixture portfolio the survival method at the default density takes 0.96 s against 0.78 s for the quantiles method at K = 32 (1.23 times; the plan's limit was five, its estimate 5 to 14), and 2.65 times on a million-row small-alpha portfolio, where the quantiles method loses 20% of the expected loss and the survival method 0.6% (the first-order bucket-0 error at that grid).

### Added

- `benchmarks/run_end_to_end.py --method`, `--nodes-per-decade`, and `--survival-sweep [--small-alpha]`: the survival method at 25, 50, 100 and 150 nodes per decade against the quantiles method at K = 32, on the mixture portfolio or a million-row small-alpha portfolio, recorded under `"suite": "survival_sweep"` in `benchmarks/results.json`.

## [1.15.0] - 2026-10-24

### Added

- [rust] The survival method in the Rust engine (`crates/eltq-core/src/curve.rs`): the same nodes, truncation, one-sided slopes, monotone cubic and per-event endpoint interpolant as the reference, in chunked passes folded in order, so results are bitwise identical across thread counts at a fixed `chunk_rows`. `method="survival"` no longer needs SciPy on the Rust backend.
- The overflow policies in the Rust engine for both methods and both outputs, with `rate_above_grid` and `mean_above_grid` reported; `overflow="error"` keeps today's `GridOverflowError`.
- `BetaFn::sf`, the survival function computed directly in each branch of the continued fraction (full relative precision in the upper tail), and `sf_and_density_times_y` sharing its logarithms.
- `Config` gains `method`, `nodes_per_decade` and `overflow`; `chunk_rows` is `Option<usize>`, resolved to 65,536 rows for `quantiles` and 4,096 for `survival`. `Meta` carries the method, node count, density, interpolation error, overflow policy and the per-output overflow figures.
- The per-event endpoint interpolant now steps the distance to the endpoint by sqrt(2) rather than halving it, in both backends: a few units in 1e-6 on an event's tail instead of 2e-5.

### Changed

- `secondary_mean_error` on the Rust backend is the secondary grid's first moment less the analytic AAL under either method (identical to the quadrature figure under `quantiles` to rounding).
- Spec test 18 runs on both backends under both methods; test 17 compares the backends' survival grids and curves.

## [1.14.0] - 2026-10-24

### Added

- [reference] The survival method in the SciPy reference: `quantize_reference(..., method="survival", nodes_per_decade=150, overflow="top_bucket")` builds the portfolio exceedance rate exactly at geometric nodes, interpolates it by a monotone cubic Hermite with exact one-sided slopes, and buckets it by the nearest rule (spec section 5A). Each event is truncated four node intervals before its endpoint and its tail is carried per event by an interpolant in the distance to the endpoint; without that, no shared interpolant followed the endpoint kinks (execution log of `dev/done/plan-single-unit-survival-route.md`). Accuracy against the exact incomplete beta at every bucket edge, down to 1e-12 of the total rate: within 4e-6 on the new small-alpha fixture and 1.1e-5 on the medium golden and the benchmark portfolio.
- `quantize_arrays`, `quantize_csv` and their wrappers take `method`, `nodes_per_decade` and `overflow`. On the Rust backend `method="survival"` runs on the reference until 1.15.0 (SciPy required); the overflow policies are honored by the Rust engine from 1.15.0 too, and until then its quantiles method raises on overflow as before.
- Overflow policies `top_bucket` (default), `drop` and `error` in the reference, for both methods and both outputs, with `meta.rate_above_grid` and `meta.mean_above_grid` per output; the derived grid length still covers every loss, so the policy acts only on an explicit window.
- `Meta` gains `method`, `nodes`, `nodes_per_decade`, `interpolation_error`, `overflow`, `rate_above_grid` and `mean_above_grid`; `quadrature_points` and `quadrature_secondary_mean` are None under the survival method. `secondary_mean_error` is defined for both methods as the secondary grid's first moment less the analytic AAL (unchanged in value for the quantiles method).
- Fixture `fixtures/small_alpha_elt.csv` (2,000 small-alpha rows, ground-up and gross-like) and spec tests 15 to 20 in `tests/test_survival.py`, on both backends.

### Changed

- The default method stays `quantiles`; the survival default lands at 2.0.0. The specification's section 5A now describes the truncated-curve design as built.

## [1.13.0] - 2026-10-24

### Changed

- [bernoulli] Rows whose stated variance is inadmissible (sigma^2 >= mu (M - mu)) now become Bernoulli rows by default: two atoms, rate (1 - p) at 0 and rate p at M with p = mu / M, the variance maximizer on [0, M]. Rate and mean are exact; the row is reported under the new `variance_bernoulli` category. The 0.999 sigma clamp stays selectable as `variance_policy="clamp"` on the one-row route until 2.0.0 and remains the multi-unit route's default.
- The multi-unit route (`multiunit_views`, `joint_two_units`) rejects `variance_policy="bernoulli"` with a typed error until it adopts the treatment (plan "Not in this plan").
- The `small` and `medium` goldens are regenerated. The secondary mean of `small` rises from 4.169 to 4.194: the two inadmissible rows no longer lose mean to the clamp's smeared beta.
- The specification was amended in the preceding commit: section 4 proves the variance bound and lists the policies, section 5A specifies the survival method that lands at 2.0.0, section 6 the overflow policies (`dev/done/plan-single-unit-survival-route.md`, now executing).

## [1.12.0] - 2026-10-22

### Added

- [release] First release on PyPI: `pip install eltq` (or `uv add eltq`) installs a prebuilt wheel on Windows x86_64, Linux x86_64 and macOS arm64, for Python 3.12 and later, with no Rust toolchain. `eltq[reference]` adds SciPy for the reference backend. Rolls up 1.11.5 to 1.11.7.
- Each wheel passes the full suite against the SciPy oracle on its own platform in CI. The 1.11.7 wheels from TestPyPI pass on this machine too, with every benchmark output bitwise identical to a local build.
- Speed of the `x86-64-v3` wheel against a local native build on Zen 5: the multi-unit route is about 8% slower. The one-row headline is 14% faster (1M rows in 0.79 s, against 0.92 s), because CI compiles with Rust 1.99.1 and this machine still has 1.98.0.
- Plan `dev/done/plan-pypi-release.md` complete on this side; the first upload happens when Steve pushes the `v1.12.0` tag.

## [1.11.7] - 2026-10-22

### Added

- [ci] GitHub Actions workflow `.github/workflows/wheels.yml`. Every push to `master` (other than docs and plans) builds the three wheels (Windows x86_64, Linux x86_64 manylinux_2_28, macOS arm64) and the sdist. It then tests each wheel on Python 3.12, and the Linux wheel also on 3.14, against the SciPy oracle with the full suite.
- The x86_64 wheels are built for `x86-64-v3` (AVX2), the ruled baseline.
- Publishing uses PyPI trusted publishing, with no stored token. A manual run with `testpypi` ticked uploads to TestPyPI. A pushed `v*` tag uploads to PyPI after Steve approves the `pypi` environment, and only if the tag matches the package version.
- CLAUDE.md declares the new delta: a pushed tag publishes, so tag pushes are Steve's.

## [1.11.6] - 2026-10-22

### Changed

- [packaging] Python 3.12 is now supported: the floor drops from 3.13 to 3.12 (`requires-python`, and the stable-ABI wheel tag `abi3-py312`), matching `aggregate`. Measured before the change: the full suite passes on 3.12 and every output is bitwise identical to the 3.13 build.
- New optional dependency `eltq[reference]` installs SciPy for the reference backend; the runtime dependency is still NumPy alone.
- PyPI metadata: MIT `LICENSE` file and license expression, readme, author, keywords, classifiers, repository and changelog URLs (`github.com/mynl/eltq`). The build needs maturin 1.14 or later.
- README gains an Install section: the platforms, the `x86-64-v3` baseline of the x86_64 wheels, and the source build for older CPUs or a build tuned to one machine.
- Checked: the sdist (112 KB) holds only the crates, the Python package, the Cargo and project files, README and LICENSE; a wheel and the sdist each install in a clean Python 3.12 environment and pass the suite.

## [1.11.5] - 2026-10-22

### Changed

- [cargo-config] The repository no longer sets compiler flags or a build directory: `.cargo/config.toml` is gone, so a fresh clone builds into `./target` at the baseline instruction set, as CI and PyPI wheels need. The old file built for the local CPU (`target-cpu=native`) into a drive-letter path outside the checkout.
- Local builds on the author's machine keep `target-cpu=native` through the machine-level `V:\dev\.cargo\config.toml`, outside any repository. Measured after the switch: the same speed as before and bitwise identical outputs.
- README "Environment" rewritten (plain `.venv` and `./target` in the checkout, how to tune a local build); the venv junction and external target directory it described are long gone.
- Plan `dev/done/plan-pypi-release.md` started.

## [1.11.4] - 2026-10-21

### Fixed

- [gauss-hermite-ties] Gauss-Hermite runs with a comonotone unit (σ_I = 0) failed with "beta quantile evaluation failed" once Q reached the hundreds. Two adjacent nodes in z ~ [8.14, 8.29] both give Φ(z) = 1 - 2^-53, and the quantile walk rejected the repeated target. `joint_two_units` failed at 19 of the 37 node counts from 600 to 2400, including Q = 700, and `multiunit_views` on the fixture failed at Q = 1001, the z-convergence study's reference size. The midpoint rule was never affected.
- `batched::beta_quantiles` now accepts non-decreasing targets: it solves the first of each run of equal values and copies that quantile to the rest of the run. All 37 node counts now succeed, with total mass within 9e-16 of 1.
- New tests: the kernel on repeated targets in both halves (including 1 - 2^-53); the joint at Q = 700 and 1001, and the fixture at Q = 1001, on both backends.

### Changed

- Plan `dev/done/plan-multiunit-fixes.md` complete (1.11.1 to 1.11.4).

## [1.11.3] - 2026-10-21

### Fixed

- [fft-nonnegative] The FFT convolution path returned negative weights of order -1e-18 in bins whose exact value is zero (below a deductible, the far tails of tight betas): up to 171 bins per view on a 200-event test portfolio, against none on the direct path. The spec requires nonnegative weights. The inverse transform now clamps each bin at zero; the mass moves by about 1e-16 per event, far below every tolerance, and the determinism contract is unaffected.
- New tests: the multi-unit fixture and a seeded 200-event, four-unit portfolio with policy and reinsurance terms, forced to FFT, have no negative bin in any view; the existing FFT-forced tests (Python and Rust) assert the same.
- Benchmark on this machine (32 threads, interleaved A/B against a 1.11.0 build): all four views at Q = 32 about 2.0 s for both builds; no measurable change from 1.11.1 to 1.11.3.

## [1.11.2] - 2026-10-21

### Fixed

- [fft-ceded] `multiunit_views(views=("ceded",), convolution="fft")` panicked (`range end index ... out of range`) on any event with a unit that has no terms. Ceded never uses the shared block of bare units, so their spectra were never built, yet the FFT path formed the block from them. `auto` could reach the same panic when large correlated bare units tipped its estimate to FFT.
- The shared bare block is now built, on both paths, only when some computed view other than ceded reads it, and the `auto` cost estimate charges it only then. Ceded-only runs skip that work; results for every other view combination are unchanged.
- New tests: direct, FFT and auto agree with each other and with the reference for each single view and for ceded plus net, on an event with a termed unit, a bare beta and a bare comonotone unit (the first test to put a comonotone unit in the shared block).

## [1.11.1] - 2026-10-21

### Fixed

- [tail-support] `multiunit_views` sized the event grid from the sum of unit supports in money, but each unit's pmf is its support rounded up to the grid, so an event with off-grid supports was longer than the grid. The Rust engine dropped the tail silently (two atoms at 13 on h = 10 returned mass 0.91, mean 22.4 instead of 1 and 26); the reference raised a NumPy broadcast error. Both now size by event span in buckets, the sum of each unit's rounded support.
- This changes numbers: an auto grid can grow by up to one node per unit, and an explicit `grid_length` that only covered the money support now raises `GridOverflowError` with the correct `required_length`. Fixtures whose supports lie on the grid, including every golden, are unchanged.
- The Rust deposit asserts that each event pmf fits the grid, so a future sizing mismatch fails loudly instead of losing mass.

### Changed

- Plan `dev/plan-multiunit-fixes.md` started and moved to `dev/done/`.

## [1.11.0] - 2026-08-26

### Added

- [ferocious-multiunit] Own `erfc` / `erf` in `eltq-core::normal` (Cody's
  rational Chebyshev approximations over the CALERF ranges, a straight
  polynomial pipeline per range that vectorizes across a row of edge
  scores), held to `libm::erfc` at 1e-15 relative over the full range by a
  dense sweep including the range switch points.
- Real FFTs: `convolve::FftCache` now plans `realfft` transforms (an n-point
  real row has an n/2 + 1 point spectrum, halving the transform work),
  cached per worker per size, with the transform scratch reused.
- Cross-view sharing in the per-event engine (`views.rs` rewritten around
  row sets): a unit materializes only the distinct row sets its terms
  produce (base for ground_up; gross after policy terms; ceded and net
  after reinsurance), so a unit without policy terms contributes its base
  rows to gross and net, one without reinsurance has net = gross and a ceded
  atom at zero that drops out of the ceded convolution entirely. Bare units
  (no terms at all) form one shared block convolved once per z node and
  reused by ground_up, gross and net; on the FFT path one event-wide size
  lets every (unit, row set, z) spectrum be transformed once and multiplied
  into each view, with the shared block's spectrum product formed once per
  z. The direct / FFT choice is made per event from the operations each
  path would actually run (`FFT_COST_FACTOR` calibrated on the benchmark so
  `auto` tracks the faster path).
- Whole-event chunks default to 2048 rows (`DEFAULT_MULTI_CHUNK_ROWS`)
  instead of 65536, so a 100k-row portfolio spreads across every core.
- `.cargo/config.toml` sets `target-cpu=native` (the wheel is built from
  source on each machine by `uv sync`; the file is now a uv cache key so
  changing it rebuilds). Per-element kernels vectorize without reordering
  any reduction, so the determinism contract is untouched.
- `benchmarks/run_multiunit.py`: the 1M unit-row harness (100k events x 10
  units, mixed regimes and typical term sparsity): all four views by
  convolution mode, ground_up only (the all-four delta), Q = 32 versus 8,
  Gauss-Hermite, thread scaling, the two-unit joint at Q = 1001 and 4001,
  bitwise determinism across thread counts, peak RSS; appends to
  `benchmarks/results.json` under `"suite": "multiunit"`. Report:
  `benchmarks/report-1.11.0.md`.

### Changed

- Plan `dev/plan-multiunit.md` completed and moved to `dev/done/`.

### Fixed

- `batched::beta_quantiles` accepts repeated endpoint targets (0 at the
  start, 1 at the end): Gauss-Hermite nodes at Q >= 32 reach |z| ~ 10, where
  `Phi(w_c z)` rounds to exactly 0 or 1 and the comonotone quantile walk
  rejected the run with a kernel failure. Regression test on the fixture
  (Q = 64 Gauss-Hermite) and the joint.

### Notes

- Performance (`benchmarks/report-1.11.0.md`, i7-7820HQ 4C/8T): ~0.91M unit
  rows (100k events x 10 units, single-unit events thinned) with all four
  views at Q = 32 in 22.5 s (40.5k rows/s); ground_up only 15.2 s; Q = 8
  in 7.9 s; threads 1 / 2 / 4 / 8: 81.9 / 43.8 / 27.7 / 22.3 s; bitwise
  identical across thread counts; the two-unit joint at Q = 1001 in 9.9 ms;
  peak RSS 231 MiB. The 1.10.0 engine did the 100k-row probe in 15.8 s;
  1.11.0 does it in 4.5 s (12.1 s single-threaded). The plan's single-digit
  headline was for the Xeon workstation, which was not measured; ~85% of the
  remaining time is the per-z convolution, so the next structural gain is
  the optional Chebyshev interpolation across z.
- The one-unit route measured 135k rows/s on this day against the 1.4.0
  history's 180-230k; an A/B against the 1.5.0 build in a separate worktree
  gave 112-124k rows/s on the same day, so the gap is machine state, not a
  regression.

## [1.10.0] - 2026-08-26

### Added

- [multiunit-views] The Rust per-event engine `eltq-core::views`
  (`multiunit_views`): pass 1 validates structure (contiguous events,
  constant rate, unique units, unit codes within the terms table),
  classifies unit rows under the one-unit diagnostic categories, and sizes
  the grid from the largest event support; pass 2 runs whole-event chunks
  in parallel with the existing bitwise-deterministic in-order fold. Per
  event: base conditional rows once per unit (Q rows for units that load on
  the shared factor, one row for independent beta units and atoms), per-view
  compiled maps (identity maps reuse the base rows; maps cached per worker),
  the independent block convolved once, correlated units folded in per z
  node by direct convolution or by FFT with the spectra accumulated over z
  and a single inverse per view (`convolution="auto"` picks by estimated
  cost; `direct` / `fft` force a path), then rate-weighted deposits into the
  chunk's per-view histograms. Per-event mass exactness (`max_mass_error`),
  analytic beta layer means (the LEV identity via the incomplete beta,
  bundles built once per unit) reconciled per view, and the optional
  roll-up diagnostic (exact ground-up total versus the rolled single-row
  beta: sd and q99 ratios per event) all come out in `MultiMeta`.
- `eltq-core::convolve`: direct convolution, rustfft plans cached per worker,
  cost estimates. `Diagnostics::merge` for chunk folds. `multiunit`:
  `classify_unit_row` / `UnitPlan::from_row` (classification stored between
  passes, plans rebuilt in the hot pass), `marginal_pmf`, `LayerMeans`.
- PyO3 `multiunit_views` binding (integer event/unit codes, structured terms
  per unit code, zero-copy columns, compute detached) and the Python
  dispatch: identifiers factorized (pandas when available, a dict pass
  otherwise), terms parsed and grid-aligned in Python, diagnostics
  translated to `event_id/unit_id` examples, roll-up entries keyed by event
  id, `MultiMeta.engine` with the engine statistics. New
  `eltq.multiunit_views_into(..., out={view: array})` following
  `quantize_into` (memmap-capable; the arrays define the views and grid).
  `multiunit_views` gains `threads`, `chunk_rows`, `convolution`.
- Tests: Rust unit tests (atom event view values, independent event equals
  direct convolution with mass = rate, direct and FFT paths agree with
  masses, diagnostics and roll-up entries checked, bitwise invariance over
  threads and chunking, structural / range / overflow / no-view /
  misalignment errors) and `tests/multiunit_golden.rs` against
  `fixtures/golden/multiunit_small.*`. The Python multi-unit suite's `eltq`
  variant now runs the Rust engine end to end (views, semantics, policies,
  structure errors, goldens); `test_determinism.py` adds the multi-unit
  thread / chunk / repeat sweep and the direct-versus-FFT agreement.

## [1.9.0] - 2026-08-26

### Added

- [terms-grammar] `eltq-core::terms`: the Rust twin of `eltq.terms`. Hand
  rolled tokenizer for `unit <id> [<limit> xs <ded>] [occurrence net of
  <clause> (and <clause>)*]`, `Clause` / `UnitTerms` with the same
  validation (share bounds, positive finite layer limits, nonnegative
  attachments, combined share at most 100% in every band), the four view
  maps, canonical text rendering (`fmt_g15` reproduces Python's `%.15g`),
  `parse_terms` over text with comments and duplicate detection, and the
  grid-alignment check. Errors surface as `Error::Terms`.
- `ViewMap::compile(terms, view, h, n)`: the per-(unit, view) deposit map,
  at most two `(index, weight)` pairs per source bucket by the linear rule,
  with `support_len` (the reachable destination length, what each view's
  convolution size will be) and an `identity` flag (ground_up, and gross
  without policy terms) so identical maps can share one spectrum in 1.10.0.
  `apply` is a gather-scatter pass.
- Goldens: `fixtures/golden/terms_parse.json` now also carries the keyword
  vocabulary; `tests/terms_golden.rs` holds the Rust parser to the Python
  parses, rejections, canonical text and keywords. The `decl.lark` glance
  (aggregate_REFACTOR, 2026-08-26) confirmed `xs`, `po` (PART_OF),
  `occurrence net of`, and `and`-joined clauses; no drift.
- Rust unit tests: parsing of full and partial lines, share conventions and
  bounds, overlap guard, canonical text round trips and `fmt_g15` cases,
  text parsing and alignment, view values and compiled maps (mass and mean
  preservation, support lengths, identity detection).

## [1.8.0] - 2026-08-26

### Added

- [joint-two-unit] Rust two-unit joint in `eltq-core::multiunit`: unit
  preparation (`prepare_unit`: kinds atom / comonotone / beta, variance
  policies, copula weights), edge scores `y_e = Phi^{-1}(I_{edge/M}(alpha,
  beta))` computed once per unit, conditional matrices `(Q x n)` by adjacent
  differences of `Phi((y_e - w_c z) / w_I)` (atoms and comonotone units by
  linear deposits, the latter through the batched quantile walk at the
  ascending targets `Phi(w_c z_q)`), and the joint `C_1^T diag(w) C_2` as
  one `matrixmultiply::dgemm`. `JointMeta` mirrors the Python dataclass.
- `eltq-core::normal`: normal cdf via `libm::erfc` (the fdlibm port, below
  one ulp; statrs's erfc loses ~1e-11 relative in the tails, caught by the
  edge-score round trip) and Wichura's AS241 quantile (~1e-16 relative).
  `eltq-core::quadrature`: midpoint quantile nodes and Gauss-Hermite by
  Golub-Welsch with a first-row-only implicit-shift QL (O(Q^2), stable at
  any Q, symmetrized bitwise).
- `batched::beta_quantiles(alpha, beta, ascending_targets, out)`: the
  midpoint kernel's warm-started walk exposed for arbitrary ascending
  targets (lower half direct, upper half on the reflected shape with exact
  complements); the comonotone path is a sorted quantile walk, no FFT.
- PyO3 `joint_two_units` binding (zero-copy `(n1, n2)` pmf, grids, meta
  dict, compute detached from the interpreter); `eltq.joint_two_units` now
  runs on Rust when the extension is present and on the reference otherwise.
  `GridOverflowError` raised by the Rust backend now carries
  `required_length` and `examples` like the reference's.
- Tests: Rust unit tests for the normal functions, both quadrature rules
  (moments exact, closed forms, symmetry), unit preparation and policies,
  conditional rows as distributions, GEMM versus the naive triple loop,
  independent units as an outer product, typed errors; `tests/joint_golden.rs`
  against `fixtures/golden/joint_small.*` (rtol 1e-9). The Python multi-unit
  suite's `eltq` variant of every joint test now exercises the Rust path.

## [1.7.0] - 2026-08-26

### Added

- [reference-n-unit] `eltq.reference_multiunit`: the SciPy reference for
  joint multi-unit secondary uncertainty, the oracle for the Rust engine
  and the backend of the new API until it lands. `secondary_two_units`
  (the two-unit joint of `dev/secondary_two_units_reference.py`, promoted
  into the package; `dev/` copy retired) and `multiunit_views_reference`:
  per event, per unit, conditional pmfs given the shared factor (beta cdf
  differences at half-bucket edges; linear deposits for atoms, comonotone
  units and every transformed view value), per-view transform, convolution
  across units, shared-factor quadrature, rate-weighted accumulation across
  events. Per-view mass, means, analytic beta layer means (the LEV identity,
  closed form via the incomplete beta) and their reconciliation, plus the
  optional roll-up diagnostic (exact total versus rolled single-row beta:
  sd and q99 ratios per event) in the new `MultiMeta`; `JointMeta` for the
  joint. Unit-level variance policies (clamp / error / drop / atom) and the
  one-unit diagnostic categories carry over with `event_id/unit_id`
  examples; misgrouped events, rates varying within an event, and repeated
  units raise `MultiUnitError`.
- [terms-grammar] `eltq.terms` (pure Python, shared by both backends): the
  DecL-style grammar `unit <id> [<limit> xs <ded>] [occurrence net of
  <clause> (and <clause>)*]`, `clause := [<share> po] <limit> xs <attach>`,
  aligned with `aggregate`'s `decl.lark` (`xs`, `po` = part of, `occurrence
  net of`, `and`). Percent shares are fractions of the layer, plain-number
  shares are amounts of the layer limit. `UnitTerms` / `Clause` with the
  four view maps (ground_up, gross, ceded, net), canonical text rendering,
  structured-input parsing, and the grid-alignment check (breakpoints must
  be multiples of `bucket_size`; no silent rounding). Clauses may overlap,
  but their combined share must not exceed 100% of gross in any band
  (`TermsError` otherwise): beyond that ceded exceeds gross and net
  decreases or goes negative, which no loss grid can represent. This
  tightens the plan's "clauses may overlap" wording; found by the suite.
- Python API: `eltq.multiunit_views(...)`, `eltq.multiunit_views_dataframe`,
  `eltq.joint_two_units(...)`, `eltq.parse_terms`, exported `UnitTerms`,
  `Clause`, `VIEWS`, `TermsError`, `MultiUnitError`, `MultiMeta`,
  `JointMeta`. Reference exceptions are translated to the package's classes
  on the Rust backend.
- Shared-factor quadrature: midpoint quantile nodes (default) or
  Gauss-Hermite (`z_rule="gauss_hermite"`, Golub-Welsch nodes, stable at
  any Q). Convergence study `benchmarks/z_convergence.py` with its report
  `benchmarks/z-convergence-1.7.0.md`: Gauss-Hermite is spectral on smooth
  units (marginal error 1e-10 at Q = 32 versus 1e-3 midpoint) but leaves
  artifacts where the conditional is step-like in z (correlation-dominant
  and comonotone units); the midpoint rule keeps the better pmf shape
  there, so it stays the default at Q = 32 (plan decision 5 validated).
- Fixtures and goldens: `fixtures/multiunit_small_elt.csv` +
  `multiunit_small_terms.txt` (every unit kind, a zero-rate event, a
  dropped row, boundary atoms, four-unit events) with per-view goldens;
  `fixtures/golden/joint_small.*` (two-unit joint at Q = 1001);
  `fixtures/golden/terms_parse.json` (parse goldens plus rejected lines)
  that the Rust parser is held to as well.
- Tests: `tests/test_terms.py` (grammar, shares, canonical text, view
  semantics, alignment, goldens) and `tests/test_multiunit.py` run against
  the reference and the package API through the `MultiImpl` registry in
  conftest: joint mass / marginals / independence outer product /
  comonotone / atom / errors / golden; view semantics on atoms (ded, limit,
  tower, amount and percent shares, overlap, off-grid fractional cession);
  independent event equals plain convolution; comonotone common-shape event
  reproduces the rolled beta (additivity Proposition 6); per-event mass
  exactly the rate for every view; view means reconcile to analytic layer
  means within the stated (h, Q) tolerance; normalize; row policies and
  diagnostics; structural and overflow errors; golden fixture agreement.

## [1.6.0] - 2026-08-26

### Changed

- **[sigma-rule] Behavioral change: the total conditional standard deviation
  is now the arithmetic sum σ = σ_I + σ_C** (Burke, Rau-Chaplin, Varghese
  2016, section III.E, step i), replacing the quadrature rule
  σ² = σ_I² + σ_C² used by every earlier version. The quadrature combination
  was incorrect; it is replaced outright with no legacy option (git history
  serves any before/after comparison). Every row with both components nonzero
  now gets a wider secondary distribution, and rows whose summed sd exceeds
  the beta bound now fall under the variance policy (clamped by default). The
  admissibility test itself is unchanged: σ² against μ(M − μ). Applied in
  `validate.rs::classify_row` and `eltq.reference`; spec section 4 working
  formula replaced and section 11 assumption 3 marked resolved with a pointer
  to `dev/secondary-uncertainty-additivity.md` (the one-factor model, the
  additivity proofs, and the numerical illustration behind the change).
- Fixture and benchmark generators split a row's total sd linearly
  (`sd_indep = σ·share`, `sd_corr = σ·(1 − share)`) instead of in quadrature,
  so the variance-fraction regime mix keeps its meaning under the new rule;
  goldens regenerated (values shift wherever both sds are nonzero).

### Added

- `dev/done/plan-multiunit.md`: the plan for 1.6.0-1.11.0 (joint multi-unit
  secondary uncertainty, DecL-style unit terms, per-view totals), plus its
  mathematical companions `dev/secondary-uncertainty-additivity.md` and
  `dev/secondary_two_units_reference.py`.

## [1.5.0] - 2026-08-14

### Added

- [fallback] The SciPy reference implementation now ships inside the package
  as `eltq.reference` and serves the full public API automatically when the
  compiled extension is not importable (`eltq.BACKEND` reports `"rust"` or
  `"reference"`); the fallback ignores `threads`/`chunk_rows` and includes a
  stdlib-csv route mirroring the Rust parser's NaN semantics. Shared `Meta` /
  `RowIssues` dataclasses moved to `eltq._meta`; `tests/reference.py` removed
  in favor of the packaged module (goldens byte-identical). New subprocess
  test exercises the fallback path end to end (81 Python tests green).
- [demo] `notebooks/eltq_demo.ipynb`: single-event intuition plot, CSV-route
  portfolio run with diagnostics, eltq vs SciPy reference comparison
  (agreement + timing), 1M rows on an exactly 2^16-node grid, bitwise
  determinism check, and memory-mapped output. Launch with
  `uv run --with jupyter --with matplotlib jupyter lab` from the repo root.

### Fixed

- `.gitignore`: ignore `*.pyd` in-tree (maturin editable builds compile the
  extension into `python/eltq/`; previously only covered by a global excludes
  file) and `.ipynb_checkpoints/`; `.venv` junction entry retained.

## [1.4.0] - 2026-08-14

### Added

- [ferocious] Own regularized incomplete-beta CDF (`incbeta.rs`):
  cephes/NR-style modified Lentz continued fraction with the symmetry switch
  at x = (α + 1)/(α + β + 2); per-event constant ln B(α, β) computed once per
  event. Cross-checked against statrs at 1e-13.
- Batched per-event midpoint-quantile kernel (`batched.rs`), now the
  production hot path: warm starts between successive quantiles via a
  second-order Taylor predictor along the quantile ODE dQ/du = 1/f(Q),
  refined by safeguarded Halley steps inside a maintained bracket with
  log-space-Newton and bisection fallbacks (hard shapes cannot diverge), a
  cubic-convergence early exit that skips the confirming CDF evaluation,
  exact reflection for upper-half quantiles, and fast paths for uniform
  (exact) and symmetric (solve half, mirror) shapes. Property-tested against
  the statrs reference kernel (1e-13 moderate shapes, 1e-10 extreme).
- Rayon chunk parallelism with a bitwise-deterministic sequential fold in
  chunk-index order: private per-chunk histograms (no locks or atomics),
  bounded waves (2x workers) capping peak partial-histogram memory, `threads`
  and `chunk_rows` config exposed through Python. Results are bitwise
  identical for any thread count; chunk_rows is part of the deterministic
  contract. Determinism tests in Rust and Python.
- Benchmarks: criterion kernel suite (`cargo bench -p eltq-core`: CDF by
  regime, batched vs reference kernel, deposits, beta_params) and the Python
  end-to-end harness `benchmarks/run_end_to_end.py` (seeded 1M-row mixture,
  K/grid sweeps, thread scaling, CSV route, memmap output, SciPy-reference
  speedup, bitwise determinism check, peak RSS, machine/compiler metadata)
  appending to the committed machine-readable `benchmarks/results.json`.
- 1M-row performance report: `benchmarks/report-1.4.0.md`. Headline on an
  i7-7820HQ (4C/8T): 1M rows at K = 32 in ~4.3 s ≈ 230k rows/s ≈ 6.6M
  beta quantiles/s, ~44x the SciPy reference; non-kernel overhead 0.02 s.

### Changed

- `quantize` now uses the batched kernel; the statrs Newton kernel remains as
  the test oracle. Full suites stay green (48 Rust, 80 Python tests).

### Notes

- Deferred with profiling justification required (1.5.0+ candidates): SIMD,
  U-shape closed-form starts, CSV parse/compute overlap, relaxed
  grid-tolerance stopping, GPU spike, large-n per-worker histogram fallback.

## [1.3.0] - 2026-08-14

### Added

- [memmap] `eltq.quantize_into(source, *, raw_out, secondary_out, ...)`:
  computes into caller-provided writable, C-contiguous float64 arrays —
  including `numpy.memmap` — validating dtype/shape/contiguity/writability
  upfront; the grid length is defined by the output arrays. Computation stays
  in Rust-owned buffers with one final copy into the borrowed memory under
  the interpreter lock (no protection released around borrowed writable
  memory).
- [memmap] `eltq.quantize_to_npy(source, path, ...)`: writes a `(2, n)`
  float64 `.npy` (row 0 raw, row 1 secondary) plus adjacent
  `<stem>.meta.json`, both transactionally (temp file + atomic replace);
  reopen with `np.load(path, mmap_mode="r")`.
- Both accept a CSV path, a DataFrame with canonical columns, or a 5-tuple
  of column arrays as `source`. New `tests/test_memmap.py` (7 tests; 78
  Python tests green).

## [1.2.0] - 2026-08-14

### Added

- [python] `eltq-python` PyO3 crate (abi3-py313, pyo3 0.29): zero-copy borrow
  of contiguous float64 NumPy columns, compute detached from the interpreter
  (`Python::detach`), results transferred `Vec<f64>` -> NumPy via
  `IntoPyArray` with no completed-array copy, Rust errors translated to
  `EltqError` / `VarianceError` / `GridOverflowError`.
- CSV input route in `eltq-core` (`csv_input.rs`, Rust `csv` crate):
  configurable column names over the strict canonical schema, optional
  event-id column, unparseable numeric cells become NaN and surface in the
  `nonfinite` diagnostic (structural CSV problems are hard typed errors).
- Python package `eltq`: `quantize_arrays`, `quantize_csv` (event-id
  diagnostic examples translated from the file), `quantize_dataframe`
  convenience (pandas never a dependency of the extension), `Meta` /
  `RowIssues` dataclasses matching the reference, numpy-style docstrings.
- maturin build backend in pyproject (agreed deviation from hatchling), uv
  cache-keys so `uv sync` rebuilds when Rust sources change; Windows wheel
  builds via `uv sync`.
- Acceptance and contract suites now run against both implementations
  (reference + eltq); new `tests/test_python_api.py` covers the CSV route
  (bitwise-equal to the array route), column mapping, error surface, and
  version sync. 71 Python + 37 Rust tests green.

### Changed

- Fixture generation and tests parse CSVs with
  `float_precision="round_trip"` so pandas-parsed inputs are bit-identical
  to Rust's correctly rounded CSV parser; goldens regenerated accordingly.

## [1.1.0] - 2026-08-14

### Added

- [rust-core] `eltq-core` scalar computational core: `Config` /
  `VariancePolicy` with validation, typed `Error` enum (thiserror), row
  classification mirroring the reference (`validate.rs`), beta
  parameterization via κ = μ(M − μ)/σ² − 1, linear mass- and mean-preserving
  grid deposit with explicit exact-node and top-endpoint handling, Neumaier
  compensated reference totals, per-category diagnostics with bounded example
  row indices, and the single-pass streaming `quantize` driver (upfront grid
  coverage check, reused K-quantile buffer, never an N×K allocation).
- Reference inverse-beta kernel (`kernel.rs`): statrs regularized
  incomplete-beta CDF/PDF driven by a safeguarded Newton solve with
  log-space fallback and bisection inside a maintained bracket; upper-half
  quantiles solved by exact reflection. Full-precision across the admissible
  shape domain, including extreme clamp-policy U-shapes; stays available as
  the test oracle for the 1.4.0 batched kernel.
- Rust test suite (33 tests green): unit tests per module, proptest
  properties (deposit mass/moment preservation, quantize mass/mean
  conservation, bitwise dyadic scale invariance, beta round trip), and
  golden-fixture agreement with the SciPy reference at 1e-9 relative
  tolerance (`tests/golden.rs`, reading the new full-precision
  `fixtures/golden/*.arrays.csv` text twins).
- `[profile.test] opt-level = 2` so the numerical suites run fast.

## [1.0.0] - 2026-08-14

### Added

- [reference] Repository scaffold: Cargo workspace (eltq-core placeholder),
  uv-managed Python environment on `V:\dev` (junction `.venv`), cargo target
  dir redirected to `V:\dev\cargo-target\secondary-corrosion`, data-firewall
  `.gitignore`.
- SciPy reference implementation (`tests/reference.py`): the transparent
  mathematical oracle implementing the full contract — validation and
  diagnostic categories, variance policies (`clamp` default with factor 0.999,
  `error`, `drop`, `atom`), degenerate-atom treatment, beta parameterization
  via κ = μ(M − μ)/σ² − 1, midpoint beta quantiles (SciPy `beta.ppf`),
  mass- and mean-preserving linear grid deposit, upfront grid-coverage check,
  exact (`math.fsum`) reference totals, optional normalization and
  `mean_is_loss_rate` mode.
- Synthetic fixtures (`fixtures/generate_fixtures.py`, seeded): handcrafted
  17-row small ELT covering every diagnostic category and fast path; 5000-row
  medium mixture across parameter regimes; committed golden outputs
  (`.npz` arrays + `.meta.json`) computed from the re-read CSVs.
- Acceptance suite (`tests/test_acceptance.py`): all fourteen mathematical
  tests of specification section 9, parameterized over available
  implementations (reference now; Rust `eltq` joins at 1.2.0). Supplementary
  contract tests (`tests/test_contract.py`): invalid-row handling, zero-rate
  and duplicate-id rows, loss-rate mode, normalization, K = 1, empty input,
  extreme clamped shapes, config validation. 33 tests green.

### Notes

- Build backend will be maturin (not hatchling) because the package is a Rust
  extension; agreed in `dev/plan-implementation.md`.
- Scale invariance is bitwise for dyadic scale factors; non-dyadic factors are
  invariant to ~1e-9 relative tolerance.
