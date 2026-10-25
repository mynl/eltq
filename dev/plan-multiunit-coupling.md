# Plan: multiunit-coupling — coupling rules, Bernoulli rows, the factor rule and bivariate outputs for the multi-unit route

Date: 2026-10-25
Status: draft for review; not started. The specification amendment it rests on (section 4A) was
written the same day and is committed with this plan.

## Goal

Bring the multi-unit route (`multiunit_views`, `joint_two_units`, 1.6.0 to 1.11.4) up to the
one-row route's standard and to the specification's new section 4A: the published coupling rules
as a named option, the Bernoulli treatment of inadmissible rows, a quadrature rule over the
common factor chosen on tail metrics with one-unit events kept out of the mixture, and the
bivariate occurrence severities (two units; each unit against the event total) as arrays for the
aggregate package to compound. Six steps, each a minor version, in the order below; the last is
optional.

## Background for a reader with no context

`eltq` is a Rust engine (`crates/eltq-core`) with PyO3 bindings and a SciPy reference in
`python/eltq/` that serves as the oracle; every acceptance test runs against both. The
mathematical contract is `secondary-uncertainty-mathematical-specification.md`. The one-row route
(`quantize_*`, `exceedance_curve`) was rebuilt between 1.13.0 and 2.3.2: Bernoulli rows, the
survival method, overflow policies, the tail cut, narrow events, and the incomplete beta and
beta quantiles at large parameters. The multi-unit route has not changed since 1.11.4
(2026-10-21) and is still the architecture of `dev/done/plan-multiunit.md`:

- per event, each unit's conditional bucket probabilities given the common factor $Z_c$ at $Q$
  quadrature nodes, by differences of the conditional distribution function at the half-bucket
  edges (beta units), or a deposited point (atoms, and comonotone units with $\sigma_I = 0$);
- per unit, the DecL-style terms applied as compiled view maps, giving ground_up, gross, ceded
  and net rows;
- per view, the convolution across units (direct or FFT), the mixture over the $Q$ nodes with the
  rule's weights, and the rate-weighted deposit into the global grid.

Two documents written on 2026-10-25 state what this plan implements. Section 4A of the
specification is the contract. The article *Secondary Uncertainty in Catastrophe Model Event
Loss Tables for Multi-Unit Portfolios* (`D:\Projects\aggregate\secondary-uncertainty\
secondary-uncertainty-multiunit.qmd`, "the multi-unit article") derives the model, collects the
published rules, proves which routes to the portfolio total agree and which do not, and computes
the joint, the annual bivariate and the conditional means on the two-line example of Homer and
Li 2017, Table 11. Its setup chunk is a 60-line SciPy construction of the joint of two units,
including two-point rows, that agrees with `joint_two_units` to $2 \times 10^{-16}$ per cell; it
is the independent check for several steps below.

### What the one-row work did and did not reach

Measured on 2026-10-25 (scratch probes, not committed) before writing this plan:

| item | multi-unit route today |
|---|---|
| own erf, incomplete beta, normal quantile, 2.3.1 $\ln B$, 2.3.2 Cornish-Fisher quantiles | shared kernels; comonotone units take `beta_quantiles` and so inherit 2.3.2 |
| beta units at $\min(\alpha,\beta)$ up to $2 \times 10^6$, $h = 100$, edges inside the bulk | Rust and reference within $4 \times 10^{-9}$ per cell; the 2026-10-25 report's row (sd $10^{-6}$) is an atom in effect and passes |
| narrow units (sd below $h$) | exact: cdf differences at the edges put the mass in one bucket |
| variance policy | default `clamp` at 0.999; `bernoulli` raises `BernoulliPolicyUnavailable` |
| $Z_c$ rule | 32-node midpoint by default; a lone beta unit (300,000, 50,000, 50,000, 2,000,000) at $h = 1000$ returns mean 299,827 (midpoint 32), 299,957 (midpoint 128), 300,000.00 (Gauss-Hermite 32 or 64) |
| one-unit events | go through the mixture, inheriting the rule's error |
| specification | had no multi-unit section until 4A |

So the retrofit is not about the special functions, which are sound on this route, but about
the variance policy, the factor rule, one-unit events, and the outputs.

## The model, in brief

Per unit $i$ of an event the marginal is the one-row model (beta, two-point, or atom). One
standard normal $Z_c$ per occurrence and one $Z_i$ per unit give the score $V_i = w_i Z_c +
\sqrt{1 - w_i^2} Z_i$ and the loss $X_i = M_i F_i^{-1}(\Phi(V_i))$, with the loading $w_i$ the only
free quantity. Conditional on $Z_c = z$ the units are independent with
$\mathsf P(X_i \le x \mid z) = \Phi((y_i(x) - w_i z)/\sqrt{1 - w_i^2})$,
$y_i(x) = \Phi^{-1}(F_i(x/M_i))$; a two-point row fits with its step $F_i$. Everything is a
conditional calculation at fixed $z$ followed by a quadrature over $z$. The published rules are
Burke's loading $\sigma_C/\sqrt{\sigma_I^2 + \sigma_C^2}$, the comonotone loading 1, the
independent loading 0, and Homer and Li's roll-up with pro rata allocation, which is not a
copula at all. The variance-matched loading, Burke's scaled by a common factor per event so that
the event total's variance equals the roll-up convention's $(\sigma_{I,R} + \sigma_{C,R})^2$, is
the one alternative with a rationale and has no source.

## Decisions taken (Steve, 2026-10-25)

1. Burke's coupling is the default; comonotone, independent and Homer are supported; the
   variance-matched loading is optional and last.
2. Homer's allocation is capped at $M_k$, as the one-row route caps at $M$; the capped
   allocations no longer sum to the rolled loss, nothing is reallocated, and the mean lost to the
   cap is reported.
3. Bivariate outputs are arrays only. The annual bivariate aggregate, its marginals and the
   conditional means are the aggregate package's (`BivariateAggregate(mode="discrete")`,
   `exeqa_df`), fed from eltq's occurrence matrix; eltq writes no DecL.
4. The multi-unit material has its own article; the method article is unchanged apart from a
   possible tweak to its last section.

## Steps

Versions continue from 2.3.2. Each step is one minor bump and one commit (plus a commit for any
golden regeneration it forces), staged by path; Claude commits, Steve pushes.

### [spec] no bump: section 4A of the specification

Done with this plan: section 4A (the multi-unit table, the model, the coupling rules, the three
routes, terms and views, the integral over the factor, bivariate outputs, metadata), section 7's
multi-unit outputs paragraph, tests 25 to 29 in section 9, and item 9 of section 11. Section 4's
`clamp` bullet now points here.

### [bernoulli] 2.4.0: Bernoulli rows on the multi-unit route

- `multiunit.rs::prepare_unit` and `reference_multiunit.prepare_unit`: `VariancePolicy::Bernoulli`
  becomes a `UnitKind::TwoPoint` plan (the conditional law of section 4A), the default for
  `multiunit_views` and `joint_two_units`; `clamp` stays selectable with a deprecation note and
  is removed at the next major, as the one-row route did between 1.13.0 and 2.0.0 (raise if
  Steve prefers outright removal now).
- Conditional matrix for a two-point unit: row $q$ has mass $1 - \Phi((\Phi^{-1}(1 - p) - w z_q)/
  \sqrt{1 - w^2})$ deposited at $M$ and the rest at 0; for $w = 1$ a step at $z = \Phi^{-1}(1-p)$.
  Atoms unchanged. The marginal and layer-mean helpers (`analytic_view_mean`, the roll-up
  diagnostic) gain the two-point case.
- Diagnostics: `variance_bernoulli` replaces `sigma_clamped` as the reported category under the
  default.
- Goldens: `fixtures/golden/multiunit_small.*` and `joint_small.*` regenerate if they contain
  inadmissible rows (check first; the fixture generator says). CHANGELOG records the change in
  numbers.
- Tests: spec test 26, both backends; the multi-unit article's event 2 of Homer's Table 11
  (personal two-point, commercial beta) against the article's construction within $10^{-12}$.

### [one-unit] 2.5.0: one-unit events leave the mixture

- `views.rs` pass 1 marks events with one contributing unit. Such an event's ground-up row is
  the unit's marginal bucketed by cdf differences at the edges (`marginal_pmf`, exact in
  probability), its views the compiled maps of that row, no $z$ loop; the engine reports the
  count (`one_unit_events`).
- Reference: the same branch in `multiunit_views_reference`.
- Tests: spec test 27 on both backends: a one-unit event under every coupling and both rules
  equals the direct bucketing exactly, and agrees with `quantize_arrays(method="survival")` on
  the same row within the survival method's bucketing tolerance (the two discretizations differ
  by the survival method's interpolation, so state the tolerance measured); a mixed portfolio's
  total is the sum of its parts.

### [z-rule] 2.6.0: the factor rule chosen on tail metrics

- Extend `benchmarks/z_convergence.py`: rules midpoint, Gauss-Hermite and stratified; $Q$ in
  8, 16, 32, 64, 128; metrics the exceedance of the event total at $10^{-2}$, $10^{-3}$,
  $10^{-4}$, $10^{-5}$ against the midpoint rule at $Q = 4001$, plus the existing mean, sd,
  q99 and marginal errors; cases A to D of the multi-unit article, the fixture, and a lone beta
  unit with $w$ from 0.5 to 0.99. Record as `benchmarks/z-convergence-2.6.0.md`.
- Implement the stratified rule: $Q$ equal-probability strata of $Z_c$, the conditional
  distribution averaged over each stratum (Gauss-Legendre within the stratum, 8 points, with the
  bivariate-normal closed form as the reference check), units independent given the stratum.
  Exact marginals and means by construction; its dependence error is what the study measures.
- Set the default rule and $Q$ from the data, as the node density was set at 1.16.0; write the
  figures into section 4A's paragraph on the integral and into test 28's tolerance.
- Goldens regenerate if the default changes; CHANGELOG says so. `dev/plan-chebyshev-z.md` stays
  parked behind this step and is re-evaluated against its result.

### [coupling] 2.7.0: the coupling option

- `coupling=` on `multiunit_views` and `joint_two_units`, values `burke` (default), `comonotone`,
  `independent`, `homer`; echoed in `MultiMeta`/`JointMeta`.
- `burke`, `comonotone`, `independent`: a loading override in `prepare_unit` ($w$ forced to 1 or
  0). A forced comonotone unit takes the existing `Comonotone` kind (an atom per node via
  `beta_quantiles`); a forced independent unit joins the independent block, which the engine
  already convolves once per event outside the $z$ loop.
- `homer`: a separate path with no $z$ loop. Per event: roll the rows up by the moment rule,
  classify the rolled row by the one-row rules (beta, two-point, atom), bucket its marginal on
  the grid by cdf differences, and push it through the composed map
  $x \mapsto \sum_k \mathrm{term}_k(\min(x \mu_k/\mu_R, M_k))$ per view, compiled like a view
  map (its breakpoints are the units' breakpoints scaled by $\mu_R/\mu_k$ and the caps). The mean
  lost to the caps, $\sum_j \lambda_j \mathsf P(\sum_k (X_{R,j}\mu_k/\mu_R - M_k)^+)$, is computed
  analytically from the rolled beta's limited expected values and reported as
  `homer_cap_mean_loss`.
- Tests: spec test 25 for the four couplings, both backends; the multi-unit article's Table 5
  (case A under the three loadings) reproduced within the rule's error.

### [joint] 2.8.0: bivariate occurrence severities

- `pair_severity(event_id, unit_id, ..., units=(u, w), bucket_size, ...)` returns the
  rate-weighted joint occurrence severity $S_{uw} = \sum_j \lambda_j J_{j,uw}$ on the common grid
  (shape $n_u \times n_w$), its two grids and metadata (mass, marginal means, the correlation at
  an occurrence). Per event with both units present, the joint is the engine's existing
  two-unit GEMM; with one present, an outer product with the atom at zero; with neither,
  nothing. Chunked and deterministic as `multiunit_views`.
- `unit_total_severity(...)` returns, for each unit $u$, $S_{u\Sigma}$: per event and node, the
  unit's conditional row against the convolution of the other units (prefix and suffix products
  of the unit spectra; the shear applied at deposit), accumulated rate-weighted. Memory is
  $n_u \times n_\Sigma$ per unit; the caller names the units it wants, all by default.
- Reference implementations of both; names are provisional (raise at review).
- Tests: spec test 29 on both backends, including $\sum_u \kappa_u(x) = x$ from the
  $S_{u\Sigma}$ at occurrence level, and the handoff: build the annual bivariate with the
  aggregate package from $S_{u\Sigma}/\Lambda$ under Poisson($\Lambda$) and check its second
  marginal against the one-dimensional compound of the ground-up total (skipped when `aggregate`
  is not installed; it is a test-only dependency).
- Notebook: the Homer Table 11 example end to end, as in the article's section 7.

### [matched] 2.9.0, optional: the variance-matched loading

- `coupling="matched"`: per event, bracket $c$ on $[0, 1/\max_i w_i]$ for the root of
  $\operatorname{Var}(\sum_i X_i) = (\sigma_{I,R} + \sigma_{C,R})^2$, the variance computed from
  the conditional means and variances per unit at the rule's nodes (no convolution needed: the
  conditional variance of a sum of conditionally independent units is the sum, and the variance
  of the conditional mean is a quadrature); about twenty evaluations per event. Saturation reported
  per event (`matched_shortfall`).
- Tests: spec test 25's `matched` clause; the multi-unit article's Table 6 (case A and Homer's
  events 1 to 3, including event 2's saturation) reproduced within the rule's error.

### [parity] with 2.9.0 or as 2.9.x: documentation and benchmarks

- README milestone rows, the Python API block, `MultiMeta` fields; CHANGELOG per version.
- `benchmarks/run_multiunit.py` rerun under the new default rule and policy; report
  `benchmarks/report-2.9.0.md`; the all-four-views delta and the `homer` path's speed recorded.

## Files

- `secondary-uncertainty-mathematical-specification.md` (done with this plan).
- `crates/eltq-core/src/multiunit.rs` (kinds, two-point plan, loadings), `views.rs` (one-unit
  branch, `homer` path, pair and unit-total accumulators), `quadrature.rs` (stratified rule),
  `config.rs` (coupling, policy default), `terms.rs` (composed Homer map), `diagnostics.rs`.
- `crates/eltq-python/` bindings for the new keywords and functions.
- `python/eltq/__init__.py`, `_meta.py`, `reference_multiunit.py`.
- `tests/test_multiunit.py` (tests 25 to 29), `fixtures/generate_fixtures.py` and goldens as
  forced, `benchmarks/z_convergence.py`, `benchmarks/run_multiunit.py`, `notebooks/`.
- `README.md`, `CHANGELOG.md`, `Cargo.toml`, `pyproject.toml`, `python/eltq/__init__.py`
  (version triple).

## Acceptance

- `cargo test -p eltq-core --release` and `uv run pytest` green at every step, both backends.
- Spec tests 25 to 29 implemented and passing; the figures they quote recorded in the
  specification where the plan leaves them open (the default rule and $Q$, the one-unit
  tolerance against the survival grid).
- The multi-unit article's Tables 4 to 6 and its agreement figure reproduced from the package
  (a test or a notebook cell per table).
- Determinism across thread counts bitwise, as today; no negative weight on any path.
- Speed: the default rule's multi-unit benchmark within the 1.11.0 figures scaled by its $Q$;
  the `homer` path at one-row speed; no regression on the one-row route (unchanged code).

## Alternatives considered

- *Raise $Q$ under the midpoint rule.* The tail cut shrinks only as $1/Q$ and the cost is linear in
  $Q$; the study decides, but the stratified rule's exact marginals are the better bet.
- *Homer uncapped, as published.* Rejected (decision 2): the published rule allocates more than a
  unit's exposure in the paper's own example.
- *Reallocate the capped excess to other units.* Rejected: a convention of ours with no source.
- *Variance matching by a pairwise loss correlation rather than a common scale.* Not identifiable
  from one equation per event for $n > 2$ units; the common scale is the minimal one-parameter
  family containing Burke.
- *Bivariate compounding inside eltq.* Rejected (decision 3): the aggregate package has the 2-D
  FFT, the marginal checks and `exeqa_df`; duplicating them adds code and a second place for the
  grid conventions to drift.
- *The one-row survival route for one-unit events inside the multi-unit engine.* The engine
  buckets on the fixed global grid and already evaluates the cdf at every edge, so cdf
  differences are exact there at the same cost; the survival curve's node machinery brings
  nothing on a fixed grid. The one-row route stays the comparison in test 27.

## Not in this plan

- Per-event grids for large exposed values on the multi-unit route (the one-row route's section
  5A window and overflow policies have no multi-unit counterpart yet); `GridOverflowError` stays.
- A survival-style, node-based formulation of the multi-unit total: the convolution is the
  honest route and nothing suggests a closed form for the total's survival function.
- The reserving application of `secondary-uncertainty-mv-cat-reserves.qmd` (open claims as
  certain events, direct input of the total sd and a correlated share): a transformation of the
  inputs, to be specified when that article is written.
- `dev/plan-chebyshev-z.md` [hybrid-rule] and [chebyshev]: re-evaluated after [z-rule].
- The deferred review items of `dev/done/plan-multiunit-fixes.md` (grid length bound, the NaN
  rate category mismatch, `1_000` in the Rust terms parser).

## Open questions for review

1. `clamp` on the multi-unit route: deprecated and selectable until 3.0.0 (proposed), or removed
   at 2.4.0 as the one-row route removed it at 2.0.0.
2. Names of the new functions (`pair_severity`, `unit_total_severity` are placeholders) and
   whether `unit_total_severity` defaults to all units.
3. Whether the convergence study's default $Q$ may differ by rule (the stratified rule at 16
   may beat Gauss-Hermite at 32 on cost), or one $Q$ is set for all rules.

## Execution log

(empty)
