# Plan: multi-unit coupling rules, Bernoulli rows, the stratified factor rule and occurrence outputs

Date: 2026-10-25, revised 2026-10-26 after the multi-unit article's analysis of the routes
Status: executing from 2026-10-26 (Steve's `/execute-plan`, with decision 2 revised to uncapped
only); see the execution log. Section 4A of the specification, the contract this plan
implements, was written with the first draft and committed with it; its routes paragraph was
revised on 2026-10-26 (commit e749f1a). The further amendments this revision needs are listed in
the steps that make them.

## Goal

Bring the multi-unit route (`multiunit_views`, `joint_two_units`) up to the one-row route's
standard and to section 4A of the specification:

- the Bernoulli treatment of inadmissible rows;
- a factor rule chosen on tail metrics, with exact marginals for every unit, including units with
  loading one and two-point units;
- one-unit events kept out of the mixture;
- the published coupling rules as a named option, with the comonotone and Homer couplings exact
  on the grid;
- the occurrence outputs that the joint aggregate route and the conditional means need, as
  arrays, with a numpy helper for the annual conditional means.

Six steps, each a minor version, in the order below; the sixth is optional. A seventh, [parity],
covers documentation and benchmarks.

## Background for a reader with no context

`eltq` is a Rust engine (`crates/eltq-core`) with PyO3 bindings and a SciPy reference in
`python/eltq/` that serves as the oracle; every acceptance test runs against both. The
mathematical contract is `secondary-uncertainty-mathematical-specification.md`. The one-row route
(`quantize_*`, `exceedance_curve`) was rebuilt between 1.13.0 and 2.4.3: Bernoulli rows, the
survival method, overflow policies, the tail cut, narrow events, the incomplete beta and beta
quantiles at large parameters, and the survival method's fine mesh. The multi-unit route keeps the
architecture of `dev/done/plan-multiunit.md` (1.6.0 to 1.11.4); since then only 2.4.1 has touched
it (the admissibility test in ratios of the row's fields), and 2.3.6 moved its golden by rounding:

- per event, each unit's conditional bucket probabilities given the common factor $Z_c$ at $Q$
  quadrature nodes, by differences of the conditional distribution function at the half-bucket
  edges (beta units), or a deposited point (atoms, and comonotone units with $\sigma_I = 0$);
- per unit, the DecL-style terms applied as compiled view maps (`terms.rs::ViewMap`: each node's
  mass moves to the node's image and splits linearly between the two nodes around it), giving
  ground_up, gross, ceded and net rows;
- per view, the convolution across units (direct or FFT), the mixture over the $Q$ nodes with the
  rule's weights, and the rate-weighted deposit into the global grid.

Two documents define what this plan implements. Section 4A of the specification is the contract.
The article *Secondary Uncertainty in Catastrophe Model Event Loss Tables for Multi-Unit
Portfolios* (`D:\Projects\aggregate\secondary-uncertainty\MVMELT.qmd`, "the multi-unit article")
derives the model, collects the published rules (`tbl-rules`), compares the routes to the
portfolio total (`sec-routes`), and computes the joint, the annual bivariate and the conditional
means on the two-line example of Homer and Li 2017, Table 11 (`tbl-homer`, `sec-joint`,
`sec-allocation`). Its setup chunk is a 60-line SciPy construction of the joint of two units,
including two-point rows, that agrees with `joint_two_units` to $2 \times 10^{-16}$ per cell
(chunk `check-eltq`); it is the independent check for several steps below. The article is cited
here by its chunk and section labels, which survive its revisions; its table numbers do not.

### What the analysis of the routes changes

Three results of the multi-unit article, revised on 2026-10-26, shape the steps below.

- **The joint aggregate route and the conditional route are one random variable.** Compounding the
  joint occurrence law of the units with one Poisson count and adding the units' annual losses
  computes $\sum_i \sum_j \sum_{n \le N_j} X_{ij}^{(n)}$; forming the total at each occurrence and
  compounding computes $\sum_j \sum_{n \le N_j} \sum_i X_{ij}^{(n)}$, the same finite sum in the
  other order (the article checks it to $10^{-10}$, chunk `routes-compare`). The joint aggregate
  route is therefore needed only for what the conditional route cannot do: terms on a unit's
  annual loss, and allocation. Both act on a unit's loss after its occurrence terms, so the pair
  joint is needed by view, not only ground-up (step [joint]).
- **The roll-up differs for four reasons**: the marginal convention ($\sigma_I + \sigma_C$ against
  $t$), copula attenuation (the map from score to loss is not linear), the beta family's failure
  to close under addition, and the variance bound, which caps a unit row under the copula and
  only the total under the roll-up. The `homer` coupling is the roll-up and keeps all four; it is
  offered because it is published, not as a model.
- **The annual conditional mean of a unit given the total needs no two-dimensional grid.**
  Occurrences form a Poisson process, so by the Mecke identity
  $\mathsf P(A_u; A = s) = \sum_y m_u(y) f_A(s - y)$, with
  $m_u(y) = \sum_j \lambda_j \mathsf P(X_{uj}; X_j = y)$ the rate-weighted first moment of unit
  $u$ at each occurrence total and $f_A$ the annual density of the total. Then
  $\kappa_u(s) = (m_u * f_A)(s) / f_A(s)$, a one-dimensional convolution, for any number of
  units. Checked on 2026-10-26 (scratch probe, not committed) on the article's Homer example
  against `exeqa_df` of the article's two-dimensional bivariate aggregate: within
  $4 \times 10^{-10}$ for totals up to 4,000 and $10^{-7}$ up to 8,000, then diverging near 10,240,
  where the two-dimensional grid's axis for the unit ends and wraps; the one-dimensional result is
  the accurate one there.

### What the one-row work did and did not reach

Measured on 2026-10-25 at 2.3.2 (scratch probes, not committed):

| item | multi-unit route today |
|---|---|
| own erf, incomplete beta, normal quantile, 2.3.1 $\ln B$, 2.3.2 Cornish-Fisher quantiles | shared kernels; comonotone units take `beta_quantiles` and so inherit 2.3.2 |
| beta units at $\min(\alpha,\beta)$ up to $2 \times 10^6$, $h = 100$, edges inside the bulk | Rust and reference within $4 \times 10^{-9}$ per cell; the 2026-10-25 report's row (sd $10^{-6}$) is an atom in effect and passes |
| narrow units (sd below $h$) | exact: cdf differences at the edges put the mass in one bucket |
| variance policy | default `clamp` at 0.999; `bernoulli` raises `BernoulliPolicyUnavailable` |
| $Z_c$ rule | 32-node midpoint by default; a lone beta unit (300,000, 50,000, 50,000, 2,000,000) at $h = 1000$ returns mean 299,827 (midpoint 32), 299,957 (midpoint 128), 300,000.00 (Gauss-Hermite 32 or 64) |
| units with loading one | one atom per node, so the marginal is a $Q$-point quantile approximation, which spec test 25 codifies ("the bucketing of $\sum_kM_kF_k^{-1}(u)$ at the nodes"); a two-point unit at loading one is a step in $z$, whose mass $p$ a node rule resolves only to the nearest cumulative node weight |
| one-unit events | go through the mixture, inheriting the rule's error |
| specification | had no multi-unit section until 4A |

So the retrofit is not about the special functions, which are sound on this route, but about the
variance policy, the factor rule (units with loading one included), one-unit events, and the
outputs.

## The model, in brief

Per unit $i$ of an event the marginal is the one-row model (beta, two-point, or atom). One
standard normal $Z_c$ per occurrence and one $Z_i$ per unit give the score $V_i = w_i Z_c +
\sqrt{1 - w_i^2} Z_i$ and the loss $X_i = M_i F_i^{-1}(\Phi(V_i))$, with the loading $w_i$ the only
free quantity. Conditional on $Z_c = z$ the units are independent with
$\mathsf P(X_i \le x \mid z) = \Phi((y_i(x) - w_i z)/\sqrt{1 - w_i^2})$,
$y_i(x) = \Phi^{-1}(F_i(x/M_i))$; a two-point row fits with its step $F_i$. Everything is a
conditional calculation at fixed $z$ followed by a quadrature over $z$. The published rules are
Burke's loading $\sigma_C/\sqrt{\sigma_I^2 + \sigma_C^2}$, the comonotone loading 1, the
independent loading 0, and Homer and Li's roll-up with pro rata allocation, whose allocated losses
are comonotone but whose marginals are scaled copies of the rolled row's beta rather than the
rows' own, so that it lies outside the family. The variance-matched loading, Burke's scaled by a
common factor per event so that the event total's variance equals the roll-up convention's
$(\sigma_{I,R} + \sigma_{C,R})^2$, is the one alternative with a rationale and has no source. The
article also notes the loading $\sigma_C/(\sigma_I + \sigma_C)$, Burke's step iii weight before
standardization, and sets it aside as less conservative than Burke's; it is not offered.

## Decisions taken

Steve, 2026-10-25 (and change 2026-10-26):

1. Burke's coupling is the default; comonotone, independent and Homer are supported; the
   variance-matched loading is optional and last.
2. Homer's allocation is offered uncapped only, as published: $X_k = X_R\mu_k/\mu_R$, the
   allocations sum to the rolled loss and keep the mean, and a unit can receive more than $M_k$.
   There is no `homer_cap` keyword and no capped mode (revised 2026-10-26 at execution; the
   first version capped always, the second offered both). The expected allocation over the
   exposures, $\sum_j\lambda_j\mathsf P(\sum_k(X_{R,j}\mu_k/\mu_R-M_k)^+)$, is still reported
   (`homer_excess_mean`) as the one figure that shows the published rule's defect.
3. Outputs are arrays only. The annual bivariate aggregate, its marginals and its conditional
   distributions are the aggregate package's (`BivariateAggregate(mode="discrete")`, `exeqa_df`),
   fed from eltq's occurrence arrays; eltq writes no DecL.
4. The multi-unit material has its own article; the method article is unchanged apart from a
   possible tweak to its last section.

Steve, 2026-10-26, on review of the first draft (its open questions 1 to 4 are decisions 5 to 8):

5. `clamp` on the multi-unit route is deprecated and stays selectable, with a deprecation warning,
   until 3.0.0; removing it at a minor version would break the interface.
6. The occurrence outputs are `occurrence_pair_joint` (two named units, by view) and
   `occurrence_unit_first_moment` (every unit by default, by view), in place of the provisional
   `pair_severity` and `unit_total_severity`. The first draft's unit-against-total matrices
   $S_{u\Sigma}$ are dropped: the conditional means need only the vectors $m_u$ (step [joint]).
7. The default $Q$ may differ by rule.
8. Withdrawn 2026-10-26 with the revision of decision 2: there is no cap to default, and
   `coupling="homer"` reproduces the paper.
9. Every one-dimensional bucketing on this route is by cdf differences at the half-bucket edges of
   the fixed grid: one-unit events and the rolled row of the `homer` coupling alike. Section 4A
   and tests 25 and 27 require the one-row survival grid instead and are amended; the survival
   grid stays a comparison at a measured tolerance.
10. A unit with loading one has an exact marginal: under the stratified rule through its closed
    form (step [z-rule]), and under `coupling="comonotone"` through an exact path with no factor
    rule at all (step [coupling]). One atom per node remains only under the node rules.
11. eltq's Python layer offers `annual_conditional_mean(first_moment, annual_density)`, a numpy
    helper for the annual conditional means by the Mecke identity (step [joint]). The caller
    supplies the annual density, so eltq still computes no aggregate; decision 3 allows it,
    since the result is an array.

## Steps

Each step is one minor bump from the version current when it executes, and one commit (plus a
commit for any golden regeneration it forces), staged by path. Versions are assigned at execution
and recorded in the execution log and the CHANGELOG. Claude commits, Steve pushes.

### [spec] no bump: section 4A of the specification

Done with the first draft (2026-10-25): section 4A (the multi-unit table, the model, the coupling
rules, the three routes, terms and views, the integral over the factor, bivariate outputs,
metadata), section 7's multi-unit outputs paragraph, tests 25 to 29 in section 9, and item 9 of
section 11; section 4's `clamp` bullet points to 4A. The routes paragraph was revised on
2026-10-26 for the joint aggregate route (e749f1a). The steps below amend 4A's integral, homer,
bivariate outputs and metadata paragraphs, tests 25 to 29 and item 9 of section 11, each in the
step that changes the behavior.

### [bernoulli]: Bernoulli rows on the multi-unit route

- `multiunit.rs::prepare_unit` and `reference_multiunit.prepare_unit`: `VariancePolicy::Bernoulli`
  becomes a `UnitKind::TwoPoint` plan (the conditional law of section 4A), the default for
  `multiunit_views` and `joint_two_units`; `clamp` stays selectable with a deprecation warning
  and is removed at 3.0.0 (decision 5).
- Conditional matrix for a two-point unit: row $q$ has mass $1 - \Phi((\Phi^{-1}(1 - p) - w z_q)/
  \sqrt{1 - w^2})$ deposited at $M$ and the rest at 0; for $w = 1$ a step at $z = \Phi^{-1}(1-p)$,
  which the node rules resolve only to the nearest cumulative node weight until [z-rule]. Atoms
  unchanged. The marginal and layer-mean helpers (`analytic_view_mean`, the roll-up diagnostic)
  gain the two-point case.
- Diagnostics: `variance_bernoulli` replaces `sigma_clamped` as the reported category under the
  default.
- Goldens: `fixtures/golden/multiunit_small.*` and `joint_small.*` regenerate if they contain
  inadmissible rows (check first; the fixture generator lists them). CHANGELOG records the change
  in numbers.
- Specification: test 26's last clause ("`clamp` is rejected with the typed error of the one-row
  route") becomes the deprecation of decision 5.
- Tests: spec test 26, both backends; the multi-unit article's event 2 (`tbl-homer`: personal
  two-point, commercial beta) against the article's construction within $10^{-12}$, at the
  article's 64 Gauss-Hermite nodes.

### [z-rule]: the stratified factor rule, chosen on tail metrics

- Exact references, in place of the midpoint rule at $Q = 4001$, which cuts the tails of $Z_c$
  itself. One unit: the direct bucketing of its marginal by cdf differences. Two units: the
  bivariate normal rectangle probabilities, $\Phi_2(y_u, y_w; w_u w_w)$ at the bucket edges
  differenced over the cells, which is the joint exactly, for beta and two-point units, with no
  quadrature over $z$; the reference backend computes $\Phi_2$ from Owen's $T$
  (`scipy.special.owens_t`), accurate to rounding. More than two units: the stratified rule at
  $Q = 4096$, with its two-unit error against the exact reference recorded beside it.
- Extend `benchmarks/z_convergence.py`: rules midpoint, Gauss-Hermite and stratified; $Q$ in 8, 16,
  32, 64, 128; metrics the exceedance of the event total at $10^{-2}$, $10^{-3}$, $10^{-4}$,
  $10^{-5}$ against the exact references, plus the existing mean, sd, q99 and marginal errors;
  cases A to D of the multi-unit article (`tbl-cases`), the three events of `tbl-homer` with their
  two-point rows, the fixture, a lone beta unit with $w$ from 0.5 to 0.99, and units at loading
  one (a beta and a two-point row, each beside a partner with $w < 1$). Record as
  `benchmarks/z-convergence-<version>.md`.
- Implement the stratified rule: $Q$ equal-probability strata of $Z_c$; within each stratum each
  unit's conditional cdf at the edges averaged over the stratum by 8-point Gauss-Legendre in the
  probability scale $u = \Phi(z)$, so that the two end strata are finite intervals; units
  independent given the stratum.
  - A unit at loading one needs no quadrature: given $U = \Phi(Z_c) \in [a, b]$ its distribution
    function is $(\min(\max(F(x), a), b) - a)/(b - a)$, its marginal restricted to the stratum.
    Its marginal is therefore exact, two-point units included, and `UnitKind::Comonotone` takes
    this closed form under the stratified rule; under the node rules it stays one atom per node.
  - A unit at loading zero has its marginal in every stratum.
  - For $0 < w < 1$ the marginal is exact to the within-stratum quadrature. The study measures it
    against the closed form: the stratum average of $\Phi((y - wz)/\sqrt{1 - w^2})$ over
    $(z_a, z_b]$ is $Q\,(\Phi_2(y, z_b; w) - \Phi_2(y, z_a; w))$. If Gauss-Legendre falls short at
    $w$ near 1, the closed form moves into the engine (an Owen's $T$ kernel in Rust), and the step
    records the decision.
  - The rule's dependence error comes from treating the units as independent within a stratum;
    it is what the study measures, and it shrinks with $Q$.
- Set the default rule, and $Q$ for each rule, from the data (decision 7), as the node density was
  set at 1.16.0; write the figures into section 4A's integral paragraph and test 28's tolerance.
- Specification: in 4A's integral paragraph, requirement (ii)'s $Q = 4001$ reference becomes the
  exact references, and the stratified rule's "exact marginals and means by construction" becomes
  exact at loadings zero and one and to the within-stratum quadrature otherwise; test 28's
  reference likewise.
- Goldens regenerate if the default changes, and the CHANGELOG records it. `dev/plan-chebyshev-z.md` stays
  parked behind this step and is re-evaluated against its result.

### [one-unit]: one-unit events leave the mixture

- `views.rs` pass 1 marks events with one contributing unit. Such an event's ground-up row is
  the unit's marginal bucketed by cdf differences at the half-bucket edges (`marginal_pmf`, exact
  in probability; decision 9), its views the compiled maps of that row, no $z$ loop; the engine
  reports the count (`one_unit_events`).
- Under the stratified rule the mixture already gives a one-unit event's marginal to the
  within-stratum quadrature, so the branch is mainly a saving of $Q$ passes per such event; under
  the node rules it also removes the rule's marginal error.
- Reference: the same branch in `multiunit_views_reference`.
- Specification: 4A's integral requirement (i), "bucketed as section 5A buckets it", becomes
  bucketed by cdf differences at the half-bucket edges; test 27 becomes agreement with the direct
  bucketing within $10^{-12}$ of the rate, and with the one-row survival grid of section 5A at the
  tolerance this step measures.
- Tests: spec test 27 on both backends: a one-unit event under every coupling and rule equals the
  direct bucketing within $10^{-12}$ of the rate, and agrees with
  `quantize_arrays(method="survival")` on the same row within the measured tolerance, recorded in
  the specification; a mixed portfolio's total is the sum of its parts.

### [coupling]: the coupling option

- `coupling=` on `multiunit_views`, `joint_two_units` and the functions of [joint], values `burke`
  (default), `comonotone`, `independent`, `homer`; echoed in `MultiMeta`/`JointMeta`.
- `burke` and `independent`: a loading override in `prepare_unit` ($w$ forced to 0 under
  `independent`); an independent unit joins the independent block, which the engine already
  convolves once per event outside the $z$ loop. Under `burke`, a unit with $\sigma_I = 0$ beside
  units with $w < 1$ stays on the factor rule, with the exact marginal of [z-rule].
- `comonotone`: a path with no $z$ loop. Every unit of the event, and every view of it, is a
  nondecreasing function of one uniform $U$, so the event total of a view is the comonotone sum
  of the units' bucketed view marginals. Merge the units' cumulative probabilities at their bucket
  edges into one sorted list of breakpoints in $u$; on each interval between breakpoints every
  unit sits at one node, and the interval's length is deposited at the sum of those nodes. The
  result is exact for the bucketed marginals, at a cost linear in the number of buckets per
  event. The stratified rule alone would give exact marginals but a dependence error, since it
  treats the units as independent within a stratum while comonotone units share $U$ exactly.
- `homer`: a separate path with no $z$ loop. Per event: roll the rows up by the moment rule,
  classify the rolled row by the one-row rules (beta, two-point, atom), bucket its marginal on the
  grid by cdf differences at the half-bucket edges (decision 9), and push it through the composed
  map per view, compiled as a `ViewMap` is: $x \mapsto \sum_k \mathrm{term}_k(x \mu_k/\mu_R)$
  uncapped, or $x \mapsto \sum_k \mathrm{term}_k(\min(x \mu_k/\mu_R, M_k))$ with
  `homer_cap=True` (the breakpoints are the units' breakpoints scaled by $\mu_R/\mu_k$, and the
  caps when capped). The expected excess over the exposures,
  $\sum_j \lambda_j \mathsf P(\sum_k (X_{R,j}\mu_k/\mu_R - M_k)^+)$, is computed analytically from
  the rolled row's limited expected values and reported in both modes, with the mode
  (`homer_excess_mean`, replacing `homer_cap_mean_loss`).
- `homer_cap` on `multiunit_views` and wherever `coupling` is accepted, default `False`
  (decision 8), ignored with a warning under the other couplings; echoed in `MultiMeta`.
- Specification: section 4A's `homer` paragraph makes the cap an option (decision 2, default
  `False`) and buckets the rolled row by cdf differences in place of "the one-row survival method
  of section 5A on the rolled table" (decision 9); the metadata paragraph renames
  `homer_cap_mean_loss` to `homer_excess_mean`; item 9 of section 11 records the cap as an
  option. Test 25 becomes:
  - `homer`: the ground-up total equals the direct bucketing of the rolled table composed with the
    allocation map within $10^{-12}$ of the rate, in both modes, and the one-row survival grid at
    a measured tolerance;
  - `comonotone`: the total equals the comonotone sum of the bucketed marginals within $10^{-12}$
    of the rate, each unit's marginal from the joint equals its direct bucketing within
    $10^{-12}$, and with shared $(\alpha, \beta)$ the total equals the rolled row's beta to
    bucketing tolerance (Proposition 6), in place of "the bucketing of $\sum_kM_kF_k^{-1}(u)$ at
    the nodes".
- Tests: spec test 25 for the four couplings, both backends, with `homer` in both modes: uncapped,
  the allocations sum to the rolled loss and the mean is exact; capped, each allocation is at
  most $M_k$ and the mean falls by the reported excess. The multi-unit article's `tbl-couplings`
  (case A under the three loadings) reproduced within the rule's error; the article's comonotone
  joint is built on 4,096 midpoint nodes, so the exact path agrees with it to that rule's error,
  recorded.

### [joint]: occurrence outputs for the joint aggregate route and the conditional means

- `occurrence_pair_joint(event_id, unit_id, ..., units=(u, w), view="ground_up", bucket_size,
  ...)` returns the rate-weighted joint occurrence severity of the view,
  $S_{uw} = \sum_j \lambda_j J_{j,uw}$, on the common grid (shape $n_u \times n_w$), its two grids
  and metadata (mass, marginal means, the correlation at an occurrence). Each unit's compiled
  `ViewMap` is applied to its conditional row before the outer product, because the joint
  aggregate route exists for terms on a unit's annual loss and for allocation, and both act on
  the unit's loss after its occurrence terms. Per event with both units present, the joint is the
  engine's two-unit GEMM over the rule's nodes or strata; with one present, an outer product with
  the atom at zero; with neither, nothing. Under `comonotone` the merged breakpoints of [coupling]
  deposit each interval at the pair of nodes; under `homer` each rolled bucket's mass goes to the
  pair of its allocations' view images. Chunked and deterministic as `multiunit_views`.
- `occurrence_unit_first_moment(..., units=None, view="ground_up", ...)` returns, for each unit $u$
  (all by default), the vector $m_u(y) = \sum_j \lambda_j \mathsf P(X_{uj}; X_j = y)$ on the grid
  of the view's total: per event and node or stratum, the convolution of $x\,f_u(x \mid z)$ with
  the convolution of the other units, formed for all the units of an event in linear time by
  prefix and suffix products of the unit spectra, accumulated rate-weighted. Memory is $n_\Sigma$
  per unit. $\sum_u m_u(y) = y\,s(y)$, with $s$ the run's rate-weighted total severity, holds
  exactly on the grid and is the run's check.
- `annual_conditional_mean(first_moment, annual_density)` (decision 11) returns
  $\kappa_u(s) = (m_u * f_A)(s) / f_A(s)$ for each unit, by the Mecke identity (Background).
  `first_moment` is the output of `occurrence_unit_first_moment`; `annual_density` is the annual
  density $f_A$ of the same view's total on the same bucket size and origin, which the caller takes
  from the aggregate package or from any compound Poisson calculation on $s$. The convolution is
  by real FFT, zero-padded to the sum of the two lengths so that nothing wraps, and truncated to
  the length of $f_A$; only $f_A$ below that length enters, so the result is as exact as $f_A$ is
  there. The result is NaN where $f_A$ is zero; filtering by a probability floor is the caller's.
  Pure numpy, about ten lines, one implementation for both backends. The two-dimensional route
  through `BivariateAggregate` remains available from `occurrence_pair_joint` and its shear, and
  is the test of the identity.
- Reference implementations of the two occurrence functions.
- Specification: 4A's bivariate outputs paragraph replaces $S_{u\Sigma}$ and its shear with $m_u$
  and the identity, with the unit's joint with the total, for two units, the shear of
  $S_{uw}$; section 7's multi-unit outputs paragraph likewise; test 29 restated as below.
- Tests: spec test 29 on both backends:
  - the marginals of $S_{uw}$ equal the two units' occurrence severities of the view within the
    rule's error, and the mixture over events of the per-event joints equals the returned array
    within $10^{-12}$ of the rate;
  - $\sum_u m_u(y) = y\,s(y)$ within $10^{-12}$ relative at every total with mass;
  - for a two-unit portfolio, $S_{uw}$ summed along its antidiagonals equals the `multiunit_views`
    total of the same view within $10^{-12}$ of the rate;
  - the handoff, route 2 against route 3: the two-dimensional compound of $S_{uw}/\Lambda$ under a
    Poisson($\Lambda$) count by the aggregate package, summed along its antidiagonals, equals the
    one-dimensional compound of the total within the aggregate package's grid tolerance (the
    article reaches $10^{-10}$);
  - $\sum_u \kappa_u(s) = s$ from `annual_conditional_mean` at every total whose annual density is
    at least $10^{-12}$, with $f_A$ from a numpy compound Poisson of $s$ built in the test;
  - $\kappa_u$ from `annual_conditional_mean` equals `exeqa_df` of the two-dimensional compound of
    the sheared $S_{uw}$ at every total where the two-dimensional grid does not wrap.
  The aggregate package is a test-only dependency; the handoff test and the last test are skipped
  without it.
- Notebook: the Homer Table 11 example end to end, as in the article's `sec-allocation`, with
  $\kappa_u$ by `annual_conditional_mean` and by the two-dimensional route.

### [matched], optional: the variance-matched loading

- `coupling="matched"`: per event, bracket $c$ on $[0, 1/\max_i w_i]$ for the root of
  $\operatorname{Var}(\sum_i X_i) = (\sigma_{I,R} + \sigma_{C,R})^2$, the variance computed from
  the conditional means and variances per unit at the rule's nodes or strata (no convolution
  needed: the conditional variance of a sum of conditionally independent units is the sum, and the
  variance of the conditional mean is a quadrature); about twenty evaluations per event.
  Saturation reported per event (`matched_shortfall`).
- Tests: spec test 25's `matched` clause; the multi-unit article's `tbl-matched` (case A and
  Homer's events 1 to 3, including event 2's saturation) reproduced within the rule's error.
- The first step to drop if time is short: the loading has no source, and the article only
  presents it.

### [parity] with [matched] or as a patch after [joint]: documentation and benchmarks

- README milestone rows, the Python API block, `MultiMeta` fields; CHANGELOG per version.
- `benchmarks/run_multiunit.py` rerun under the new default rule and policy, with the report
  `benchmarks/report-<version>.md`; the all-four-views delta and the speed of the `comonotone` and
  `homer` paths recorded.

## Files

- `secondary-uncertainty-mathematical-specification.md` (4A done; amended by the steps).
- `crates/eltq-core/src/multiunit.rs` (kinds, two-point plan, loadings, the closed form at loading
  one), `views.rs` (one-unit branch, `comonotone` and `homer` paths, pair and first-moment
  accumulators), `quadrature.rs` (stratified rule), `config.rs` (coupling, policy default),
  `terms.rs` (composed Homer map), `diagnostics.rs`.
- `crates/eltq-python/` bindings for the new keywords and functions.
- `python/eltq/__init__.py` (including `annual_conditional_mean`), `_meta.py`,
  `reference_multiunit.py`.
- `tests/test_multiunit.py` (tests 25 to 29), `fixtures/generate_fixtures.py` and goldens as
  forced, `benchmarks/z_convergence.py`, `benchmarks/run_multiunit.py`, `notebooks/`.
- `README.md`, `CHANGELOG.md`, `Cargo.toml`, `pyproject.toml`, `python/eltq/__init__.py`
  (version triple).

## Acceptance

- `cargo test -p eltq-core --release` and `uv run pytest` green at every step, both backends.
- Spec tests 25 to 29 implemented and passing; the figures they quote recorded in the
  specification where the plan leaves them open (the default rule and $Q$ for each rule, the
  one-unit and `homer` tolerances against the survival grid).
- Exact marginals for units at loading one: under `coupling="comonotone"` every unit's marginal
  from the joint equals its direct bucketing within $10^{-12}$; beside units with $w < 1$ under
  the stratified rule, within the rule's measured tolerance.
- The multi-unit article's `tbl-cases`, `tbl-couplings`, `tbl-matched` and `fig-route-compare`
  reproduced from the package (a test or a notebook cell each).
- Determinism across thread counts bitwise, as today; no negative weight on any path.
- Speed: the default rule's multi-unit benchmark within the 1.11.0 figures scaled by its $Q$;
  the `homer` and `comonotone` paths at one-row speed; no regression on the one-row route
  (unchanged code).

## Alternatives considered

- *Raise $Q$ under the midpoint rule.* The tail cut shrinks only as $1/Q$ and the cost is linear in
  $Q$; the study decides, but the stratified rule's exact marginals are the better bet.
- *The midpoint rule at $Q = 4001$ as the study's reference.* Replaced: it cuts the tails of $Z_c$
  itself, while the bivariate normal rectangles are exact for two units.
- *Bivariate normal rectangles as the production rule for two units.* Exact, but $n_u n_w$
  evaluations of $\Phi_2$ per event against the GEMM's $Q(n_u + n_w)$ special-function evaluations
  and $Q n_u n_w$ multiply-adds; kept as the reference.
- *Comonotone through the stratified rule alone.* Exact marginals but a dependence error; the
  merged breakpoints are exact at lower cost.
- *Homer capped only.* The first version of decision 2, revised: the published rule allocates
  more than a unit's exposure in the paper's own example, but it is the published rule, so it
  stays available and the cap is an explicit choice.
- *Reallocate the capped excess to other units, or offset the mean.* Rejected: a convention of
  ours with no source.
- *Variance matching by a pairwise loss correlation rather than a common scale.* Not identifiable
  from one equation per event for $n > 2$ units; the common scale is the minimal one-parameter
  family containing Burke.
- *Bivariate compounding inside eltq.* Rejected (decision 3): the aggregate package has the 2-D
  FFT, the marginal checks and `exeqa_df`; duplicating them adds code and a second place for the
  grid conventions to drift.
- *The unit-against-total matrices $S_{u\Sigma}$ of the first draft.* Dropped (decision 6): for the
  conditional mean the Mecke identity needs only $m_u$, of memory $n_\Sigma$ against
  $n_u n_\Sigma$ per unit, for any number of units; for two units the joint of a unit with the
  total is the shear of $S_{uw}$.
- *The one-row survival method for one-unit events and for the rolled `homer` row.* Rejected
  (decision 9): the engine buckets on the fixed global grid and already evaluates the cdf at every
  edge, so cdf differences are exact there at the same cost; the survival curve's node machinery
  brings nothing on a fixed grid. The survival grid stays the comparison in tests 25 and 27.

## Not in this plan

- Per-event grids for large exposed values on the multi-unit route (the one-row route's section
  5A window and overflow policies have no multi-unit counterpart yet); `GridOverflowError` stays.
- A survival-style, node-based formulation of the multi-unit total: the convolution is the
  honest route and nothing suggests a closed form for the total's survival function.
- The annual joint distribution of more than two units, and a unit's distribution given the total
  beyond its mean for more than two units: `BivariateAggregate` is two-dimensional, and the
  conditional means of any number of units are covered by $m_u$.
- The loading $\sigma_C/(\sigma_I + \sigma_C)$, which the multi-unit article sets aside.
- The reserving application of `MV-cat-reserves.qmd` (open claims as certain events, direct input
  of the total sd and a correlated share): a transformation of the inputs, to be specified when
  that article is written.
- `dev/plan-chebyshev-z.md` [hybrid-rule] and [chebyshev]: re-evaluated after [z-rule].
- The deferred review items of `dev/done/plan-multiunit-fixes.md` (grid length bound, the NaN
  rate category mismatch, `1_000` in the Rust terms parser).

## Open questions for review

None. The first draft's four questions are decisions 5 to 8, and the revision's one question,
the helper for the annual conditional mean, is decision 11.

## Execution log

Claude commits, Steve pushes. The Rust gate is `cargo test -p eltq-core` at the plain test
profile (the acceptance section above says `--release`; the project's delta of 2026-10-26 moved
the gate, and that is the one used). `dev/plan-chebyshev-z.md`, which [z-rule] above says stays
parked, had already been closed as superseded in `dev/done/` on 2026-10-25; nothing to park.

### [bernoulli] 2.5.0, 2026-10-26

- `UnitKind::TwoPoint` (Rust) and `_TWO_POINT` (reference) with `two_point_conditional` on both
  backends; `UnitPlan::depends_on_factor` replaces the ad hoc test in `views.rs` pass 2. The
  default policy of `MultiConfig`, `JointConfig`, both Python entry points and the reference
  is `bernoulli`; the roll-up diagnostic's rolled row follows it.
- `Error::BernoulliPolicyUnavailable` removed. The deprecation warning for `clamp` is raised in
  the Python layer (`reference_multiunit.warn_if_clamp`, called by the reference functions and
  by the package wrapper on the Rust path), since the engine has no warning channel.
- Divergence: the plan's test for event 2 of Homer's table against the article's construction
  is run for all three events; the article's `Unit`, `conditional_pmf` and `joint` are ported
  into `tests/test_multiunit.py` (a test must not import the article), and both backends agree
  with them within $10^{-12}$ per cell. The views route's mean reconciliation on that table is
  checked at $10^{-3}$ relative, not to rounding: Gauss-Hermite at 64 nodes misses the marginal
  of event 2's commercial row ($w = 0.96$) at the $10^{-4}$ level, the rule's error that
  [z-rule] addresses; a two-point row beside an atom reconciles within $10^{-12}$.
- No golden moved: the multi-unit fixture has no inadmissible row (the generator's comment on
  e4/b said "clamped"; $900^2 < 600\times1400$, and the comment is corrected).
- Spec: section 4's `clamp` bullet and test 26's last clause amended as the step says.

### [z-rule] 2.6.0, 2026-10-26

- The plan's 8-point Gauss-Legendre stratum average in the probability scale was measured first
  (scratch probe, not committed) and falls short everywhere, not only near loading one: marginal
  errors of $3\times10^{-5}$ at $w = 0.3$ to $4\times10^{-3}$ at $w = 0.999$, from the two end
  strata, where $\Phi^{-1}$ is singular in $u$. Per the step's own fallback the closed form went
  into the engine: `normal::bivariate_cdf` and `BivariateUpper` (Genz 2004, `bvnu`: Drezner and
  Wesolowsky's integral by Gauss-Legendre, 6, 12 or 20 points by $|\rho|$, the transformed
  integrand from $|\rho| = 0.925$), held to the reference's Owen's $T$ form
  (`reference_multiunit.bivariate_normal_cdf`, `scipy.special.owens_t`) within $5\times10^{-15}$,
  which itself agrees with `scipy.stats.multivariate_normal` to $2\times10^{-16}$. The stratum
  average is formed as a difference of upper-tail probabilities $U(y, z) = \mathsf P(V \le y, Z_c >
  z)$ on both backends, so the narrow tail strata keep their relative accuracy; the two backends
  agree on the fixture's views within $10^{-12}$ of the total rate.
- **Finding, and the decision it forced.** With $Q$ equal-probability strata, as the plan and
  section 4A described the rule, the marginals are exact but the event total's exceedance at the
  $10^{-4}$ and $10^{-5}$ levels is short by 50% to 99% even at $Q = 128$ in every dependent case:
  the whole upper tail of the total lies inside the top stratum, where the units are treated as
  independent. Equal strata are the wrong partition for tail metrics. A second probe (not
  committed) measured strata refined geometrically in the upper tail of the factor, the one-row
  survival method's ladder idea: the error is then set by the ratio alone, at every level
  ($1.2\times10^{-2}$ at ratio 2, $3\times10^{-3}$ at $\sqrt2$, $7.5\times10^{-4}$ at $2^{1/4}$),
  and the body count hardly matters. **Adopted:** `z_nodes` equal strata over $[0, 3/4]$, a
  ladder of ratio $\sqrt2$ from $1/4$ down to $2^{-24}$ (44 strata) in the upper tail, then the top
  stratum; 76 strata at the default `z_nodes = 32`. The lower tail keeps the body's width: every
  view is nondecreasing in the loss, so every exceedance lives in the upper tail of the factor
  (a symmetric ladder measured the same tails at 118 strata). Constants `STRATA_BODY_TOP`,
  `STRATA_TAIL_RATIO`, `STRATA_TAIL_FLOOR` in `quadrature.rs` and `_meta.py`. Section 4A's
  integral paragraph is rewritten accordingly; `z_strata` in `MultiMeta` and `JointMeta` reports
  the count.
- **Default** (decision 7): `z_rule="stratified"`, `z_nodes=32`, on `multiunit_views`,
  `joint_two_units` and the reference; the joint's old default of 1001 midpoint nodes goes
  (76 strata, exact marginals, tails within $7\times10^{-3}$; midpoint at 1001 had marginal errors
  of $4\times10^{-5}$). Measured against the exact references in
  `benchmarks/z-convergence-2.6.0.md`: worst tail error over the nine two-unit cases
  $6.7\times10^{-3}$ (case B at $10^{-4}$), case D (both loadings one) $1.7\times10^{-2}$ at
  $10^{-2}$ and exact beyond, standard deviations within $7\times10^{-4}$; Gauss-Hermite at 32 is
  $10^{-9}$ on case A but $0.15$ off in the marginal of a unit at loading one; midpoint is off by
  $10^{-1}$ to $1$ in every tail.
- **Cost, the trade Steve should weigh.** On the 0.91M unit-row benchmark portfolio the
  stratified default takes 8.6 s against 2.5 s (midpoint 32) and 2.0 s (Gauss-Hermite 32); it was
  11.4 s before the bivariate normal's per-unit constants were hoisted into `BivariateUpper`
  (which halved the two-unit joint's construction, 2.56 ms to 1.36 ms at $101 \times 201$).
  `z_nodes=8` (52 strata) takes 6.6 s with the same tails. The plan's acceptance line allows the
  1.11.0 figures scaled by $Q$, 2.4 times for 76 strata; the default is 3.4 times, the bivariate
  normal construction being the excess. The node rules stay one keyword away for a portfolio
  whose loadings are all well below one; [parity] records the final figures.
- `z_quadrature` keeps its node-rule contract and refuses the stratified rule; `FactorRule`
  (both backends) carries nodes or strata to the conditional matrices, with `from_edges` for
  explicit partitions (the study's equal strata). The study, `benchmarks/z_convergence.py`, is
  rewritten as the step says, with the exact two-unit reference `exact_two_unit_joint` (reference
  backend) in place of the midpoint rule at $Q = 4001$; the fixture's reference is the stratified
  rule at $Q = 4096$ body strata.
- Goldens: `multiunit_small` regenerated under the new default (params `z_rule="stratified"`;
  the e3 one-unit event and every mixture move by the rule's error); `joint_small` keeps its
  explicit 1001 midpoint nodes and gains `z_strata` in its metadata.
- `dev/plan-chebyshev-z.md`, closed before this plan started, stays closed: the rule's cost is
  the strata count times the bivariate normal construction, and neither a hybrid node rule nor a
  Chebyshev interpolant in $z$ addresses the within-stratum independence that the ladder fixes.

### [one-unit] 2.7.0, 2026-10-26

- Engine: the event's live rows are collected first; with one of them the base row is
  `marginal_pmf` with a single row set, and the existing convolution machinery (a unit with one
  row is "independent") carries it through the views with no loop. Reference: the same branch
  with unit weights. `one_unit_events` on `ChunkPartial`, `MultiMeta` (both backends) and the
  bindings.
- Divergence, recorded: the plan says the goldens regenerate only if the default rule changes;
  the multi-unit golden regenerates here too, since e3 is a one-unit event and e6 has one live
  unit after its dropped row. Under the stratified
  default it moves only by rounding (the mixture already returned the marginal), so the
  regeneration is a formality; the generator's comment says why.
- The survival-grid comparison the step asks to measure: on e3 at $h = 10$ the one-row survival
  grid of `quantize_arrays` differs from the direct bucketing by $5.6\times10^{-8}$ of the rate;
  test 27 fixes $10^{-6}$. Test 27 also covers a two-point, an atom and a comonotone row under all
  three rules, and the termed unit's gross view against the direct bucketing pushed through its
  tower.

### [coupling] 2.8.0, 2026-10-26

- As ruled at execution (decision 2 revised): no `homer_cap`, no capped mode; the step's text
  above still describes both modes and `homer_cap_mean_loss`, and the specification's `homer`
  paragraph, metadata paragraph, item 9 of section 11 and test 25 were rewritten for the uncapped
  rule with `homer_excess_mean`. The expected allocation over the exposures is kept as a
  diagnostic, the one figure that shows the published rule's defect; on Homer's Table 11 it is
  positive (event 1's commercial line holds 40% of the mean and 25% of the exposure).
- `Coupling` in `multiunit.rs` with `UnitPlan::couple`; `comonotone_merge` (a callback per
  combination of bucket indices, shared by the total and the pair joint), `rolled_spec`,
  `terms_on_rolled`, `homer_view_values`, `homer_excess`; `ViewMap::from_values` for the composed
  allocation map. The reference mirrors each (`couple`, `comonotone_merge`, `rolled_unit`,
  `terms_on_rolled`, `homer_view_values`, `homer_excess`, `_transform_values`).
- The homer path's grid: pass 1 sizes an event under `homer` to the summed exposures as well as
  the units' supports, since the rolled beta reaches $M_R$ while an atom's support is its mean;
  the two-unit joint checks each unit's grid against $s_k M_R$ and raises the overflow error
  with the required length.
- Divergence, recorded: test 25's `comonotone` clause asked for the total within $10^{-12}$ of
  "the comonotone sum of the bucketed marginals"; the test checks that identity through an
  independent inverse-cdf construction on a uniform grid of $10^{-6}$, which agrees within
  $2\times10^{-6}$, and the exactness of the marginals and the mass within $10^{-12}$. The view
  means under `comonotone` equal the bucketed marginals' means exactly and differ from the
  analytic means by the bucketing of the betas, as under every rule; likewise the `homer`
  ground-up mean is the rolled row's to its bucketing ($3\times10^{-4}$ at $h = 10$ on Homer's
  table), not "exact" as the step's text says: cdf differences at the half-bucket edges are exact
  in probability, not in mean (decision 9). The tests fix $10^{-3}$ for those means. Proposition 6
  by the exact path: the comonotone sum of two U-shaped bucketed marginals is coarser than the
  rolled beta's bucketing at the origin (both units in bucket 0 land at 0), $1.6\times10^{-2}$ in
  the first cell of the cdf at $h = 5$, while the standard deviation and the 99th percentile agree
  within $10^{-3}$ and $5\times10^{-3}$.
- The article's `tbl-couplings` (case A under the three loadings): the `independent` and `burke`
  rows are the existing joint tests; the `comonotone` row, built in the article on 4,096 midpoint
  nodes with each unit's whole sd on the factor, agrees with the exact path on the total's
  standard deviation within $10^{-3}$ and on its 99th percentile within $5\times10^{-3}$, the
  midpoint rule's error at that count.

### [joint] 2.9.0, 2026-10-26

- The occurrence outputs are computed inside the existing event loop of `multiunit_views`
  (`MultiConfig.pair`, `MultiConfig.first_moments`; `MultiResult.pair`,
  `MultiResult.first_moments`), sharing the classification, the materialized view rows and the
  chunk fold, rather than in a second driver; the engine's Python binding returns a third value
  with them, and `occurrence_pair_joint` / `occurrence_unit_first_moment` run the engine with
  `views=(view,)` and build `PairMeta` / `FirstMomentMeta`. The reference mirrors this through
  `pair=` and `first_moments=` on `multiunit_views_reference`, which then returns a third value.
- Divergence, recorded: the pair joint over the factor rule is a weighted sum of outer products
  in plain loops, not the two-unit GEMM the step names; the flops are the same and the loop
  keeps the three couplings and the one-present case in one place. Grid lengths of the pair come
  from each unit's largest support over the events (its share of the rolled exposure under
  `homer`), tracked in pass 1.
- The step's "with neither, nothing" is kept: an event exposing neither unit of the pair
  contributes nothing, so `PairMeta.mass` is the rate of the pair's own occurrences, the
  Poisson rate to compound with (thinning makes this equivalent to the full rate with an atom at
  the origin).
- The identity $\sum_u m_u(y) = y\,s(y)$ is checked against the view's own per-event pmf (the
  convolution path's), not a second inverse transform, and reported rate-weighted in loss units
  (`identity_error`, $10^{-16}$ on the fixture). Test 29's "within $10^{-12}$ relative at every
  total with mass" is applied as $10^{-10}$ of the rate-weighted mean, since the relative form
  fails where a cell's mass is itself at rounding level.
- Under `homer` the pair joint deposits a rolled bucket's two allocations on their own axes, so
  the antidiagonal sums of $S_{uw}$ reproduce the view's total in mass and mean but not cell by
  cell (the index sum of two linear deposits spreads by up to a bucket); test 29 states the
  antidiagonal identity for the other three couplings and the mass and mean under `homer`.
- The aggregate package is not in the project's environment; the handoff test and the `exeqa_df`
  comparison are skipped without it and were run locally with `uv run --with aggregate` (see the
  report). Adding it to the `dev` group so that CI runs them is Steve's call.
- `notebooks/multiunit_homer_example.ipynb`: Homer's Table 11 end to end; its last cell needs the
  aggregate package.
