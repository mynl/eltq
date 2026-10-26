# Plan: peer-review-response, the numerical and scheduling corrections from the 2026-10-25 review

Date: 2026-10-25. Status: proposed, for Steve's review and decisions before any implementation.
Author: Claude, from the independent peer review `dev/peer-review.qmd` (version 2.3.2, commit
`cf28041`), after reproducing each of its counterexamples and one measurement of its own.

Versions are assigned when the plan executes; each step below is one bump, patch or minor as
marked. `dev/plan-multiunit-coupling.md` has 2.4.0 to 2.9.0 written into its steps; whichever
plan lands first takes the next numbers and the other renumbers in its execution log.

## Goal

Make the survival method accurate to a stated contract over its accepted parameter domain, so
that the claims "exact in rate, exceedance within $10^{-4}$" hold for adversarial single-event
portfolios and not only for the benchmark portfolios, and take the demonstrated scheduling gain in
the quantiles method. Eight findings of the review were reproduced on the installed 2.3.2 build;
every one is to be either corrected and pinned by a test, or documented as a limit of the domain.

## Background for a reader with no context

`eltq` converts an event loss table (rate $\lambda_j$, exposed value $M_j$, conditional mean
$\mu_j$, conditional standard deviation $\sigma_j$ per event) into a rate-weighted loss grid. Each
event's conditional loss is a beta on $[0, M_j]$ with shape parameters
$(\alpha_j,\beta_j)=\kappa_j(p_j,1-p_j)$, $p_j=\mu_j/M_j$,
$\kappa_j=\mu_j(M_j-\mu_j)/\sigma_j^2-1$. The default method (spec section 5A) builds the
portfolio exceedance rate $\Lambda(x)=\sum_j\lambda_jS_j(x)$ and buckets it at the half-bucket
edges $e_i=(i+\tfrac12)h$. The parts that matter here:

- **Shared curve.** Geometric nodes at `nodes_per_decade` (default 100) from $x_0=h/2$ to
  $\max_jM_j$. Each event's survival and slope are evaluated exactly at its active nodes; between
  nodes the sum is a monotone cubic Hermite in $x$ (`curve.rs`, `build_curve` and
  `Curve::interpolate`).
- **Tails.** Each event is cut at its truncation node $x_{k_j}$, four intervals below $M_j$;
  above it the event is its own interpolant in $u=\log(M_j-x)$ on a ladder of nodes shrinking the
  distance to $M_j$ by $\sqrt2$ per step down to $x_0/2$, then the power law $(M_j-x)^{\beta_j}$
  (`TailInterpolant`). A tail is kept only if $S_j(x_{k_j})>0$ in double precision.
- **Narrow events.** An event with $\sigma_j$ below one node interval at its mean and
  $\alpha_j,\beta_j>1$ is kept out of the curve and bucketed by quadrature of its density
  (`NarrowKernel`, `narrow_bulk`). Above $\alpha_j+\beta_j=10^{15}$ it is replaced by a step at
  its mean (`NARROW_STEP_PARAMETER_SUM`).
- **Incomplete beta.** The engine's own continued fraction (`incbeta.rs`, `betacf`), 400
  iterations, returns whatever it has when the cap is reached.
- **Overflow.** When the grid ends below $\max_jM_j$, the mass above the top edge is reported
  with its displaced first moment. The curve's share of that moment is computed per event from
  the kept tails (`Curve::grid`, the loop over `self.tails`).
- **Diagnostic.** The curve's interpolation error is sampled at 64 log-midpoints against the
  exact total (`sample_intervals`); the spec calls it a sample, not a bound.
- **Scheduling.** Rows are processed in fixed chunks folded in order; the quantiles method's
  default chunk is 65,536 rows, the survival method's 256 (`config.rs`). `run_chunks` collects
  partials in waves of twice the thread count.

## The findings, checked

Each reproduction in the review's appendix was rerun on the installed 2.3.2 build
(scratch script, outside the repository). All eight reproduce to the digits the review gives.

| ID | Finding | Reproduced | Assessment |
|:---|:---|:---|:---|
| A1 | Step at the mean above $\alpha+\beta=10^{15}$: a symmetric event with its mean on a bucket edge goes wholly to one bucket; the same event with $\sigma=0$ splits half and half. | Yes: grid `[0,0,0,1,0,0]` for $\sigma=10^{-8}$, `[0,0,.5,.5,0,0]` for $\sigma=0$. | Real but bounded: a half-bucket misplacement, $\lambda h/2$ in the mean at worst, the nearest rule's own order. The inconsistency with $\sigma=0$ and the lost overflow moment (I1, step case) are the parts to fix. The threshold's justification in the spec is wrong: the standardized coordinate resolves the density to about $10^{-7}$ at $10^{15}$. |
| A2 | Tail power law used before its asymptotic regime: $\operatorname{Beta}(10^6,0.5)$ at an edge $5\times10^{-7}$ below $M$. | Yes: grid tail 0.0540, exact 0.6827, curve query exact. | Real. The ladder stops at distance $x_0/2$ whatever $\alpha$ is; the first correction $(\alpha-1)\beta t/(\beta+1)$ is 0.17 there. The public curve is unaffected (it evaluates tails directly). |
| A3 | Shared cubic 22.8% off in the upper tail of $\operatorname{Beta}(1800,10^6)$ with the diagnostic at $1.6\times10^{-5}$. | Yes. | Real, and the review understates it; see the measurement below. |
| I1 | Overflow moment zero for events whose endpoint survival underflows; the reference gets it. | Yes: Rust 0, reference and exact $5.796\times10^{-4}$; step case rate 1, moment 0, exact 0.15. | Real and material: with $M$ at $10^6$ the missing moment is 580 per unit rate. Most events of a real ELT have an underflowing endpoint survival, so this affects every windowed grid. |
| I2 | Unconverged continued fraction returned silently; $a=b=10^9$ at 200,000 nodes per decade gives bucket weights $[-0.45, 1.45]$. | Yes. | Real. The narrow routing shields the default configuration, but the routing depends on node density, not on the kernel's validity, and a negative weight breaks spec section 8. |
| I3 | Tiny-$\alpha$ small survival by cancellation; validation overflows at scale $10^{160}$ and underflows at $10^{-200}$; `Cumulants::of` overflows near $s=10^{103}$; the top half-bucket atom is unreported first-moment loss. | Yes, all four. | Domain edges. Two are one-line fixes (ratios in `validate.rs`, the cumulant formula); two are documentation. |
| P1 | Quantiles method: four chunks for 262,144 rows leaves 28 of 32 threads idle; `chunk_rows=4096` is 4.74 times faster. | Not rerun; the mechanism is confirmed in `config.rs` and `run_chunks`. | Real. The review's memory caveat is also right: `run_chunks` holds a wave of $2\times$`threads` partials, each two full grids, so smaller chunks at a large grid cost memory. |
| P2 | Redundant work: per-edge binary search in `Curve::grid`, repeated grid scans, a thread pool per `run_chunks` call, normal scores recomputed per event, `log(d)` computed twice per tail edge. | Confirmed by inspection. | Take the cheap ones; measure. |

### The measurement that changes the priority

The review's A3 is one point. The sweep below is one bell-shaped event, $\beta=10^6$, the mean
$\mu$, `min_loss` $=10^{-3}\mu$, 100 nodes per decade, evaluated at every node-interval midpoint
against SciPy's `betaincc`, the largest relative error by band of $S$. The first column is the
event's standard deviation in node intervals (one interval is 2.33% of the loss at this density;
the narrow route takes events below 1.0).

| $\sigma$ / interval | $S\in[10^{-2},1)$ | $[10^{-4},10^{-2})$ | $[10^{-6},10^{-4})$ | $[10^{-8},10^{-6})$ | $[10^{-12},10^{-8})$ |
|---:|---:|---:|---:|---:|---:|
| 1.05 | $3.1\times10^{-2}$ | $1.6\times10^{-1}$ | $1.2\times10^{-1}$ | $6.7\times10^{-1}$ | $1.6$ |
| 2 | $2.1\times10^{-3}$ | $1.7\times10^{-2}$ | $6.5\times10^{-2}$ | $1.8\times10^{-1}$ | $1.4\times10^{-1}$ |
| 4 | $1.7\times10^{-4}$ | $2.1\times10^{-3}$ | $7.8\times10^{-3}$ | $1.7\times10^{-2}$ | $5.6\times10^{-2}$ |
| 8 | $2.4\times10^{-5}$ | $2.4\times10^{-4}$ | $9.2\times10^{-4}$ | $2.1\times10^{-3}$ | $6.9\times10^{-3}$ |
| 12 | $6.5\times10^{-6}$ | $7.6\times10^{-5}$ | $2.9\times10^{-4}$ | $7.3\times10^{-4}$ | $2.5\times10^{-3}$ |
| 20 | $1.7\times10^{-6}$ | $2.2\times10^{-5}$ | $8.6\times10^{-5}$ | $2.3\times10^{-4}$ | $9.9\times10^{-4}$ |
| 40 | $3.5\times10^{-7}$ | $5.5\times10^{-6}$ | $2.7\times10^{-5}$ | $9.0\times10^{-5}$ | $4.4\times10^{-4}$ |

$\beta=10^3$ gives the same figures within a factor 1.3; 200 nodes per decade gives the same
figures at the same $\sigma$ / interval (so the ratio, not the density, is the variable). The
error falls roughly as $(\sigma/\text{interval})^{-2.7}$: a cubic in $x$ cannot follow a
Gaussian tail, whose log falls as $-z^2/2$, once $S$ drops several orders of magnitude per
interval. In the current configuration, every bell-shaped event with a coefficient of variation
below about 90% misses $10^{-4}$ relative somewhere above $S=10^{-6}$, and below 28% it misses
$10^{-4}$ in the band $[10^{-4},10^{-2})$ where the thousand-year losses of a portfolio of a few
hundred events sit. Test 15 passes on the fixtures because they are small-$\alpha$ events, whose
survival in $\log x$ is smooth, and because in a portfolio the error at any $x$ is dominated by the
events in their bulk there, where it is small. A portfolio whose tail is one large concentrated
policy does not have that protection.

This is the finding to design around. It subsumes A3, and it decides what to do about the narrow
threshold (A1, I2): the question is not where the narrow route starts but how dense the mesh must be
around each bell-shaped event.

### What the test portfolios contain

The share of each portfolio's rate in the regime each step touches, from the rows' $(\alpha,\beta)$
and coefficient of variation (scratch count, 2026-10-25):

| Regime | Step | `medium_elt` | `small_alpha_elt` | `synthetic_elt(10^6)` |
|:---|:---|---:|---:|---:|
| $\alpha\le1$ | none | 53.8% | 91.1% | 53.5% |
| bell, CV $\ge0.28$ | none | 22.9% | 8.9% | 22.1% |
| bell, $0.023\le$ CV $<0.28$ | mesh | 12.7% (535 rows) | 0 | 12.7% (107,674 rows) |
| bell, CV $<0.023$ | narrow, unchanged | 0.2% (4 rows) | 0 | 0.2% |
| $\min(\alpha,\beta)\ge2\times10^5$ | kernel | 0 | 0 | 0 |
| $\alpha>100$, $\beta<5$ | tails | 0 | 0 | 0 |
| CV $<10^{-8}$ | atoms | 0 | 0 | 0 |

So at default settings only [mesh] moves any grid of the test portfolios, through the 12.7% of
rate in bell events of moderate width; the other corrections touch regimes the portfolios do not
hold. Test 15 bounds the size of that move: the current grids are within $1.1\times10^{-5}$
relative of the exact $\Lambda$ at every bucket edge of the medium golden and the benchmark
portfolio, and [mesh] moves them toward the exact value, so by at most that much.

A single shared mesh under the density envelope would carry 3,164 nodes on the medium golden (4.0
times the 784 base nodes) and 7,561 on the million-row portfolio (9.5 times), with 40% of the
node range above 500 per decade, and every small-$\alpha$ event, active across the whole range,
would pay for all of them. That design is rejected (Alternatives). With the requesting events
evaluated on their own fine curve and everyone else on the base curve, the extra evaluations are
about 21% of the base curve's on both portfolios, less the 12% of events that leave the base
curve: a net cost of roughly 10% on the curve build at $T=12$, proportional to $T$.

## Decisions needed (Steve)

**Ruled 2026-10-25 (Steve): all eight as recommended.** $T$ is a knob with default 12; the 15%
gate; the quantiles default chunk changes in rounding; the order below; a separate
`atom_by_resolution` category; a 1 GiB wave budget; a typed error on kernel failure; the cap at
5,000 per decade. The items below are decisions, not proposals. Spec tests are numbered in landing
order: 30 kernel, 31 atoms, 32 tails, 33 the mesh contract (the draft had the contract as 30).

1. **The accuracy contract.** Proposed: for every beta event, the curve at any loss is within
   $10^{-4}$ relative of the event's exact survival where $S_j\ge10^{-4}$, within $10^{-3}$ where
   $S_j\ge10^{-8}$, and within $3\times10^{-3}$ where $S_j\ge10^{-12}$; the rate exact; the
   portfolio criterion of test 15 unchanged. This is what $T=12$ nodes per standard deviation buys
   (the table). $T=20$ buys one more decade of $S$ at each tolerance for $1.7$ times the nodes in
   each bell event's range. The knob stays public.
2. **Cost ceiling for [mesh].** The fine curve costs about $10T$ evaluations per requesting
   event's range, shared among events at similar means: roughly 10% on the curve build of the
   benchmark portfolio at $T=12$ (the count above), 17% at $T=20$, and nothing on
   `small_alpha_elt`. The step is gated on the three benchmark portfolios' survival times staying
   within 15% of 2.3.2; say if that is the wrong ceiling. The grids of the medium golden and the
   benchmark portfolio move by at most $1.1\times10^{-5}$ relative at any edge, toward the exact
   beta model (test 15's current figure); the goldens are regenerated.
3. **The quantiles default chunk** ([schedule]) changes the quantiles method's results in
   rounding for inputs over the new chunk size, as 2.2.0 did for the survival method. Confirm.
4. **Order.** Recommended below: the four small corrections first, the mesh second, scheduling
   third, documentation and the article last. The review's order puts the mesh question (its A3)
   second as well.

## Steps

Each step is one commit by Claude, staged by path, with its CHANGELOG entry; Steve pushes. Each
step adds its own tests, built from the review's reproductions with the review's numbers, so that
the suite pins the exact failure.

### [kernel] patch: the continued fraction reports failure, and routing follows validity (I2)

- `incbeta.rs`: `betacf` returns `Option<f64>`, `None` when the 400 iterations are exhausted
  without $|\delta-1|\le\varepsilon$. `sf`, `cdf`, `sf_and_density_times_y` return NaN on `None`
  (so that no caller mistakes it for a value); a new `BetaFn::is_valid_regime()` returns false
  when $\min(\alpha,\beta)\ge$ `LARGE_PARAMETER_FROM` ($2\times10^5$, the quantiles method's
  Cornish-Fisher threshold of 2.3.2, where the fraction is already unreliable), so that the curve
  never asks the fraction to work where it fails.
- `curve.rs`: the narrow criterion becomes `(sigma < mu * local_interval || !bf.is_valid_regime())
  && alpha > 1 && beta > 1`; an event in the invalid regime that fails the bulk search (cannot
  happen for a near-normal law, kept as a guard) and any NaN from the kernel raise a typed
  `Error::IncompleteBetaDiverged { row }`. Spec section 8 forbids a silent NaN or negative weight;
  this is the typed failure it asks for.
- `batched.rs`: the inverse walk's iteration cap likewise returns an error, not the last iterate
  (the review's remark on the inverse routines; `plan-quantiles-large-parameters.md` moved the
  large-parameter rows off this path, so the error should be unreachable, and a test asserts so).
- Tests: the review's `BetaFn::new(1e9, 1e9).sf(0.5)` table; the public I2 case at 200,000 nodes
  per decade now gives $[0.5, 0.5]$ via the narrow route; a one-row sweep over
  $\min(\alpha,\beta)\in\{10^4,10^5,2\times10^5,10^6,10^9\}$ at three node densities agrees with
  SciPy within $10^{-10}$ relative in the bulk.
- `_meta.py` and `reference.py` mirror the routing.

### [atoms] patch: the narrow kernel beyond $10^{16}$ and the atom by resolution (A1, I1 step case)

- `curve.rs`: `NARROW_STEP_PARAMETER_SUM` is retired. The density kernel's two logarithms cancel
  to about $4\times10^{-15}\sqrt{\alpha+\beta}$ in $\log f$, so the kernel holds to a few units in $10^{-7}$ up
  to $\alpha+\beta=10^{16}$ (`NARROW_NORMAL_FROM`); above it the kernel is the standardized normal
  in $z=(y-\bar y)/\sigma_y$, whose skewness there is below $2\times10^{-8}\sqrt{(1-p)/p}$, with
  its integral and partial expectation in closed form through the engine's own `erf`
  (`normal.rs`). Quadrature is not needed for the normal; `NarrowKernel` gains the branch.
- `validate.rs`: a beta row whose standard deviation is below $2^{-48}\mu$ (the standardized
  coordinate no longer resolves the mean to one part in $2^{4}$) is an atom at $\mu$ under a new
  diagnostic category `atom_by_resolution`; it is deposited by the linear rule like any atom, so
  the A1 case gives $0.5/0.5$ for $\sigma=10^{-8}$ exactly as for $\sigma=0$, and the step case
  of I1 gets its overflow moment from the atom path. The spec's sentence on $10^{15}$ is replaced.
- Tests: the A1 table (three rows); the I1 step case, moment $0.15$; symmetric events with
  $\alpha+\beta\in\{10^{14},10^{16},10^{18},10^{24}\}$ on grids with the mean on an edge and
  displaced from it by a few ulps, against the exact symmetric split and against SciPy's normal
  approximation where SciPy's beta no longer evaluates.

### [overflow] patch: the curve's overflow moment from the interpolant (I1)

- `curve.rs`, `Curve::grid`: the per-event loop over `self.tails` is replaced by the identity
  $\sum_j\lambda_j\int_e^{x_{k_j}}(x-c)\,dF_j=(e-c)\tilde\Lambda(e)+\int_e^{x_m}\tilde\Lambda(x)\,dx$
  with $e$ the top edge and $c$ the top node (integration by parts; the tails' own part above
  $x_{k_j}$ is unchanged, by partial expectations as now). The integral of the cubic Hermite is
  closed-form per interval, $O(\text{nodes})$, and it covers the events whose endpoint survival
  underflowed, since their whole contribution is in $\tilde\Lambda$. The moment is then accurate
  to the interpolation accuracy, consistent with the grid it accompanies.
- Spec section 5A, Bucketing: the sentence on the overflow's first moment names the integral.
- Tests: the I1 case against $5.796480662\times10^{-4}$ within the interpolation tolerance; a
  windowed grid on the medium golden against the reference's figure; test 18 extended with a
  window that cuts through events whose endpoint survival underflows.
- `reference.py` keeps its per-event exact computation and so remains the oracle for this figure.

### [tails] patch: the ladder reaches the asymptotic regime (A2)

- `curve.rs`, `TailInterpolant::new`: the ladder continues below `min_distance` while the first
  correction of the endpoint expansion, $c\,t=(\alpha-1)\beta\,t/(\beta+1)$ with $t=d/M$, exceeds
  $10^{-8}$ at the last node, up to a cap of 200 nodes or an underflowing survival. Where $c$ is
  large the ladder step shrinks from $\sqrt2$ so that $c\,t$ changes by at most a quarter per
  step (the cubic in $u$ of $\log(1-c\,e^u)$ needs that to hold $10^{-5}$). Below the last node the
  power law applies only when the criterion held there; otherwise the one edge that can lie below
  it (edges are $h$ apart and the last node is within $h/4$ of $M$) is evaluated directly by `sf`.
- Tests: the A2 case ($\operatorname{Beta}(10^6,0.5)$, grid tail $0.6827$ within $10^{-6}$);
  $\operatorname{Beta}(10^6,1)$ against $-\operatorname{expm1}(10^6\log x)$; a sweep
  $\alpha\in\{10^2,10^4,10^6\}$, $\beta\in\{0.3,0.5,1,2,5\}$ with the bucket-edge check of test
  15 on every edge of the tail interval; the ladder's node count reported in the test log.

### [mesh] minor: node density set by the events, with the accuracy knob (A3, the measurement)

The base mesh stays geometric at `nodes_per_decade`, $n_d$, and carries every event as now except
the **requesting events**: bell-shaped rows ($\alpha_j,\beta_j>1$) whose standard deviation is
below $T$ base intervals at the mean, $\sigma_j<T\mu_j(10^{1/n_d}-1)$, a coefficient of variation
below $0.28$ at $T=12$ and 100 per decade. Each requesting event asks for a node spacing of
$\sigma_j/T$ at its mean, $d_j=T\ln10\cdot\mu_j/\sigma_j$ nodes per decade, on
$[\mu_j-3\sigma_j,\ \mu_j+7\sigma_j]\cap[x_0,M_j]$ (the upper end is where $S_j\approx10^{-12}$
for a near-normal law). The **fine mesh** is the base nodes plus the envelope of those requests,
$D(u)=\min\{D_{\max},\max(n_d,\max_j d_j\,\mathbf 1[u\in\text{range}_j])\}$ in $u=\ln x$,
computed by sorting the requests by their lower end and sweeping, $O(N\log N)$, and placed by
integrating $D$ with $x_0$ and $\max_jM_j$ exact as `geometric_nodes` has them. The requesting
events live entirely on a second curve $\tilde\Lambda_{\text{fine}}$ over the fine mesh; the base
curve never sees them, and the base events never see the fine nodes. $\tilde\Lambda$ at any $x$
is the sum of the two cubics. A requesting event is evaluated at the fine nodes between its
lower-tail skip (where $F_j<2^{-54}$, about $8.5\sigma_j$ below the mean) and its truncation
node, a window of about $10T$ nodes; the plateau below the skip is one difference, not a visit per
node.

The cap $D_{\max}$ (proposed $5{,}000$ per decade) is where the fine curve hands over to the
narrow route: an event whose request exceeds the cap, $\sigma_j<T\ln10\cdot\mu_j/D_{\max}$, a
coefficient of variation below $0.0055$ at $T=12$, goes narrow, so that no event is ever carried at
fewer than $T$ intervals per standard deviation. (Today the narrow route begins at $0.023$; the
band between is where the sweep's worst rows sit.) The kernel-validity routing of [kernel]
applies first.

- `curve.rs`: `Curve` gains the fine mesh and its curve and slope vectors; `build_curve` classifies
  the requesting events in the pre-pass that computes the envelope, folds a second pair of partial
  vectors per chunk, and deposits each event's plateau as a difference ($+\lambda_j$ at node 0,
  $-\lambda_j$ at the first evaluated node, prefix-summed at the fold); `interpolate` sums the two
  cubics; `TAIL_INTERVALS`, `sample_intervals`, the tails and the narrow events are untouched. The
  diagnostic samples the total of both curves.
- `config.rs`, `__init__.py`, `_meta.py`: `nodes_per_sigma` (default $T=12$) and
  `max_nodes_per_decade` (default $5{,}000$) beside `nodes_per_decade`; `CurveMeta` reports the
  node count and the number of events that requested density.
- `reference.py`: the same mesh in NumPy (the two backends share nodes to the rounding of `exp`,
  as now).
- Calibration before the defaults are fixed: the sweep above rerun under the mesh for
  $\alpha\in\{1.5,3,10,10^2,10^3,10^4\}$, coefficients of variation from $5\times10^{-4}$ to $1$,
  $T\in\{8,12,16,20\}$, on both backends, recorded in `benchmarks/report-<version>.md` as the
  table of this plan with the chosen $T$ marked. The contract of decision 1 is then spec test 33.
- Spec section 5A, Nodes and Narrow events: the mesh rule, the knob, the contract, and the narrow
  criterion in local terms; the sentence "the default density, 100, is the smallest whose
  exact-edge error is within $10^{-4}$ on every benchmark portfolio" gains "for the small-$\alpha$
  events those portfolios hold; bell-shaped events set their own density".
- Tests: test 33 (the contract on one-row portfolios); the A3 case; a block of 300 policies at a
  common limit with coefficients of variation from $10^{-3}$ to $10^{-1}$ (the regime between the
  old narrow threshold and the base density) against SciPy at every bucket edge; determinism
  (test 17) at the new default; the survival goldens regenerated where bell events move them,
  with the shift recorded in the CHANGELOG as 2.3.0 did.
- Gate: the three benchmark portfolios within 15% of their 2.3.2 survival times (decision 2);
  the medium golden's grid within $1.1\times10^{-5}$ relative of 2.3.2's at every edge where
  $\Lambda>10^{-12}\sum_j\lambda_j$, and closer to the exact $\Lambda$ there than 2.3.2 was.

### [ratios] patch: the domain edges (I3)

- `validate.rs`: the admissibility test in ratios, $(\sigma/\mu)^2\ge M/\mu-1$, and the
  degenerate test on $\sigma/\mu$ and $\mu/M$, so that no product of two monetary amounts is
  formed; the scale-$10^{160}$ and scale-$10^{-200}$ rows then classify as the unit-scale row
  does (test 10, scale invariance, extended to those scales).
- `batched.rs`, `Cumulants::of`: `sd = sqrt((a/s) * (b/s) / (s + 1))`; a test at $s=10^{150}$.
- Spec section 6 and `README.md`: the top half-bucket, an atom in $(\text{top node},\ \text{top
  edge}]$ goes to the top node and its displacement is not overflow; the raw first-moment
  statement gains that clause.
- Spec section 5A, Nodes: the tiny-$\alpha$ limit, $S$ computed as a complement below the switch
  point with absolute error of order $\varepsilon$, which matters only below the $10^{-12}$ floor;
  documented, not changed (the review rates it lowest).

### [schedule] minor: the quantiles chunk and the wave (P1)

- `config.rs`: the quantiles default chunk is a rule of the input, not of the thread count, so
  results stay a function of input and configuration: `chunk_rows = clamp(ceil(n_rows / 64),
  4096, 65536)`; 262,144 rows then make 64 chunks. The survival default stays 256.
- `curve.rs`, `run_chunks`: the wave is bounded by memory as well as by threads,
  `wave = min(2 * threads, max(1, budget / partial_bytes))` with the partial size supplied by the
  caller (two grids of $G$ doubles for the quantiles method, the span-limited histogram for the
  survival method) and a budget of 1 GiB by default, configurable; an empty pass returns before
  any pool is built; when `threads` is given the pool is built once per top-level call and passed
  down, not once per `run_chunks`.
- `README.md`: the sentence "grid length is irrelevant to throughput" is qualified: deposits
  are local, but each chunk's two partial grids are initialized and folded at the grid's length,
  so at $2^{24}$ buckets the chunk count and the wave are a memory and time term; the formula.
- Measurement: the review's table rerun (262,144 and $10^6$ rows, 1 to 32 threads, grids
  $2^{16}$ to $2^{24}$) with peak memory, recorded in `benchmarks/results.json` and the report.
- Tests: determinism across thread counts at the new default; the largest bin difference against
  the 65,536-chunk result within $10^{-14}$ of total rate (the review measured
  $4.7\times10^{-15}$).

### [tidy] patch: redundant work that is cheap to remove (P2)

Only the items whose change is local and measurable; each is kept if it measures faster on the
benchmark portfolios and dropped otherwise, with the figure in the execution log.

- `Curve::grid`: the ascending edges walk the intervals with a cursor and per-interval
  coefficients instead of a binary search per edge; the differences are written directly.
- `TailInterpolant::at_ascending` passes $\log d$ into `eval`.
- `batched.rs`: the $K$ normal scores and their Cornish-Fisher polynomial terms computed once per
  quantization, not per event.
- `quantize.rs`: the secondary first moment computed once; the mass and moment scans combined
  where the summation order is preserved (the determinism test guards it).
- Not taken here: `front / (1 - y)` for the density (an underflow fallback would be needed; the
  gain is one `exp` per node), tail-query caching (the public query path, rare), direct output
  filling in `quantize_into` (peak memory only).

### [benchmark] no bump: the equal-accuracy suite

- `benchmarks/run_accuracy.py`: the sweep of this plan (bell events by width and shape, endpoint
  tails by $\alpha$ and $\beta$, large parameters, narrow events, mixtures) on both methods and
  both backends, reporting time and the error bands together, so that a speed comparison is read
  at a stated accuracy. The $K=32$ comparison in the README keeps its historical place and gains
  the sentence that it is a cost comparison, not an efficiency one.
- `benchmarks/report-<version>.md`: the calibration table of [mesh], the scheduling table of
  [schedule], and the tidy figures.

### [article] no bump, outside this repository: three qualifications

`D:\Projects\aggregate\secondary-uncertainty\ELTQ-implementation.qmd`, by the published-sources
rule of this project's `CLAUDE.md` (the review itself is not a source; the mathematics here is
derived):

- The table of five problems and the small-$\alpha$ section: "$S$ exact at every bucket edge, down
  to $10^{-300}$" becomes "exact at the nodes; between them within the contract of spec test 33".
- The compound Poisson paragraph: $\Lambda(0)=\sum_j\lambda_j$ holds with zero losses kept as
  atoms, which the code does; a sentence says so.
- The parallel section: the compensated sums are the scalar totals and the chunk fold; the
  per-node vectors are ordinary additions. One clause.

## Files

| File | Steps |
|:---|:---|
| `crates/eltq-core/src/incbeta.rs` | kernel |
| `crates/eltq-core/src/curve.rs` | kernel, atoms, overflow, tails, mesh, schedule, tidy |
| `crates/eltq-core/src/validate.rs`, `diagnostics.rs`, `error.rs` | kernel, atoms, ratios |
| `crates/eltq-core/src/normal.rs` | atoms |
| `crates/eltq-core/src/batched.rs` | kernel, ratios, tidy |
| `crates/eltq-core/src/config.rs`, `quantize.rs` | mesh, schedule, tidy |
| `crates/eltq-python/`, `python/eltq/__init__.py`, `_meta.py` | kernel, atoms, mesh, schedule |
| `python/eltq/reference.py` | kernel, atoms, mesh |
| `tests/test_survival.py`, `test_contract.py`, `test_determinism.py`; `crates/eltq-core/tests/` | every step |
| `fixtures/` | mesh (regenerated survival goldens), new adversarial one-row fixtures |
| `secondary-uncertainty-mathematical-specification.md` | atoms, overflow, tails, mesh, ratios |
| `README.md`, `CHANGELOG.md` | every bump; ratios and schedule for the README |
| `benchmarks/run_accuracy.py`, `report-<version>.md`, `results.json` | mesh, schedule, benchmark |
| `D:\Projects\aggregate\secondary-uncertainty\ELTQ-implementation.qmd` | article |

## Acceptance

- Every reproduction in the review's appendix, rerun as a test with the review's numbers, gives
  the exact or SciPy answer within the step's stated tolerance, on both backends.
- Spec test 33 (the contract of decision 1) passes on one-row portfolios across the calibration
  grid; tests 15 to 24 still pass; test 17's determinism holds at the new defaults.
- The sweep table of this plan, rerun under [mesh] at the chosen $T$, is within the contract in
  every band.
- The three benchmark portfolios' survival times are within 15% of 2.3.2; the quantiles method at
  262,144 rows and 32 threads is at least 3 times faster than 2.3.2 at the default chunk; peak
  memory at $2^{24}$ buckets is within the stated budget.
- No test, fixture or benchmark reads anything outside `fixtures/` and the synthetic generators.

## Alternatives considered

- **Interpolating $\log\tilde\Lambda$ instead of $\tilde\Lambda$.** A single Gaussian tail is a
  quadratic in $\log$ and the cubic would be exact. But $\tilde\Lambda$ is a sum: where two events
  with different tail slopes cross inside an interval, $\log$ of the sum has a corner of width
  $1/\Delta k$ that the cubic misses by up to $\log2$, and at a truncation node where the dominant
  event drops out the log has a near-singularity, which is what the 2.0.0 trial found (spec
  section 5A, "a log-log interpolant was planned and rejected"). The mesh controls the error of
  the $x$-cubic without a coordinate change, and keeps every downstream routine as it is.
- **One shared mesh under the envelope**, every event on it. Counted on the test portfolios: 4.0
  times the base nodes on the medium golden and 9.5 times on the million-row portfolio, and the
  small-$\alpha$ events, active across the whole node range, would be evaluated at all of them.
  The curve build would slow several-fold. The two-curve form confines the extra nodes to the
  events that asked for them.
- **Raising the narrow threshold** to several intervals. The narrow route's cost is one quadrature
  per bucket edge across $\pm9\sigma$, so at $h=25$ and $\sigma=10^5$ it is $7\times10^4$ edges
  per event; it is the right route only where $\sigma$ is a few $h$. The mesh keeps such events
  on the shared curve at a cost of about $10T$ nodes each.
- **Per-event deep tails** through the existing tail mechanism, truncating bell events at
  $z\approx3$ as well as near $M$. Also one cubic per edge per event across $6\sigma$; the same
  cost objection, and a second interpolant coordinate to validate.
- **Local refinement from an a posteriori indicator** (compare two interpolants at each midpoint,
  add nodes where they disagree, rerun the chunk pass on the new nodes). Correct and general, but
  a second pass with node merging through `tails`, `group_start` and `narrow`, for an error that
  the a priori rule already predicts from $(\mu_j,\sigma_j)$.
- **Direct evaluation at requested thresholds** for the public curve (the review's "compute only
  what the caller requests"). The curve is built once and queried many times in the downstream
  use; the tail queries already evaluate directly. Not taken.
- **Keeping every event in `tails`** for the overflow moment. Correct but costs a per-event
  incomplete beta pair on every windowed grid; the integral of the interpolant is $O(\text{nodes})$
  and consistent with the grid.

## Not in this plan

- SIMD kernels, AVX-512 dispatch, profile-guided optimization, GPU: the review itself gives no
  measured basis; a profiling plan after the algorithmic steps, if the benchmark suite shows the
  scalar kernel still dominating.
- Caching or deduplication of public tail queries (20 ms per query on 10,000 identical narrow
  events): rare path; a note in the README.
- Direct output filling in `quantize_into` and sequential `.npy` writing.
- A linear-projection discretization preserving the first moment (the review's alternative to
  the nearest rule): the spec chose the nearest rule for its variance and records why.
- The multi-unit route, financial terms and the `aggregate` FFT, which the review did not audit.
- The tiny-$\alpha$ small-survival complement (documented limit, [ratios]).

## Open questions for review

- Is $T=12$ the right default, or $16$ or $20$ (decision 1)? The cost is linear in $T$ and only
  in bell events' ranges.
- Should `atom_by_resolution` be a separate diagnostic category or folded into `degenerate_atom`?
  Separate is proposed so that a consumer can tell a stated zero from a resolved one.
- The memory budget for the wave: 1 GiB default, or derived from the grid length alone?
- Whether [kernel] should fall back to the narrow kernel on a NaN rather than raise: raising is
  proposed because the routing makes the NaN unreachable, and a silent fallback is what the review
  objects to.

## Execution log

### [kernel] 2.3.3 (2026-10-25)

Landed as planned, with three divergences, each recorded here and in the CHANGELOG:

- **The iteration cap rose from 400 to 1,000.** Measured on the way: at the mean the fraction needs about $1.1\sqrt{\alpha+\beta}$ iterations when both parameters are large, 315 at $(2\times10^5,2\times10^5)$ but 443 at $(2\times10^5,6\times10^7)$, so the regime below the $2\times10^5$ threshold was not fully covered by 400 and the existing continuity test at the threshold failed once the fraction reported failure. At 1,000 the fraction converges for $\min(\alpha,\beta)$ to about $2\times10^6$ at every ratio.
- **The robust inverse walk accepts rather than raises at its cap, under a criterion.** The plan asked for an error. With the fraction converged, its value near the threshold is still only about $5\times10^{-10}$ relative, so the walk's bracket cannot close to $2\varepsilon$ and the cap is reached legitimately; the best point is accepted when its CDF residual is within $10^{-8}$ (`ROBUST_ACCEPT_CDF_ERROR`), otherwise the walk fails typed. This is the justified acceptance criterion the reviewer asked for.
- **Test 30's backend agreement is $10^{-8}$ of the rate below the threshold** (measured $1.5\times10^{-10}$ at $\min(\alpha,\beta)=10^5$, ratio 20, 1,000 nodes per decade: the fraction's own accuracy against SciPy), $10^{-10}$ at and above it.

Gates: `cargo test -p eltq-core --release` (82 lib tests and the golden, property and parser targets; the release profile because of the rustc 1.99.1 `libm` failure at the test profile), `uv sync --reinstall-package eltq` then `uv run pytest` (181 passed, 1 skipped as before). Grids unchanged at default settings on every fixture; the goldens stand.

### [atoms] 2.3.4 (2026-10-25)

Landed as planned, with these divergences:

- **The normal law's criterion is $\min(\alpha,\beta)\ge10^{16}$, not $\alpha+\beta>10^{16}$.** Working the cancellation through: the two logarithms of the density are each about $(\alpha+\beta)\,d$ with $d\le40\sigma_y=40\sqrt{\alpha\beta/(\alpha+\beta)^3}$, so each is about $40\sqrt{\alpha\beta/(\alpha+\beta)}\approx40\sqrt{\min(\alpha,\beta)}$, and the quadrature kernel's error in $\log f$ is $4\times10^{-15}\sqrt{\min(\alpha,\beta)}$, governed by the smaller parameter. The normal's error is its skewness $2/\sqrt{\min(\alpha,\beta)}$ times a polynomial in $z$ (a lopsided beta is a gamma in the short side, skewed at any $\alpha+\beta$). The two cross near $\min=10^{16}$ at about $10^{-6}$ each; the plan's $\alpha+\beta$ form would have handed a $\operatorname{Beta}(10^{16},100)$ to the normal with skewness $0.2$. Measured: the kernels agree to $5\times10^{-6}$ on mass and moment at the handover (Rust unit test).
- **SciPy fails there too.** Its incomplete beta is off by $1.4\times10^{-5}$ at $z=\tfrac12$ and $6\times10^{-4}$ at $z=5$ for $\alpha=\beta=10^{24}$, and returns NaN at the mean of $\operatorname{Beta}(10^{16}-2,3\times10^{17})$. The reference takes the normal law from $10^{16}$ as well and guards a non-finite SciPy value point by point with the normal; the backends then agree to $4\times10^{-9}$ at the handover itself (where they can recover $\alpha$ on opposite sides of $10^{16}$) and to $10^{-9}$ elsewhere; test 31 asks $10^{-8}$.
- **The resolution of the mean bounds the split.** At $\min(\alpha,\beta)=10^{24}$ the mean's share is recovered from $(M,\mu,\sigma)$ to a few ulps, $10^{-4}$ of a standard deviation, so a mean on a bucket edge splits $0.5005/0.4995$ rather than half and half on both backends alike. Test 31's tolerance carries that term; it is what the $2^{-48}$ atom threshold caps at a few percent.
- `Diagnostics` holds a fixed array per category; it grew to ten with the new category, and `NARROW_STEP_PARAMETER_SUM` left the crate's re-exports.

Gates: `cargo test -p eltq-core --release` (84 lib tests, the other targets as before); `uv run pytest` after the rebuild (182 passed, 1 skipped). Grids unchanged at default settings on every fixture (no fixture row has a coefficient of variation below $2.3\%$ at $\min(\alpha,\beta)\ge10^{16}$ or below $2^{-48}$); the goldens stand.

### [overflow] 2.3.5 (2026-10-25)

Landed as planned. `Curve::integral_above(e)` integrates the cubic Hermite in closed form, the partial interval by the antiderivatives of the four basis functions and each whole interval by $\Delta x\,(y_0+y_1)/2+\Delta x^2(m_0-m_1)/12$, with a compensated sum; the per-event loop over `self.tails` is gone from `Curve::grid`, and `tail_moment` stays for the tails' own overflow. Measured: on the review's $\operatorname{Beta}(2,1000)$ case the moment is within $10^{-4}$ relative of the exact $5.796480662\times10^{-4}$ (it was 0) and the integral within $10^{-6}$ of a 200,000-step Riemann sum of the interpolant; on the medium fixture windowed at a third of its largest exposure the engine and the reference agree within $10^{-4}$ relative. No divergence from the plan. The reference keeps its exact per-event sum as the oracle.

Gates: `cargo test -p eltq-core --release` (85 lib tests); `uv run pytest` after the rebuild (183 passed, 1 skipped). Grids unchanged everywhere; the goldens stand.

### [tails] 2.3.6 (2026-10-26)

Landed with two divergences, both from measuring before building:

- **The ladder is not extended below $x_0/2$; the one edge below the last node is evaluated exactly instead.** The plan offered both. Edges are $h$ apart and the last node is within $h/4$ of $M_j$, so at most one edge lies below it, and the walk reaches it only when $\lambda_jS_j$ there is above the rate floor, which for $\beta_j$ above a few it never is. Measured on the reference's interpolant with the power law switched off: the pure power law's error at half the last distance is $c_jt_L/2$ to the digit ($1.7\times10^{-3}$ at $\alpha=10^6$, $\beta=\tfrac12$, $t_L=10^{-8}$), so the gate is $c_jt_L\le10^{-6}$ (`TAIL_POWER_LAW_LIMIT`) and the interpolant keeps a copy of its `BetaFn` for the direct evaluation.
- **The ladder refines by $2^{1/4}$ across $c_jt\in[10^{-2},10^2]$ when $c_j>1$**, rather than a step set by $c_j$ alone. Measured Hermite error across that zone with $\sqrt2$ steps: $6\times10^{-6}$ at $\beta=0.3$ rising to $3\times10^{-4}$ at $\beta=5$, the same at every $\alpha$ from $10^2$ to $10^6$; below the zone $3\times10^{-7}$. With the finer step $2\times10^{-5}$ at $\beta=5$ (Rust unit test, 5e-5 asserted). The nodes are $d_0\,2^{-k/4}$ for an integer sequence $k$ shared by both backends, so the unrefined ladder's nodes are bitwise the old ones.

Also found: at distances below $10^{-12}M_j$ the formulation's rounding of $1-y$ costs up to $10^{-2}$ relative in the survival on both backends (SciPy included); no grid query goes there ($h/4$ over $M_j$ is $6\times10^{-9}$ at $h=25$, $M_j=10^9$), and it is the same residual spec section 5A records for the fraction above the switch point.

The survival goldens moved: the medium golden by up to $3\times10^{-6}$ of the total rate in the top bucket below the exposed value of its bell-shaped events (the power law was extrapolated from a quarter bucket away, $c_jt_L$ up to a few percent for $\alpha_j$ in the thousands and $M_j$ in the tens of thousands at $h=5{,}000$), the small golden by $4\times10^{-8}$. Checked on the medium fixture's 4,347 beta rows alone (atoms deposit linearly and cannot be compared to a step): the new tail sums are within $4.9\times10^{-5}$ of the exact exceedance at every one of the 999 bucket edges, mean $1.1\times10^{-6}$. Regenerated with `fixtures/generate_fixtures.py`; the quantiles goldens' metadata gained the `narrow_events` key they lacked since 2.3.0 and the multi-unit golden moved by $4\times10^{-16}$ relative, rounding of the regeneration, both committed with the step.

Gates: `cargo test -p eltq-core --release` (86 lib tests, goldens regenerated); `uv run pytest` after the rebuild (184 passed, 1 skipped).

### [mesh] 2.4.0 (2026-10-26): built, tested, held at the gate

The two-curve design landed as planned: `Route` per row decided in a pre-pass, `refine_mesh` splitting each base interval into equal pieces in $\ln x$, `deposit_event` shared by both meshes with the plateau as a difference and the upper skip at $2^{-60}$, a second set of partial vectors folded alongside the first, `cubic_at` and `cubic_integral_above` summing the two curves. Divergences:

- **No `max_nodes_per_decade`.** The knob failed validation whenever a caller raised `nodes_per_decade` above it (test 30 at 200,000) and was redundant: the narrow route takes over at one base interval, where a requesting event's density would reach $T$ times the base, so the pieces per base interval are capped at $T$ instead. The ruling's cap of 5,000 was moot at the defaults ($12\times100=1{,}200$).
- **The narrow route keeps its one-base-interval criterion** beside the kernel-validity one, rather than handing the band from $0.55\%$ to $2.3\%$ to the fine curve as drafted: there the quadrature is exact ($10^{-12}$) where the fine curve would be $10^{-5}$, and test 22 asks $10^{-8}$ of it.
- **The narrow route requires both parameters at least 2** (`NARROW_MIN_PARAMETER`). Measured: its 8-point rule is off by $1.2\times10^{-4}$ to $4.8\times10^{-4}$ relative for $\beta$ from $1.02$ to $1.6$ at $\alpha=100$, $10^{-13}$ from $\beta=2$ (the density's derivative is infinite at the endpoint below 2). Such an event stays on a curve: its mass sits against the endpoint, inside the tail interval, and the ladder carries it to $10^{-5}$. Found by test 33's sweep.
- **A grid bug of the two-curve design, caught by the sweep.** `Curve::grid` clipped the edge values from `self.curve[0]`, the base curve's first value, so a requesting event's plateau at the lowest node was clipped to zero and its mass landed in bucket 0 ($10^{-3}$ of the rate on $\operatorname{Beta}(100,2)$). The clip now starts from the full curve's value at the lowest node, on both backends.
- **The contract at $T=12$ has one exception**, $\operatorname{Beta}(10,7.3)$ at a coefficient of variation of $0.2$: $1.5\times10^{-4}$ in the band $[10^{-4},10^{-2})$, a skewed small-parameter bell whose upper tail is steeper than a normal's; recorded in the spec and the report rather than special-cased.

Calibration (`benchmarks/run_accuracy.py`, in `benchmarks/report-2.4.0.md`): at $T=12$ the bands are $8.6\times10^{-6}$, $1.5\times10^{-4}$, $3.9\times10^{-5}$, $1.0\times10^{-4}$, $2.1\times10^{-4}$ against the base mesh alone at $1.1\times10^{-2}$, $0.13$, $0.23$, $0.31$, $1.4$. On the medium fixture's beta rows the tail sums are within $2.0\times10^{-5}$ of the exact exceedance at every edge ($4.9\times10^{-5}$ at 2.3.6); the medium golden moved by at most $6.9\times10^{-7}$ of the total rate.

**The speed gate (decision 2, 15%) fails on one of the three portfolios.** Survival method, same machine, medians of three:

| portfolio | 2.3.6 | 2.4.0 | change |
|:---|---:|---:|---:|
| mixture, $10^6$ rows, $h=5{,}000$ | 1.049 s | 1.897 s | $+81\%$ |
| small-alpha, $10^6$ rows, $h=100$ | 2.779 s | 2.684 s | $-3\%$ |
| large exposure, $10^5$ rows, $h=25$, $2^{22}$ buckets | 1.415 s | 1.310 s | $-7\%$ |

The mixture's cost is the fine evaluations: 105,779 of its events are requesting (10.6%), about 7 microseconds each, which is roughly 130 incomplete-beta evaluations; with the fine mesh off (`nodes_per_sigma=1`) the build takes 1.00 s. Half of those events have coefficients of variation between 0.14 and 0.28 and ask for two pieces per base interval (about 40 extra nodes each); the quarter below 0.07 ask for five to twelve (about 100 each). The cost is intrinsic to the contract: it buys the accuracy exactly where the shared cubic was wrong, in proportion to how wrong it was, and nowhere else. Steve's ruling, 2026-10-26: accept the cost ("unlikely to feel it in reality"); the 15% ceiling of decision 2 is replaced for this step by the measured table above, and `nodes_per_sigma=1` remains the route back to the 2.3.6 speed. Committed as 2.4.0.

Gates: `cargo test -p eltq-core --release` (88 lib tests, goldens regenerated); `uv run pytest` after the rebuild (186 passed, 1 skipped).

### [ratios] 2.4.1 (2026-10-26)

Landed as planned, with these notes:

- **The row plan carries the standard deviation, not the variance** (`RowPlan::sd`, `Classified::sd_eff`, and `beta_params(M, mu, sigma)` on both backends). The plan asked only for the test in ratios, but a classification that passes a row at $10^{160}$ then hands `beta_params` a variance that is `inf`, and at $10^{-200}$ one that is `0`; the parameters had to come from the same ratios, $\kappa=\bigl((M-\mu)/\mu\bigr)/(\sigma/\mu)^2-1$, and the cleanest way to keep every product of two monetary amounts out of the pipeline was to never form the variance. The Bernoulli plan's sd is $\mu\sqrt{M/\mu-1}=\sqrt{\mu(M-\mu)}$ in the same form. The typed error still reports $\sigma^2$ and $\mu(M-\mu)$ in monetary units, formed on the error path only.
- **The multi-unit route takes the same ratio test** (two sites in `multiunit.rs`, including the clamp's $\sigma$), so the spec's section 4A claim that a unit row is classified exactly as a one-row row is true at every scale. Not in the plan's file list; three lines each.
- **Rounding moved one test off a knife edge.** Test 30 placed a row exactly at `LARGE_PARAMETER_FROM` by reconstructing $(\mu,\sigma)$ from $(2\times10^5, 4\times10^6)$; the new $\kappa$ rounds that $\alpha$ to $199{,}999.99999999994$, below the line, where the old form gave $200{,}000$ exactly. Neither is more right: the parameters come back from the moments to a few ulps either way. The threshold case now sits $10^{-12}$ above the line (spec test 30 says so). No golden moved beyond its $10^{-9}$ tolerance.
- **Test 10 compares diagnostics and valid-row counts as well as grids** at the four factors, and adds the review's row at the three scales on both methods.
- The cumulants' sd and the two documentation items (top half-bucket, tiny-$\alpha$ complement) as planned.

Gates: `cargo test -p eltq-core --release` (91 lib tests); `uv run pytest` after the rebuild (186 passed, 1 skipped).

### [schedule] 2.4.2 (2026-10-26)

Landed with two divergences from the plan's rule, both forced by the measurement the plan asked for (`benchmarks/run_schedule.py`, `benchmarks/report-2.4.2.md`):

- **The chunk rule has a grid-length term**, $\min(\max(\lceil N/64\rceil,\lceil n/64\rceil,4096),65536)$, not $\lceil N/64\rceil$ alone. With the row term alone, 262,144 rows at $2^{24}$ buckets took 6.4 s against 3.0 s at the fixed chunk on one thread and 4.4 s against 1.6 s on 32: 64 chunks each allocate, zero and fold two 128 MiB vectors. The review's recommendation was a rule joint in row count and grid length; the plan's draft dropped the grid. A divisor of 256 (chunk at least $n/256$) still lost 23% at $2^{22}$ on a million rows; 64 gives the old chunk from $2^{22}$ and 16 chunks at $2^{20}$, where the gain survives (0.67 s to 0.24 s at 262,144 rows on 32 threads).
- **The quantiles fold is parallel across the grid.** The fold of a wave's partials was serial, and with 32 threads on the kernel it dominated at $2^{20}$ and above (61 chunks of 16 MiB vectors). `run_waves` hands each wave to the caller; the quantiles pass adds the partials' two vectors over disjoint blocks of the grid in parallel, each element in chunk order, so the result is bitwise the serial fold's (the determinism tests hold). The scalar totals fold serially as before.

Other notes:

- The memory budget is a `Config` field (`wave_memory_bytes`, 1 GiB), not a Python keyword: the Python API's scheduling knobs stay `threads` and `chunk_rows`; a caller who needs the budget can ask. At the budget, the peak working set on 32 threads at $2^{20}$ buckets is about 1.1 GiB (the wave of 64 partials of 16 MiB), where the fixed chunk's four partials used 140 MiB; at $2^{24}$ the wave is four partials and the peak is the same as before.
- The survival method's column of the sweep is slow at large grids for a reason outside this step: at 65,536 buckets and above every tail is walked across its edges (the review's 4.2 s single-threaded on 262,144 rows was at about 1,000 buckets; this sweep's 12.5 s at the same rows is at 65,536, 8.6 s with the fine mesh off). The survival cells at $2^{24}$ below 16 threads were skipped, one run taking minutes; `dev/done/plan-survival-tail-cost.md` is where that cost is treated.
- The two measured constants behind the grid term, for the record: about 3.4 ns per grid element per chunk for the partial's allocation, zeroing and fold against about 9.7 microseconds per row for the kernel at $K=32$, single-threaded on this machine.

Gates: `cargo test -p eltq-core --release` (94 lib tests); `uv run pytest` after the rebuild (187 passed, 1 skipped).
