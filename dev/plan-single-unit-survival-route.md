# Plan: single-unit-survival-route, bucketing each event's beta by its survival function

Date: 2026-10-23
Status: draft, not started.

## Goal

Replace the K-point midpoint-quantile rule as the default way `quantize` builds the `secondary`
grid. The new route evaluates the portfolio exceedance rate
$\Lambda(x)=\sum_j\lambda_j\Pr(X_j>x)$ exactly at log-spaced loss nodes, interpolates it
monotonically, and buckets it onto the even grid by differences at half-bucket edges. The
K-point rule stays available as `method="quantiles"`. Alongside: a named overflow policy that
replaces the requirement that the grid reach the largest $M$, and a public exceedance-curve
object from which the grid (at any $h$ and $n$) and the OEP points both come.

This plan covers one-row events only (`quantize` and its wrappers). The multi-unit route is
untouched; see "Not in this plan".

## Background for a reader with no context

`eltq` converts an event loss table (ELT) into rate-weighted distributions on the grid
$g_i=ih$. Each nondegenerate row $j$ has rate $\lambda_j$, mean $\mu_j$, standard deviations
$\sigma_{I,j}$ and $\sigma_{C,j}$, and exposed value $M_j$; its conditional loss is
$X_j=M_jB_j$ with $B_j$ beta, moment-matched to $\mu_j$ and $\sigma_j=\sigma_{I,j}+\sigma_{C,j}$
(spec §4). Today the `secondary` grid comes from spec §5: event $j$ is replaced by $K$ equally
weighted points $M_jQ_j((k+\tfrac12)/K)$, deposited by the linear rule of §6. The engine is
`crates/eltq-core/src/quantize.rs` (row classification, chunked deposit, in-order fold), with
the quantile kernel in `batched.rs`, the incomplete beta in `incbeta.rs`, metadata in
`diagnostics.rs`, bindings in `crates/eltq-python/src/lib.rs`, the Python API in
`python/eltq/__init__.py`, and the SciPy oracle in `python/eltq/reference.py`. The acceptance
suite in `tests/` runs every test on both backends through the `impl` fixture in
`tests/conftest.py`.

### The defect in the K-point rule

The fraction of event $j$'s $K$ points above a loss $x$ is the number of $k$ with
$(k+\tfrac12)/K>F_j(x)$, divided by $K$. That is $K\,S_j(x)$ rounded to the nearest integer and
divided by $K$, where $S_j=1-F_j$. So the rule sets every event's survival function to zero
wherever it is below $1/(2K)$: at $K=32$ every event is cut off above its 98.4% quantile.

For a beta with small $\alpha$ (roughly $\alpha\approx(\mu/\sigma)^2$ when $\mu\ll M$), most of
the probability sits near zero and much of the mean sits in an upper tail of small probability,
exactly where the rule puts nothing. Synthetic examples with $\mu/M=10^{-4}$:

| $\alpha$ | $\beta$ | mean error, K = 32 | K = 256 | K = 2048 |
|---|---|---|---|---|
| 0.003 | 30 | -96.9% | -27.8% | -4.0% |
| 0.01 | 100 | -56.7% | -9.7% | -1.3% |
| 0.03 | 300 | -23.9% | -3.5% | -0.5% |
| 0.1 | 1,000 | -8.3% | -1.1% | -0.1% |
| 1 | 10,000 | -1.1% | -0.1% | 0.0% |

The cut falls only in proportion to $1/K$, so more points help slowly, and the tails beyond
$1/(2K)$ are always gone. Gross ELTs make it worse: deductibles shrink $\mu$ and leave $M$
unchanged, so $\alpha$ falls.

The defect surfaced in a downstream tie-out of two real ELTs (ground-up and gross) against a
vendor's OEP and AEP. Data and figures are kept outside this repository. The exact beta model
with $\sigma=\sigma_I+\sigma_C$ tied the vendor's OEP to within 0.1% at every return period from
5 years, in every peril and view, which confirms the model `eltq` implements. The K = 32 grid
fell short by tens of percent in the perils whose events have small $\alpha$ (severe convective
storm, gross winter storm), and its `secondary_mean_error` showed the same loss of expected
value. The same calculation done by the route in this plan tied the vendor's AEP in most blocks.

## The survival route

**Exceedance curve.** Split the valid rows as now (`validate.rs::classify_row`): atoms
(degenerate rows, policy atoms) and betas. For the betas, compute
$$
\Lambda(x)=\sum_{j\ \text{beta}}\lambda_j S_j(x),\qquad
\Lambda'(x)=-\sum_{j\ \text{beta}}\frac{\lambda_j}{M_j}f_j\!\left(\frac{x}{M_j}\right),
$$
exactly, at nodes $x_0<x_1<\dots<x_m$ spaced geometrically from a lower loss $x_0$ to
$\max_jM_j$, at `nodes_per_decade` per decade. $S_j$ comes from the incomplete beta and $f_j$
from the beta density; both share the factor $x^\alpha(1-x)^\beta/B(\alpha,\beta)$ already
computed in `incbeta.rs`, so the slope costs little beyond the survival value. Nodes at or above
$M_j$ contribute exactly zero for event $j$ and are skipped.

**Interpolation.** Between nodes, $\log\Lambda$ is a monotone cubic Hermite interpolant in
$\log x$ using the exact slopes $d\log\Lambda/d\log x=x\Lambda'(x)/\Lambda(x)$, limited by the
Fritsch-Carlson conditions so that it never increases. Above the last node with $\Lambda>0$,
$\Lambda=0$. On the interval ending at $\max_jM_j$ (where $\Lambda=0$) interpolate linearly in
$\Lambda$. The interpolation error is measured on every run: exact $\Lambda$ at 64 log-midpoints
spread over the node range, against the interpolant, reported as the largest relative error
where $\Lambda>10^{-12}\sum\lambda_j$.

**Bucketing (the nearest rule).** With edges $e_i=(i+\tfrac12)h$, the beta part of the grid is
$$
p_0=\textstyle\sum_{\text{beta}}\lambda_j-\Lambda(e_0),\qquad
p_i=\Lambda(e_{i-1})-\Lambda(e_i)\quad(1\le i\le n-2),\qquad
p_{n-1}=\Lambda(e_{n-2}),
$$
so the top node holds all the rate above $e_{n-2}$ (the overflow policy below). Atoms are added
by the existing linear deposit, as today. The lowest node is $x_0=e_0=h/2$, the smallest loss
the grid uses.

Properties: every $p_i\ge0$ because the interpolant is monotone; the total rate is exact because
the sum telescopes; the first moment differs from the analytic expected loss by the
bucketing error, of order $h^2$ times the density's slope, plus any rate moved down into the
top node. Both are reported. This is the rule the multi-unit route already uses for beta units
("cdf differences at half-bucket edges"), and the one `aggregate` uses for continuous
severities, so all three agree.

**The curve is the primary object.** Its non-uniform nodes cover everything up to
$\max_jM_j$ for about $150\times\log_{10}(\max M/x_0)$ points, whatever $h$ is; the even grid is
made only when needed, at whatever $h$ and $n$ the caller wants, and the OEP comes straight
from the curve, $\mathrm{OEP}(x)=1-e^{-\Lambda(x)-\Lambda_{\text{atoms}}(x)}$. Because
$\Lambda$ adds over events, a later multi-unit or portfolio-combining route can add its events'
contributions on the same nodes.

**Parallelism and determinism.** Rows are processed in the existing fixed chunks; each chunk
produces partial $\Lambda$ and $\Lambda'$ vectors of length $m+1$, folded in chunk order with
Neumaier sums, so results stay bitwise identical for any thread count. The partials are about a
thousand doubles, not the grid length, so peak memory no longer scales with $n$. Bucketing is
elementwise over edges and parallelizes without a reduction.

### Alternatives considered

- **More quantile points.** The cut moves only as $1/K$ (table above), and every tail beyond it
  is still lost.
- **Tail-graded quantile cells with conditional-mean points.** This keeps the linear deposit and
  can make each event's mean exact, but to resolve survival down to $10^{-12}$ with a cell
  ratio of 1.1 takes about 260 cells per event plus partial expectations, no cheaper than the
  survival route and more machinery.
- **Linear (first-moment matching) bucketing of the continuous beta.** Through the stop-loss
  transform $\Pi(x)=\sum_j\lambda_j\mathsf{P}(X_j-x)^+$ the bucket weights are second
  differences of $\Pi$, and the grid mean is exact by telescoping. Nonnegativity then needs a
  convexity-preserving interpolant of $\Pi$, and each evaluation needs a second incomplete beta
  at $\alpha+1$. Rejected for a gain of order $10^{-5}$ in the mean; it remains the route if an
  exact mean is ever required.
- **Exact evaluation at every bucket edge.** $N\times n$ evaluations, infeasible at production
  $n$; used only as the test oracle on small fixtures.
- **Per-event routing** (survival only for events where the K rule fails, judged by each event's
  K-point mean error). On the real ELTs above, a 0.1% tolerance routed 90% to 100% of the events
  in the affected perils, so routing saves nothing there.

## Cost: the survival route is slower than K = 32

This is a speed-for-correctness trade, and it needs to be measured before the default changes.

- The K = 32 route costs 32 quantile inversions per event, about 2 to 4 incomplete-beta
  evaluations each (`benchmarks/report-1.4.0.md`: 6.55M quantiles per second, 1M rows in about
  4.3 s on the 4-core laptop).
- The survival route costs one incomplete beta (and a density) per event per active node. On the
  real ELTs, at 150 nodes per decade over 7 to 8 decades, events had 560 to 880 active nodes
  after skipping nodes where the survival is 0 or 1 to double precision. That is 5 to 14 times
  the K = 32 work.
- Exact slopes make the interpolant fourth order rather than the third order of a PCHIP with
  estimated slopes, which should allow about 50 nodes per decade at the same accuracy, bringing
  the cost to 2 to 5 times K = 32. That is an expectation, not a measurement: [benchmark]
  settles it, and it is a gate.

## Decisions (Steve, 2026-10-23)

- The survival route becomes the default method; `method="quantiles"` keeps the K-point rule.
- The requirement that the grid reach $\max_jM_j$ is replaced by a named overflow policy,
  `overflow="top_bucket"` by default, reporting the rate placed in the top node.
  `overflow="error"` keeps today's `GridOverflowError`. With `grid_length=None` the grid still
  covers $\max_jM_j$, so nothing overflows.
- Bucketing uses the nearest rule (differences at half-bucket edges), not the linear rule.

## Steps

Each step is one commit; each version bump is its own commit with its CHANGELOG entry.

### [spec] no bump: amend the mathematical specification

`secondary-uncertainty-mathematical-specification.md`:

- New section after §5, "Survival-function discretization": the definitions above for
  $\Lambda$, the nodes, the monotone interpolant, the bucket weights and the atom deposit, with
  the exactness statement (rate exact, mean error reported, interpolation error reported).
- §5 states that the K-point rule is the `quantiles` method and records the $1/(2K)$ cut-off.
- §6: overflow policies `top_bucket` and `error`, named and reported; resolves §11 item 8.
- §6 last paragraph and §7: the outputs for each method; metadata adds the method, node count,
  nodes per decade, interpolation error, overflow policy and rate above the grid.
- §9: tests 15 to 19 below.
- §11 item 5: the default method is `survival`.

### [reference] 1.13.0: the survival route in the SciPy reference

- `python/eltq/reference.py`: `quantize_reference(..., method="quantiles", nodes_per_decade=150,
  overflow="top_bucket")`. The survival branch computes $\Lambda$ and $\Lambda'$ at the nodes
  with `scipy.special.betaincc` and the beta density, builds the same limited Hermite
  interpolant (written out, not `PchipInterpolator`, so that both backends compute the same
  function), buckets, deposits atoms, and fills the new metadata. A private
  `exact_edges=True` switch evaluates $\Lambda$ at every edge with no interpolation, for tests on
  small fixtures only.
- `python/eltq/__init__.py`: `quantize_arrays` and its wrappers (`quantize_csv`,
  `quantize_into`, `quantize_to_npy`, `quantize_dataframe`) take `method`, `nodes_per_decade` and
  `overflow`. On the Rust backend, `method="survival"` dispatches to the reference until
  [rust] lands, as `_via_reference` does for the multi-unit route.
- `python/eltq/_meta.py`: `Meta` gains `method`, `nodes`, `nodes_per_decade`,
  `interpolation_error`, `overflow`, `rate_above_grid`, with defaults so older call sites keep
  working. For the survival method `quadrature_points` is 0, `quadrature_secondary_mean` is the
  first moment of the bucketed secondary measure in rate units, and `secondary_mean_error` is
  that less `analytic_aal`, which for the quantile method matches today's definition.
- The default method stays `quantiles` in this step.
- Tests (below) on the reference backend.

### [rust] 1.14.0: the survival route in the Rust engine

- `incbeta.rs`: `BetaFn::sf(x)` returning the survival directly in each branch of the
  continued fraction (no `1 - cdf` cancellation in the upper tail), and a combined
  `sf_and_pdf(x)` sharing the front factor.
- New `curve.rs`: `survival_curve(&EltInput, &Config) -> Curve` (nodes, $\Lambda$, slopes, atom
  losses and rates, interpolation error) using the existing classification and chunked fold;
  `Curve::exceedance(x)` (the interpolant plus atoms); `bucket_curve(&Curve, h, n, overflow)`
  returning the secondary grid and the rate above it.
- `quantize.rs`: `Config` gains `method` (enum `Quantiles`, `Survival`), `nodes_per_decade` and
  `overflow` (enum `TopBucket`, `Error`). The quantile path takes `overflow` too: `TopBucket`
  lets points beyond the top node land in it through `deposit`'s existing top-node branch, and
  reports their rate. `Meta` in `diagnostics.rs` gains the new fields.
- `crates/eltq-python/src/lib.rs`: the new keyword arguments and metadata keys; the Python
  dispatch to the reference for `survival` is removed.
- Rust unit tests: `sf` against statrs across the regimes in `incbeta.rs`, including survival
  values down to $10^{-300}$; the interpolant is nonincreasing on random curves; thread and chunk
  invariance of the curve, bitwise.

### [benchmark] no bump: measure, set the node density, and gate the default

- `benchmarks/run_end_to_end.py` gains `--method` and `--nodes-per-decade`. Run the 1M-row
  suite for `quantiles` at K = 32 and `survival` at 25, 50, 100 and 150 nodes per decade, with
  the interpolation error and the time split into curve build and bucketing.
- Add a small-$\alpha$ synthetic case to the performance suite (spec §10 already asks for
  strongly skewed shapes): $\alpha\in[0.003,1]$, $M/\mu\in[10^3,10^5]$.
- Set the default `nodes_per_decade` to the smallest density whose interpolation error is at
  most $10^{-4}$ on every benchmark case.
- Write `benchmarks/report-<version>.md`.
- **Gate:** if `survival` at the default density takes more than 5 times K = 32 on the 1M-row
  case, stop and talk before [default]. Speedups to look at then, in order: skip the nodes below
  each event's effective support (bisection for the last node with $S_j<1-2^{-53}$, then a
  plateau sum), the batched kernel layout in `batched.rs`, and a lower density with an adaptive
  refinement wherever the midpoint check fails.

### [curve] 1.15.0: public exceedance curve

```python
curve = eltq.exceedance_curve(rate, max_loss, mean_loss, sd_indep, sd_corr, *, min_loss,
                              nodes_per_decade=None, mean_is_loss_rate=False,
                              variance_policy="clamp", clamp_factor=0.999,
                              threads=None, event_ids=None)
curve.loss, curve.rate_above, curve.log_slope   # beta part at the nodes
curve.atom_loss, curve.atom_rate                 # degenerate rows
curve.meta                                       # row diagnostics, node count, interpolation error
curve.exceedance(x)                              # Lambda(x), atoms included
curve.oep_pml(return_periods)                    # loss where Lambda = -log(1 - 1/T)
curve.grid(bucket_size, grid_length, *, normalize=False, overflow="top_bucket")
```

`quantize_arrays(method="survival")` becomes `exceedance_curve(min_loss=h/2)` followed by
`curve.grid(h, n)` plus the raw grid. Both backends. A curve built once serves any number of
grids, for example a coarse one for a quick look and a fine one for the FFT.

### [default] 1.16.0: survival becomes the default

- `method` defaults to `survival` in Rust, the bindings, the reference and every Python
  wrapper.
- Tests that depend on the K-point rule (spec tests 4, 5 and 7, and the existing golden
  fixtures) pass `method="quantiles"` explicitly; new survival goldens are generated by
  `fixtures/generate_fixtures.py`.
- `README.md`, the module docstring in `python/eltq/__init__.py` and the description at the top
  of `CLAUDE.md` describe the secondary grid by the survival route, with the K-point rule as an
  option.

### [tie-out] outside this repository

The downstream tie-out project installs this checkout editable, replaces its own exact-severity
code with `eltq`'s survival route, and checks that the OEP and AEP points agree with its exact
calculation within $10^{-3}$ at return periods from 5 years, in every block. The PyPI release
follows when Steve pushes the tag.

## Files

| file | steps |
|---|---|
| `secondary-uncertainty-mathematical-specification.md` | [spec] |
| `python/eltq/reference.py`, `python/eltq/_meta.py`, `python/eltq/__init__.py` | [reference], [curve], [default] |
| `crates/eltq-core/src/incbeta.rs`, `curve.rs` (new), `quantize.rs`, `config.rs`, `diagnostics.rs`, `lib.rs` | [rust], [curve], [default] |
| `crates/eltq-python/src/lib.rs` | [rust], [curve] |
| `tests/test_acceptance.py`, `tests/test_contract.py`, `tests/conftest.py`, new `tests/test_survival.py` | [reference] onward |
| `fixtures/generate_fixtures.py`, `fixtures/golden/` | [reference], [default] |
| `benchmarks/run_end_to_end.py`, `benchmarks/report-<version>.md` | [benchmark] |
| `README.md`, `CLAUDE.md`, `CHANGELOG.md`, versions in `Cargo.toml`, `pyproject.toml`, `python/eltq/__init__.py` | each bump |

## Acceptance

New spec tests, on both backends, in `tests/test_survival.py` with a new synthetic fixture
`fixtures/small_alpha_elt.csv` (about 2,000 rows, $\alpha$ from 0.003 to 3, $M/\mu$ from
$10^2$ to $10^5$, some atoms and clamped rows, and a gross-like copy with $\mu$ shrunk and $M$
kept):

15. **Exceedance accuracy.** The tail sums of the survival grid match the exact incomplete beta
    at every bucket edge (the `exact_edges` oracle) within $10^{-4}$ relative wherever
    $\Lambda>10^{-12}\sum\lambda_j$, and `meta.interpolation_error` is at most $10^{-4}$.
16. **Rate and mean.** The secondary mass equals the total rate within $10^{-14}$ relative; with
    `grid_length=None`, $|$`secondary_mean_error`$|/$`analytic_aal` is at most $10^{-4}$ at the
    fixture's $h$ and falls roughly as $h^2$ when $h$ is halved twice.
17. **Nonnegativity and determinism.** No negative weight; bitwise identical across thread
    counts 1, 2, 3 and 7 and across `chunk_rows`.
18. **Overflow.** With `grid_length` short of $\max M$: `top_bucket` puts $\Lambda(e_{n-2})$ in
    the top node and reports it in `rate_above_grid`; `error` raises `GridOverflowError` as
    today; both for both methods.
19. **The K-point cut.** On the same fixture, `method="quantiles"` at K = 32 has
    $\Lambda=0$ above each event's $1-1/64$ quantile and a `secondary_mean_error` below $-5\%$,
    while `survival` is within the tolerance of test 16. This documents the defect.

Also: tests 1 to 14 pass on both methods where they apply (tests 4, 5 and 7 are K-point tests);
Rust and reference survival grids agree within $10^{-12}$ relative on the medium fixture; the
[benchmark] gate is met or discussed; the downstream [tie-out] passes.

## Not in this plan

These belong to the later portfolio-combining phase and are recorded so they are not lost.

- `multiunit_views` sends one-unit events through the shared-factor mixture. A test with
  one-unit events, where the mixture should reproduce the marginal beta exactly, found the
  default midpoint rule over $Z_c$ (32 nodes) cutting event tails: survival ratios of 0.40 to
  0.82 at $10^{-3}$ and as low as 0 for events whose $w_C=\sigma_C/\sqrt{\sigma_I^2+\sigma_C^2}$
  is near 1. The midpoint rule at 1001 nodes still fails beyond $10^{-4}$; Gauss-Hermite at 32
  nodes is within 1% to 4%, and at 64 nodes it is exact to two decimals. One-unit events should
  take the survival route, and the default $Z_c$ rule should be reconsidered with tail metrics.
- `dev/plan-chebyshev-z.md` [hybrid-rule] sends correlation-dominant units to the midpoint rule
  at 32 nodes, which is where the tail cut is worst, and its accuracy evidence is the 99th
  percentile (`benchmarks/z-convergence-1.7.0.md`). Hold it until it is redone with tail
  metrics.
- The multi-unit route evaluates each unit across all of $[0,M]$ at the global $h$; large
  exposed values need per-event grids before the deposit to the global grid.
- An a posteriori tail check for the quantile method (exact $\Lambda$ at a few loss points,
  reported in `Meta`).

## Execution log

(empty)
