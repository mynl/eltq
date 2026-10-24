# Plan: single-unit-survival-route, bucketing each event's beta by its survival function

Date: 2026-10-23, revised 2026-10-24 after review.
Status: executing from 2026-10-24 (moved to `dev/done/` at the first step, per the house rule).

## Goal

Replace the K-point midpoint-quantile rule as the default way `quantize` builds the `secondary`
grid. The new route evaluates the portfolio exceedance rate
$\Lambda(x)=\sum_j\lambda_j\Pr(X_j>x)$ exactly at log-spaced loss nodes, interpolates it
monotonically, and buckets it onto the even grid by differences at half-bucket edges (the
nearest rule). The K-point rule stays available as `method="quantiles"`. Alongside, three
contract changes decided in review:

- rows whose stated variance is inadmissible become **Bernoulli** rows (two atoms), replacing
  the 0.999 sigma clamp;
- a named **overflow policy** replaces the requirement that the grid reach the largest $M$;
- a public **exceedance-curve object** from which the even grid (at any $h$ and $n$) and the
  OEP points both come.

This plan covers one-row events only (`quantize` and its wrappers). The multi-unit route is
untouched; see "Not in this plan".

## Background for a reader with no context

`eltq` converts an event loss table (ELT) into rate-weighted distributions on the grid
$g_i=ih$. Each nondegenerate row $j$ has rate $\lambda_j$, mean $\mu_j$, standard deviations
$\sigma_{I,j}$ and $\sigma_{C,j}$, and exposed value $M_j$; its conditional loss is
$X_j=M_jB_j$ with $B_j$ beta, moment-matched to $\mu_j$ and $\sigma_j=\sigma_{I,j}+\sigma_{C,j}$
(spec §4, Burke 2016 §III.E). Today the `secondary` grid comes from spec §5: event $j$ is
replaced by $K$ equally weighted points $M_jQ_j((k+\tfrac12)/K)$, deposited by the linear rule
of §6. The engine is `crates/eltq-core/src/quantize.rs` (row classification, chunked deposit,
in-order fold), with the quantile kernel in `batched.rs`, the incomplete beta in `incbeta.rs`,
configuration in `config.rs`, metadata in `diagnostics.rs`, bindings in
`crates/eltq-python/src/lib.rs`, the Python API in `python/eltq/__init__.py`, and the SciPy
oracle in `python/eltq/reference.py`. The acceptance suite in `tests/` runs every test on both
backends through the `impl` fixture in `tests/conftest.py`.

### The defect in the K-point rule

The fraction of event $j$'s $K$ points above a loss $x$ is the number of $k$ with
$(k+\tfrac12)/K>F_j(x)$, divided by $K$. That is $K\,S_j(x)$ rounded to the nearest integer and
divided by $K$, where $S_j=1-F_j$. So the rule sets every event's survival function to zero
wherever it is below $1/(2K)$: at $K=32$ every event is cut off above its 98.4% quantile.

For a beta with small $\alpha$ (roughly $\alpha\approx(\mu/\sigma)^2$ when $\mu\ll M$), most of
the probability sits near zero and much of the mean sits in an upper tail of small probability,
exactly where the rule puts nothing. Synthetic examples with $\mu/M=10^{-4}$ (checked against
SciPy on 2026-10-24):

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
vendor's OEP and AEP; that data is test evidence only and is kept outside this repository. The
exact beta model with $\sigma=\sigma_I+\sigma_C$ tied the vendor's OEP to within 0.1% at every
return period from 5 years, in every peril and view, which confirms the model `eltq` implements.
The K = 32 grid fell short by tens of percent in the perils whose events have small $\alpha$
(severe convective storm, gross winter storm), and its `secondary_mean_error` showed the same
loss of expected value. The same calculation done by the route in this plan tied the vendor's
AEP in most blocks.

### The defect in the sigma clamp

On $[0,M]$, $X^2\le MX$, so $\operatorname{Var}X\le M\mu-\mu^2=\mu(M-\mu)$, with equality
only for the Bernoulli law on $\{0,M\}$. A row with $\sigma^2\ge\mu(M-\mu)$ asserts a variance
no distribution on $[0,M]$ can have. Today's default repairs it by setting
$\sigma=0.999\sqrt{\mu(M-\mu)}$, which makes a beta with $\alpha+\beta\approx0.002$: the
Bernoulli to double precision, except that about 5% of its upper mass is smeared over the
interior ($S/p=1.027$ at $10^{-6}M$ and $0.947$ at $M(1-10^{-12})$, $p=\mu/M$). The 0.999 is an
arbitrary constant standing in for the variance maximizer, and in the survival route the
near-vertical drop of $S_j$ at $M_j$ is a step the interpolant cannot resolve. The honest
projection of an inadmissible row is the maximizer itself: two atoms, $\lambda(1-p)$ at 0 and
$\lambda p$ at $M$. Rate and mean are exact and the variance is the admissible supremum.

## The survival route

**Exceedance curve.** Split the valid rows as now (`validate.rs::classify_row`): atoms
(degenerate rows, policy atoms, and both atoms of a Bernoulli row) and betas. For the betas,
compute
$$
\Lambda(x)=\sum_{j\ \text{beta}}\lambda_j S_j(x),\qquad
\Lambda'(x)=-\sum_{j\ \text{beta}}\frac{\lambda_j}{M_j}f_j\!\left(\frac{x}{M_j}\right),
$$
exactly, at nodes $x_0<x_1<\dots<x_m$ spaced geometrically from a lower loss $x_0$ to
$\max_jM_j$, at `nodes_per_decade` per decade. $S_j$ comes from the incomplete beta and $f_j$
from the beta density; both share the factor $x^\alpha(1-x)^\beta/B(\alpha,\beta)$ already
computed in `incbeta.rs`, so the slope costs little beyond the survival value. Two skips bound
the work per event: nodes at or above $M_j$ contribute exactly zero; nodes below
$M_j\exp(\texttt{ln\_lower\_tail\_start}(2^{-53}))$ (where $F_j<2^{-53}$, closed form, already in
`incbeta.rs`) contribute exactly $\lambda_j$ to $\Lambda$ and nothing to $\Lambda'$, as a
plateau. The active-node counts in the cost section were measured with both skips.

**Interpolation.** Between nodes, $\log\Lambda$ is a cubic Hermite interpolant in $\log x$
using the exact slopes $d\log\Lambda/d\log x=x\Lambda'(x)/\Lambda(x)$, limited by the
Fritsch-Carlson conditions so that it never increases. Above the last node with $\Lambda>0$,
$\Lambda=0$. On the interval ending at that last positive node's successor, interpolate
linearly in $\Lambda$. The interpolation error is a reported diagnostic: exact $\Lambda$ at 64
log-midpoints spread over the node range, against the interpolant, as the largest relative error
where $\Lambda>10^{-12}\sum\lambda_j$. It is a sample, not a bound; the reference backend's
tests check every interval midpoint (below).

**Bucketing (the nearest rule).** With edges $e_i=(i+\tfrac12)h$, the beta part of the grid is
$$
p_0=\textstyle\sum_{\text{beta}}\lambda_j-\Lambda(e_0),\qquad
p_i=\Lambda(e_{i-1})-\Lambda(e_i)\quad(1\le i\le n-1),
$$
and the overflow $\Lambda(e_{n-1})$, the rate beyond the top bucket's upper edge, goes where
the overflow policy says (below). Atoms are added by the existing linear deposit, as today. The
lowest node is $x_0=e_0=h/2$, the smallest loss the grid uses, so $p_0$ is exact.

Properties: every $p_i\ge0$ because the interpolant is monotone; the total rate is exact
because the sum telescopes; the first moment differs from the analytic expected loss by the
bucketing error. In the interior that error is of order $h^2$ times the density's slope. From
bucket 0, whose mass sits at loss 0, it is $\sum_j\lambda_j\int_0^{h/2}x\,dF_j\approx
\sum_j\lambda_j\frac{\alpha_j}{\alpha_j+1}\frac h2F_j(h/2)$, first order in $h$ for the
small-$\alpha$ events this plan exists for (measured: $4.5\times10^{-5}$ relative at
$h=\mu/100$ for $\alpha=0.01$, halving when $h$ halves). Both are reported through
`secondary_mean_error`. This is the rule the multi-unit route already uses for beta units
("cdf differences at half-bucket edges"), and the one `aggregate` uses for continuous
severities, so all three agree. The linear rule was considered and rejected: it is exact in the
mean and inflates the variance by about $h^2/6$ per unit mass against the nearest rule's
Sheppard $h^2/12$, an illusion of exactness.

**The curve is the primary object.** Its non-uniform nodes cover everything up to
$\max_jM_j$ for about `nodes_per_decade` $\times\log_{10}(\max M/x_0)$ points, whatever $h$ is;
the even grid is made only when needed, at whatever $h\ge2x_0$ and $n$ the caller wants, and the
OEP comes straight from the curve under Poisson frequency,
$\mathrm{OEP}(x)=1-e^{-\Lambda(x)-\Lambda_{\text{atoms}}(x)}$. Because $\Lambda$ adds over
events, a later multi-unit or portfolio-combining route can add its events' contributions on
the same nodes.

**Parallelism and determinism.** Rows are processed in fixed chunks; each chunk produces
partial $\Lambda$ and $\Lambda'$ vectors of length $m+1$, folded in chunk order with Neumaier
sums, so results are bitwise identical for any thread count at a fixed `chunk_rows`. (A
different `chunk_rows` changes the fold order and is not bitwise, which is today's contract
too.) The partials are about a thousand doubles, not the grid length, so peak memory no longer
scales with $n$, and the chunk can be small: `chunk_rows=None` resolves to 65,536 rows for
`quantiles` (each fold adds grid-length vectors) and 4,096 for `survival`, so a 1M-row ELT
makes about 250 chunks and keeps a 32-thread machine busy. Bucketing is elementwise over edges
and parallelizes without a reduction.

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
  at $\alpha+1$. Rejected: a gain of order $10^{-5}$ in the mean, bought with the linear rule's
  variance inflation.
- **Exact evaluation at every bucket edge.** $N\times n$ evaluations, infeasible at production
  $n$; used only as the test oracle on small fixtures.
- **Per-event routing** (survival only for events where the K rule fails, judged by each event's
  K-point mean error). On the real ELTs above, a 0.1% tolerance routed 90% to 100% of the events
  in the affected perils, so routing saves nothing there.
- **Nodes inserted at each inadmissible row's $M_j$** to resolve the clamp's step. Rejected in
  favor of the Bernoulli treatment, which removes the step and the 0.999 constant together.
- **A separate overflow bin** (spec §11 item 8). Rejected: it changes the array-length contract
  for a quantity better reported than carried.

## Overflow policies

Today the grid must reach $\max_jM_j$ or `quantize` raises `GridOverflowError`. Ground-up
exposed values run to tens of billions while the losses that matter sit far lower, so callers
truncate by hand after the fact, unreported. The curve itself always runs to $\max_jM_j$; only
the even grid is windowed, and the window is a choice made for the FFT downstream. The policy
applies to both outputs, `raw` (atoms at $\mu_j$) and `secondary`, and to both methods. The
rate beyond the top bucket's upper edge $e_{n-1}=(n-\tfrac12)h$ is the **overflow**: for
`survival`, $\Lambda(e_{n-1})$ plus any atoms above it; for `quantiles`, the weight of the
points (and raw atoms) above it.

- `overflow="top_bucket"` (default): the overflow is added to the top node $g_{n-1}$. Rate is
  exact, so the Poisson frequency fed to `aggregate` is unchanged; the occurrence exceedance is
  exact for every $x\le e_{n-2}$; the first moment is understated by the overflow's
  displacement to $g_{n-1}$.
- `overflow="drop"`: the overflow is discarded. The rate falls short by the overflow and the
  occurrence exceedance is low by it everywhere; the grid holds nothing it does not mean.
- `overflow="error"`: today's `GridOverflowError` whenever the overflow would be nonzero.

Both non-error policies report `rate_above_grid` and `mean_above_grid` (the overflow's rate
and its first moment in excess of $g_{n-1}$) for each output, so a user who prefers to
post-process the blob can. With `grid_length=None` the grid covers $\max_jM_j$ and both figures
are exactly zero, so the default only acts when the caller asks for a window.

## Cost: the survival route is slower than K = 32

This is a speed-for-correctness trade, and it needs to be measured before the default changes.

- The K = 32 route costs 32 quantile inversions per event, about 2 to 4 incomplete-beta
  evaluations each. Current basis: 1M rows in 0.79 s on this machine (CHANGELOG 1.12.0, the
  `x86-64-v3` wheel; 0.92 s native).
- The survival route costs one incomplete beta (and a density) per event per active node. On the
  real ELTs, at 150 nodes per decade over 7 to 8 decades, events had 560 to 880 active nodes
  after both skips. That is 5 to 14 times the K = 32 work.
- Exact slopes make the interpolant fourth order rather than the third order of a PCHIP with
  estimated slopes, which should allow about 50 nodes per decade at the same accuracy, bringing
  the cost to 2 to 5 times K = 32. That is an expectation, not a measurement: [benchmark]
  settles it, and it is a gate.

## Decisions (Steve, 2026-10-23 and 2026-10-24)

- The survival route becomes the default method; `method="quantiles"` keeps the K-point rule.
- Bucketing uses the nearest rule, not the linear rule (confirmed 2026-10-24: "linear is a bit
  evil, illusion of exactness, and it distorts SD").
- Inadmissible-variance rows become Bernoulli rows by default; the 0.999 clamp goes.
- The max-$M$ requirement is replaced by the overflow policies above, `top_bucket` by default,
  with the overflow's rate and mean reported (a "grudging yes": in `aggregate` lumping the tail
  at the top made aggregates that looked wrong, and leaving probability sums below one was
  preferred; `drop` is that option, and the reported figures allow post-processing).
- Steve makes every commit (CLAUDE.md delta, 2026-10-24). Each step below is left in the working
  tree with its files and intended commit subject listed in the execution log.

## Steps

Each step is one commit, made by Steve; each version bump has its own CHANGELOG entry and its
own commit. Versions 1.13.0 to 1.16.0 are interim and are not tagged.

### [spec] no bump: amend the mathematical specification

`secondary-uncertainty-mathematical-specification.md`:

- §4: the variance bound $\operatorname{Var}X\le\mu(M-\mu)$ with its one-line proof, the
  Bernoulli as the unique maximizer, and the `bernoulli` policy as the default treatment of
  inadmissible rows (two atoms $\lambda(1-p)$ at 0 and $\lambda p$ at $M$); the other policies
  `error`, `drop`, `atom`; `clamp` recorded as the former default and removed at 2.0.0.
- New section after §5, "Survival-function discretization": the definitions above for
  $\Lambda$, the nodes, the two skips, the monotone interpolant, the bucket weights and the atom
  deposit, with the exactness statement (rate exact; mean error reported, first order in $h$ from
  bucket 0 for $\alpha<1$ and second order in the interior; interpolation error reported as a
  sampled diagnostic).
- §5 states that the K-point rule is the `quantiles` method and records the $1/(2K)$ cut-off.
- §6: the sentence "must stop with a clear error unless the caller has explicitly selected and
  named an overflow policy" is replaced by the three named policies, with `top_bucket` the
  default and the overflow's rate and mean reported; resolves §11 item 8.
- §6 last paragraph and §7: the outputs for each method; metadata adds the method, node count,
  nodes per decade, interpolation error, overflow policy, and `rate_above_grid` and
  `mean_above_grid` per output. The OEP formula is stated with its Poisson assumption.
- §9: tests 15 to 20 below.
- §11 items 5, 6 and 8 marked resolved: default method `survival`, default variance policy
  `bernoulli`, overflow by named policy.

### [bernoulli] 1.13.0: Bernoulli rows replace the sigma clamp

- `validate.rs`: `VariancePolicy::Bernoulli` (default); `RowKind::Bernoulli` with $p=\mu/M$;
  diagnostics category `variance_bernoulli`. `ClampSigma` stays selectable until 2.0.0.
- `quantize.rs`: a Bernoulli row deposits $\lambda$ at $\mu$ in `raw` and $\lambda(1-p)$ at 0,
  $\lambda p$ at $M$ in `secondary`; its quadrature mean is $\lambda\mu$ exactly.
- `reference.py`, `_meta.py`, `__init__.py`, `crates/eltq-python/src/lib.rs`:
  `variance_policy="bernoulli"` as the default string; the new category.
- `fixtures/generate_fixtures.py` regenerates the goldens (`small`, `medium`); the multi-unit
  goldens are untouched because the multi-unit route keeps its own classification (see "Not in
  this plan").
- Tests: spec test 11 gains the Bernoulli case; test 20 below.

### [reference] 1.14.0: the survival route in the SciPy reference

- `python/eltq/reference.py`: `quantize_reference(..., method="quantiles", nodes_per_decade=150,
  overflow="top_bucket")`. The survival branch computes $\Lambda$ and $\Lambda'$ at the nodes
  with `scipy.special.betaincc` and the beta density, builds the same limited Hermite
  interpolant (written out, not `PchipInterpolator`, so that both backends compute the same
  function), buckets, deposits atoms, applies the overflow policy, and fills the new metadata. A
  private `exact_edges=True` switch evaluates $\Lambda$ at every edge with no interpolation, for
  tests on small fixtures only; a private `check_every_interval=True` evaluates the exact
  $\Lambda$ at every node-interval midpoint for the interpolation-error figure.
- `python/eltq/__init__.py`: `quantize_arrays` and its wrappers (`quantize_csv`,
  `quantize_into`, `quantize_to_npy`, `quantize_dataframe`) take `method`, `nodes_per_decade` and
  `overflow`. On the Rust backend, `method="survival"` dispatches to the reference until
  [rust] lands, as `_via_reference` does for the multi-unit route (SciPy required meanwhile).
- `python/eltq/_meta.py`: `Meta` gains `method`, `nodes`, `nodes_per_decade`,
  `interpolation_error`, `overflow`, `rate_above_grid` and `mean_above_grid` (each a
  `{"raw": float, "secondary": float}` dict), with defaults so older call sites keep working.
  `secondary_mean_error` is defined for both methods as the secondary grid's first moment in
  rate units less `analytic_aal`; under the linear rule that equals today's value up to
  rounding. `quadrature_points` and `quadrature_secondary_mean` are `None` for `survival`.
- The default method stays `quantiles` in this step.
- Tests 15 to 19 on the reference backend.

### [rust] 1.15.0: the survival route in the Rust engine

- `incbeta.rs`: `BetaFn::sf(x)` returning the survival directly in each branch of the
  continued fraction (no `1 - cdf` cancellation in the upper tail), and a combined
  `sf_and_pdf(x)` sharing the front factor.
- New `curve.rs`: `survival_curve(&EltInput, &Config) -> Curve` (nodes, $\Lambda$, slopes, atom
  losses and rates, interpolation error) using the existing classification and chunked fold
  with both skips; `Curve::exceedance(x)` (the interpolant plus atoms); `bucket_curve(&Curve,
  h, n, overflow)` returning the secondary grid and the overflow's rate and mean.
- `config.rs`: `Config` gains `method` (enum `Quantiles`, `Survival`), `nodes_per_decade` and
  `overflow` (enum `TopBucket`, `Drop`, `Error`); `chunk_rows` becomes `Option<usize>` resolved
  per method. `quantize.rs` applies the overflow policy to `raw` and to the quantile path
  (points above $e_{n-1}$ counted and placed or dropped). `Meta` in `diagnostics.rs` gains the
  new fields.
- `crates/eltq-python/src/lib.rs`: the new keyword arguments and metadata keys; the Python
  dispatch to the reference for `survival` is removed.
- Rust unit tests: `sf` against statrs across the regimes in `incbeta.rs`, including survival
  values down to $10^{-300}$; the interpolant is nonincreasing on random curves; thread
  invariance of the curve, bitwise, at fixed `chunk_rows`; the lower skip agrees with the
  unskipped sum bitwise where $F_j<2^{-53}$.

### [benchmark] no bump: measure, set the node density, and gate the default

- `benchmarks/run_end_to_end.py` gains `--method` and `--nodes-per-decade`. Run the 1M-row
  suite for `quantiles` at K = 32 and `survival` at 25, 50, 100 and 150 nodes per decade, with
  the interpolation error and the time split into curve build and bucketing.
- Add a small-$\alpha$ synthetic case to the performance suite (spec §10 already asks for
  strongly skewed shapes): $\alpha\in[0.003,1]$, $M/\mu\in[10^3,10^5]$.
- Set the default `nodes_per_decade` to the smallest density whose interpolation error is at
  most $10^{-4}$ on every benchmark case, measured with `check_every_interval` on the reference
  for a subsample and the 64-point diagnostic in Rust.
- Write `benchmarks/report-<version>.md`.
- **Gate:** if `survival` at the default density takes more than 5 times K = 32 on the 1M-row
  case (about 4 s on this machine), stop and talk before [default]. Speedups to look at then,
  in order: the batched kernel layout in `batched.rs` for the node loop, and a lower density
  with adaptive refinement wherever the midpoint check fails.

### [curve] 1.16.0: public exceedance curve

```python
curve = eltq.exceedance_curve(rate, max_loss, mean_loss, sd_indep, sd_corr, *, min_loss,
                              nodes_per_decade=None, mean_is_loss_rate=False,
                              variance_policy="bernoulli",
                              threads=None, chunk_rows=None, event_ids=None)
curve.loss, curve.exceedance_rate, curve.log_slope   # beta part at the nodes
curve.atom_loss, curve.atom_rate                     # degenerate and Bernoulli atoms
curve.meta                                           # row diagnostics, node count, interpolation error
curve.exceedance(x)                                  # Lambda(x), atoms included; NaN below min_loss
curve.oep_pml(return_periods)                        # loss where Lambda = -log(1 - 1/T), Poisson
curve.grid(bucket_size, grid_length, *, normalize=False, overflow="top_bucket")
```

`min_loss` is a required keyword (no magic constant): `quantize_arrays(method="survival")`
becomes `exceedance_curve(min_loss=h/2)` followed by `curve.grid(h, n)` for the secondary grid
plus the raw grid from a cheap per-row pass (the curve does not store $\mu_j$). `curve.grid`
refuses `bucket_size < 2 * min_loss`, since the curve knows nothing below its lowest node.
`oep_pml` returns an atom's loss when the step at that atom jumps the target. Both backends. A
curve built once serves any number of grids with $h\ge2\,$`min_loss`, for example a coarse one
for a quick look and a fine one for the FFT.

### [default] 2.0.0: survival becomes the default

Major bump: the default `secondary` output changes, and the `clamp` policy with its
`clamp_factor` keyword and `sigma_clamped` category are removed.

- `method` defaults to `survival` in Rust, the bindings, the reference and every Python
  wrapper.
- Tests that depend on the K-point rule (spec tests 4, 5 and 7, and the existing golden
  fixtures) pass `method="quantiles"` explicitly; new survival goldens are generated by
  `fixtures/generate_fixtures.py`.
- `README.md`, the module docstring in `python/eltq/__init__.py` and the description at the top
  of `CLAUDE.md` describe the secondary grid by the survival route, with the K-point rule as an
  option, and the Bernoulli treatment of inadmissible rows.

### [tie-out] outside this repository

The downstream tie-out project installs this checkout editable, replaces its own exact-severity
code with `eltq`'s survival route, and checks that the OEP and AEP points agree with its exact
calculation within $10^{-3}$ at return periods from 5 years, in every block. The PyPI release
follows when Steve pushes the tag.

## Files

| file | steps |
|---|---|
| `secondary-uncertainty-mathematical-specification.md` | [spec] |
| `crates/eltq-core/src/validate.rs`, `config.rs`, `diagnostics.rs`, `quantize.rs` | [bernoulli], [rust], [default] |
| `crates/eltq-core/src/incbeta.rs`, `curve.rs` (new), `lib.rs` | [rust], [curve] |
| `crates/eltq-python/src/lib.rs` | [bernoulli], [rust], [curve], [default] |
| `python/eltq/reference.py`, `python/eltq/_meta.py`, `python/eltq/__init__.py` | [bernoulli], [reference], [curve], [default] |
| `tests/test_acceptance.py`, `tests/test_contract.py`, `tests/conftest.py`, new `tests/test_survival.py` | [bernoulli] onward |
| `fixtures/generate_fixtures.py`, `fixtures/golden/`, new `fixtures/small_alpha_elt.csv` | [bernoulli], [reference], [default] |
| `benchmarks/run_end_to_end.py`, `benchmarks/report-<version>.md` | [benchmark] |
| `README.md`, `CLAUDE.md`, `CHANGELOG.md`, versions in `Cargo.toml`, `pyproject.toml`, `python/eltq/__init__.py` | each bump |

## Acceptance

New spec tests, on both backends, in `tests/test_survival.py` with a new synthetic fixture
`fixtures/small_alpha_elt.csv` (about 2,000 rows, $\alpha$ from 0.003 to 3, $M/\mu$ from
$10^2$ to $10^5$, some atoms and inadmissible rows, and a gross-like copy with $\mu$ shrunk
and $M$ kept):

15. **Exceedance accuracy.** The tail sums of the survival grid match the exact incomplete beta
    at every bucket edge (the `exact_edges` oracle) within $10^{-4}$ relative wherever
    $\Lambda>10^{-12}\sum\lambda_j$; the reference's `check_every_interval` error is at most
    $10^{-4}$; the 64-point `meta.interpolation_error` is at most that figure.
16. **Rate and mean.** The secondary mass equals the total rate within $10^{-14}$ relative; with
    `grid_length=None`, $|$`secondary_mean_error`$|/$`analytic_aal` is at most $10^{-4}$ at the
    fixture's $h$ and falls at least linearly when $h$ is halved twice.
17. **Nonnegativity and determinism.** No negative weight; bitwise identical across thread
    counts 1, 2, 3 and 7 at fixed `chunk_rows`; Rust and reference agree within $10^{-12}$
    relative.
18. **Overflow.** With `grid_length` short of $\max M$, for both methods and both outputs:
    `top_bucket` adds $\Lambda(e_{n-1})$ (and the atoms above it) to the top node and reports it
    in `rate_above_grid` with its displaced moment in `mean_above_grid`; `drop` leaves the mass
    short by exactly that and reports the same figures; `error` raises `GridOverflowError`.
    With `grid_length=None` both figures are exactly zero.
19. **The K-point cut.** On the same fixture, `method="quantiles"` at K = 32 has
    $\Lambda=0$ above each event's $1-1/64$ quantile (plus one bucket of linear spread) and a
    `secondary_mean_error` below $-5\%$, while `survival` is within the tolerance of test 16.
    This documents the defect.
20. **Bernoulli rows.** An inadmissible row contributes $\lambda(1-p)$ at 0 and $\lambda p$ at
    $M$ in both methods, with rate and mean exact and grid variance $\mu(M-\mu)$ to bucketing
    tolerance; under `survival` the exceedance curve has an exact step of $\lambda p$ at $M$;
    the row appears under `variance_bernoulli`.

Also: tests 1 to 14 pass on both methods where they apply (tests 4, 5 and 7 are K-point tests);
the [benchmark] gate is met or discussed; the downstream [tie-out] passes.

## Not in this plan

These belong to the later portfolio-combining phase and are recorded so they are not lost.

- The multi-unit route keeps the sigma clamp for now; moving it to the Bernoulli treatment is
  a one-line change in its classification but changes its goldens, and it belongs with the
  multi-unit work below.
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
- Admissible rows with very small $\kappa=\alpha+\beta$ (legal near-Bernoullis) stay betas. If
  the exhaustive interpolation check on the fixture shows their step at $M_j$ breaking test 15,
  the remedy is a node at each such $M_j$; no threshold constant is introduced pre-emptively.

## Execution log

Steve rules at execution (2026-10-24): the plan lands as 2.0.0 as written. Two gaps found at review and ruled the same day: the clamp removal at 2.0.0 applies to the one-row route only (`quantize_*`), because the Rust `VariancePolicy` enum, the `Category` enum and the policy parser are shared with the multi-unit route, which keeps `clamp` with `clamp_factor` and the `sigma_clamped` category and rejects `bernoulli` with a typed error until it adopts the Bernoulli treatment; and the sentence in `reproductions/Homer2017a.qmd` describing the clamp default is rewritten at 2.0.0.

Divergences recorded as they are made:

- **Rust gate.** `cargo test -p eltq-core` fails to compile `libm` 0.2.16 under rustc 1.99.1 at the `[profile.test]` opt-level 2 (`Missing .seh_endepilogue`, a compiler bug); release and dev profiles compile. The Rust gate for this execution is `cargo test -p eltq-core --release`, 66 unit tests plus the integration tests, green at the start.
- **Test 17 tolerance.** Bucket weights are differences of $\Lambda$, so a per-element relative tolerance is unattainable where the difference cancels. The test checks the curve pointwise at $10^{-12}$ relative where $\Lambda>10^{-12}\sum\lambda_j$ and the grids at $10^{-12}$ of the total rate per element; spec test 17 is written that way.
- **Lower skip threshold.** $1-F$ rounds to exactly 1 only when $F<2^{-54}$, so the plateau cut uses a probability with a margin below $2^{-53}$; the Rust unit test checks the bitwise claim.
- **The `error` overflow policy** raises on the upfront check that some contributing $M_j$ exceeds $e_{n-1}$, for both methods, rather than after computing the points; this is the condition under which the overflow can be nonzero.
- **`crates/eltq-core/tests/golden.rs`** was not in the Files table; it reads the golden params and gains `method` at 2.0.0.
- **Spec numbering.** The new section is `5A` rather than renumbering 6 to 11, so references to those sections elsewhere stay valid.

### [spec] (no bump)

`secondary-uncertainty-mathematical-specification.md`: section 1 names both methods; section 4 proves the variance bound, names the Bernoulli maximizer and lists the policies with `clamp` recorded as the former default; section 5 records the $1/(2K)$ cut and names the quantiles method; section 5A specifies the survival route (nodes, skips, interpolant, bucketing, curve, OEP, determinism); section 6 replaces the coverage requirement with the three overflow policies and the reported figures; section 7 lists the outputs and metadata by method and the exposed curve; section 9 adds tests 15 to 20; section 11 marks items 5, 6 and 8 resolved. Plan moved to `dev/done/`.

Files for the commit: `secondary-uncertainty-mathematical-specification.md`, `dev/done/plan-single-unit-survival-route.md` (moved from `dev/`). Subject: `amend the specification for the survival method, Bernoulli rows and overflow policies`.

### [bernoulli] 1.13.0

As planned, with these notes:

- `VariancePolicy::Bernoulli` is the Rust default (`Config::new`); `MultiConfig::new` and `JointConfig` use the new `VariancePolicy::clamp_default()` so the multi-unit routes are unchanged, and both their classifiers return the new `Error::BernoulliPolicyUnavailable` for the Bernoulli variant. The roll-up diagnostic's rolled row in `views.rs` used `VariancePolicy::default()` and now names the clamp explicitly, otherwise an inadmissible rolled row would have errored.
- `Category::ALL` has nine entries; the Python `CATEGORIES` tuple matches.
- `fixtures/generate_fixtures.py` regenerates every golden. The multi-unit goldens came back with last-digit differences (about 1e-17 relative, the SciPy build in this environment, not this change); they were restored with `git checkout` so the commit carries only the one-row goldens.
- Test 20's quantiles-method half is in the new `tests/test_survival.py`; the survival half follows at [reference]. Test 11 gains `test_11_variance_policy_bernoulli`; the clamp tests select `clamp` explicitly.
- Gates: `uv run pytest` (both backends) and `cargo test -p eltq-core --release`, both green.

Files for the commit: `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, `uv.lock`, `python/eltq/__init__.py`, `python/eltq/reference.py`, `crates/eltq-core/src/{config,diagnostics,error,validate,quantize,multiunit,views}.rs`, `crates/eltq-python/src/lib.rs`, `fixtures/generate_fixtures.py`, `fixtures/golden/{small,medium}.*`, `tests/test_acceptance.py`, `tests/test_contract.py`, `tests/test_multiunit.py`, `tests/test_survival.py`, `benchmarks/run_end_to_end.py`, `README.md`, `CHANGELOG.md`, `CLAUDE.md` (the version line; the rest of that file's diff is Steve's), `dev/done/plan-single-unit-survival-route.md`. Subject: `1.13.0 inadmissible-variance rows become Bernoulli rows, replacing the sigma clamp default`.

### [reference] 1.14.0: the design changed under test

**The plan's interpolant does not work.** The plan's survival route (geometric nodes, one log-log Hermite of $\Lambda$) was built first and measured against the exact incomplete beta at every bucket edge. Every beta's survival function has an infinite log-slope at its endpoint $M_j$, so $\Lambda$ has a kink at every endpoint; between geometric nodes a smooth interpolant misses by the share of $\Lambda$ that the events ending nearby carry. Relative error of the bucketed tail sums at 150 nodes per decade, by exceedance level ($\Lambda$ as a fraction of the total rate):

| portfolio | > 1e-1 | > 1e-2 | > 1e-3 | > 1e-4 | > 1e-6 | > 1e-12 |
|---|---|---|---|---|---|---|
| small-alpha fixture (2,000 rows) | - | - | - | - | 2.8e-2 | 270 |
| medium golden (5,000 rows) | 9e-4 | 1.4e-2 | 2.0e-2 | 0.31 | 0.65 | 0.78 |
| benchmark portfolio (20k subsample) | 7e-4 | 7e-3 | 0.10 | 0.29 | 7.0 | 7.0 |

The medium and benchmark portfolios are adversarial (mean damage ratios up to 0.95, half the betas below 1), but a 1% miss at the 100-year level on the project's own golden is not a deep-tail curiosity. Tried and measured before stopping: nodes at every endpoint (the plan's own remedy for near-Bernoulli rows; the slope singularity is one-sided, so the interval before the endpoint still missed by 15%); endpoint refinement with nodes geometric in $M_j-x$ for events whose share exceeds 1e-3 (the fixture improves, the J-shaped portfolios reach 1e-3 at four times the nodes, and the flagged count grows with rows); routing endpoint-dominant events to the K-point rule (it routes exactly the small-alpha events the method exists for, with 58% mean error); exact evaluation at every edge (correct and infeasible at production grid lengths, as the plan said). The real-ELT tie-out cited in the plan used exact evaluation, so the interpolant had never met real data.

**Ruling (Steve, 2026-10-24).** Build a kink-free design rather than pause or lower the target.

**What was built.** Each event is split at a truncation node, `TAIL_INTERVALS` = 4 node intervals below the first node at or above its endpoint: the shared curve carries $\lambda_j(S_j(x)-S_j(x_k))^+$, continuous and vanishing at a node, with one-sided slopes at the nodes; the tail mass $\lambda_jS_j(x_k)$ on $(x_k,M_j]$ stays with the event and is bucketed by a per-event monotone cubic Hermite of $\log S_j$ against $\log(M_j-x)$ on nodes halving the distance to the endpoint, with the exact power law below the last node. Three findings along the way, each measured: a one-interval zone is not enough (the collective endpoint curvature of the events sharing a truncation node still gave 1e-2); tail atoms at the conditional mean with a per-event share tolerance do not bound the collective error (900 atoms under 1e-4 each added to 2%); and the plain cubic in $x$ beats the log-log one decisively (1.1e-5 against 1e-2 on the medium golden), because the logarithm magnifies the curvature where several truncated functions vanish together. Final figures, every bucket edge where $\Lambda>10^{-12}\sum\lambda_j$, 150 nodes per decade: small-alpha fixture 4e-6, medium golden 1.1e-5, benchmark portfolio 1.1e-5. The 50 nodes per decade figures are in the [benchmark] step.

**Divergences from the plan, recorded:**

- Spec section 5A rewritten for the truncated-curve design (truncation, one-sided slopes, the per-event tail interpolant, the plain cubic); the plan's log-log interpolant and "nodes alone" text is superseded.
- Two shared constants in `python/eltq/_meta.py`: `DEFAULT_NODES_PER_DECADE` (150 until [benchmark]) and `TAIL_INTERVALS` (4).
- The lower skip uses the probability $2^{-60}$ (`LOWER_SKIP_PROBABILITY`), the margin noted at review.
- Test 17's thread sweep applies to the Rust backend only (the reference takes no thread keywords); its backend comparison is trivial until [rust] lands, since both backends run the reference for the survival method.
- Test 18's quantiles case is skipped on the Rust backend until 1.15.0 (the engine raises on overflow regardless of the policy); `Meta.overflow` reports `"error"` from the Rust engine until then.
- The fixture's bucket is $h=1$ with max $M=10^6$ (a million-node grid); the survival tests take about four minutes on the reference, mostly test 16's halvings. To be trimmed when the Rust engine serves them.
- The multi-unit goldens again came back with last-digit differences from `generate_fixtures.py` and were restored.

- Spec test 13 now selects `overflow="error"` for its raise; under the default policy a short grid is a window, which test 18 covers.

Gates: `uv run pytest` (both backends; 168 passed, 1 skipped) and `cargo test -p eltq-core --release`, both green.

Files for the commit: `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, `uv.lock`, `python/eltq/__init__.py`, `python/eltq/_meta.py`, `python/eltq/reference.py`, `fixtures/generate_fixtures.py`, `fixtures/small_alpha_elt.csv`, `tests/test_acceptance.py`, `tests/test_contract.py`, `tests/test_survival.py`, `secondary-uncertainty-mathematical-specification.md` (section 5A), `README.md`, `CHANGELOG.md`, `CLAUDE.md` (version line), `dev/done/plan-single-unit-survival-route.md`. Subject: `1.14.0 survival method and overflow policies in the SciPy reference`.

### [rust] 1.15.0

As planned, against the design as built at 1.14.0 rather than the plan's text:

- `curve.rs` holds `geometric_nodes`, `limit_slopes`, `hermite`, `tail_survival` and `survival_pass`; the quantize driver classifies once and dispatches by method, both passes sharing the overflow-aware `deposit::place`. The curve partials are node-length, but the tail and atom deposits need a grid-length partial histogram per chunk as the quantiles route has, so peak memory still scales with `n` (the plan's "no longer scales with n" does not hold for this design).
- The interpolation-error diagnostic samples the same 64 interval log-midpoints as the reference (`round_ties_even` reproduces NumPy's rounding of the sample indices); exact curve and exact total at those points are accumulated in the chunk pass.
- `BetaFn::sf` against statrs: held to 2e-12 relative, because statrs's own survival function loses digits near 1 and the two readings of a symmetric 500-500 beta at its midpoint straddle 0.5 by 5e-13.
- The endpoint interpolant's node spacing changed to sqrt(2) steps in both backends after the Rust unit test showed 2e-5 on a bell-shaped event's first halving; the spec and docstrings say so.
- Test 17's raw grids are compared at 1e-12 relative rather than bitwise (the chunked fold's summation order).
- The quantiles test 18 case runs on the Rust backend; `Meta.overflow` is no longer defaulted by the wrapper.
- `tests/test_memmap.py`: outputs too short for the input are a window under the default policy (reported overflow), and raise only under `overflow="error"`. Found at the 1.16.0 gate: the 1.15.0 suite run had been read through a pipe that hid pytest's exit code, so its green was not verified; the 1.16.0 run below was.
- Gates: `cargo test -p eltq-core --release` (73 unit tests plus the integration tests) green; `uv run pytest` green once the memmap test was corrected (169 passed on both backends).

Files for the commit: `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, `uv.lock`, `python/eltq/__init__.py`, `python/eltq/reference.py`, `crates/eltq-core/src/{curve (new),config,deposit,diagnostics,error,incbeta,lib,quantize}.rs`, `crates/eltq-core/tests/{golden,properties}.rs`, `crates/eltq-python/src/lib.rs`, `tests/test_survival.py`, `tests/test_memmap.py`, `secondary-uncertainty-mathematical-specification.md`, `README.md`, `CHANGELOG.md`, `CLAUDE.md` (version line), `dev/done/plan-single-unit-survival-route.md`. Subject: `1.15.0 survival method and overflow policies in the Rust engine`.

### [benchmark] 1.16.0 (the plan said no bump)

- **Divergence: a bump.** Setting the default node density is a behavior change, and the house rule bumps every capability change, so this step is 1.16.0 and [curve] moves to 1.17.0; [default] stays 2.0.0.
- **Gate met.** Survival at the chosen density: 1.23 times K = 32 on the million-row mixture portfolio, 2.65 times on a million-row small-alpha portfolio (limit five). The plan's estimate of 5 to 14 times assumed nodes above every event's endpoint; the truncated curve has none.
- **Density 100.** The 64-point diagnostic passes 1e-4 at 50 nodes per decade, but the exact-edge criterion of test 15 (the reference against the exact incomplete beta at every edge) gives 2.8e-4 and 4.7e-4 on the medium golden and the mixture subsample at 50, and 4.8e-5 and 5.9e-5 at 100. The plan's rule names the exact check, so 100 it is; the fixture alone would have allowed 50.
- The time split into curve build and bucketing is not observable from outside the engine (tails are deposited in the node pass) and is not reported.
- `benchmarks/report-1.16.0.md`; the two sweep records are appended to `benchmarks/results.json`.
- Gates: `cargo test -p eltq-core --release` (73 unit tests plus the integration tests) and `uv run pytest` (169 passed, exit code checked), both green at the new default.

Files for the commit: `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, `uv.lock`, `python/eltq/__init__.py`, `python/eltq/_meta.py`, `crates/eltq-core/src/config.rs`, `benchmarks/run_end_to_end.py`, `benchmarks/report-1.16.0.md`, `benchmarks/results.json`, `secondary-uncertainty-mathematical-specification.md`, `README.md`, `CHANGELOG.md`, `CLAUDE.md` (version line), `dev/done/plan-single-unit-survival-route.md`. Subject: `1.16.0 survival node density 100 from the benchmark, speed gate met`.

### [curve] 1.17.0 (the plan said 1.16.0)

The API as planned, with these divergences:

- **No `log_slope` attribute.** The curve's slopes are one-sided and belong to the interpolant; what the curve exposes at the nodes is the exact exceedance rate of the beta events (`exceedance_rate`), accumulated in the build pass from each event's truncated survival, its tail mass and its exact survival at the nodes its tail covers.
- **The curve is the one implementation.** `quantize(method="survival")` builds the curve at `min_loss = h / 2` and takes its grid, in both backends; the 1.15.0 survival pass was removed rather than kept as a second copy. The curve keeps each event's tail parameters (about 56 bytes per beta row) so that a grid at another spacing, and the exceedance queries, can bucket or evaluate the tails exactly; `grid` runs its own chunk pass over them, folded in order.
- `curve.grid` returns `(secondary, GridMeta)`; the raw grid is not part of the curve (it does not store the means), so `quantize` makes it in a separate per-row pass.
- `oep_pml`: NaN where the target rate exceeds the rate at `min_loss` (the curve's range), 0 for return periods of a year or less, and an atom's loss when the bisection bracket closes on an atom.
- The Rust binding sums the grid's mass and moment with the compensated sums `quantize` uses, so the two `secondary_mean_error` figures agree bitwise.
- Gates: `cargo test -p eltq-core --release` (74 unit tests plus the integration tests) and `uv run pytest` (both backends; the curve tests run on the loaded backend and check it against the reference's curve), both green.

Files for the commit: `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, `uv.lock`, `python/eltq/__init__.py`, `python/eltq/_meta.py`, `python/eltq/reference.py`, `crates/eltq-core/src/{curve,error,lib,quantize}.rs`, `crates/eltq-python/src/lib.rs`, `tests/test_survival.py`, `secondary-uncertainty-mathematical-specification.md`, `README.md`, `CHANGELOG.md`, `CLAUDE.md` (version line), `dev/done/plan-single-unit-survival-route.md`. Subject: `1.17.0 public exceedance curve: exceedance_curve, grids at any spacing, OEP points`.

### [default] 2.0.0

As planned, under the review ruling that the clamp leaves the one-row route only:

- `Config::new` defaults to `Method::Survival`; the bindings and every Python wrapper default to `"survival"`. `classify_row` returns `Error::ClampPolicyRemoved` for `VariancePolicy::ClampSigma`, which stays in the enum for the multi-unit route; the one-row binding has its own policy parser that refuses `"clamp"`, and `clamp_factor` is gone from the one-row signatures (kept on `multiunit_views` and `joint_two_units`). The reference's `CATEGORIES` and `VARIANCE_POLICIES` drop the clamp; `Category::SigmaClamped` stays in Rust for the multi-unit route.
- Goldens: `small` and `medium` carry `method: quantiles` in their params and a `fixture` key; `small_survival` and `medium_survival` are new. `crates/eltq-core/tests/golden.rs` reads the fixture name and the optional `K` from the metadata and checks all four.
- Test 10's dyadic bitwise invariance is a quantiles property (the survival nodes are geometric in the loss, not dyadic); it runs both methods, bitwise for quantiles and to rounding for survival. Test 11's clamp case checks the refusal; the contract test of far-inadmissible rows checks the Bernoulli pair.
- `reproductions/Homer2017a.qmd` line 254 rewritten as ruled; the chapter's computed figures came from 1.12.0 and the sentence says so. The two mentions of clamping on the multi-unit route (lines 418 and 437) stay true.
- Not done, flagged: `notebooks/eltq_demo.ipynb` and `dev/Manually.ipynb` were not reviewed for the old default or the removed keyword.
- Gates: `cargo test -p eltq-core --release` (74 unit tests, 4 golden tests, the integration tests) and `uv run pytest` (175 passed, 1 skipped, exit code checked), both green.

Files for the commit: `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, `uv.lock`, `python/eltq/__init__.py`, `python/eltq/reference.py`, `crates/eltq-core/src/{config,error,validate,quantize}.rs`, `crates/eltq-core/tests/{golden,properties}.rs`, `crates/eltq-python/src/lib.rs`, `fixtures/generate_fixtures.py`, `fixtures/golden/{small,medium,small_survival,medium_survival}.*`, `tests/test_acceptance.py`, `tests/test_contract.py`, `tests/test_python_api.py`, `reproductions/Homer2017a.qmd`, `README.md`, `CHANGELOG.md`, `CLAUDE.md` (the description and the version line; the rest of that file's diff is Steve's), `dev/done/plan-single-unit-survival-route.md`. Subject: `2.0.0 survival is the default method, the clamp leaves the one-row route`.

### Plan complete

All steps executed on 2026-10-24: [spec], [bernoulli] 1.13.0, [reference] 1.14.0, [rust] 1.15.0, [benchmark] 1.16.0, [curve] 1.17.0, [default] 2.0.0. [tie-out] is outside this repository and remains Steve's. Nothing is committed; every step's files and subject are listed above. The 2.0.0 tag, which publishes, is Steve's.
