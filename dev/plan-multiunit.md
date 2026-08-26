# Plan: multiunit — joint secondary uncertainty, unit terms, and view totals

Date: 2026-08-26
Status: finalized 2026-08-26, the five open questions resolved (see Resolved
decisions); in execution. Progress: 1.6.0 [sigma-rule], 1.7.0 [reference-n-unit], 1.8.0 [joint-two-unit], 1.9.0 [terms-grammar] done 2026-08-26.

## Context

`dev/secondary_two_units_reference.py` (now `python/eltq/reference_multiunit.py`) and `dev/secondary-uncertainty-additivity.md`
settled the mathematics: Burke's one-factor construction (shared $Z_c$ per event,
independent $Z_i$ per unit, beta marginals with total sd $\sigma_I + \sigma_C$),
conditional independence given $Z_c$, and the proof that unit-level and rolled ELTs
never agree at FFT precision, so the unit-level route is the model. This plan brings
that into eltq at eltq speed. Three deliverables:

1. **[joint-two-unit]** Exact bivariate bucketed pmf for one event and two units
   (the reference's `secondary_two_units`, in Rust).
2. **[multiunit-views]** For a multi-unit ELT (event x unit rows) with per-unit
   policy terms and per-unit occurrence reinsurance specified DecL-style: the total
   event severity by view (ground_up, gross, ceded, net), rate-accumulated across
   events into aggregate-ready severities.
3. **[sigma-rule]** Correct the total-sd convention throughout eltq from
   $\sigma^2 = \sigma_I^2 + \sigma_C^2$ to the documented Burke rule
   $\sigma = \sigma_I + \sigma_C$ (spec section 11 assumption 3, now resolved).
   Outright, no legacy option: the quadrature combination was incorrect.

Non-goals: event-level occurrence and aggregate treaty terms (aggregate owns those
via DecL on the totals, and applying them downstream is exact), year event tables,
cross-event dependence, GPU (bounded spike later at most).

## The mathematical pipeline (fixed by the reference implementation)

Per unit $i$ of an event: $p = \mu/M$, $\sigma = \sigma_I + \sigma_C$,
$s = \sigma/M$, beta $(\alpha, \beta)$ by moment match; score weights
$w_C = \sigma_C/t$, $w_I = \sigma_I/t$, $t = \operatorname{hypot}(\sigma_I, \sigma_C)$.
Conditional on $Z_c = z$ the units are independent with
$$
P(X_i \le x \mid z) = \Phi\!\left(\frac{y(x) - w_C z}{w_I}\right),
\qquad y(x) = \Phi^{-1}\big(I_{x/M}(\alpha, \beta)\big),
$$
and the whole engine is four steps per event:

1. per unit, once: edge scores $y_e$ at the $n{+}1$ half-bucket edges
   (incomplete beta plus normal quantile, **independent of the z count**);
2. per z node: conditional pmf by adjacent differences of $\Phi((y_e - w_C z)/w_I)$
   (pure erf work), then per-view transform (precompiled deposit maps);
3. per z node: convolve the per-unit view pmfs across units (real FFT);
4. average over z (midpoint quantile nodes $z_q = \Phi^{-1}((q + \tfrac12)/Q)$,
   equal weights; Gauss-Hermite as an option).

The two-unit joint is step 1 and 2 twice plus one GEMM:
$\mathrm{pmf} = C_1^{\mathsf T} C_2 / Q$ with $C_i$ the $(Q \times n_i)$ conditional
matrices. Fresh $(Z_c, Z_i)$ per event occurrence (Burke's $z(E)$, $z(Prog,E)$)
means occurrences are iid, so rate-weighted mixture severity plus Poisson
$(\Lambda)$ reproduces the compound distribution exactly; accumulation across
events is the existing eltq pattern unchanged.

## Workload analysis (what actually costs time)

Symbols: $R$ event-unit rows, $E$ events, $U$ units per event, $n$ per-unit grid
nodes, $N$ per-event total grid, $Q$ z nodes, $V$ views.

- **Special functions, once per row**: $n$ incomplete-beta CDFs plus $n$ normal
  quantiles. $R = 10^6$, $n = 128$: about $1.3\times10^8$ evaluations at
  50-200 ns, roughly 2 s on 16 cores. The structural win: this is $O(Rn)$, not
  $O(RnQ)$; nothing downstream of step 1 touches the beta again.
- **The erf wall, per row per z**: $n$ erf evaluations. $R = 10^6$, $Q = 32$,
  $n = 128$: $4\times10^9$ erfs. Scalar 10 ns each is 40 s single core; a
  branch-free rational erf auto-vectorizes to 1-4/ns, giving roughly 1-3 s on 16
  cores. This is why we write our own erf (Cody-style rational, tested to 1e-15
  against a reference), the exact analog of owning incbeta in 1.4.0.
- **The FFT wall, per event per z per view**: $U$ forward transforms at the
  view's own padded size, spectra accumulated over z in the frequency domain
  (the z mixture is linear), so only **one inverse per view per event**. Dense
  worst case $E = 10^5$, $U = 10$, $N = 1024$, $Q = 32$, four distinct view
  maps on every unit: about $1.3\times10^8$ forward real FFTs at 1-2 us,
   15-25 s on 16 cores. Three mitigations pull typical portfolios well under
  that: identical view maps share one spectrum (ground_up equals gross
  wherever a unit has no policy terms; ceded is the atom at zero and net
  equals gross wherever it has no reinsurance), each view convolves at its own
  support size (ceded towers are usually far smaller than $M$; ground_up is
  the largest), and the fast paths below remove whole units or events from the
  z loop.
- Everything else (view maps, mixing, accumulation) is linear passes.

Fast paths that change the complexity class, all common in real ELTs:

- **Independent block**: units with $\sigma_C = 0$ have z-independent conditional
  pmfs. Convolve them once per event into a cached spectrum; only the correlated
  units re-enter the z loop. An all-independent event needs no z loop at all,
  one convolution total.
- **Comonotone units** ($\sigma_I = 0$): the conditional distribution given $z$
  is an atom at $M\,Q_\beta(\Phi(w_C z))$. Reuse the batched quantile kernel
  (its walk already supports arbitrary ascending targets; small refactor to
  expose that entry point). An all-comonotone event is a sorted quantile walk,
  no FFT.
- **Atoms** (degenerate rows): a fixed index shift in the convolution.
- **All four views are captured by default** (resolved decision 4): the edge
  scores and the erf wall are view-independent, so the all-four delta is
  confined to the FFT component, and spectrum sharing, per-view support sizes,
  and the single inverse per view per event keep it modest in practice.
  `views` stays caller-restrictable for FFT-bound edge cases, and the
  benchmark suite reports the measured all-four delta explicitly.

Headline target, to be confirmed by the probe benchmarks not asserted: a 1M
unit-row portfolio ELT (100k events x 10 units, $n \approx 128$, $Q = 32$, all
four views, typical term sparsity) end to end in **single-digit seconds** on
the Xeon workstation; the dense worst case with four distinct maps on every
unit may reach the low tens of seconds; the two-unit joint at reference
accuracy in milliseconds.

## Data model

- Multi-unit ELT, structure of arrays, rows grouped contiguously by event
  (validated, not sorted for the caller): `event_id`, `unit_id`, `rate`
  (constant within event, validated), `mean_loss`, `sd_indep`, `sd_corr`,
  `max_loss`. Event ids stay out of the hot path as today.
- Terms table keyed by `unit_id`: parsed term sets, compiled once to view maps
  per (unit, event grid). Distinct term sets are few; cache compiled maps.
- `MultiConfig` extends `Config`: `z_nodes` ($Q$, default 32, matching the
  existing quadrature-points default), `z_rule` (`midpoint` default,
  `gauss_hermite` option), `views` (default all four: ground_up, gross, ceded,
  net), plus the existing `bucket_size`, `grid_length`, `normalize`,
  `variance_policy`, `threads`, `chunk_rows` (chunks are whole events here).

## Terms grammar (DecL-like)

One line per unit, borrowing DecL idioms so the eye reads it like aggregate:

```
unit <unit_id> [<limit> xs <ded>] [occurrence net of <clause> (and <clause>)*]
clause := [<share> po] <limit> xs <attach>
share := <number> | <number>%
```

Examples:

```
unit plant_a 750 xs 250 occurrence net of 50% po 400 xs 100 and 500 xs 500
unit plant_b 1000 xs 0 occurrence net of 100 po 400 xs 100
unit fleet
```

Semantics, stated once and tested: `ground_up` $= X$ (prior to policy terms);
`gross` $= \min(\max(X - d, 0), \ell)$ (after ded and limit); `ceded` $=$ sum
of clauses applied to gross, each $sh \cdot \min(\max(g - a, 0), \lim)$;
`net` $=$ gross minus ceded. Shares follow current aggregate DecL: `po` reads
"part of" (`so` and `share of` are gone from the lark); a percent share is the
fraction of the layer directly (`50% po 400 xs 100` cedes half the layer),
while a plain-number share is an amount of the layer limit with fraction
amount over limit (`100 po 400 xs 100` cedes $100/400 = 25\%$; `.2 po 5 xs 1`
cedes $0.2/5$). A clause with no share takes the whole layer. All views are
nondecreasing piecewise-linear maps of $X$. Missing layer means $d = 0$,
$\ell = \infty$ capped at $M$; missing occurrence clause means ceded is zero
and net equals gross.

Rules: monetary breakpoints must be multiples of `bucket_size` (error
otherwise; no silent rounding); percent shares in $(0, 100]$ and amount shares
in $(0, \lim]$; clauses may overlap (each applies to gross independently,
standard XL tower reading). The parser is a small hand-rolled Rust tokenizer
shared by the Python string route; Python may alternatively pass structured
tuples. The keyword set already matches current aggregate DecL; one final
glance at `decl.lark` when the parser freezes keeps the dialects from
drifting.

View compilation: for each source bucket value $x_k = kh$, the view value
$v(x_k)$ deposits via the existing linear mass-and-mean-preserving rule, so a
compiled map is at most two `(index, weight)` pairs per source bucket, applied
as a gather-scatter pass. A whole-layer clause is a pure index shift; fractional
shares land off grid and the linear rule carries them exactly in mean. Base conditional pmfs come from CDF differences (exact in probability);
transformed point values use the linear deposit; that mixed convention is
deliberate and documented, matching eltq's deposit rule.

## Kernels

- **Edge scores**: reuse `incbeta::BetaFn` for $I_{x}(\alpha,\beta)$ at the
  edges; add a normal quantile (Wichura AS241, double precision) for
  $\Phi^{-1}$. Cost is per row, cold path, no heroics needed.
- **Own erf/erfc** (`normal.rs`): branch-free rational approximation, tested to
  1e-15 against statrs and mpmath-generated goldens; the hot path is a straight
  polynomial pipeline the autovectorizer can chew (benchmark scalar vs
  `f64x4`-style manual batch before committing to explicit SIMD, same posture
  as the 1.4.0 kernel work).
- **Comonotone walk**: generalize `batched::walk_map` to public
  `beta_quantiles_at(&bf, ascending_targets, ...)`; the midpoint entry becomes
  a special case. No new solver code.
- **FFT**: `rustfft` (plus `realfft` for the real transforms), plan cache per
  worker per size, per-event padding to the next fast size (power of two or
  5-smooth). Direct convolution crossover for small $N$ or $U = 2$ with tiny
  supports (benchmark sets the threshold).
- **Two-unit joint**: `matrixmultiply::dgemm` on the conditional matrices;
  $Q \times n_1 \times n_2$ flops, milliseconds even at reference-grade
  $Q = 4001$ and $n$ in the hundreds.

## The per-event engine

Per event, inside a worker with reused scratch arenas (no allocation in the
loop): classify units (beta, comonotone, atom, drop; variance policy as
configured) -> edge scores for beta units -> compile or fetch view
maps -> independent-block spectra once -> for each z node: conditional pmfs for
correlated units, view transform, forward FFTs, spectrum product with the
independent block, weighted add into per-view frequency-domain accumulators
-> one inverse FFT per view per event
-> per-view event severities deposited into per-chunk global accumulators
weighted by $\lambda$ -> per-event diagnostics: conditional mass exactness,
pipeline view means versus analytic beta layer means (closed form via
incomplete beta, the LEV identity), roll-up comparison (optional flag: sd and
q99 ratio of exact total versus rolled-row beta, the additivity note's
diagnostic, off by default).

## Parallelism and determinism

Identical contract to `quantize.rs`: fixed event chunks, private per-view
partial histograms and compensated totals per chunk, sequential fold in
chunk-index order, bitwise-deterministic for any thread count; bounded waves
cap memory at `waves x V x N x 8` bytes plus per-worker scratch. The z loop
lives inside the event, so no cross-thread reduction ever mixes floating-point
orders.

## Python API

```python
views, meta = eltq.multiunit_views(
    event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss,
    *, terms,                      # str | list[str] | parsed dicts, by unit_id
    bucket_size, grid_length=None,
    views=("ground_up", "gross", "ceded", "net"),
    z_nodes=32, z_rule="midpoint", **options)
# views: dict view -> float64 array on the global grid (rate-weighted;
# normalize option as today); meta: Meta extended with per-view mass/mean,
# analytic-mean reconciliation, z convergence info, term echo, diagnostics.

pmf, x1, x2, meta = eltq.joint_two_units(unit1, unit2, *, bucket_size,
    n1, n2, z_nodes=1001, **options)   # unit = (mu, sd_i, sd_c, M)
```

Zero-copy array movement, `allow_threads`, typed exceptions, memmap-capable
`multiunit_views_into` following the `quantize_into` pattern. The reference
backend serves both functions when the extension is absent, exactly as today.

## [sigma-rule] correction detail

- `validate.rs::classify_row` and `eltq.reference`: total sd
  $\sigma = \sigma_I + \sigma_C$ (then $\sigma^2$ against $\mu(M - \mu)$).
  Outright replacement, no legacy option (resolved decision 1): the quadrature
  combination was incorrect, and git history serves any before/after need.
- Spec edits: `secondary-uncertainty-mathematical-specification.md` section 4
  working formula replaced, section 11 assumption 3 marked resolved (Burke
  2016, section III.E, step i), with a pointer to the additivity note.
- Golden fixtures regenerate (values shift wherever both sds are nonzero);
  CHANGELOG records the behavioral change prominently. The one-unit outputs get
  wider secondary distributions; that is the correction, not a regression.

## Testing and validation

Reference first, as always:

- **[reference-n-unit]** extend the Python reference to n units, the terms
  grammar (Python mirror of the parser), views, and totals; promote the
  two-unit reference from `dev/` into `python/eltq/` as part of the packaged
  fallback; golden fixtures for joint, views, and parses.
- Acceptance: joint matches reference to quadrature tolerance; marginals of
  the joint match direct beta bucketing; independent event equals plain
  convolution; comonotone common-shape event reproduces the rolled beta
  exactly (additivity Proposition 6 as a test); all-atom events; term
  semantics goldens (ded, limit, tower, shares, overlap); per-event mass
  exactly $\lambda$; view means reconcile to analytic beta layer means within
  a stated $(h, Q)$ tolerance; grid overflow typed errors; misgrouped events
  and inconsistent rates rejected.
- Rust proptest: erf and normal-quantile accuracy sweeps; view-map mass and
  mean preservation; spectrum-product convolution equals direct convolution;
  fast-path equivalence (independent block versus general path) within fp
  tolerance; determinism thread sweep bitwise.
- z convergence: $Q$ doubling study on hard cases (correlation-dominant,
  $\sigma_I \ll \sigma_C$) reported in the benchmark history; pick the default
  from the data.

## Benchmarks

Criterion: erf batch, edge-score build, view-map apply, FFT sizes, GEMM.
End-to-end Python harness: synthetic 100k-event x 10-unit ELT (mixed regimes
including independent-only and comonotone-only events), rows/s, erf/s, FFTs
per event, peak RSS, scaling at 1/2/4/8/all cores, appended to the JSON
history with full machine context. Correctness gates performance: no benchmark
from a build failing acceptance.

## Milestones -> versions (semver, CHANGELOG per version, one-line commits, no push)

- **1.6.0 [sigma-rule]** Convention correction, outright with no option; spec
  edits, reference and core updates, regenerated goldens.
- **1.7.0 [reference-n-unit]** Extended Python reference: n units, terms
  parser, views, totals; packaged; fixtures. The oracle for everything below.
- **1.8.0 [joint-two-unit]** Rust conditional matrices + GEMM joint,
  `joint_two_units` bindings, acceptance vs reference.
- **1.9.0 [terms-grammar]** Rust parser, view-map compiler, validation and
  diagnostics, parse goldens, decl.lark keyword alignment check.
- **1.10.0 [multiunit-views]** The per-event engine end to end: conditional
  pipeline, fast paths, per-view accumulation, extended Meta,
  `multiunit_views` (+`_into`) bindings, acceptance and determinism green.
- **1.11.0 [ferocious-multiunit]** Own erf, Gauss-Hermite option, plan caching
  and real FFTs, crossover tuning, committed 1M unit-row performance report.
- **[optional, profiling-justified]** Chebyshev interpolation of spectra
  across z; contract grouping (per-contract subtotals with contract occurrence
  terms inside the pipeline, the reinsurer multi-treaty view); joint
  (gross, ceded) bivariate output; GPU spike.

## Resolved decisions (Steve, 2026-08-26)

1. Sigma rule: sum, outright, no legacy option; the quadrature combination
   was incorrect.
2. Shares use `po` (part of), matching current aggregate DecL, which dropped
   `so` and `share of`. Percent share means the fraction of the layer; a plain
   number means an amount of the layer limit, fraction amount over limit
   (`.2 po 5 xs 1` cedes $0.2/5$).
3. Confirmed: ground_up precedes policy terms; ded and limit make gross;
   reinsurance applies to gross.
4. Capture all four views by default: cheaper than any recompute, the delta is
   confined to the FFT component, and spectrum sharing, per-view support
   sizes, and one inverse per view per event keep it small in practice. The
   benchmark suite reports the measured all-four delta.
5. $Q$ default 32, matching the existing routine's quadrature default; the
   convergence study validates rather than picks.

## Verification

- `uv run pytest`: acceptance suite vs the extended reference, both backends.
- `cargo test -p eltq-core`: unit + property tests including the new kernels.
- `uv run python benchmarks/run_multiunit.py`: end-to-end synthetic portfolio,
  results appended to the history.
- Determinism: same input twice plus thread-count sweep, bitwise identical.
- `maturin develop`; smoke test both new entry points on Windows/Python 3.13.
