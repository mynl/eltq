# eltq — ferociously fast ELT secondary-uncertainty quantization

Converts a portfolio event loss table (ELT, ~1M rows) into two discrete, evenly
spaced, rate-weighted loss-grid distributions:

- **raw** — each event contributes an atom of rate λ at its conditional mean μ;
- **secondary** — each event's conditional loss is modeled as a beta on
  [0, M] matching (μ, σ = σ_I + σ_C); the portfolio exceedance curve of those
  betas is evaluated exactly on geometric nodes, each event's endpoint handled
  per event, and bucketed onto the grid g_i = i·h by the nearest rule (the
  survival method). The K-point midpoint-quantile rule
  x_k = M·Q(α, β; (k+½)/K) of versions before 2.0.0 remains selectable.

Atoms and the raw grid use a linear deposit rule that preserves mass and first
moment exactly (on a windowed grid, an atom in the top half-bucket goes to the
top node and is not overflow; spec section 6); the survival grid is exact in
rate and reports its first-moment error. The exceedance curve is public (`exceedance_curve`). The mathematical
contract is `secondary-uncertainty-mathematical-specification.md`; the
implementation plans are `dev/done/plan-implementation.md` (1.0.0-1.5.0),
`dev/done/plan-multiunit.md` (1.6.0-1.11.0: joint multi-unit secondary
uncertainty, unit terms, view totals) and
`dev/done/plan-single-unit-survival-route.md` (1.13.0-2.0.0: the survival
method, Bernoulli rows, overflow policies, the exceedance curve) and
`dev/done/plan-survival-tail-cost.md` (2.0.1-2.3.0: the tail rate cut, the
balanced tail pass, narrow events), `dev/done/plan-ln-beta-large-parameters.md` and
`dev/done/plan-quantiles-large-parameters.md`
(2.3.1: the incomplete beta at large parameters).

## Install

```
pip install eltq                # or: uv add eltq
pip install "eltq[reference]"   # adds SciPy for the reference backend
```

Wheels are built for Python 3.12 and later on Windows x86_64, Linux x86_64
(manylinux) and macOS arm64; no Rust toolchain is needed. The only runtime
dependency is NumPy. The reference backend (`eltq.reference`,
`eltq.reference_multiunit`) is the SciPy oracle the test suite checks the Rust
engine against; it also serves as the fallback when the compiled extension is
missing.

The x86_64 wheels target the `x86-64-v3` instruction set (AVX2: Intel Haswell,
2013, and AMD Excavator, 2015, onward). On an older CPU the import fails with an
illegal-instruction error; build from source instead, which needs a Rust
toolchain:

```
pip install --no-binary eltq eltq
```

Setting `RUSTFLAGS="-C target-cpu=native"` for that build tunes it to the
machine, which is not always a gain: on a Zen 5 CPU the one-row routes run
4% to 6% slower than the `x86-64-v3` wheel under native (AVX-512 encodings on
a scalar kernel) and the multi-unit route 3% to 9% faster; every output is
bitwise identical either way (`benchmarks/report-2.4.2.md`).

## Status

| Milestone | Version | State |
| --- | --- | --- |
| [reference] SciPy oracle, golden fixtures, acceptance suite | 1.0.0 | done |
| [rust-core] eltq-core scalar path | 1.1.0 | done |
| [python] PyO3 bindings + CSV route | 1.2.0 | done |
| [memmap] quantize_into / quantize_to_npy | 1.3.0 | done |
| [ferocious] parallelism + batched kernel + benchmarks | 1.4.0 | done |
| [fallback] SciPy backend in-package + demo notebook | 1.5.0 | done |
| [sigma-rule] total sd = σ_I + σ_C (Burke 2016), outright | 1.6.0 | done |
| [reference-n-unit] n-unit reference, terms grammar, views | 1.7.0 | done |
| [joint-two-unit] Rust bivariate joint pmf | 1.8.0 | done |
| [terms-grammar] Rust parser + view-map compiler | 1.9.0 | done |
| [multiunit-views] per-event engine, `multiunit_views` | 1.10.0 | done |
| [ferocious-multiunit] own erf, real FFTs, cross-view sharing, 1M unit-row report | 1.11.0 | done |
| [pypi-release] wheels on PyPI: Windows, Linux, macOS; CI against the oracle | 1.12.0 | done |
| [bernoulli] inadmissible-variance rows become Bernoulli rows (two atoms) | 1.13.0 | done |
| [reference] survival method and overflow policies in the SciPy reference | 1.14.0 | done |
| [rust] survival method and overflow policies in the Rust engine | 1.15.0 | done |
| [benchmark] node density set, speed gate met (`benchmarks/report-1.16.0.md`) | 1.16.0 | done |
| [curve] public exceedance curve: `exceedance_curve`, grids at any spacing, OEP points | 1.17.0 | done |
| [default] survival is the default method; the clamp leaves the one-row route | 2.0.0 | done |
| [oep-pml] `oep_pml` above every atom | 2.0.1 | done |
| [tail-cut] tail walks cut at the rate floor and the grid window | 2.1.0 | done |
| [edge-cost] span-limited chunk histograms balance the tail pass | 2.2.0 | done |
| [narrow-events] betas narrower than a node interval walked by quadrature | 2.3.0 | done |
| [ln-beta] `ln B` by Stirling at large parameters; the survival exact to 1e-13 at beta = 8e7 | 2.3.1 | done |
| [large-parameters] beta quantiles by the Cornish-Fisher expansion from min(alpha, beta) = 2e5; the quantiles method right and finite at every parameter | 2.3.2 | done |
| [kernel] the incomplete beta reports failure instead of a value; events beyond the fraction's regime routed around it whatever the density | 2.3.3 | done |
| [atoms] the normal law for narrow events from min(alpha, beta) = 1e16 replaces the step at the mean; rows too narrow to resolve are atoms under `atom_by_resolution` | 2.3.4 | done |
| [overflow] the windowed grid's excess moment from the integral of the curve, so events whose endpoint survival underflows keep theirs | 2.3.5 | done |
| [tails] the endpoint ladder's power law gated on its asymptotic criterion, the one edge below it evaluated exactly, quarter-octave steps across the transition | 2.3.6 | done |
| [mesh] bell-shaped events on a fine curve of `nodes_per_sigma` nodes per standard deviation; the accuracy contract of spec test 33 | 2.4.0 | done |
| [ratios] the admissibility test, the shape parameters and the cumulants' sd formed in ratios, so a row classifies the same at every monetary scale; the top half-bucket and the tiny-alpha complement documented | 2.4.1 | done |
| [schedule] the quantiles default chunk a rule of the row count, waves bounded by a 1 GiB memory budget, one thread pool per call | 2.4.2 | done |
| [tidy] the survival grid walked with an interval cursor, the normal scores shared on the Cornish-Fisher route, the output scans combined; bitwise neutral | 2.4.3 | done |
| [bernoulli] inadmissible unit rows are two-point units given the shared factor on the multi-unit route; the clamp deprecated there | 2.5.0 | done |
| [z-rule] the stratified factor rule with a tail ladder is the default: exact marginals, the event total's tail within 7e-3; the bivariate normal in the engine; the convergence study with tail metrics | 2.6.0 | done |
| [one-unit] events with one contributing unit bucketed directly, outside the mixture; `one_unit_events` | 2.7.0 | done |
| [coupling] `coupling=` burke, comonotone, independent, homer; the comonotone and Homer paths exact with no factor rule; the Homer allocation uncapped with `homer_excess_mean` | 2.8.0 | done |
| [joint] `occurrence_pair_joint`, `occurrence_unit_first_moment` and `annual_conditional_mean`: the joint aggregate route's inputs and the conditional means by the Mecke identity | 2.9.0 | done |
| [matched] `coupling="matched"`, Burke's loadings scaled per event to the roll-up convention's variance, with `matched_shortfall`; [parity] the multi-unit benchmark rerun under the stratified default (`benchmarks/report-2.10.0.md`) | 2.10.0 | done |
| [narrow-units] beta units past the continued fraction's regime evaluated by the narrow kernel, no NaN; units below the resolution are atoms; the reference's unit classification in ratios | 2.10.1 | done |
| [model-moments] `moments_arrays`: the one-row ELT's rate, raw moments, zero rate and support with no grid, classified as the grid route classifies | 2.11.0 | done |

## Layout

```
crates/eltq-core/     pure Rust core (no PyO3/NumPy/CLI types)
crates/eltq-python/   PyO3 + numpy bindings
python/eltq/          pure-Python layer: API wrappers, SciPy reference/fallback
                      (one-unit: reference.py; multi-unit: reference_multiunit.py),
                      DecL-style unit terms (terms.py)
tests/                pytest acceptance suite (runs both backends)
fixtures/             synthetic ELTs + golden outputs (never real data)
benchmarks/           bench harness + machine-readable results history
notebooks/            demo notebook (eltq_demo.ipynb)
dev/                  plans (moved to dev/done when complete)
.github/workflows/    CI: build and test wheels, publish to PyPI (wheels.yml)
```

## Environment (uv)

`uv sync` creates `.venv` in the checkout and builds the extension into
`./target`; both are gitignored. The repository sets no compiler flags, so a
source build targets the baseline instruction set of the Rust target. To tune
local builds to the machine, set the flag in a cargo config outside the
repository, which cargo reads for every project below it (the author's is
`V:\dev\.cargo\config.toml`), or in `RUSTFLAGS`, which overrides it:

```toml
[build]
rustflags = ["-C", "target-cpu=x86-64-v3"]
```

That is the level the published wheels use, so local timings are the ones
users get; `target-cpu=native` was measured slower on the one-row routes on a
Zen 5 (see Installing), and under rustc 1.99.1 it also breaks the test
profile's build of `libm`.

uv does not rebuild when that file changes; force it with
`uv sync --reinstall-package eltq`.

## Commands

```powershell
uv sync                                        # build wheel (maturin) + deps;
                                               # rebuilds when Rust sources change
uv run pytest                                  # acceptance suite (reference + eltq)
uv run python fixtures/generate_fixtures.py    # regenerate fixtures + goldens
cargo test -p eltq-core                        # Rust unit + property tests
cargo bench -p eltq-core                       # criterion kernel benchmarks
uv run python benchmarks/run_end_to_end.py     # 1M-row end-to-end benchmark;
                                               # appends benchmarks/results.json
uv run python benchmarks/run_multiunit.py      # 1M unit-row multi-unit benchmark
                                               # (100k events x 10 units); appends
uv run python benchmarks/z_convergence.py      # shared-factor quadrature study
uv run --with jupyter --with matplotlib jupyter lab
                                               # demo: notebooks/eltq_demo.ipynb
```

## Performance

Headline (i7-7820HQ 4C/8T, Windows 11, see `benchmarks/report-1.4.0.md`):
one million mixed-regime rows at K = 32 quantized in ~4.3 s — ~230k rows/s,
~6.6M beta quantiles/s, ~44x the SciPy reference — with bitwise-identical
results for any thread count. Non-kernel overhead (validation + deposits +
reduction) is ~0.02 s; the inverse-beta kernel is everything. Grid length
barely matters to the deposits (each touches two nodes whatever n is: n =
1,001, n = 65,536 (2^16) and n = 100,000 run at the same speed), but each
chunk's two partial grids are initialized and folded at the grid's length, so
at 2^24 buckets the chunk count is a memory and time term. A pass runs its
chunks in waves of min(2 x threads, 1 GiB / partial size), so the partials in
flight never exceed 1 GiB whatever the grid (`benchmarks/report-2.4.2.md`).

Determinism contract: for a fixed input, config (including `chunk_rows`, by
default clamp(max(ceil(rows / 64), ceil(n / 64)), 4096, 65536) for the
quantiles method and 256 for the survival method, both functions of the input
and the configuration alone) and build, output arrays are bitwise
reproducible regardless of `threads` and work stealing.
Reordering input rows changes results only within floating-point summation
tolerance.

Survival method (`benchmarks/report-1.16.0.md`, Zen 5 desktop): the
million-row mixture portfolio at the default 100 nodes per decade in 0.96 s
against 0.78 s for the quantiles method at K = 32, and a million-row
small-alpha portfolio in 2.7 s against 1.0 s, where the quantiles method
loses 20% of the expected loss and the survival method 0.6%. That is a cost
comparison, not an efficiency one: `benchmarks/run_accuracy.py` reports the
error each method's time buys, by band of the exceedance rate, and at K = 32
the quantiles method is wrong by its whole value from the hundred-year band
on where the survival method is within 1e-5 (`benchmarks/report-2.4.2.md`).

Multi-unit route (`benchmarks/report-2.10.0.md`, the Zen 5 desktop,
`x86-64-v3` build): ~0.91M unit rows (100k events x 10 units, typical term
sparsity) with all four views under the stratified default (76 strata) in
8.4 s, ground_up only in 7.0 s, `z_nodes=8` in 5.6 s; the midpoint and
Gauss-Hermite rules at Q = 32 in 2.0 s and 1.9 s; the exact `comonotone` and
`homer` paths in 0.38 s; bitwise identical across thread counts (whole-event
chunks, default 2048 rows), 14.4x on 16 cores. The stratified default buys
exact marginals and tails within 7e-3 for 4.3 times the node rules' time
(`benchmarks/z-convergence-2.6.0.md`). The `x86-64-v3` wheel matches or
beats a native build except on the multi-unit route, where it is 3% to 9%
slower (`dev/done/plan-pypi-release.md`, `benchmarks/report-2.4.2.md`).

## Python API

```python
import eltq

raw, secondary, meta = eltq.quantize_arrays(
    rate, max_loss, mean_loss, sd_indep, sd_corr,
    *, bucket_size, grid_length=None,
    method="survival", quadrature_points=32, nodes_per_decade=None, nodes_per_sigma=None,
    overflow="top_bucket", normalize=False, mean_is_loss_rate=False,
    variance_policy="bernoulli", event_ids=None)

raw, secondary, meta = eltq.quantize_csv(path, *, bucket_size,
    columns={"rate": "annual_rate"}, ...)      # optional column-name mapping

raw, secondary, meta = eltq.quantize_dataframe(df, *, bucket_size, ...)

meta = eltq.quantize_into(source, *, raw_out, secondary_out, ...)
    # writable float64 arrays incl. np.memmap; their length defines the grid

meta = eltq.quantize_to_npy(source, "grids.npy", *, bucket_size, ...)
    # transactional (2, n) .npy + grids.meta.json; np.load(..., mmap_mode="r")

curve = eltq.exceedance_curve(rate, max_loss, mean_loss, sd_indep, sd_corr,
    *, min_loss, nodes_per_decade=None, nodes_per_sigma=None, ...)   # the survival curve, built once
curve.loss, curve.exceedance_rate              # nodes and the exact rate there
curve.atom_loss, curve.atom_rate, curve.meta   # atoms; CurveMeta
curve.exceedance(x)                            # Lambda(x), atoms included
curve.oep_pml([10, 100, 250])                  # OEP losses, Poisson occurrence
secondary, gmeta = curve.grid(bucket_size, grid_length=None, overflow="top_bucket")
    # any bucket_size >= 2 min_loss; what quantize(method="survival") returns
moments = eltq.moments_arrays(rate, max_loss, mean_loss, sd_indep, sd_corr,
    *, mean_is_loss_rate=False, variance_policy="bernoulli", event_ids=None)
    # ModelMoments with no grid: total_rate, raw_moments (C_1..C_3), severity_moments,
    # rate_at_zero, support per row, counts by law, diagnostics (spec section 4B)
```

`source` for the last two: a CSV path, a DataFrame with canonical columns, or
a `(rate, max_loss, mean_loss, sd_indep, sd_corr)` tuple.

**Methods.** `method="survival"` (the default from 2.0.0) evaluates the
portfolio exceedance rate exactly at `nodes_per_decade` geometric nodes (and
the bell-shaped events on a finer mesh of `nodes_per_sigma` nodes per standard
deviation around their means, the method's accuracy knob),
interpolates it monotonically and buckets it by the nearest rule, with each
event's endpoint handled per event, its tail walked only while its rate
exceeds 1e-20 of the total, and events narrower than a node interval kept
out of the curve and walked by quadrature of their density (spec section
5A); both backends compute the same function. `method="quantiles"` (the default before 2.0.0) replaces
each beta event by K midpoint quantiles deposited by the linear rule; it cuts
every event's survival function at 1/(2K), which loses tens of percent of the
expected loss for small-alpha events (spec section 5). `overflow` names what happens to mass
above the top bucket when `grid_length` is a window: `top_bucket` (default,
rate exact), `drop`, or `error`; the overflow's rate and displaced moment are
reported in `meta` either way.

Arrays are borrowed zero-copy (contiguous float64; the wrapper converts
explicitly otherwise), compute releases the interpreter, and results hand
Rust-owned buffers to NumPy without a copy. Errors raise `eltq.EltqError` or
its subclasses `VarianceError` / `GridOverflowError`. `meta` is a `Meta`
dataclass with the fields listed under Output below.

**Backends.** `eltq.BACKEND` reports which implementation serves the API:
`"rust"` (the compiled extension) or `"reference"` — the pure-Python SciPy
implementation in `eltq.reference`, used automatically when the extension is
not importable (requires scipy; ~40x slower; ignores `threads`/`chunk_rows`).
`eltq.reference.quantize_reference` is also importable directly as the
independent oracle the acceptance suite tests both backends against.

## Multi-unit route (1.7.0)

For ELTs at unit level (event x unit rows) the units of one event share a
common factor (Burke 2016, section III.E): per unit the conditional loss is a
beta on [0, M] with total sd σ_I + σ_C; one shared standard normal Z_c per
event occurrence and an independent Z_i per unit give a one-factor Gaussian
copula, and the units are independent given Z_c. `dev/secondary-uncertainty-
additivity.md` proves why this cannot be reduced to rolled rows, so unit-level
rows are kept and per-unit terms applied before summation.

```python
views, meta = eltq.multiunit_views(
    event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss,
    *, terms, bucket_size, grid_length=None,
    views=("ground_up", "gross", "ceded", "net"),
    z_nodes=32, z_rule="stratified", normalize=False,
    variance_policy="bernoulli", clamp_factor=0.999, roll_up_diagnostic=False,
    coupling="burke")   # or comonotone, independent, homer
# views: dict view -> rate-weighted array on the grid i * h; meta: MultiMeta

pmf, x1, x2, meta = eltq.joint_two_units(
    (mu1, sd_i1, sd_c1, M1), (mu2, sd_i2, sd_c2, M2),
    *, bucket_size, n1, n2, z_nodes=32, z_rule="stratified", coupling="burke")

# The joint aggregate route's inputs (2.9.0): the rate-weighted joint
# occurrence severity of two units for a view, and every unit's first-moment
# vector at each total, from which the annual conditional mean of a unit given
# the portfolio total is a one-dimensional convolution (Mecke identity).
pmf, x_u, x_w, meta = eltq.occurrence_pair_joint(
    event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss,
    *, units=("u", "w"), view="ground_up", terms, bucket_size)
moments, x, meta = eltq.occurrence_unit_first_moment(
    event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss,
    *, view="ground_up", terms, bucket_size)      # dict unit -> m_u(y)
kappa = eltq.annual_conditional_mean(moments, annual_density)   # f_A from the caller
```

Rows of one event must be contiguous with one row per unit and a constant
rate (`MultiUnitError` otherwise). Terms are DecL-style, one line per unit
(`eltq.parse_terms` / `eltq.terms`):

```
unit plant_a 750 xs 250 occurrence net of 50% po 400 xs 100 and 500 xs 500
unit plant_b 1000 xs 0 occurrence net of 100 po 400 xs 100
unit fleet
```

`ground_up` = X; `gross` = min(max(X − ded, 0), limit); `ceded` = Σ share ·
min(max(gross − attach, 0), layer) over the clauses (each applied to gross;
overlaps allowed as long as the combined share stays ≤ 100% in every band, so
ceded never exceeds gross); `net` = gross − ceded. A percent share is the fraction of
the layer; a plain number is an amount of the layer limit (`100 po 400 xs
100` cedes 25%). Breakpoints must be multiples of `bucket_size`. Units
without a line get no terms. The shared factor is integrated by the
stratified rule (default from 2.6.0: `z_nodes` equal strata over the lower
three quarters of the factor's probability and a geometric ladder of ratio
√2 in the upper tail, 76 strata at `z_nodes=32`; every unit's marginal exact,
the event total's exceedance within 7e-3 at the 1e-2 to 1e-5 levels) or by
midpoint quantile or Gauss-Hermite nodes, which are 4 to 6 times faster and
cut or miss the tails of units with a loading near one;
`benchmarks/z-convergence-2.6.0.md` records the study behind the default and
`meta.z_strata` the strata used. Both functions run in Rust
on the compiled backend (`joint_two_units`: conditional matrices plus one
GEMM, 1.8.0; `multiunit_views`: the per-event engine of 1.10.0, whole-event
chunks folded bitwise-deterministically like `quantize`, direct or FFT
convolution across units chosen by cost or forced with `convolution=`) and
on the SciPy reference `eltq.reference_multiunit` otherwise.
`multiunit_views_into(..., out={view: array})` writes into caller-provided
(memmap-capable) arrays, the keys selecting the views and the length the grid.

## Input schema

Canonical columns (CSV route allows a column-name mapping from 1.2.0):

| column | symbol | meaning |
| --- | --- | --- |
| `event_id` | e | optional identifier (row index fallback), audit only |
| `rate` | λ | annual event rate / nonnegative measure weight |
| `max_loss` | M | beta upper bound (maximum possible loss) |
| `mean_loss` | μ | conditional mean loss (or a rate on [0, 1] with `mean_is_loss_rate`) |
| `sd_indep` | σ_I | independent std-dev component |
| `sd_corr` | σ_C | correlated std-dev component |

Admissibility: λ, M, σ_I, σ_C ≥ 0 and 0 ≤ μ ≤ M. Total conditional sd is the
arithmetic sum σ = σ_I + σ_C (Burke, Rau-Chaplin, Varghese 2016, section
III.E; the root-sum-of-squares rule used before 1.6.0 was incorrect). The
model behind the two components, and why unit-level rows must be kept when
unit-level terms exist, is in `dev/secondary-uncertainty-additivity.md`.

## Row handling and policies

Structurally invalid rows are dropped and reported (never silently): nonfinite
fields, negative fields, μ > M. Zero-rate rows carry no mass and are skipped.
Degenerate rows (σ = 0, μ = 0, μ = M, or M = 0) contribute a single atom at μ.

Rows with inadmissible beta variance (σ² ≥ μ(M − μ), a variance no
distribution on [0, M] can have) follow `variance_policy`:

- `bernoulli` (default) — the variance maximizer on [0, M]: two atoms, rate
  (1 − p) at 0 and rate p at M with p = μ/M; rate and mean exact, reported;
- `error` — typed error;
- `drop` — row dropped from both outputs, reported;
- `atom` — collapsed to an atom at μ, reported.

The `clamp` of versions before 2.0.0 (σ clamped to 0.999 of the bound, the row
kept as a beta) is refused on this route. The multi-unit route has `bernoulli`
as its default from 2.5.0 (the two-point law given the shared factor, spec
section 4A) and keeps `clamp`, with `clamp_factor`, deprecated until 3.0.0.

Diagnostics report per-category counts plus up to 10 representative event ids:
`nonfinite`, `negative_field`, `mean_exceeds_max`, `zero_rate`,
`degenerate_atom`, `variance_dropped`, `variance_atom`, `variance_bernoulli`
(and `sigma_clamped` on the multi-unit route).

With `grid_length=None` the length is derived as ceil(max M / h) + 1, which
covers every loss. A shorter explicit grid is a window: the mass above the top
bucket's upper edge follows `overflow` (`top_bucket`, `drop`, or `error`,
which raises `GridOverflowError`), and its rate and displaced moment are
reported either way. No silent truncation, no overflow bin.

## Output

Two length-n `float64` arrays (weights at losses i·h; rate-weighted by
default, `normalize=True` divides both by the total contributing rate Λ) plus
metadata: h, n, the method (with K and the quadrature secondary mean under
`quantiles`, the node count, density and interpolation error under
`survival`), normalized flag, Λ, analytic AAL, both output masses and first
moments, the secondary mean error vs the AAL, the overflow policy and figures,
row counts, and the diagnostics above.

## Open questions (spec §11)

- Confirm the real file's exposure field is exactly the beta upper bound M.
- `mean_is_loss_rate` scales only μ by M (per spec). If real ELTs give the
  standard deviations as damage ratios too, that convention needs a decision.
