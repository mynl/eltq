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
moment exactly; the survival grid is exact in rate and reports its first-moment
error. The exceedance curve is public (`exceedance_curve`). The mathematical
contract is `secondary-uncertainty-mathematical-specification.md`; the
implementation plans are `dev/done/plan-implementation.md` (1.0.0-1.5.0),
`dev/done/plan-multiunit.md` (1.6.0-1.11.0: joint multi-unit secondary
uncertainty, unit terms, view totals) and
`dev/done/plan-single-unit-survival-route.md` (1.13.0-2.0.0: the survival
method, Bernoulli rows, overflow policies, the exceedance curve) and
`dev/done/plan-survival-tail-cost.md` (2.0.1-2.3.0: the tail rate cut, the
balanced tail pass, narrow events) and `dev/done/plan-ln-beta-large-parameters.md`
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
machine. On the multi-unit route that is about 8% faster than the wheel on a
Zen 5 CPU; elsewhere the two are equally fast, and every output is bitwise
identical.

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
rustflags = ["-C", "target-cpu=native"]
```

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
reduction) is ~0.02 s; the inverse-beta kernel is everything. Grid length is
irrelevant to throughput (each deposit touches two nodes whatever n is):
n = 1,001, n = 65,536 (2^16) and n = 100,000 all run at the same speed.

Determinism contract: for a fixed input, config (including `chunk_rows`,
default 65536 for the quantiles method and 256 for the survival method) and
build, output arrays are bitwise reproducible regardless of `threads` and
work stealing. Reordering input rows changes results only within
floating-point summation tolerance.

Survival method (`benchmarks/report-1.16.0.md`, Zen 5 desktop): the
million-row mixture portfolio at the default 100 nodes per decade in 0.96 s
against 0.78 s for the quantiles method at K = 32, and a million-row
small-alpha portfolio in 2.7 s against 1.0 s, where the quantiles method
loses 20% of the expected loss and the survival method 0.6%.

Multi-unit route (`benchmarks/report-1.11.0.md`, same laptop): ~0.91M unit
rows (100k events x 10 units, typical term sparsity) with all four views at
Q = 32 in 22.5 s, ground_up only in 15.2 s, Q = 8 in 7.9 s; bitwise identical
across thread counts (whole-event chunks, default 2048 rows); the two-unit
joint at reference accuracy (Q = 1001) in 10 ms. These figures are for local
builds with `target-cpu=native` (see Environment), so rebuild after moving
machines. On an AMD Ryzen 9 9950X3D2 (Zen 5, 16C/32T) the 1M-row headline takes
0.92 s, and the `x86-64-v3` wheel matches a native build except on the
multi-unit route, where it is 8% slower (`dev/done/plan-pypi-release.md`).

## Python API

```python
import eltq

raw, secondary, meta = eltq.quantize_arrays(
    rate, max_loss, mean_loss, sd_indep, sd_corr,
    *, bucket_size, grid_length=None,
    method="survival", quadrature_points=32, nodes_per_decade=None,
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
    *, min_loss, nodes_per_decade=None, ...)   # the survival curve, built once
curve.loss, curve.exceedance_rate              # nodes and the exact rate there
curve.atom_loss, curve.atom_rate, curve.meta   # atoms; CurveMeta
curve.exceedance(x)                            # Lambda(x), atoms included
curve.oep_pml([10, 100, 250])                  # OEP losses, Poisson occurrence
secondary, gmeta = curve.grid(bucket_size, grid_length=None, overflow="top_bucket")
    # any bucket_size >= 2 min_loss; what quantize(method="survival") returns
```

`source` for the last two: a CSV path, a DataFrame with canonical columns, or
a `(rate, max_loss, mean_loss, sd_indep, sd_corr)` tuple.

**Methods.** `method="survival"` (the default from 2.0.0) evaluates the
portfolio exceedance rate exactly at `nodes_per_decade` geometric nodes,
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
    z_nodes=32, z_rule="midpoint", normalize=False,
    variance_policy="clamp", clamp_factor=0.999, roll_up_diagnostic=False)
# views: dict view -> rate-weighted array on the grid i * h; meta: MultiMeta

pmf, x1, x2, meta = eltq.joint_two_units(
    (mu1, sd_i1, sd_c1, M1), (mu2, sd_i2, sd_c2, M2),
    *, bucket_size, n1, n2, z_nodes=1001, z_rule="midpoint")
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
without a line get no terms. The shared factor is integrated by midpoint
quantile nodes (default) or Gauss-Hermite; `benchmarks/z-convergence-1.7.0.md`
records the convergence study behind the default. Both functions run in Rust
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
kept as a beta) is refused on this route. The multi-unit route keeps `clamp`
as its default, with `clamp_factor`, and does not accept `bernoulli` yet.

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
