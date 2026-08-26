# eltq — ferociously fast ELT secondary-uncertainty quantization

Converts a portfolio event loss table (ELT, ~1M rows) into two discrete, evenly
spaced, rate-weighted loss-grid distributions:

- **raw** — each event contributes an atom of rate λ at its conditional mean μ;
- **secondary** — each event's conditional loss is modeled as a beta on
  [0, M] matching (μ, σ = σ_I + σ_C) and approximated by K equally weighted
  midpoint quantiles x_k = M·Q(α, β; (k+½)/K).

Both are mapped to the grid g_i = i·h by a linear deposit rule that preserves
mass and first moment exactly. The mathematical contract is
`secondary-uncertainty-mathematical-specification.md`; the implementation plans
are `dev/done/plan-implementation.md` (1.0.0-1.5.0) and `dev/plan-multiunit.md`
(1.6.0 onward: joint multi-unit secondary uncertainty, unit terms, view totals).

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
| [joint-two-unit] Rust bivariate joint pmf | 1.8.0 | planned |
| [terms-grammar] Rust parser + view-map compiler | 1.9.0 | planned |
| [multiunit-views] per-event engine, `multiunit_views` | 1.10.0 | planned |
| [ferocious-multiunit] own erf, real FFTs, 1M unit-row report | 1.11.0 | planned |

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
```

## Environment (Windows, uv, V:\dev)

Build churn stays off the Synology-synced folder: the venv lives at
`V:\dev\venvs\secondary-corrosion` (junction `.venv` in the repo) and the cargo
target dir at `V:\dev\cargo-target\secondary-corrosion` (`.cargo/config.toml`).

```powershell
uv venv V:\dev\venvs\secondary-corrosion --python 3.13
New-Item -ItemType Junction -Path .venv -Target V:\dev\venvs\secondary-corrosion
uv sync
```

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
default 65536) and build, output arrays are bitwise reproducible regardless
of `threads` and work stealing. Reordering input rows changes results only
within floating-point summation tolerance.

## Python API

```python
import eltq

raw, secondary, meta = eltq.quantize_arrays(
    rate, max_loss, mean_loss, sd_indep, sd_corr,
    *, bucket_size, grid_length=None, quadrature_points=32,
    normalize=False, mean_is_loss_rate=False,
    variance_policy="clamp", clamp_factor=0.999, event_ids=None)

raw, secondary, meta = eltq.quantize_csv(path, *, bucket_size,
    columns={"rate": "annual_rate"}, ...)      # optional column-name mapping

raw, secondary, meta = eltq.quantize_dataframe(df, *, bucket_size, ...)

meta = eltq.quantize_into(source, *, raw_out, secondary_out, ...)
    # writable float64 arrays incl. np.memmap; their length defines the grid

meta = eltq.quantize_to_npy(source, "grids.npy", *, bucket_size, ...)
    # transactional (2, n) .npy + grids.meta.json; np.load(..., mmap_mode="r")
```

`source` for the last two: a CSV path, a DataFrame with canonical columns, or
a `(rate, max_loss, mean_loss, sd_indep, sd_corr)` tuple.

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
records the convergence study behind the default. Until the Rust engine
lands (1.10.0) both functions run on the SciPy reference
`eltq.reference_multiunit`.

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

Rows with inadmissible beta variance (σ² ≥ μ(M − μ)) follow `variance_policy`:

- `clamp` (default) — σ clamped to 0.999·√(μ(M − μ)), reported;
- `error` — typed error;
- `drop` — row dropped from both outputs, reported;
- `atom` — collapsed to an atom at μ, reported.

Diagnostics report per-category counts plus up to 10 representative event ids:
`nonfinite`, `negative_field`, `mean_exceeds_max`, `zero_rate`,
`degenerate_atom`, `sigma_clamped`, `variance_dropped`, `variance_atom`.

Grid coverage requires (n − 1)·h ≥ max contributing M; with `grid_length=None`
the length is auto-derived as ceil(max M / h) + 1. An insufficient explicit
grid raises `GridOverflowError` — no silent truncation, no overflow bin.

## Output

Two length-n `float64` arrays (weights at losses i·h; rate-weighted by
default, `normalize=True` divides both by the total contributing rate Λ) plus
metadata: h, n, K, normalized flag, Λ, analytic AAL, both output masses and
first moments, the quadrature secondary mean and its finite-K error vs the
AAL, row counts, and the diagnostics above.

## Open questions (spec §11)

- Confirm the real file's exposure field is exactly the beta upper bound M.
- `mean_is_loss_rate` scales only μ by M (per spec). If real ELTs give the
  standard deviations as damage ratios too, that convention needs a decision.
