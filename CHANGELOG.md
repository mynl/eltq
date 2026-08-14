# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/) and semantic versioning.

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
