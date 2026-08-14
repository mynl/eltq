# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/) and semantic versioning.

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
