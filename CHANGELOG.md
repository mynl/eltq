# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/) and semantic versioning.

## [1.11.0] - 2026-08-26

### Added

- [ferocious-multiunit] Own `erfc` / `erf` in `eltq-core::normal` (Cody's
  rational Chebyshev approximations over the CALERF ranges, a straight
  polynomial pipeline per range that vectorizes across a row of edge
  scores), held to `libm::erfc` at 1e-15 relative over the full range by a
  dense sweep including the range switch points.
- Real FFTs: `convolve::FftCache` now plans `realfft` transforms (an n-point
  real row has an n/2 + 1 point spectrum, halving the transform work),
  cached per worker per size, with the transform scratch reused.
- Cross-view sharing in the per-event engine (`views.rs` rewritten around
  row sets): a unit materializes only the distinct row sets its terms
  produce (base for ground_up; gross after policy terms; ceded and net
  after reinsurance), so a unit without policy terms contributes its base
  rows to gross and net, one without reinsurance has net = gross and a ceded
  atom at zero that drops out of the ceded convolution entirely. Bare units
  (no terms at all) form one shared block convolved once per z node and
  reused by ground_up, gross and net; on the FFT path one event-wide size
  lets every (unit, row set, z) spectrum be transformed once and multiplied
  into each view, with the shared block's spectrum product formed once per
  z. The direct / FFT choice is made per event from the operations each
  path would actually run (`FFT_COST_FACTOR` calibrated on the benchmark so
  `auto` tracks the faster path).
- Whole-event chunks default to 2048 rows (`DEFAULT_MULTI_CHUNK_ROWS`)
  instead of 65536, so a 100k-row portfolio spreads across every core.
- `.cargo/config.toml` sets `target-cpu=native` (the wheel is built from
  source on each machine by `uv sync`; the file is now a uv cache key so
  changing it rebuilds). Per-element kernels vectorize without reordering
  any reduction, so the determinism contract is untouched.
- `benchmarks/run_multiunit.py`: the 1M unit-row harness (100k events x 10
  units, mixed regimes and typical term sparsity): all four views by
  convolution mode, ground_up only (the all-four delta), Q = 32 versus 8,
  Gauss-Hermite, thread scaling, the two-unit joint at Q = 1001 and 4001,
  bitwise determinism across thread counts, peak RSS; appends to
  `benchmarks/results.json` under `"suite": "multiunit"`. Report:
  `benchmarks/report-1.11.0.md`.

### Changed

- Plan `dev/plan-multiunit.md` completed and moved to `dev/done/`.

### Fixed

- `batched::beta_quantiles` accepts repeated endpoint targets (0 at the
  start, 1 at the end): Gauss-Hermite nodes at Q >= 32 reach |z| ~ 10, where
  `Phi(w_c z)` rounds to exactly 0 or 1 and the comonotone quantile walk
  rejected the run with a kernel failure. Regression test on the fixture
  (Q = 64 Gauss-Hermite) and the joint.

### Notes

- Performance (`benchmarks/report-1.11.0.md`, i7-7820HQ 4C/8T): ~0.91M unit
  rows (100k events x 10 units, single-unit events thinned) with all four
  views at Q = 32 in 22.5 s (40.5k rows/s); ground_up only 15.2 s; Q = 8
  in 7.9 s; threads 1 / 2 / 4 / 8: 81.9 / 43.8 / 27.7 / 22.3 s; bitwise
  identical across thread counts; the two-unit joint at Q = 1001 in 9.9 ms;
  peak RSS 231 MiB. The 1.10.0 engine did the 100k-row probe in 15.8 s;
  1.11.0 does it in 4.5 s (12.1 s single-threaded). The plan's single-digit
  headline was for the Xeon workstation, which was not measured; ~85% of the
  remaining time is the per-z convolution, so the next structural gain is
  the optional Chebyshev interpolation across z.
- The one-unit route measured 135k rows/s on this day against the 1.4.0
  history's 180-230k; an A/B against the 1.5.0 build in a separate worktree
  gave 112-124k rows/s on the same day, so the gap is machine state, not a
  regression.

## [1.10.0] - 2026-08-26

### Added

- [multiunit-views] The Rust per-event engine `eltq-core::views`
  (`multiunit_views`): pass 1 validates structure (contiguous events,
  constant rate, unique units, unit codes within the terms table),
  classifies unit rows under the one-unit diagnostic categories, and sizes
  the grid from the largest event support; pass 2 runs whole-event chunks
  in parallel with the existing bitwise-deterministic in-order fold. Per
  event: base conditional rows once per unit (Q rows for units that load on
  the shared factor, one row for independent beta units and atoms), per-view
  compiled maps (identity maps reuse the base rows; maps cached per worker),
  the independent block convolved once, correlated units folded in per z
  node by direct convolution or by FFT with the spectra accumulated over z
  and a single inverse per view (`convolution="auto"` picks by estimated
  cost; `direct` / `fft` force a path), then rate-weighted deposits into the
  chunk's per-view histograms. Per-event mass exactness (`max_mass_error`),
  analytic beta layer means (the LEV identity via the incomplete beta,
  bundles built once per unit) reconciled per view, and the optional
  roll-up diagnostic (exact ground-up total versus the rolled single-row
  beta: sd and q99 ratios per event) all come out in `MultiMeta`.
- `eltq-core::convolve`: direct convolution, rustfft plans cached per worker,
  cost estimates. `Diagnostics::merge` for chunk folds. `multiunit`:
  `classify_unit_row` / `UnitPlan::from_row` (classification stored between
  passes, plans rebuilt in the hot pass), `marginal_pmf`, `LayerMeans`.
- PyO3 `multiunit_views` binding (integer event/unit codes, structured terms
  per unit code, zero-copy columns, compute detached) and the Python
  dispatch: identifiers factorized (pandas when available, a dict pass
  otherwise), terms parsed and grid-aligned in Python, diagnostics
  translated to `event_id/unit_id` examples, roll-up entries keyed by event
  id, `MultiMeta.engine` with the engine statistics. New
  `eltq.multiunit_views_into(..., out={view: array})` following
  `quantize_into` (memmap-capable; the arrays define the views and grid).
  `multiunit_views` gains `threads`, `chunk_rows`, `convolution`.
- Tests: Rust unit tests (atom event view values, independent event equals
  direct convolution with mass = rate, direct and FFT paths agree with
  masses, diagnostics and roll-up entries checked, bitwise invariance over
  threads and chunking, structural / range / overflow / no-view /
  misalignment errors) and `tests/multiunit_golden.rs` against
  `fixtures/golden/multiunit_small.*`. The Python multi-unit suite's `eltq`
  variant now runs the Rust engine end to end (views, semantics, policies,
  structure errors, goldens); `test_determinism.py` adds the multi-unit
  thread / chunk / repeat sweep and the direct-versus-FFT agreement.

## [1.9.0] - 2026-08-26

### Added

- [terms-grammar] `eltq-core::terms`: the Rust twin of `eltq.terms`. Hand
  rolled tokenizer for `unit <id> [<limit> xs <ded>] [occurrence net of
  <clause> (and <clause>)*]`, `Clause` / `UnitTerms` with the same
  validation (share bounds, positive finite layer limits, nonnegative
  attachments, combined share at most 100% in every band), the four view
  maps, canonical text rendering (`fmt_g15` reproduces Python's `%.15g`),
  `parse_terms` over text with comments and duplicate detection, and the
  grid-alignment check. Errors surface as `Error::Terms`.
- `ViewMap::compile(terms, view, h, n)`: the per-(unit, view) deposit map,
  at most two `(index, weight)` pairs per source bucket by the linear rule,
  with `support_len` (the reachable destination length, what each view's
  convolution size will be) and an `identity` flag (ground_up, and gross
  without policy terms) so identical maps can share one spectrum in 1.10.0.
  `apply` is a gather-scatter pass.
- Goldens: `fixtures/golden/terms_parse.json` now also carries the keyword
  vocabulary; `tests/terms_golden.rs` holds the Rust parser to the Python
  parses, rejections, canonical text and keywords. The `decl.lark` glance
  (aggregate_REFACTOR, 2026-08-26) confirmed `xs`, `po` (PART_OF),
  `occurrence net of`, and `and`-joined clauses; no drift.
- Rust unit tests: parsing of full and partial lines, share conventions and
  bounds, overlap guard, canonical text round trips and `fmt_g15` cases,
  text parsing and alignment, view values and compiled maps (mass and mean
  preservation, support lengths, identity detection).

## [1.8.0] - 2026-08-26

### Added

- [joint-two-unit] Rust two-unit joint in `eltq-core::multiunit`: unit
  preparation (`prepare_unit`: kinds atom / comonotone / beta, variance
  policies, copula weights), edge scores `y_e = Phi^{-1}(I_{edge/M}(alpha,
  beta))` computed once per unit, conditional matrices `(Q x n)` by adjacent
  differences of `Phi((y_e - w_c z) / w_I)` (atoms and comonotone units by
  linear deposits, the latter through the batched quantile walk at the
  ascending targets `Phi(w_c z_q)`), and the joint `C_1^T diag(w) C_2` as
  one `matrixmultiply::dgemm`. `JointMeta` mirrors the Python dataclass.
- `eltq-core::normal`: normal cdf via `libm::erfc` (the fdlibm port, below
  one ulp; statrs's erfc loses ~1e-11 relative in the tails, caught by the
  edge-score round trip) and Wichura's AS241 quantile (~1e-16 relative).
  `eltq-core::quadrature`: midpoint quantile nodes and Gauss-Hermite by
  Golub-Welsch with a first-row-only implicit-shift QL (O(Q^2), stable at
  any Q, symmetrized bitwise).
- `batched::beta_quantiles(alpha, beta, ascending_targets, out)`: the
  midpoint kernel's warm-started walk exposed for arbitrary ascending
  targets (lower half direct, upper half on the reflected shape with exact
  complements); the comonotone path is a sorted quantile walk, no FFT.
- PyO3 `joint_two_units` binding (zero-copy `(n1, n2)` pmf, grids, meta
  dict, compute detached from the interpreter); `eltq.joint_two_units` now
  runs on Rust when the extension is present and on the reference otherwise.
  `GridOverflowError` raised by the Rust backend now carries
  `required_length` and `examples` like the reference's.
- Tests: Rust unit tests for the normal functions, both quadrature rules
  (moments exact, closed forms, symmetry), unit preparation and policies,
  conditional rows as distributions, GEMM versus the naive triple loop,
  independent units as an outer product, typed errors; `tests/joint_golden.rs`
  against `fixtures/golden/joint_small.*` (rtol 1e-9). The Python multi-unit
  suite's `eltq` variant of every joint test now exercises the Rust path.

## [1.7.0] - 2026-08-26

### Added

- [reference-n-unit] `eltq.reference_multiunit`: the SciPy reference for
  joint multi-unit secondary uncertainty, the oracle for the Rust engine
  and the backend of the new API until it lands. `secondary_two_units`
  (the two-unit joint of `dev/secondary_two_units_reference.py`, promoted
  into the package; `dev/` copy retired) and `multiunit_views_reference`:
  per event, per unit, conditional pmfs given the shared factor (beta cdf
  differences at half-bucket edges; linear deposits for atoms, comonotone
  units and every transformed view value), per-view transform, convolution
  across units, shared-factor quadrature, rate-weighted accumulation across
  events. Per-view mass, means, analytic beta layer means (the LEV identity,
  closed form via the incomplete beta) and their reconciliation, plus the
  optional roll-up diagnostic (exact total versus rolled single-row beta:
  sd and q99 ratios per event) in the new `MultiMeta`; `JointMeta` for the
  joint. Unit-level variance policies (clamp / error / drop / atom) and the
  one-unit diagnostic categories carry over with `event_id/unit_id`
  examples; misgrouped events, rates varying within an event, and repeated
  units raise `MultiUnitError`.
- [terms-grammar] `eltq.terms` (pure Python, shared by both backends): the
  DecL-style grammar `unit <id> [<limit> xs <ded>] [occurrence net of
  <clause> (and <clause>)*]`, `clause := [<share> po] <limit> xs <attach>`,
  aligned with `aggregate`'s `decl.lark` (`xs`, `po` = part of, `occurrence
  net of`, `and`). Percent shares are fractions of the layer, plain-number
  shares are amounts of the layer limit. `UnitTerms` / `Clause` with the
  four view maps (ground_up, gross, ceded, net), canonical text rendering,
  structured-input parsing, and the grid-alignment check (breakpoints must
  be multiples of `bucket_size`; no silent rounding). Clauses may overlap,
  but their combined share must not exceed 100% of gross in any band
  (`TermsError` otherwise): beyond that ceded exceeds gross and net
  decreases or goes negative, which no loss grid can represent. This
  tightens the plan's "clauses may overlap" wording; found by the suite.
- Python API: `eltq.multiunit_views(...)`, `eltq.multiunit_views_dataframe`,
  `eltq.joint_two_units(...)`, `eltq.parse_terms`, exported `UnitTerms`,
  `Clause`, `VIEWS`, `TermsError`, `MultiUnitError`, `MultiMeta`,
  `JointMeta`. Reference exceptions are translated to the package's classes
  on the Rust backend.
- Shared-factor quadrature: midpoint quantile nodes (default) or
  Gauss-Hermite (`z_rule="gauss_hermite"`, Golub-Welsch nodes, stable at
  any Q). Convergence study `benchmarks/z_convergence.py` with its report
  `benchmarks/z-convergence-1.7.0.md`: Gauss-Hermite is spectral on smooth
  units (marginal error 1e-10 at Q = 32 versus 1e-3 midpoint) but leaves
  artifacts where the conditional is step-like in z (correlation-dominant
  and comonotone units); the midpoint rule keeps the better pmf shape
  there, so it stays the default at Q = 32 (plan decision 5 validated).
- Fixtures and goldens: `fixtures/multiunit_small_elt.csv` +
  `multiunit_small_terms.txt` (every unit kind, a zero-rate event, a
  dropped row, boundary atoms, four-unit events) with per-view goldens;
  `fixtures/golden/joint_small.*` (two-unit joint at Q = 1001);
  `fixtures/golden/terms_parse.json` (parse goldens plus rejected lines)
  that the Rust parser is held to as well.
- Tests: `tests/test_terms.py` (grammar, shares, canonical text, view
  semantics, alignment, goldens) and `tests/test_multiunit.py` run against
  the reference and the package API through the `MultiImpl` registry in
  conftest: joint mass / marginals / independence outer product /
  comonotone / atom / errors / golden; view semantics on atoms (ded, limit,
  tower, amount and percent shares, overlap, off-grid fractional cession);
  independent event equals plain convolution; comonotone common-shape event
  reproduces the rolled beta (additivity Proposition 6); per-event mass
  exactly the rate for every view; view means reconcile to analytic layer
  means within the stated (h, Q) tolerance; normalize; row policies and
  diagnostics; structural and overflow errors; golden fixture agreement.

## [1.6.0] - 2026-08-26

### Changed

- **[sigma-rule] Behavioral change: the total conditional standard deviation
  is now the arithmetic sum σ = σ_I + σ_C** (Burke, Rau-Chaplin, Varghese
  2016, section III.E, step i), replacing the quadrature rule
  σ² = σ_I² + σ_C² used by every earlier version. The quadrature combination
  was incorrect; it is replaced outright with no legacy option (git history
  serves any before/after comparison). Every row with both components nonzero
  now gets a wider secondary distribution, and rows whose summed sd exceeds
  the beta bound now fall under the variance policy (clamped by default). The
  admissibility test itself is unchanged: σ² against μ(M − μ). Applied in
  `validate.rs::classify_row` and `eltq.reference`; spec section 4 working
  formula replaced and section 11 assumption 3 marked resolved with a pointer
  to `dev/secondary-uncertainty-additivity.md` (the one-factor model, the
  additivity proofs, and the numerical illustration behind the change).
- Fixture and benchmark generators split a row's total sd linearly
  (`sd_indep = σ·share`, `sd_corr = σ·(1 − share)`) instead of in quadrature,
  so the variance-fraction regime mix keeps its meaning under the new rule;
  goldens regenerated (values shift wherever both sds are nonzero).

### Added

- `dev/done/plan-multiunit.md`: the plan for 1.6.0-1.11.0 (joint multi-unit
  secondary uncertainty, DecL-style unit terms, per-view totals), plus its
  mathematical companions `dev/secondary-uncertainty-additivity.md` and
  `dev/secondary_two_units_reference.py`.

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
