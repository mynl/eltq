# Plan: eltq — ferociously fast ELT secondary-uncertainty quantization

Date: 2026-08-14
Status: approved, awaiting go-ahead for milestone 1.0.0

## Context

`secondary-uncertainty-mathematical-specification.md` defines the mathematical contract:
convert a ~1M-row portfolio ELT into two evenly spaced, rate-weighted loss-grid
distributions — `raw` (atom at μⱼ per event) and `secondary` (K midpoint beta quantiles
per event) — via a mass- and mean-preserving linear deposit rule. The GPT brief
(`gpt/secondary-uncertainty-rust-build-brief.md`) proposes a Rust/PyO3 build. This plan
merges that brief with Claude's independent design. Primary requirement: **ferociously
fast**, without compromising validation, determinism, or Python usability.

Decisions already made by Steve:

- Package/crate/import name: **`eltq`**.
- Default inadmissible-variance policy: **clamp σ** to c·√(μ(M−μ)), c = 0.999, fully
  reported (strict-error, drop-and-report, degenerate-atom selectable).

## Workload analysis (what actually costs time)

1M rows × K=32 = **32M inverse-beta solves**. Each solve is a few regularized
incomplete-beta CDF evaluations (Lentz continued fraction, ~50–200 ns each). Everything
else — CSV parse (~100 MB text), 33M two-element grid deposits, reduction — is
sub-second territory. So the plan optimizes, in order: (1) CDF evaluations per
quantile, (2) thread scaling, (3) parse overlap. Kernel design is the heart of this
plan; the rest is disciplined engineering.

## Architecture

Cargo workspace + maturin. Two crates to start (KISS; CLI deferred until wanted):

```
secondary-corrosion/
├── Cargo.toml               # workspace
├── pyproject.toml           # maturin backend (see Build notes)
├── crates/
│   ├── eltq-core/           # pure Rust: no PyO3/NumPy/CLI types anywhere
│   └── eltq-python/         # PyO3 + numpy crate: conversion + error translation only
├── python/eltq/             # thin pure-Python: pandas/path conveniences, __init__
├── tests/                   # pytest + SciPy reference + golden fixtures
├── fixtures/                # synthetic ELTs only (data firewall)
├── benchmarks/              # bench harness + machine-readable results history
├── dev/                     # plans (move to dev/done when complete)
├── README.md / CHANGELOG.md
```

`eltq-core` owns: schema, validation, beta parameterization, the inverse-beta kernel,
grid deposit, deterministic parallel reduction, diagnostics. Rust unit tests +
criterion benches live with it.

## Core data model

- Structure-of-arrays input: `rate, max_loss, mean_loss, sd_indep, sd_corr` as
  `&[f64]`; event IDs optional (row index fallback), never in the hot path.
- All math in `f64`. No `f32` without a benchmark plus end-to-end error analysis.
- `Config`: `bucket_size (h)`, `grid_length (n, default auto = ceil(max M/h)+1)`,
  `quadrature_points (K, default 32)`, `normalize (default false: rate-weighted)`,
  `mean_is_loss_rate (default false; if true μ is a rate on [0,1], scaled by M)`,
  `variance_policy (default ClampSigma{c: 0.999})`, `overflow_policy (default Error)`,
  `deterministic (default true)`, `threads`, `memory_budget`, `tolerance (default
  full precision; relaxed grid-relative mode opt-in)`.
- `QuantizeResult`: `raw: Vec<f64>`, `secondary: Vec<f64>`, metadata (h, n, K,
  normalized?, Λ, analytic AAL, both output masses and first moments, finite-K
  secondary mean error, valid-row count) and per-category diagnostics (counts + ≤10
  representative event IDs each: nonfinite, negative field, μ>M, σ-clamped,
  zero-rate, degenerate-atom, overflow).
- Typed errors (`thiserror`); invalid rows never panic; no NaN/Inf/negative weight can
  reach the outputs.

## Streaming pipeline (single pass, both histograms)

Per fixed-size chunk (default 64Ki rows): validate + classify → raw deposit at μⱼ →
beta (αⱼ, βⱼ) for nondegenerate rows → K midpoint quantiles into a stack/reused buffer
→ scale by Mⱼ → secondary deposits immediately → discard buffer. Never allocate N×K.
Accumulate Σλ, Σλμ, and realized secondary moment Σ(λ/K)Σxⱼₖ in-pass for diagnostics
(pairwise/compensated for validation totals). Grid coverage checked upfront when max M
is available before compute (array route; CSV route detects during streaming and
reports without unsafe indexing).

## Inverse-beta kernel (the heart)

Small interface, swappable implementations, explicit status returns.

1. **Reference kernel first**: scalar `statrs::Beta::inverse_cdf`. Full-domain
   correctness vs SciPy before any optimization. Stays available as a test oracle.
2. **Own incomplete-beta CDF**: cephes-style Lentz continued fraction with the
   symmetry switch at x > (α+1)/(α+β+2); per-event constants (ln B(α,β) via
   `ln_gamma`) computed **once** per event, not per probability.
3. **Batched per-event kernel** `beta_midpoint_quantiles(α, β, K, tol, &mut out)` —
   walk the sorted midpoints u₀ < … < u_{K−1} using the quantile ODE
   `dQ/du = 1/f(Q)` as predictor: second-order Taylor step from x_k (both derivatives
   analytic; `f'/f = (α−1)/x − (β−1)/(1−x)`), then 1–2 **safeguarded Halley**
   iterations with a maintained bracket [x_k, 1) so hard shapes cannot diverge.
   Cold-start only u₀ (AS 109-style or tail closed form). Expected ~1–2 CDF
   evaluations per quantile vs ~5–8 cold. Tail asymptotes for extreme shapes
   (α or β ≪ 1): x ≈ (uαB(α,β))^{1/α} as start, log-space near endpoints.
4. **Fast paths**: σ=0 / μ=0 / μ=M → atoms; α=β=1 uniform → x_k = M(k+½)/K exact;
   α=β symmetric → solve K/2, mirror; M=0 → atom at 0.
5. **Tolerance**: default effectively machine precision. Opt-in relaxed mode stops on
   grid-relative accuracy (h/Mⱼ-scaled), with the induced histogram-weight error bound
   (≤ w·|Δx|/h per deposit) tested, not assumed. No adaptive K, no lookup tables in
   the default mode (they change the stated quadrature); only as explicit alternative
   modes later, if ever.
6. SIMD: not assumed to help (branchy, divergent convergence). Only after profiling
   shows batch uniformity. Same posture for GPU: bounded spike after CPU release, off
   the critical path.

## Grid deposit

Tiny `#[inline]` function: `r = x * inv_h; i = floor(r); hist[i] += w·(i+1−r);
hist[i+1] += w·(r−i)`; exact-node and top-endpoint (i = n−1) handled explicitly; can
never index out of bounds. Reciprocal spacing precomputed; FMA measured before
adoption (reproducibility).

## Parallelism + bitwise determinism

Rayon threads. Fixed chunk grid; each chunk produces **private** raw+secondary partial
histograms in parallel (no locks, no atomics); partials folded **sequentially in
chunk-index order** → bitwise-deterministic result independent of thread count and
work stealing. Memory: bounded chunks-in-flight from `memory_budget` (n=100K grid →
~3.2 MB/chunk pair; 1M rows/64Ki = 16 chunks ≈ 51 MB — fine). Large-n fallback:
per-worker histograms with static contiguous partition (deterministic for fixed T;
documented narrower guarantee). Nondeterministic faster mode only if benchmarks show
material gain.

## Input routes

- **Array route** (primary for Steve's workflow): contiguous 1-D `float64` NumPy
  arrays borrowed without copy; wrapper does explicit contiguous conversion or clear
  rejection. Parquet/feather → pandas/pyarrow → arrays happens in Python.
- **CSV route**: Rust `csv` crate streaming into reusable column buffers,
  `fast-float`-style parsing, configurable column names + strict canonical schema,
  bounded diagnostics. Parse thread overlaps compute via chunk pipeline. Faster
  formats (Arrow/Parquet in Rust) only if profiling says input is material.

## Python API + packaging

```python
raw, secondary, meta = eltq.quantize_arrays(rate, max_loss, mean_loss, sd_indep,
    sd_corr, *, bucket_size, grid_length=None, quadrature_points=32, **options)
raw, secondary, meta = eltq.quantize_csv(path, *, bucket_size, ...)
meta = eltq.quantize_into(source, *, raw_out, secondary_out, ...)   # memmap-capable
eltq.quantize_to_npy(source, path, ...)   # (2,n) .npy + adjacent JSON metadata,
                                          # transactional write; reopen via np.load(mmap_mode="r")
```

- Compute detached from the interpreter (PyO3 `allow_threads`); results move
  `Vec<f64>` → NumPy via `IntoPyArray` (no completed-array copy).
- `quantize_into` validates dtype/shape/contiguity/writability incl. `np.memmap`;
  compute into Rust-owned buffers then one final copy into borrowed memory (no
  releasing protection around borrowed writable memory).
- Metadata as a small dataclass-like object; Rust errors → specific Python exceptions.
- pandas never a dependency of the extension; `python/eltq` has an explicit
  DataFrame convenience.
- **maturin** build backend, `abi3-py313` wheel, Windows first. uv-managed venv.

## Build/environment notes (Steve-specific)

- **hatchling → maturin**: standing instructions say hatchling, but hatchling cannot
  build Rust extensions; maturin is the pyproject build backend here. (Flagged; this
  plan assumes agreement.)
- **V:\dev**: venv symlinked per standing practice, **and** `CARGO_TARGET_DIR =
  V:\dev\cargo-target\secondary-corrosion` via `.cargo/config.toml` — Rust `target/`
  churn on a Synology-synced folder is far worse than venv churn.
- `.gitignore` (currently empty): target/, .venv, wheels, benchmark scratch, data
  files except `fixtures/`, `.env`. Sanity-check staged list before each commit.
- Config via TOML where the tool needs any; no env vars.

## Testing & validation

- Python/SciPy **reference implementation** first (`tests/reference.py`): short,
  transparent, `scipy.stats.beta.ppf`; generates golden fixtures (small/medium
  synthetic ELTs, committed under `fixtures/`).
- All 14 acceptance tests from spec §9 (zero uncertainty, exact nodes, linear-rule
  mass+moment, uniform beta, symmetry, moment reconstruction, K-convergence, mass
  conservation Σ=Λ, mean conservation, scale invariance, invalid-moment policy,
  boundary means, grid overflow, large-file reference).
- Rust proptest: mass conservation, deposit mean preservation, scaling, admissible
  beta round-trip; extreme shapes (α,β ≪ 1 and ≫ 1), u near 0/1, tiny/huge h/M,
  exact-M endpoints, zero-rate rows, duplicate event IDs.
- Relaxed-tolerance mode: measure quantile error **and** induced histogram error
  (quantiles can cross grid boundaries while numerically close).

## Benchmarks

Criterion for kernels; Python harness for end-to-end; results appended to a
machine-readable JSON history in `benchmarks/` with CPU/cores/memory/compiler/flags/
Python/grid/K/format recorded. Measure: CSV parse alone; beta parameterization;
scalar vs batched kernel by parameter regime; deposits alone; end-to-end in-memory and
memmap; scaling at 1/2/4/8/all physical cores; peak RSS; deterministic-reduction cost;
Python boundary cost. Headline benchmark: **1M rows, K=32**, rows/s and quantiles/s,
speedup vs SciPy reference and vs scalar Rust baseline. Correctness gates performance:
no benchmark accepted from a build failing acceptance tests.

## Milestones → versions (semver, CHANGELOG per version, one-line commits, no push)

- **1.0.0 [reference]** Repo scaffold, .gitignore, workspace, SciPy reference +
  golden fixtures, acceptance-test skeleton, README/CHANGELOG.
- **1.1.0 [rust-core]** eltq-core scalar path: validation/policies, statrs reference
  kernel, deposits, diagnostics; all golden + property tests green.
- **1.2.0 [python]** PyO3/maturin bindings, `quantize_arrays`/`quantize_csv`,
  zero-copy returns, Windows wheel via uv+maturin.
- **1.3.0 [memmap]** `quantize_into` + `quantize_to_npy` (transactional, (2,n)).
- **1.4.0 [ferocious]** Rayon chunk parallelism + deterministic fold; own
  incomplete-beta; batched ODE-warm-start Halley kernel; fast paths; tuning from
  benchmarks; 1M-row performance report committed.
- **1.5.0+ [optional]** Relaxed grid-tolerance mode, SIMD/format investigations, GPU
  spike — each only with profiling justification.

## Resolved / open assumptions (spec §11)

Resolved: λ in ELT directly (joins are upstream prep); σ² = σ_I² + σ_C²; default
output rate-weighted (normalization explicit); h required, n auto-derived default,
K default 32; clamp-σ default policy; no finite-K moment correction in default mode
(optional later, separately identified); no overflow bin — full coverage or error.
Open (confirm when real file spec available): exposure field ≡ Mⱼ.

## Verification

- `uv run pytest` — acceptance suite vs SciPy reference + golden fixtures.
- `cargo test -p eltq-core` — unit + property tests.
- `uv run python benchmarks/run_end_to_end.py` — 1M-row synthetic, prints rows/s,
  quantiles/s, mass/moment diagnostics; results file appended.
- Determinism check: same input twice + thread-count sweep → bitwise-identical arrays.
- `maturin develop` then `import eltq; eltq.quantize_arrays(...)` smoke test on
  Windows/Python 3.13.
