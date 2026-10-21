# Plan: multiunit-fixes — four defects in the multi-unit route

Date: 2026-10-21
Status: in execution from 2026-10-21 (Steve: "execute in four 0.0.1 bumps"); see the execution log.

## Goal

Fix four confirmed defects in `multiunit_views` / `joint_two_units` (the
multi-unit route, 1.7.0–1.11.0). One drops probability mass without telling
anyone, one crashes, one breaks the nonnegativity contract, and one aborts runs
at large quadrature counts. The one-row `quantize` route is not affected by any
of them.

Each fix is one step, one patch bump, one commit, so each release can be
reverted alone: 1.11.1 to 1.11.4 in the order below, most serious first. The
docstring fix in [tidy] is pure tidying and does not bump.

## Background for a reader with no context

`eltq` is a Rust engine (`crates/eltq-core`) with PyO3 bindings
(`crates/eltq-python`) and a pure-Python SciPy reference
(`python/eltq/reference_multiunit.py`). The reference is the oracle: the
acceptance suite (`tests/`) runs every test against both backends through the
`multi` fixture. The mathematical contract is
`secondary-uncertainty-mathematical-specification.md`; the multi-unit model
is described in `dev/secondary-uncertainty-additivity.md` and the module
docstring of `reference_multiunit.py`.

Multi-unit model, in brief: per event, rows are units. Unit i has a beta
damage ratio on [0, M_i] moment-matched to (μ_i, σ_I,i + σ_C,i). One shared
standard normal Z_c per event and an independent Z_i per unit give
U_i = Φ((σ_C,i Z_c + σ_I,i Z_i) / t_i). Conditional on Z_c = z the units are
independent, so the engine:

1. builds each unit's conditional pmf on the grid k·h for each of Q
   quadrature nodes z_q (cdf differences at half-bucket edges for beta units,
   a linear deposit for atoms and for comonotone units with σ_I = 0);
2. maps each unit's rows through its terms to get the views
   (ground_up, gross, ceded, net);
3. convolves across units per view, directly or by FFT (`convolution=` is
   `auto`, `direct` or `fft`; `auto` picks by estimated cost);
4. mixes over z with the quadrature weights and adds λ × (event pmf) into
   the output arrays.

Units with no terms at all are "bare". The engine convolves them once per
event into a shared block that ground_up, gross and net reuse. Ceded never
uses bare units.

All four defects below were reproduced on 2026-10-21 against the 1.11.0
build (scratch scripts, not committed).

## [tail-support] 1.11.1 — grid sized too short, tail mass silently dropped

**Current behavior.** The event's grid requirement is computed from the sum
of real-valued unit supports, n = ⌈Σ s_i / h⌉ + 1, where s_i is μ_i for an
atom unit and M_i otherwise. But each unit's own pmf has length
⌈s_i / h⌉ + 1, so the convolution has length Σ⌈s_i / h⌉ + 1, which can
exceed n. In the deposit loop, `views.rs:961` zips the event pmf against the
output array, and the `zip` stops at the shorter length, which drops the
tail entries silently. `max_mass_error` measures the event pmf before
deposit, so it still reports 0.

Repro: one event, two units, μ = 13, σ_I = σ_C = 0, M = 20, h = 10, rate 1.
Each unit is an atom deposited as [0, 0.7, 0.3]. The true ground-up pmf is
[0, 0, 0.49, 0.42, 0.09] (mean 26). The Rust engine returns
[0, 0, 0.49, 0.42], mass 0.91, mean 22.4. The reference raises a NumPy
broadcast `ValueError` at `reference_multiunit.py:682` instead. The two
backends disagree, and both are wrong. Three beta units with M = 16 and
h = 10 lose an entry the same way. An explicit `grid_length` that passes the
overflow check truncates the same way. The spec forbids silent truncation
(§6). No fixture catches this because every fixture's supports are multiples
of h.

**Change.** Measure event support in buckets, not money. The event span is
Σ_i (⌈s_i / h⌉), and the grid needs span + 1 nodes.

- Rust, `views.rs` pass 1 (`:806-841`): accumulate an integer
  `span += grid_length_for(s, h) - 1` per live unit. Then
  `auto_length = max_span + 1`. The explicit-length check becomes
  `max_span + 1 > n`, and `GridOverflow` reports
  `required = max_span + 1` and `max_loss = max_span as f64 * h`. Event
  example rows are selected by span. Per-unit lengths at `views.rs:915`
  already use `grid_length_for(plan.support(), h)` and stay consistent.
- Rust, deposit (`views.rs:961`): replace the silent `zip` with an
  explicit invariant `assert!(pmf.len() <= n, ...)` before the loop, so any
  future sizing mismatch fails loudly instead of dropping mass.
- Reference, `reference_multiunit.py:621-651`: the same span arithmetic
  (`span += grid_length_for(prep.support, h) - 1`, `n = max_span + 1`).
  Error messages quote the span in buckets and `required_length`.
- Docstrings: `python/eltq/__init__.py:574` and `reference_multiunit.py:531`
  describe the auto length as "the largest event support (sum of unit
  maxima)". Change that to "the largest event span: each unit's support
  rounded up to the grid, summed".

**Tests** (`tests/test_multiunit.py`, both backends):
- the two-atom repro returns exactly [0, 0, 0.49, 0.42, 0.09], mass 1, mean 26;
- three beta units with M = 16, h = 10: mass 1 within 1e-12 per view;
- the same event with `grid_length=4` raises `GridOverflowError` with
  `required_length == 5`.

## [fft-ceded] 1.11.2 — FFT path panics when no computed view uses the bare block

**Current behavior.** `fft_views` builds a unit's spectra only for row sets
some computed view needs (`views.rs:612-627`). Ceded never uses bare units,
so with `views=("ceded",)` a bare unit has no spectra. The shared-block loop
(`views.rs:631-645`) runs whenever a bare unit exists and slices that empty
spectrum: `PanicException: range end index 17 out of range for slice of
length 0`.

Repro: units a and b in one event, terms only on a
(`unit a 1000 xs 0 occurrence net of 50% po 500 xs 0`), `views=("ceded",)`,
`convolution="fft"`. Direct gives the correct answer. `auto` can also reach
this: `estimate` (`views.rs:422-436`) charges the unused shared block to the
direct path, so a few large correlated bare units tip the choice to FFT.

**Change.** Compute one flag per event,
`shared_needed = bare_exists && computed.iter().any(|&v| v != View::Ceded)`,
and use it in all three places:
- `fft_views`: build `shared_spec` only when `shared_needed`;
- `direct_views`: build `shared_indep` / `shared_corr` only when
  `shared_needed`. This is wasted work today, not a bug;
- `estimate`: charge the shared-block cost (`c_ind + q * c_corr` and
  `shared_products`) only when `shared_needed`.
The per-view `uses_shared` stays as it is (`bare_exists && view != Ceded`),
because it is true only when `shared_needed` is.

**Tests:**
- Python, both backends: for each single view, and for
  `("ceded", "net")`, on an event with one termed unit and two bare units
  (one beta and one comonotone with σ_I = 0), `convolution` in
  {direct, fft, auto} agree with each other within 1e-13 and with the
  reference within the existing tolerance. This also covers a test gap: no
  existing test puts a comonotone unit in the shared bare block.
- Rust unit test in `views.rs`: the same event with `views = [Ceded]` and
  `Convolution::Fft` returns `Ok`.

## [fft-nonnegative] 1.11.3 — FFT path returns negative weights

**Current behavior.** `FftCache::inverse_into` (`convolve.rs:130-132`)
copies the inverse transform out unclamped. Bins whose exact value is 0
(gaps in a ceded tower, the region below a deductible) carry symmetric
round-off of order 1e-17, so about half of them come out negative. On a
200-event, 4-unit test with policy and reinsurance terms, about 22 bins per
view were negative (min ≈ −4e-18). The direct path returned none. The spec
requires nonnegative weights (§7, §8). `tests/test_determinism.py:77` has a
tolerance of atol 1e-14, which is loose enough to let these through.

**Change.** Clamp at zero on the way out:
`out.extend(self.real[..len].iter().map(|v| (v * scale).max(0.0)))`.
The mass change is at most the sum of the clamped magnitudes, around 1e-16
per event, far below every stated tolerance. Clamping is per element, so the
determinism contract is unaffected. A threshold relative to the largest bin
was considered and rejected: it adds a tuning constant to fix an error that
is already at machine precision.

**Tests:** the multi-unit golden fixture and the 200-event case with
`convolution="fft"`: every view has `min() >= 0.0`. Add the same assertion to
the existing FFT-forced tests.

## [gauss-hermite-ties] 1.11.4 — equal interior targets abort the run

**Current behavior.** `batched::beta_quantiles` rejects equal consecutive
interior targets (`batched.rs:92-97`). For a comonotone unit (σ_I = 0,
so w_c = 1) the targets are Φ(z_q). For z in roughly [8.14, 8.29], Φ(z)
evaluates to exactly 1 − 2⁻⁵³. Gauss-Hermite nodes reach that range once Q
is in the hundreds, and two adjacent nodes can land on the same value. The
whole call then fails with "beta quantile evaluation failed". Repro:
`joint_two_units((200, 0, 250, 1000), (500, 300, 200, 2000), bucket_size=10,
n1=101, n2=201, z_rule="gauss_hermite", z_nodes=Q)` fails at Q = 700 and
Q = 1001, and at 19 of the 37 node counts from 600 to 2400 in steps of 50.
The z-convergence study's reference size is Q = 1001. SciPy accepts ties, so
the reference never fails. Tests only go up to Q = 64. The midpoint rule is
immune because its largest node is Φ⁻¹(1 − 1/(2Q)), about 3.5 at Q = 4001.

**Change.** Accept non-decreasing targets. In `beta_quantiles`, solve only
the first of each run of equal interior targets, then copy that quantile to
the rest of the run, so `walk_map` keeps seeing strictly increasing targets.
The validity check becomes `w[1] < w[0]` only. Update the doc comment above
`beta_quantiles`, and the `walk_map` comment, to state the tie handling.

**Tests:**
- `batched.rs` test at `:415`: `[0.4, 0.4]` now returns two equal quantiles
  instead of an error. Add `[0.1, 0.4, 0.4, 0.4, 0.7]`, with the middle three
  equal and matching a lone solve at 0.4.
- Python, both backends: the `joint_two_units` repro at Q = 700 and
  Q = 1001 succeeds with total mass 1 within 1e-12. Extend
  `test_gauss_hermite_with_comonotone_units_at_large_q` to cover
  Q = 1001 on `multiunit_views` as well.

## [tidy] no bump — docstring default

`python/eltq/__init__.py:592` (`multiunit_views`) gives the `chunk_rows`
default as 65536. The Rust default is 2048 (`DEFAULT_MULTI_CHUNK_ROWS`,
`views.rs:53`). Chunk size is part of the determinism contract, so the
docstring must match. Line 219 (`quantize`, 65536) is correct and stays.

## Files

- `crates/eltq-core/src/views.rs` — [tail-support], [fft-ceded]
- `crates/eltq-core/src/convolve.rs` — [fft-nonnegative]
- `crates/eltq-core/src/batched.rs` — [gauss-hermite-ties]
- `python/eltq/reference_multiunit.py` — [tail-support]
- `python/eltq/__init__.py` — docstrings ([tail-support], [tidy])
- `tests/test_multiunit.py`, `tests/test_determinism.py` — new tests
- `Cargo.toml`, `pyproject.toml` (both carry the version; `Cargo.lock`
  follows on build), `CHANGELOG.md` — each bump

## Acceptance

Per step, before its commit:
1. `uv sync` (rebuilds the wheel), `uv run pytest`, and
   `cargo test -p eltq-core --release` all pass, including the new tests.
2. The golden fixtures pass without regeneration. Their supports are on the
   grid, so [tail-support] does not change them, and the clamp in
   [fft-nonnegative] only removes values far below the golden tolerance. If
   a golden fails, stop and investigate. Do not regenerate to make it pass.
3. Version bumped in both `Cargo.toml` and `pyproject.toml`, with a
   CHANGELOG entry of a few bullets.

After [fft-nonnegative] and again at the end:
`uv run python benchmarks/run_multiunit.py`, with wall time within noise
(about 3%) of `benchmarks/report-1.11.0.md` on the same machine. None of
the fixes is expected to cost measurable time, and [fft-ceded] removes work
on ceded-only runs. Note: the 1.11.0 report was measured on the previous
machine (i7-7820HQ), so first take a 1.11.0 baseline on this one.

## Not in this plan

- Housekeeping found in the same review: the hard-coded
  `[build] target-dir` in `.cargo/config.toml`, the README "Environment"
  section prescribing the old Synology venv junction, and the missing
  multi-unit section in the spec. These go in a separate plan.
- The stratum-averaged Z_c conditionals (exact marginals, exact mean, about
  half the nodes for the same accuracy). That is an algorithm change and
  goes in its own plan, alongside `dev/plan-chebyshev-z.md`.
- Lower-priority items from the review, deferred: no upper bound on grid
  length (M/h ≥ 2⁶⁴ wraps; M/h around 1e10 tries a multi-GB allocation);
  a NaN rate over several rows of one event is a nonfinite diagnostic in
  Rust but a `MultiUnitError` in the reference; the Rust terms number
  parser rejects `1_000`, which Python accepts.

## Execution log

Review 2026-10-21 against 1.11.0 (`440ce86`): every cited path, line number and symbol matched the code. Baseline before any change: `uv run pytest` 141 passed, `cargo test -p eltq-core --release` 77 passed.

Divergences recorded at review:

- The version also lives in `python/eltq/__init__.py` (`__version__`), which the Files list omits; each bump updates it with `Cargo.toml` and `pyproject.toml`. The project `CLAUDE.md` line "Currently 1.11.0 in both" is kept current at each bump and now names the third file.
- The reference takes no `convolution=` keyword, so the [fft-ceded] path sweep runs on the package API and compares each path with the reference; the reference itself is the direct oracle.
- The [fft-nonnegative] "200-event case" was a scratch script, not committed. A seeded synthetic equivalent (200 events, 4 units, policy and reinsurance terms) is built inside the test.
- Benchmarks run with `--no-append`; the numbers are recorded here rather than in `benchmarks/results.json`.

Benchmark baseline, 1.11.0 on DOOB (32 threads), `uv run python benchmarks/run_multiunit.py --no-append`, 909,937 rows, event grid 874: all four auto Q = 32 2.03 s; ground_up only 1.42 s; Q = 8 0.70 s; forced direct 2.22 s; forced FFT 8.90 s; Gauss-Hermite 2.14 s; threads 1 / 2 / 4 / 32 25.13 / 12.99 / 6.68 / 2.02 s; joint Q = 1001 3.2 ms, Q = 4001 12.2 ms; peak RSS 393 MiB; bitwise identical across threads.

**1.11.1 [tail-support]**, 2026-10-21. Landed as specified. The new tests were run against the 1.11.0 build first and failed on both backends as the plan describes (Rust grid 4 instead of 5; reference broadcast `ValueError`). Divergences: the two-atom pmf is checked with `atol=1e-15` rather than exact equality, since 0.7 × 0.7 does not round to 0.49 in binary; the three-beta test uses μ = 12 (σ_I = 2, σ_C = 1) on M = 16 rather than an unspecified mean, so the dropped top bucket carries visible mass (the test also asserts it is above 1e-6). The `GridOverflowError` message in the reference now quotes the span in buckets; the Rust message keeps its wording with `max_loss = span × h`. Gate: `uv run pytest` 145 passed, `cargo test -p eltq-core --release` 77 passed, goldens unchanged.
