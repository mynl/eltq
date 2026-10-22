# Plan: pypi-release, eltq on PyPI as prebuilt wheels

Date: 2026-10-22
Status: ruled 2026-10-22 (see "Rulings"), not executed.

## Goal

Make `pip install eltq` (and `uv add eltq`) work on Windows, Linux and macOS without a Rust
toolchain, so that pages which use `eltq`, starting with `reproductions/Homer2017a.qmd`, can
build from PyPI packages alone and move into the aggregate Reproductions book. Do it without
losing the speed this machine gets from building for its own CPU.

## Background for a reader with no context

`eltq` is a Rust engine (`crates/eltq-core`) with PyO3 bindings (`crates/eltq-python`) and a
Python layer (`python/eltq`), built by maturin (`pyproject.toml`, `[build-system]`). Today it is
installed only from source: `uv sync` in this repository, or a path dependency, as in
`reproductions/pyproject.toml` (`eltq = { path = ".." }`). Building from source needs the Rust
toolchain.

A wheel is a prebuilt binary package. Its file name carries a Python tag, an ABI tag and a
platform tag, for example `eltq-1.11.4-cp312-abi3-win_amd64.whl`. The platform tag names the OS
and the CPU family (x86_64, arm64) but not the instruction set level within a family, so pip
cannot choose between an AVX-512 build and an AVX2 build of the same package. A wheel is
therefore compiled for one baseline, and anything above it must be chosen at run time inside the
binary. (A "wheel variants" proposal would add such tags; it is not usable yet.)

The binding is built against the Python stable ABI (`abi3` in `crates/eltq-python/Cargo.toml`),
so one wheel per platform serves every Python from the floor up; there is no wheel per Python
version. The only runtime dependency is numpy; SciPy is needed only by the pure-Python reference
backend (`reference.py`, `reference_multiunit.py`), which `__init__.py` loads only when the
compiled extension is missing.

PyPI trusted publishing lets a GitHub Actions workflow upload without any stored API token: PyPI
is told in advance which repository, workflow file and GitHub environment may publish a project,
and the workflow proves its identity with a short-lived OpenID Connect token. For a project that
does not exist yet, PyPI accepts a "pending publisher", and the first upload creates the project.

## Current state (read 2026-10-22)

- `.cargo/config.toml` sets `rustflags = ["-C", "target-cpu=native"]` and
  `target-dir = "V:/dev/cargo-target/secondary-corrosion"`. Both must change before any wheel
  leaves this machine. `native` here is Zen 5 (`znver5`, AMD Ryzen 9 9950X3D2), which includes
  AVX-512; a wheel built with it fails with an illegal-instruction error on any CPU without those
  instructions. The `target-dir` is a drive-letter path left over from the Synology era, which
  the `V:\dev` rules forbid (build churn stays in the checkout; no drive letters), and it would
  misbehave on a CI runner. The comment in the file still describes the old i7-7820HQ. The
  `[tool.uv] cache-keys` in `pyproject.toml` list `.cargo/config.toml`.
- `README.md`, "Environment", still describes a venv junction to `V:\dev\venvs` and the
  external target dir. `.venv` is in fact a plain directory in the checkout. The Performance
  section says the wheel is built with `target-cpu=native`. There is no Install section.
- No hand-written SIMD anywhere in `crates/eltq-core/src`: no `mul_add`, no
  `#[target_feature]`, no `is_x86_feature_detected!`. All vector speed comes from LLVM
  auto-vectorization under the `target-cpu` flag. `rustfft` (used through `realfft`) selects its
  AVX path at run time on its own, whatever the flags.
- No `LICENSE` file, although the Cargo workspace declares `license = "MIT"`. `pyproject.toml`
  has no `license`, `readme`, `authors`, project URLs or classifiers. `[build-system]` requires
  `maturin>=1.7,<2`; 1.14.1 is installed.
- `requires-python = ">=3.13"` and `abi3-py313`; `aggregate` 1.6.0 requires `>=3.12`.
- No CI, no `.github/` directory, no git tags. None of the sibling Rust projects under `V:\dev`
  (`aggregate-ntt`, `eisenstein`, `skimmatch`) has a workflow to copy, and `aggregate-ntt` is
  not on PyPI. Its own design (`aggregate-ntt/dev/plan-rust-engine.md`, [Rust-Build-Local])
  also wants `target-cpu=native` for local builds and a baseline for anything published.
- The GitHub repository `mynl/eltq` (renamed from `mynl/secondary_corrosion`, default branch
  `master`) was made public on 2026-10-22. An audit of every path in its history (18 commits, 90 paths) found data-like files
  only under `fixtures/` (synthetic by construction, `fixtures/generate_fixtures.py`) plus
  `benchmarks/results.json` (timings), and no credentials, e-mail addresses or user paths.
- The name `eltq` is unclaimed on PyPI (`https://pypi.org/pypi/eltq/json` returned 404 on
  2026-10-22).
- `tests/conftest.py` lines 106 and 217 pass `float_precision="round_trip"` to `pd.read_csv`,
  which pandas 3.1 deprecates with a warning (not an error).

## Measurement: what the CPU baseline costs

Measured 2026-10-22 on this machine (AMD Ryzen 9 9950X3D2, 16 cores, 32 threads, Windows 11,
Python 3.14.7, numpy 2.5.2), on `eltq` 1.11.4 built four ways. Method, so it can be repeated:

- Four release wheels from the same source, each with `RUSTFLAGS="-C target-cpu=<level>"`
  and its own `CARGO_TARGET_DIR`, by `maturin build --release`, for levels `native` (Zen 5,
  AVX-512 with full-width datapath), `x86-64-v4` (generic AVX-512), `x86-64-v3` (AVX2, FMA,
  BMI2; Intel Haswell 2013 and AMD Excavator 2015 onward) and `x86-64-v2` (SSE4.2).
- Each wheel in its own clean venv. A timing script reuses the seeded generators of
  `benchmarks/run_end_to_end.py` and `benchmarks/run_multiunit.py`, takes best of 3 per run
  (best of 5 for the two-unit joint), and hashes every output array.
- Five rounds, each running all four builds in rotated order; the table gives the median over
  rounds and its ratio to native.

| Workload | native | v4 | v3 | v2 |
|---|---:|---:|---:|---:|
| `quantize_arrays`, 1M rows, K = 32, all threads | 0.921 s | 1.00 | 0.98 | 0.98 |
| `quantize_arrays`, 200k rows, one thread | 2.398 s | 0.99 | 1.00 | 1.00 |
| `multiunit_views`, 100k events x 10 units, ground_up, all threads | 1.153 s | 1.07 | 1.08 | 1.38 |
| `multiunit_views`, 10k events x 10 units, one thread | 1.655 s | 1.09 | 1.08 | 1.35 |
| `joint_two_units`, Q = 1001 | 0.101 s | 0.99 | 1.00 | 1.00 |

Round-to-round spread was within 5% on every cell. Findings:

- **Every output is bitwise identical across all four builds and all twenty runs.** The
  determinism contract holds across instruction sets, as the comment in `.cargo/config.toml`
  claimed (no reordered reductions, no fused multiply-add without `mul_add`).
- **The headline route gains nothing from the instruction set.** `quantize_arrays` is the
  inverse-beta kernel, iterative scalar code that does not vectorize; v2 runs it as fast as
  Zen 5 native. The two-unit joint is FFT bound, and `rustfft` dispatches at run time.
- **The multi-unit route needs AVX2.** v2 is 35% to 38% slower than native; v3 is 8% slower.
- **AVX-512 itself buys nothing.** Generic v4 is no faster than v3. The 8% that native keeps
  over both comes from tuning for Zen 5 (instruction scheduling and the cost model), not from
  wider vectors, so it cannot be recovered by compiling extra AVX-512 copies of the kernels.
- For scale: the 1M-row headline that took ~4.3 s on the i7-7820HQ (`report-1.4.0.md`) takes
  0.92 s here.

## Measurement: what a Python 3.12 floor costs

Measured 2026-10-22, on a copy of HEAD (1.11.4) in a scratch directory with `abi3-py312` and
`requires-python = ">=3.12"`, built at `x86-64-v3`:

- `maturin build --release` against CPython 3.12.14 compiles without a warning and produces
  `eltq-1.11.4-cp312-abi3-win_amd64.whl`.
- In a clean 3.12.14 venv (numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, pytest 9.1.1) the wheel
  imports with `BACKEND == "rust"`, and the full suite passes: 151 tests, the same count as the
  project venv.
- The timing script of the CPU measurement gives the same SHA-256 for every output as the
  `abi3-py313` v3 build, with timings within noise.
- The Python sources use no feature newer than 3.12.

No impact found, so the floor drops to 3.12.

## Rulings (Steve, 2026-10-22)

1. **Visibility: public**, with an sdist. Done on 2026-10-22 (see Current state).
2. **Python floor: 3.12** (`abi3-py312`, `requires-python = ">=3.12"`), conditional on no
   impact; the measurement above found none. If any step turns up an impact, revert to 3.13
   rather than work around it.
3. **Platforms:** Windows x86_64, Linux x86_64 (manylinux), macOS arm64. No Linux aarch64, no
   macOS x86_64.
4. **CPU baseline: `x86-64-v3`** for the x86_64 wheels, no multiversioning.
5. **Bumps:** a patch bump each for [cargo-config], [packaging] and [ci], then the minor bump at
   [release], the first upload. ([cpu-baseline] is a ruling that [ci] implements.) Version
   numbers are assigned at execution; `dev/plan-chebyshev-z.md` may land in between.
6. **Repository name: `mynl/eltq`**, matching the package. Renamed from
   `mynl/secondary_corrosion` on 2026-10-22, before the trusted publishers were configured,
   since the package page links to the repository and the publisher is keyed by repository
   name. GitHub redirects the old URL; the local remote `mynl` points at
   `https://github.com/mynl/eltq.git`. The local directory stays `secondary-corrosion`. The
   package name `eltq` is kept: a PyPI name is permanent once uploaded.

## Steps

### [cargo-config] build settings that travel

- Remove `rustflags` and `target-dir` from the committed `.cargo/config.toml`; nothing else is
  in it, so delete the file, and its entry in `[tool.uv] cache-keys`. The build goes to
  `./target`, already in `.gitignore`.
- Keep native speed on this machine without committing it: create `V:\dev\.cargo\config.toml`
  holding `[build] rustflags = ["-C", "target-cpu=native"]`. Cargo reads it for every project
  under `V:\dev`, because it searches the parent directories of the working directory, so the
  setting is machine-specific and lives on the machine, per the `V:\dev` rules. It applies to
  `aggregate-ntt`, `eisenstein` and `skimmatch` too, which matches the [Rust-Build-Local] design
  of `aggregate-ntt`. A `RUSTFLAGS` environment variable overrides it, which is how CI and any
  deliberate baseline build escape it. Changes to this file do not trigger a `uv sync` rebuild;
  use `uv sync --reinstall-package eltq`.
- Delete the orphaned `V:\dev\cargo-target\secondary-corrosion` (derived build output).
- Rewrite `README.md` "Environment" (plain `.venv` in the checkout, `./target`, the machine
  config above) and the sentence in "Performance" about `target-cpu=native`, to say that local
  source builds use the machine's CPU through `V:\dev\.cargo\config.toml` and that PyPI wheels
  use the `x86-64-v3` baseline.
- Acceptance: `uv sync --reinstall-package eltq` builds into `./target`; `uv run pytest` green;
  the benchmark headline on this machine within noise of the `native` column above.
- Bump: patch, since the build configuration of a fresh clone changes.

### [packaging] metadata, Python floor and contents

- Add `LICENSE` (MIT, Stephen J. Mildenhall, 2026).
- `pyproject.toml`: `license = "MIT"`, `license-files = ["LICENSE"]`, `readme = "README.md"`,
  `authors`, `keywords`, `[project.urls]` (Repository `https://github.com/mynl/eltq`,
  Changelog `https://github.com/mynl/eltq/blob/master/CHANGELOG.md`), and classifiers:
  Programming Language Rust, Python 3 only, 3.12, 3.13, 3.14, Operating System Microsoft
  Windows, POSIX Linux, MacOS, Intended Audience Science/Research and Financial and Insurance
  Industry, Topic Scientific/Engineering Mathematics. No license classifier: PEP 639 forbids one
  beside a license expression. Raise the `[build-system]` floor to `maturin>=1.14,<2`, the
  version in use, which writes `License-Expression`.
- The Python floor of ruling 2: `abi3-py312` in `crates/eltq-python/Cargo.toml`,
  `requires-python = ">=3.12"`; relock `uv.lock`. The project `.venv` stays on its current
  Python.
- An optional dependency for the reference backend:
  `[project.optional-dependencies] reference = ["scipy>=1.14"]`, so that
  `pip install eltq[reference]` makes `eltq.reference_multiunit` importable.
- `README.md`: an Install section (`pip install eltq`, `uv add eltq`, the `reference` extra);
  the CPU baseline in one sentence, with `pip install --no-binary eltq eltq` (needs a Rust
  toolchain) as the route for CPUs older than the baseline, and
  `RUSTFLAGS="-C target-cpu=native"` in front of it for a build tuned to one machine.
- Audit the sdist from a clean clone (the working tree holds a built `_eltq.pyd` in
  `python/eltq/`, gitignored): `maturin sdist`, list its contents, and exclude what a user does
  not need (`dev/`, `notebooks/`, `reproductions/`, `benchmarks/`, `gpt/`) through
  `[tool.maturin] exclude` if maturin includes them. Confirm no data file outside `fixtures/`.
- Acceptance: `maturin build --release` and `maturin sdist` succeed; `uvx twine check dist/*`
  passes on both; the wheel's `METADATA` has `License-Expression: MIT` and
  `Requires-Python: >=3.12`; a clean venv installs the built wheel and imports `eltq` with
  `BACKEND == "rust"`; a second clean venv installs from the sdist (building with Rust) and does
  the same.
- Bump: patch.

### [cpu-baseline] the instruction set of the x86_64 wheels

Ruled: **build the x86_64 wheels at `x86-64-v3` and do not multiversion.**

- v3 gives up 8% on the multi-unit route only, against Zen 5 native, and nothing on the
  one-row route or the joint. v4 would gain nothing over v3 and would exclude every Intel
  consumer CPU from Alder Lake onward, which lacks AVX-512.
- Multiversioning (the `multiversion` crate, or `#[target_feature]` plus
  `is_x86_feature_detected!`) would add code and a dispatch layer to recover at most the 8%,
  only on Zen 5, and only on one route. Not worth it, per the KISS rule; revisit only if a
  future kernel is written with explicit SIMD.
- This machine keeps native speed through `V:\dev\.cargo\config.toml` ([cargo-config]), since
  Steve builds from source.
- The cost of v3: CPUs older than Intel Haswell (2013) or AMD Excavator (2015), and some
  Pentium, Celeron and Atom parts, which Intel shipped without AVX until about 2022, cannot run
  the wheel and fail with an illegal-instruction error at import. The README sentence of
  [packaging] gives them the source route.
- Implemented in [ci]; acceptance there.

### [ci] build, test and publish on GitHub Actions

One workflow, `.github/workflows/wheels.yml`. The file name and the environment names are part
of the trusted publisher configuration on PyPI, so they do not change without changing it.

- **Triggers:** push to `master` with `paths-ignore` for `**.md`, `dev/**`, `reproductions/**`,
  `notebooks/**` and `benchmarks/**` (build and test only); pull requests (build and test only);
  a pushed tag `v*` (build, test, publish to PyPI; path filters do not apply to tags); and
  `workflow_dispatch` with a boolean input `testpypi` (build, test, and publish to TestPyPI when
  set).
- **build**, a matrix of three jobs with `PyO3/maturin-action` (`build --release --out dist`):
  `ubuntu-latest` x86_64 with `manylinux: auto`, `windows-latest` x64, `macos-latest` (arm64).
  The two x86_64 jobs set `RUSTFLAGS: -C target-cpu=x86-64-v3`; the arm64 job sets nothing, its
  default target CPU is Apple M1. maturin-action forwards `RUST*` and `CARGO_*` variables into
  its manylinux container; confirm from the job log, and pass it with `docker-options` if not.
  One abi3 wheel per job, uploaded as an artifact.
- **sdist**: `maturin sdist --out dist`, uploaded as an artifact.
- **test**, a matrix: each of the three platforms on Python 3.12 (the floor), plus
  `ubuntu-latest` on 3.14. Each job checks out the repository (for `tests/` and `fixtures/`),
  downloads its platform's wheel, installs it with
  `pip install --find-links dist "eltq[reference]" pytest pandas`, checks
  `python -c "import eltq; assert eltq.BACKEND == 'rust'"`, and runs `pytest`. pytest imports
  `eltq` from site-packages, since the repository root holds no `eltq` package (it sits under
  `python/`, which is not on the path), so the installed wheel is what is tested. The suite runs every test against both backends, so each
  wheel is checked against the SciPy oracle on its platform. The golden comparisons in
  `tests/test_acceptance.py` use a tolerance (`rtol=1e-9`), so last-bit differences between
  platform math libraries (`exp`, `log`) do not fail them; the determinism tests compare runs
  within one process. The pandas `float_precision` warning is expected and harmless.
- **publish-testpypi**: runs only on `workflow_dispatch` with `testpypi` set; needs build,
  sdist and test; `environment: testpypi`; `permissions: id-token: write`;
  `pypa/gh-action-pypi-publish@release/v1` with
  `repository-url: https://test.pypi.org/legacy/` and `skip-existing: true`.
- **publish-pypi**: runs only on a `v*` tag; needs build, sdist and test; first checks that the
  tag equals `v` plus the version in `pyproject.toml` and fails otherwise; `environment: pypi`;
  `permissions: id-token: write`; `pypa/gh-action-pypi-publish@release/v1`.
- Use the current major version of every action (`actions/checkout`, `actions/setup-python`,
  `actions/upload-artifact`, `actions/download-artifact`, `PyO3/maturin-action`).
- **CLAUDE.md delta.** A pushed `v*` tag publishes to PyPI, so per the house rule this project
  declares a delta: Claude pushes commits as usual, but creates and pushes no `v*` tag; the tag
  push is Steve's. The `pypi` environment's required reviewer is a second gate.
- Acceptance: a push of this step's commit runs build and test green on all four test jobs; a
  manual run with `testpypi` set publishes to TestPyPI; on this machine a clean venv with
  `uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url
  https://pypi.org/simple/ eltq` imports with `BACKEND == "rust"`, passes the suite run from the
  repository, and, with the timing script of the CPU measurement, matches the `x86-64-v3`
  column (multi-unit 100k about 1.24 s; a build without the flag would be at least the 1.6 s of
  v2).
- Bump: patch.

### [publisher-setup] Steve's one-time setup on PyPI and GitHub

Needed before the [ci] acceptance (TestPyPI) and before [release] (PyPI). Settings for both
publishers: owner `mynl`, repository `eltq`, workflow `wheels.yml`.

- **PyPI** (`https://pypi.org/manage/account/publishing/`, logged in as the owner of
  `aggregate`): add a pending publisher, PyPI project name `eltq`, environment `pypi`.
- **TestPyPI** (`https://test.pypi.org/manage/account/publishing/`; a separate account from
  PyPI, with its own 2FA): the same, environment `testpypi`.
- **GitHub** (repository Settings, Environments): create `pypi` with Steve as required reviewer
  and deployment tags limited to `v*`; create `testpypi` with no rules. Actions are enabled by
  default on a public repository; no secrets are needed.

### [release] first upload to PyPI

- Bump the minor version (first installable release) in the three version files, CHANGELOG
  entry, commit, push. Steve tags `v<version>` and pushes the tag, then approves the `pypi`
  deployment in the Actions run.
- Verify on this machine: clean venv, `uv pip install eltq`, `pytest` from the repository,
  `eltq.__version__`, `eltq.__rust_version__`; and that `reproductions/` renders with `eltq`
  from PyPI.
- Switch `reproductions/pyproject.toml` from the path dependency to `eltq>=<that version>`, or
  keep the path dependency for development and note in its README that PyPI works; Steve's call
  at the time.

## Not in scope

- Wheel variants or separate packages per CPU level.
- Linux aarch64 and macOS x86_64 wheels (ruling 3); each is one more matrix entry if wanted.
- conda-forge.
- Changes to numerics. On one platform, every instruction-set level must produce output bitwise
  identical to every other (see the measurement); a difference is a bug to stop on.
- The pandas `float_precision` deprecation in `tests/conftest.py`; revisit when a pandas release
  removes the argument.

## Execution log

(empty)
