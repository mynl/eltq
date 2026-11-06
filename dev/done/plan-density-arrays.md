# Plan: `density_arrays`, the exact event-loss density of a one-row ELT

`[Eltq-Density-Arrays]`. Drafted 2026-11-06 against eltq 2.14.0 (commit `cfe22b7`), from an aggregate session. **Status: ruled 2026-11-06, ready to execute.** The eltq half of a pair: the aggregate half is `V:/worktrees/aggregate_REFACTOR/dev/plan-elt-severity-functions.md`, which cannot land until this one is released. One feature, one bump, to the release `2.15.0`.

## Goal

A grid-free function, beside `tail_moments_arrays`, that returns at each loss $x$ the rate density of the continuous part of a one-row ELT's event-loss measure,

$$
\nu_c(x) = \sum_j \lambda_j f_j(x),
$$

with $f_j$ the density of row $j$'s law under section 4's classification, and the atoms excluded. It is $-dR/dx$ wherever $R(t) = \sum_j \lambda_j \mathsf P(X_j > t)$ has no atom, so the pair `tail_moments_arrays` and `density_arrays` describes the measure exactly with no grid, which is what aggregate needs for a pointwise severity (`pdf` beside `sf`).

## Why this is in eltq

aggregate measured three routes on 2026-11-06, on a synthetic table of $10^4$ to $10^6$ beta rows with standard deviations near the Bernoulli bound:

- **Differentiating the exceedance curve** (spec section 5A) is off by up to $10^{-2}$ relative at $10^4$ rows and $0.95$ at $10^6$, near the endpoints of rows with $\beta < 1$, whose density has a pole at $M$. The curve interpolates $R$, not its derivative.
- **Differencing the exact $R$** from `tail_moments_arrays` fails at the same points (up to $0.9$).
- **An exact row sum in SciPy inside aggregate** is correct and costs 20 ms, 0.22 s and 2.3 s for 30 points at $10^4$, $10^5$ and $10^6$ rows, but it would copy section 4's classification (atom, beta, Bernoulli two-point, $\sigma = \sigma_I + \sigma_C$, the variance policy) and section 5A's regimes (narrow kernel, normal law) into a second package, where the two can drift.

The row sum belongs here, where the classification and the regimes already live.

## Current behavior, 2.14.0

- `tail_moments_arrays` (`python/eltq/__init__.py:438`, Rust `tails::tail_moments`, `crates/eltq-core/src/tails.rs:222`) classifies every row with `classify_rows_with`, builds its `RowTail` (`Atom`, `TwoPoint`, `Regular` beta, `Narrow` with the narrow kernel or, from $\min(\alpha, \beta) = 10^{16}$, the normal law), and sums $\lambda_j$ times each row's contribution per edge, in chunks over rows on rayon, with Neumaier summation, results bitwise identical for any thread count. Edges must be finite and nonnegative (`Error::BadTailEdge`).
- `BetaFn` has `pdf` and `ln_pdf` (`crates/eltq-core/src/incbeta.rs:188`, `194`).
- Nothing returns a density. `rg -i "density_arrays|def density|fn density"` over `python` and `crates` finds nothing.

## The change

- **Python:** `eltq.density_arrays(rate, max_loss, mean_loss, sd_indep, sd_corr, *, x, mean_is_loss_rate=False, variance_policy="bernoulli", threads=None, event_ids=None) -> np.ndarray`, shape `(len(x),)`, the columns and keywords exactly as `tail_moments_arrays` takes them. `x` is finite, any order and any sign; a non-finite value raises `EltqError`, as a bad edge does.
- **Per row:** a `Regular` beta contributes $\lambda f_{\alpha,\beta}(x/M)/M$ for $0 < x < M$, from `BetaFn::ln_pdf` exponentiated; a `Narrow` row its kernel's density on the bulk and 0 outside it; the normal law $\lambda\varphi((x-\mu)/\sigma)/\sigma$; `Atom` and `TwoPoint` rows contribute nothing. At a pole ($\alpha < 1$ at $0^+$, $\beta < 1$ at $M^-$) the value is $+\infty$, as SciPy returns; $x \le 0$ and $x \ge M$ contribute 0.
- **Rust:** `tails::density` (or a sibling module) on the `tail_moments` pattern: the same classification, the same chunks, rayon and Neumaier, and a row's work stopping at the first sorted $x$ beyond its support. A PyO3 binding `_eltq.density_arrays` in `crates/eltq-python/src/lib.rs`.
- **Reference backend:** `reference.density_reference` with `scipy.stats.beta.pdf`, the reference narrow kernel and `scipy.stats.norm.pdf` over `_classify_rows`, served when the extension is absent, as for every other function.
- **Spec:** section 4B gains a paragraph, **Density**, defining $\nu_c$, stating that $\nu_c = -dR/dx$ away from atoms and that atoms are excluded, and listing the regimes. Section 7 lists the output. Section 9 gains the correctness test below.
- **Docs:** `README.md` API table and the CHANGELOG.

## Tests

One new correctness test in spec section 9, and the backend-agreement suite extended to the new function:

1. Against SciPy on random beta rows (small, moderate and large $\alpha$, $\beta$, including $\alpha < 1$ and $\beta < 1$): relative $10^{-12}$ away from poles.
2. Integration: $\int_a^b \nu_c = R(a) - R(b) - \sum_{\text{atoms in }(a,b]} \lambda$, by adaptive quadrature at several $(a, b)$, relative $10^{-9}$.
3. Atom and Bernoulli rows contribute exactly 0; a table of only atoms returns zeros.
4. The narrow-kernel and normal-law regimes against SciPy's beta density and the normal density at their thresholds.
5. Bitwise identical results for `threads` 1, 2 and all cores; Rust against the reference backend within the suite's tolerance.
6. $x < 0$ and $x \ge \max M$ give 0; a non-finite $x$ raises.

## Performance

At 30 points on the 0.91M-row benchmark portfolio, no slower than `tail_moments_arrays` at the same 30 edges (aggregate measured that at 90 ms on $10^6$ synthetic rows). Recorded in `benchmarks/`.

## Ruling

- **[Eltq-Release-For-Aggregate]** Ruled by the author, 2026-11-06: this feature takes eltq straight to the release `2.15.0`, with no pre-release, in `Cargo.toml`, `pyproject.toml` and `__version__` together, and the author publishes it to PyPI. The reason: aggregate's `elt` extra installs eltq from PyPI, and `pip` skips pre-releases unless asked, so aggregate can require the function only from a published release.

## Execution log

Executed 2026-11-06 as 2.15.0, from eltq 2.14.0 (commit `cfe22b7`).

### Review findings and rulings

- **[Eltq-Density-Large-Parameters]** Test 1's $10^{-12}$ was out of reach of the route the plan named, `BetaFn::ln_pdf` exponentiated: measured against 50-digit arithmetic at the mean and up to six standard deviations, $7.1\times10^{-13}$ at $(10^3, 10^3)$, $1.8\times10^{-11}$ at $(10^4, 3\times10^4)$, $2.9\times10^{-10}$ at $(10^5, 10^5)$ and $3.9\times10^{-10}$ at $(1.9\times10^5, 6\times10^5)$, the three logarithms of size $(\alpha+\beta)H(\bar y)$ cancelling; SciPy's `beta.pdf` (Boost) is within $10^{-12}$ throughout, and with one parameter small the naive form is within $10^{-14}$. Ruled by the author 2026-11-06: from $\min(\alpha,\beta)=100$ the density takes the Stirling form, $\ln f(y)=(\alpha-1)\log(1+g/\alpha)+(\beta-1)\log(1-g/\beta)+\tfrac12\ln\bigl((\alpha+\beta)^3/(\alpha\beta)\bigr)-\tfrac12\ln2\pi-\sigma(\alpha)-\sigma(\beta)+\sigma(\alpha+\beta)$ with $g=y(\alpha+\beta)-\alpha$ held exact by a two-sum and a fused multiply-add, $\sigma$ the six-term Stirling correction already used for $\ln B$ (Press et al. 1992, section 6.1). Prototyped in Python before the ruling: within $6.6\times10^{-13}$ up to the narrow threshold.
- **[Eltq-Density-Usage-Doc]** No eltq folder existed under `D:/Projects/Hacks` or `D:/Projects/Using`. Ruled by the author 2026-11-06: `D:/Projects/Using/eltq/using-density-arrays.qmd`.

### Divergences

- **Endpoints.** SciPy returns $+\infty$ at $x=0$ exactly for $\alpha<1$ and at $x=1$ for $\beta<1$, the finite $\beta$ at $x=0$ for $\alpha=1$, and raises `OverflowError` at a subnormal share for a tiny $\alpha$. The plan's own rule, the density on the open interval with 0 at $x\le0$ and $x\ge M$, is kept: $+\infty$ appears only where an interior value overflows. The reference masks the endpoints and maps an overflow to $+\infty$, and the tests compare at interior points.
- **The complement near $M$.** $1-y$ is formed as $(M-x)/M$, not $1-x/M$, so the pole of a $\beta<1$ row keeps its digits within a few units in the last place of $M$, where aggregate's differencing failed.
- **No reference narrow kernel.** The one-row reference has none: it takes SciPy below $10^{16}$ and the normal law from there (`reference._law`), and the reference density does the same.
- **Benchmark portfolio.** The 0.91M-row portfolio is the multi-unit one. The comparison at 30 points runs on `run_moments.py`'s three seeded one-row portfolios of $10^6$ rows, and on the 0.91M unit rows read as a one-row table.
- `CLAUDE.md` gave the version as 2.10.0; corrected to 2.15.0 in the bump.
- **Signature details.** The Rust function is `tails::density(input, options, x, threads) -> Result<Vec<f64>, Error>`, beside `tail_moments`, with a new `Error::BadDensityPoint` for a non-finite point. `event_ids` is accepted for parity with `tail_moments_arrays`, but the result is a bare array with no diagnostics, so it reaches only the reference backend's error messages.
- **Waves.** The pass folds its chunks in waves of 64 (`DENSITY_WAVE_CHUNKS`), so that the partials in flight stay bounded when a caller asks at every grid node: $2^{16}$ points cost 32 MiB a wave, where collecting every chunk at once, as `tail_moments` does, would cost 0.25 GiB on a million rows. The fold is in chunk order whatever the wave, so the wave never changes a result.
- **Test 40 tolerances**, set from measurement: the backends agree within $10^{-11}$ on a table with no narrow row, and within $10^{-9}$ with narrow rows to $\alpha+\beta$ of $10^{13}$, where both are accurate only to the rounding of the share (about $\varepsilon\sqrt{\alpha+\beta}$ per standard score, $7.5\times10^{-10}$ measured); clause 1 runs below $2\times10^5$, and clause 4 covers the thresholds. Clause 1 also gained a monetary-scale case and a pole case against 50-digit arithmetic, which the plan did not list.

### Steps

- **[engine]** `incbeta::BetaDensity` (the direct route below 100, the Stirling route from there), `NarrowLaw::bulk` and `NarrowLaw::density` in `law.rs`, and `tails::density` with four unit tests (one beta's closed form, the Stirling route against the direct one at the switch, atoms, poles and bad points, bitwise across 1, 2 and 7 threads on three chunks and a partial one).
- **[bindings]** `_eltq.density_arrays` in `crates/eltq-python/src/lib.rs`; `eltq.density_arrays` in `python/eltq/__init__.py`, served by `reference.density_reference` without the extension.
- **[reference]** `reference.density_reference` and `_beta_share_density`: SciPy's `beta.pdf` below one half and by reflection above it, with `1 - y` as `(M - x) / M`, an overflow as $+\infty$ and the normal law from $10^{16}$.
- **[tests]** `tests/test_density.py` (spec test 40, both backends) and a clause in `tests/test_fallback.py`.
- **[spec]** Section 4B retitled "Model moments, tail measures and density" and given a **Density** paragraph; section 7 lists the output; section 9 gains test 40.
- **[benchmark]** `run_moments.py --density`, `benchmarks/report-2.15.0.md`.
- **[docs]** `README.md` milestone row and API block, the CHANGELOG, and the usage page `D:/Projects/Using/eltq/using-density-arrays.qmd` (outside this repository, not committed).

### Measurements and gates

- Accuracy (50-digit arithmetic, the mean and up to four standard deviations): the engine within $5\times10^{-13}$ from $\alpha,\beta$ of $10^{-3}$ to $2\times10^5$ on both sides of the Stirling switch, SciPy likewise; $2.8\times10^{-11}$ at $(10^9, 3\times10^9)$ and $1.3\times10^{-8}$ at $(10^{15}, 10^{15})$ on the narrow kernel, the rounding of the share.
- At 30 points, all threads (`benchmarks/report-2.15.0.md`): 0.031 to 0.040 s on the three one-row portfolios of a million rows and 0.034 s on the 909,937 multi-unit rows, against 0.092 to 0.148 s for `tail_moments_arrays` at the same points; bitwise across thread counts. At every node of a $2^{16}$ grid on the mixture portfolio: 0.86 s on $10^5$ rows and 8.0 s on $10^6$, peak RSS 72 and 168 MiB, so aggregate's grid-scale callers ([ELT-Grid-Scale-Callers]) cost seconds rather than the minutes that ruling allowed for.
- Gates: `cargo test -p eltq-core` green (118 unit tests and the integration suites); `uv run pytest` 487 passed, 3 skipped (none in `tests/test_density.py`), 4 min 49 s.

### The macOS arm64 test fix (after 2.15.0, no bump)

- The wheels run of `09c6a0b` on master (run 42303032765) failed seven of test 40's cases on macOS arm64 alone, 480 passing: the engine against SciPy at `min(alpha, beta)` of 5.8e4 to 2e5 (relative 1.9e-12 to 2.6e-11), the reference's narrow-row backend comparison, and the reference just below $10^{16}$ against the normal density (absolute $1.7\times10^7$). Windows and Linux were green. The pattern is SciPy's, already recorded at 2.14.0 for the incomplete beta (off by about $\min(a,b)\times3\times10^{-18}$ there): it grows with the parameters, and the engine passed the same check just below $10^{16}$ where the reference failed. Not reproduced here, since no arm64 machine is available.
- Ruled by the author 2026-11-06 ("yes"). From $\min(\alpha,\beta)=10^3$ the oracle of clauses 1 and 4 is 50-digit arithmetic in place of SciPy on every platform, so the engine is checked against exact values on arm64 too. On macOS arm64 the reference is skipped from $10^3$ up to the normal law, and the narrow-row backend comparison is skipped. The flag `SCIPY_LARGE_PARAMETER_BETA_INEXACT` moved from `tests/test_survival.py` to `tests/conftest.py`, so both files share one definition. Checked locally with the flag forced on through a throwaway plugin: 14 reference cases skip and every engine case passes. Committed without a version bump, as `05c7abb` was at 2.14.0; the release tag goes on that commit.

### Left for the author

- Push the arm64 test fix, wait for a green wheels run on master, then tag that commit `v2.15.0` and approve the PyPI publish; aggregate's half (`[ELT-Severity-Functions]`) waits on the release.
- Whether `D:/Projects/Using/eltq` should become a third Using site (`_quarto.yml`, `style.py`, `site.toml`, an environment and an index): the page stands alone and has not been rendered.
- Tests 34 and 35 of the specification contain a tab where `\t` of `\times` was lost (`5.8<tab>imes10^{-11}` and seven more), from an earlier edit; not fixed here.
