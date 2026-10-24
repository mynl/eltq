# Plan: the incomplete beta at large parameters

Status: specified 2026-10-24, awaiting Steve's review. Version at execution: 2.3.1 (a fix). Sources: Numerical Recipes joins the allowed set (Steve, 2026-10-24: "pull in NR").

## Goal

The engine's regularized incomplete beta, `BetaFn` in `crates/eltq-core/src/incbeta.rs`, is the survival function of every beta event on the regular routes of the survival method and the quantile inversion of the quantiles method. For events with a huge second parameter it is accurate only to about 3e-6 relative. The goal is double precision near the bulk and 1e-8 or better in the tail for every event a real ELT contains, at no cost in speed, with a test that pins the figure and a source for the method.

## Current behavior and the finding

The downstream tie-out of 2.3.0 (recorded in `dev/done/plan-survival-tail-cost.md`, "Downstream check of 2.3.0 and a follow-up") found that real ELTs carry tens of thousands of events with $\alpha+\beta$ above $10^4$ that are not narrow, and a few hundred above $10^7$: J-shaped, $\alpha$ from 0.13 to 1.5, $\beta$ from $10^7$ to $8\times10^7$, the limit $M$ from $10^7$ to $3.5\times10^8$ times the mean. On the forty largest the engine's node values agree with SciPy only to 3.1e-6 relative. At portfolio level the grid was within 7.7e-7 of exact where $\Lambda\ge10^{-6}$, so nothing shows in the OEP yet.

Such an event is, to all purposes, a gamma with shape $\alpha$ and scale $M/\beta$. Nothing about its shape enters the method; the question is only numerical.

### Diagnosis, measured 2026-10-24

A Python port of the engine's fraction (`betacf`, the modified Lentz evaluation of Numerical Recipes section 6.4 with the symmetry switch at $x=(\alpha+1)/(\alpha+\beta+2)$), run against SciPy and against mpmath at 40 digits, separates two effects:

1. **The continued fraction is not the problem.** For $\alpha\le2$ and $\beta$ up to $8\times10^7$ it converges in 4 to 55 iterations at every point from 0.3 to 30 times the mean, far below the cap of 400. The cap matters only when both parameters are huge (the narrow regime of 2.3.0, which no longer uses the fraction).
2. **$\ln B(\alpha,\beta)$ is the problem.** `BetaFn::new` computes it as $\ln\Gamma(\alpha)+\ln\Gamma(\beta)-\ln\Gamma(\alpha+\beta)$. At $\beta=10^8$ the two large terms are near $1.7\times10^9$ and cancel to a result of order $-\alpha\ln\beta$; seven digits go with them. The absolute error of that sum, against mpmath:

| $\beta$ | $10^3$ | $10^4$ | $10^6$ | $10^8$ |
|---|---|---|---|---|
| naive sum | 7e-13 | 2e-11 | 1e-9 | 2e-7 |

   and the survival function carries it as a relative error amplified by $F/S$ in the lower branch (the error sits in the front factor, the survival is the complement): 2e-6 at $\beta=8\times10^7$, three times the mean, with SciPy's $\ln\Gamma$; 30 times the $\ln B$ error just below the switch point. The same probe in Rust (a temporary unit test, 2026-10-24, reverted): `statrs::ln_gamma` is within 2e-16 of mpmath at 0.13, 0.5 and 1.5 and within 3e-8 at $10^7$, and the engine's naive $\ln B$ is 3.5e-8 off at $(0.13,8\times10^7)$ and 3.8e-8 at $(0.5,10^7)$, which the amplification takes to the 1e-6 range. That is consistent with the 3.1e-6 of the finding in magnitude; the proof that it is the whole of it is the tie-out rerun under Acceptance.

With $\ln B$ exact, the converged fraction against mpmath is within 1e-14 relative at and below the switch point (0.3 to 3 times the mean) for every $\beta$, and in the upper branch within 1e-13 at $\beta=10^4$, 1e-10 at $10^6$ to $10^7$, and 4.5e-9 at $8\times10^7$. That residual is the conditioning of the upper branch, which evaluates $I_{1-x}(\beta,\alpha)$: the survival's sensitivity to its argument is about $\beta$ relative per unit, and $1-x$ carries the rounding of 1, so the error is about $\varepsilon\beta$. It is independent of the iteration count and is a property of the formulation, not of its evaluation.

SciPy's `betaincc` is within 2e-15 of mpmath throughout, so the reference's oracle stands. SciPy's `betaln` is not better than the naive sum until $\beta$ passes about $10^7$ (1e-9 at $10^6$), so it is not the oracle for $\ln B$; mpmath is.

## The change: [ln-beta]

Compute $\ln B(a,b)$, $a\le b$ after ordering (the function is symmetric), as

$$\ln B(a,b)=\ln\Gamma(a)+\bigl[\ln\Gamma(b)-\ln\Gamma(a+b)\bigr],$$

with the bracket from Stirling's series, $\ln\Gamma(z)\sim(z-\tfrac12)\ln z-z+\tfrac12\ln2\pi+\sigma(z)$, $\sigma(z)=\sum_{k\ge1}B_{2k}/(2k(2k-1)z^{2k-1})$, the two large expansions subtracted symbolically:

$$\ln\Gamma(b)-\ln\Gamma(a+b)=a-\bigl(b-\tfrac12\bigr)\log(1+a/b)-a\ln(a+b)+\sigma(b)-\sigma(a+b).$$

Every term is $O(a)$ or $O(a\ln b)$ and none cancels against another of size $b$: the first two cancel to $O(a^2/b)$ from quantities of size $a$, which costs nothing, and $\log(1+a/b)$ is taken by `ln_1p`. Six terms of $\sigma$ (coefficients $1/12$, $-1/360$, $1/1260$, $-1/1680$, $1/1188$, $-691/360360$) are exact to double at $b\ge100$; the branch applies there, and below it the present sum stays, where its cancellation costs at most 1e-13 (measured 4e-14 at $b=100$). Measured against mpmath for $a\in\{0.13,0.5,1.5,20\}$ and $b$ from $10^2$ to $10^8$: within 6e-14 absolute everywhere, within 4e-15 for $a\le1.5$, in the Python port; in Rust with `statrs::ln_gamma` for the small term, within 3e-14 at $(0.13,8\times10^7)$, $(0.5,10^7)$, $(1.5,10^4)$, $(20,10^6)$ and $(0.13,100)$.

The change is confined to `BetaFn::new`. The fraction, the branches, `sf`, `cdf`, `sf_and_density_times_y`, `ln_lower_tail_start` and every caller are untouched; the per-event constant is computed once per event as before, so the cost is nil. $\ln\Gamma(a)$ for the smaller parameter stays with the `statrs` Lanczos evaluation.

When both parameters are large (a narrow event evaluated by a regular route, or the quantiles method on such an event) the bracket is of order $b$ and its absolute error is $\varepsilon b$, which is the best any formulation of $\ln B$ can do in double precision (measured at $(10^7,10^7)$: 4.5e-8 against 1.5e-8 for the present sum, both the rounding of a result near $-1.4\times10^7$); those events are the narrow route's business (2.3.0).

### Sources

- Press, Teukolsky, Vetterling, Flannery, Numerical Recipes in C, second edition (1992), bibliography tag `Press1992`: section 6.1 for $\ln\Gamma$ and Stirling's formula, section 6.4 for the incomplete beta, its continued fraction (Abramowitz and Stegun 26.5.8) and the modified Lentz evaluation, which is what `incbeta.rs` already implements under a "Cephes/Numerical-Recipes-style" header.
- The subtraction of the two Stirling expansions is a derivation and goes in the specification; the coefficients are the Bernoulli numbers.

`CLAUDE.md`, section Sources, adds `Press1992` to the allowed set with the two section numbers and the date; the citation at the point of use goes in the `incbeta.rs` header and in the specification.

## Files

- `crates/eltq-core/src/incbeta.rs`: `ln_beta_large(a, b)` (the bracket), the branch in `BetaFn::new` at `max(a,b) >= LN_BETA_STIRLING_FROM` = 100, the header citation. Unit test `ln_beta_matches_mpmath_at_large_parameters`: a table of eight values from mpmath at 40 digits (the generating snippet in the test's comment), tolerance 1e-13 absolute.
- `secondary-uncertainty-mathematical-specification.md`: section 5A, the Nodes paragraph, one sentence on $\ln B$ with the formula and the source; the Narrow events paragraph's sentence on the engine's fraction gains the distinction (the 400-iteration cap and the garbage above $10^7$ are the both-large regime; with one parameter small the fraction converges in tens of iterations and the error was $\ln B$); test 23 in section 9.
- `tests/test_survival.py`: test 23 (below). `tests/test_incbeta.py` is not created; the engine's survival function reaches Python through `eltq.exceedance_curve` on a one-row portfolio, whose `exceedance(x)` is $\lambda S(x)$ exactly (2.1.0).
- `pyproject.toml`: `mpmath` in the dev dependency group (it was installed ad hoc into `.venv` for the measurement above and is not yet in `uv.lock`); the version to 2.3.1, with `Cargo.toml` and `python/eltq/__init__.py`.
- `CHANGELOG.md`: 2.3.1, Fixed. `README.md`: milestone row. `CLAUDE.md`: the source, and "Currently 2.0.0" in the Deltas to the version at hand.
- `dev/done/plan-survival-tail-cost.md`: a line under the follow-up pointing here.

## Test 23

On one-row portfolios with $\alpha\in\{0.13,0.5,1,1.5,2\}$, $\beta\in\{10^4,10^6,10^7,8\times10^7\}$, $M=10^6$ and the rate 1: both backends' `exceedance(x)` against SciPy `betaincc` at $x$ from 0.3 to 30 times the mean (21 points, geometric), within 1e-8 relative where $S\ge10^{-15}$, and within 1e-13 at and below three times the mean. Then a portfolio of 2,000 base rows (the recipe of test 22) with 300 such events added, rates $10^{-5}$ to $10^{-3}$: the survival grid's tail sums against the reference's exact $\Lambda$ at every edge where $\Lambda\ge10^{-12}\sum\lambda_j$, within 1e-6 relative (the figure test 15 asks of the whole method is 1e-4; the cubic between nodes, not the node values, sets this bound, see below), and the two backends' grids within $10^{-12}$ of the total rate.

The fixture question of the follow-up is answered here rather than by a fixture change: `fixtures/small_alpha_elt.csv` has 154 rows with $\beta>10^4$ and none above $3\times10^5$, so test 15 does not reach the regime; test 23 builds its portfolio in code, as test 22 does, and the fixtures and their goldens stay. The benchmark recipe `small_alpha_elt` in `benchmarks/run_end_to_end.py` is likewise untouched.

## Acceptance

- `cargo test -p eltq-core --release` and `uv run pytest` green, including test 23 on both backends.
- The goldens: the fixtures' betas are at most 827 (`small_elt`, `medium_elt`), so `ln B` moves by at most $10^{-12}$ and the survival goldens are expected to pass unchanged; if one moves, regenerate it, read the diff and record the largest change.
- The downstream tie-out, rerun on the eight real blocks by the session that found the figure: the engine's node values on the forty largest events within 1e-8 of SciPy (from 3.1e-6), the OEP and AEP points unchanged to their present tolerances.
- Speed: no sweep is needed; a single `benchmarks/run_end_to_end.py --survival-sweep --small-alpha --rows 100000` run confirms the build pass is unchanged within noise, recorded in the log, not in `results.json`.

## Alternatives considered and not taken

- **Quadrature of the density for $\alpha+\beta>3000$**, the switch Numerical Recipes' third edition makes, proposed before the measurement. The measurement withdrew it: the fraction converges in tens of iterations in this regime, so quadrature would be no faster and, for $\alpha<1$, would need the singularity at 0 handled. It remains the right tool if the upper branch's $\varepsilon\beta$ residual (4.5e-9 at $\beta=8\times10^7$) is ever required to go: a Gauss and Legendre walk of the log-density from $x$ outward, on pieces over which it changes by at most 4, about 80 density evaluations per point, accuracy 1e-14, at roughly the cost of the present fraction. Not worth a change for a 1e-9 ceiling that the OEP cannot see.
- **The gamma limit** as the evaluation: Beta$(\alpha,\beta)$ at scale $M/\beta$ is Gamma$(\alpha)$ with a relative error of order $(\beta x)^2/\beta$, 1e-7 at the mean but 1e-4 at thirty means, worse than what we have; its correction terms are DiDonato and Morris's asymptotic expansion (ACM TOMS Algorithm 708, 1992), heavier than the problem.
- **A cap that grows with the parameters**: irrelevant here, since the fraction converges; the both-large regime is the narrow route's.
- **The cubic between nodes** on these events: 3.7e-4 relative at interval midpoints deep in the tail, where one node interval spans 0.6 of an e-folding at $S=10^{-12}$, 7.7e-7 at portfolio level. This is the interpolant of section 5A, shared by all events, and the node density is its control (`nodes_per_decade`); it is out of scope here and is recorded in the tail-cost plan's follow-up.

## Execution log

(Filled at execution.)
