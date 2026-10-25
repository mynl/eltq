# Plan: the quantiles method at large parameters

Status: specified 2026-10-25; approved and executed the same day as 2.3.2 (Steve: "A&S in the sources is fine, @Abramowitz1970. Pls execute."). Prompted by the ELTA agent's report of the same day that `method='quantiles'` on 2.3.1 fails with "beta quantile evaluation failed" on rows with $\alpha$ between about $5\times10^{20}$ and $5\times10^{22}$ that 2.3.0 accepted. Sources: Abramowitz and Stegun joins the allowed set (tag `Abramowitz1970`).

## Goal

The quantiles method evaluates $x_{jk}=M_jQ_{\alpha_j,\beta_j}(u_k)$ by inverting the engine's own incomplete beta (`crates/eltq-core/src/batched.rs`). When both parameters are large the inversion is wrong, and beyond that it fails. The goal is a quantile that is correct to better than $10^{-8}$ standard deviations for every admissible $(\alpha,\beta)$, with no error path reachable from a finite row, at no cost in speed, with a test that pins the figure and a source for the method. The multi-unit routes share the kernel and are covered by the same change.

## Current behavior and the finding

The report: a one-row portfolio with exposed value $10^6$, mean $250{,}000$ and standard deviation $10^{-6}$ (so $\alpha=4.7\times10^{22}$, $\beta=1.4\times10^{23}$) fails under `method='quantiles'` on 2.3.1 and passed on 2.3.0; it passes on 2.3.1 at standard deviations $10^{-3}$ and $10^{-7}$. The survival method is unaffected (such an event is narrow and goes by quadrature, 2.3.0).

Reproduced 2026-10-25 and widened (`quantize_arrays`, bucket size 1000, one row, mean on node 250). The rows that "pass" are not right:

| standard deviation | $\alpha+\beta$ | 2.3.1 result |
|---|---|---|
| $10^{-3}$ | $1.9\times10^{17}$ | passes; mass in buckets 250 and 251 (right) |
| $10^{-4}$ | $1.9\times10^{19}$ | passes; mass spread over 31 buckets from 125 to 1000 (wrong) |
| $10^{-5}$ to $3\times10^{-7}$ | $1.9\times10^{21}$ to $2\times10^{24}$ | fails |
| $10^{-7}$ | $1.9\times10^{25}$ | passes; the same 31 wrong buckets |
| $10^{-8}$ | $1.9\times10^{27}$ | fails |

The 31 buckets are the dyadic points $0.25-0.125/2^i$ and $1-0.375/2^i$: the bisection midpoints of a solver whose function returns nonsense. A row whose true conditional loss is $250{,}000\pm10^{-6}$ has a third of its rate deposited near the exposed value.

The same probe built from the 2.3.0 commit (`2d524c6`, in a temporary worktree) shows the same pattern: wrong quantiles from $\alpha+\beta$ about $10^{12}$, failures from about $10^{18}$, with the failing and the passing windows in different places from 2.3.1 (at a mean share of 0.25, 2.3.0 fails from $10^{18}$ to $10^{23}$ and passes, wrongly, at $10^{24}$ and $10^{25}$). **2.3.1 did not introduce the defect.** It changed the rounding pattern of $\ln B$, which moves the windows in which the failure surfaces as an error rather than as wrong numbers. Where 2.3.0 "handled" such a row it deposited the wrong points, silently.

### Diagnosis, measured 2026-10-25

Two effects, both independent of the $\ln B$ change, measured with the 2.3.1 kernel against SciPy's `betaincinv` (sound in this range, see below) over 32 midpoint quantiles, errors in units of the event's standard deviation $\sigma$:

1. **The continued fraction's cap.** `betacf` stops at 400 iterations (`MAX_CF_ITERS`). Near the mean it needs about $\sqrt{\min(\alpha,\beta)}$ of them, so from $\min(\alpha,\beta)$ around $5\times10^5$ it returns an unconverged value and the quantiles drift:

   | $\min(\alpha,\beta)$ | $\le3.2\times10^5$ | $5.6\times10^5$ | $10^6$ | $1.8\times10^6$ | $3.2\times10^6$ | $10^7$ | $10^8$ | $1.3\times10^9$ |
   |---|---|---|---|---|---|---|---|---|
   | worst quantile error, $\sigma$ | 4e-9 | 3e-7 | 1e-5 | 2e-4 | 1e-3 | 1e-2 | 4e-2 | 2 |

   (the worst over ratios $\beta/\alpha$ of 1, 3, 300 and $3\times10^4$). Below $3.2\times10^5$ the kernel is as good as the oracle. This is the regime the specification already describes as "nothing usable above $10^7$ when both parameters are large" (section 5A, Narrow events), where the survival method stopped using the fraction in 2.3.0.

2. **The exponent's rounding.** Each evaluation forms $\exp(a\ln x+b\ln(1-x)-\ln B)$. The three terms are of size $a+b$ and their true sum is of order $\ln\sqrt{a+b}$, so the sum carries an absolute rounding of about $\varepsilon(a+b)$: $10^{-4}$ at $a+b=10^{12}$, 1 at $10^{16}$, and beyond 709 (about $5\times10^{18}$) the exponential overflows or underflows, the CDF becomes NaN and the walk returns `KernelError`, which `quantize` reports as "beta quantile evaluation failed". Between, the CDF is off by factors and the bracketing solver converges to the dyadic points above. Which rows hit NaN depends on the last bits of three large numbers, which is why the failing windows move between 2.3.0 and 2.3.1 and why the report's row passed at $10^{-7}$ and failed at $10^{-6}$.

Neither effect can be fixed by a better $\ln B$: the first is the fraction's convergence, the second the formulation itself (section 5A makes the same point for the survival of a narrow event). SciPy's `beta.ppf` breaks too, later: at $\alpha=4.7\times10^{20}$ it returns non-monotone points up to 287 $\sigma$ from the mean, and $2\times10^5\sigma$ at $4.7\times10^{26}$. So the reference backend is not an oracle in this regime either, and the tests that compare the two backends on such rows have no stake in the ground today.

What such a row is: a beta with both parameters large is, to the order of the correction below, a normal with the beta's mean and variance, and its quantile has an asymptotic expansion in the beta's cumulants that gets better exactly where the fraction gets worse.

## The change: [large-parameters]

In `batched.rs`, when $\min(\alpha,\beta)\ge T$ with $T=2\times10^5$ (`LARGE_PARAMETER_FROM`), both `beta_midpoint_quantiles` and `beta_quantiles` evaluate every target by the Cornish-Fisher expansion to second order instead of the walk:

$$Q(u)\approx\mu+\sigma\Bigl[z+\frac{\gamma_1}{6}(z^2-1)+\frac{\gamma_2}{24}(z^3-3z)-\frac{\gamma_1^2}{36}(2z^3-5z)\Bigr],\qquad z=\Phi^{-1}(u),$$

with the beta's mean $\mu=\alpha/(\alpha+\beta)$, variance $\sigma^2=\alpha\beta/((\alpha+\beta)^2(\alpha+\beta+1))$, skewness $\gamma_1=2(\beta-\alpha)\sqrt{\alpha+\beta+1}/((\alpha+\beta+2)\sqrt{\alpha\beta})$ and excess kurtosis $\gamma_2=6[(\alpha-\beta)^2(\alpha+\beta+1)-\alpha\beta(\alpha+\beta+2)]/(\alpha\beta(\alpha+\beta+2)(\alpha+\beta+3))$, all closed forms from the beta's moments. $\Phi^{-1}$ is `normal::quantile`, Wichura's AS241, already in the crate at $10^{-16}$. Targets of exactly 0 and 1 keep their endpoint treatment. The result is clipped to the open unit interval, which the formula never approaches at these parameters (it would need $|z|$ above $\sqrt{\min(\alpha,\beta)}$, about 450).

Measured against SciPy's `betaincinv`, worst over the same ratios and the 32 midpoint targets, in units of $\sigma$:

| $\min(\alpha,\beta)$ | $10^4$ | $3.2\times10^4$ | $10^5$ | $1.8\times10^5$ | $3.2\times10^5$ | $5.6\times10^5$ | $10^6$ to $10^8$ |
|---|---|---|---|---|---|---|---|
| first order (skewness only) | 4e-5 | 1e-5 | 4e-6 | 2e-6 | 1e-6 | 8e-7 | 4e-7 to 3e-9 |
| second order (the change) | 3e-7 | 5e-8 | 9e-9 | 4e-9 | 2e-9 | 7e-10 | below 1e-7 (the oracle's own floor at huge $\beta$) |

The second-order error falls as $0.3/\min(\alpha,\beta)^{3/2}$. At $T=2\times10^5$ both routes are within about $4\times10^{-9}\sigma$ of the oracle, so the handover is continuous to that level; first order alone would open a gap of three decades there, which is why the second order is taken (two more terms, no more evaluations). Above $10^8$ the oracle itself is the limit of the comparison; the expansion keeps improving and at $\alpha+\beta=10^{26}$ its terms beyond $z$ are below $10^{-13}$, so the points are the normal's, and all of them sit within $2.2\sigma$ of the mean, as they must. The report's row then deposits its whole rate on node 250 (within $3\times10^{-9}$, the linear rule's share of a $10^{-6}$ offset at bucket size 1000).

Cost: one normal quantile per target and no continued fraction, cheaper than the walk it replaces; nothing changes below $T$, so the goldens are unchanged (their rows have $\min(\alpha,\beta)$ far below $2\times10^5$) and the benchmark need not move.

The threshold is on the smaller parameter, not the sum, because both the fraction's iteration count and the expansion's error are governed by $\min(\alpha,\beta)$ (the one-parameter-small regime, $\alpha\le2$ with $\beta$ to $8\times10^7$, converges in tens of iterations and keeps the walk; 2.3.1).

### What else the change touches

- **Multi-unit routes.** `multiunit.rs` (the comonotone unit's `beta_quantiles` on $\Phi(w_cz)$ targets) and `views.rs` (`conditional_matrix`) call the same two functions and take the new route automatically. Their targets can be 0, 1 and repeated; the expansion handles them by the same endpoint and run rules as today.
- **The reference backend.** `python/eltq/reference.py` evaluates `beta_dist.ppf(u, a, b)`; it takes the same expansion above the same threshold, in a small public helper (`large_parameter_quantiles(a, b, u)`), so that the two backends agree to rounding and the helper can be tested directly against SciPy where SciPy is sound.
- **The survival method.** Untouched. An event with $\min(\alpha,\beta)\ge2\times10^5$ has coefficient of variation at most $2.2\times10^{-3}$, below one node interval at 100 nodes per decade (2.3 percent), so it is narrow and already goes by quadrature; the regular routes never see it.
- **Diagnostics.** No new counter. A `large_parameter_events` count in the metadata, like `narrow_events`, would be useful for the tie-out but widens the metadata contract across both backends and the Python layer; it is a follow-up if Steve wants it.

### Sources

- Abramowitz and Stegun, Handbook of Mathematical Functions (tag `Abramowitz1970` in `D:\Projects\Biblio\uber-library.bib`), section 26.2, the Cornish-Fisher asymptotic expansion of a quantile in the cumulants (26.2.49 in the 1964 numbering; the equation number is to be checked against the copy at execution). The book is already the source of the continued fraction (26.5.8) through Numerical Recipes.
- The beta's skewness and excess kurtosis are derived from its moments; the derivation goes in the specification, not to a source.
- Wichura's AS241 for $\Phi^{-1}$ is already in `normal.rs` with its citation.

`CLAUDE.md`, section Sources, adds `Abramowitz1970` with the section number and the date, as it did `Press1992`.

## Files

- `crates/eltq-core/src/batched.rs`: `LARGE_PARAMETER_FROM`, `large_parameter_quantile(alpha, beta, u)` and its moments helper, the branch at the top of `beta_midpoint_quantiles` and `beta_quantiles` after the argument checks, the module header's citation. Unit tests: `large_parameters_match_scipy` (a table of SciPy `betaincinv` values at $\min(\alpha,\beta)$ of $3\times10^5$, $10^6$, $10^7$ and $10^8$ with ratios 1, 3 and 300, generating snippet in the comment, tolerance $10^{-8}\sigma$); `large_parameters_continuous_at_the_threshold` (quantiles at $T(1\pm10^{-9})$ within $10^{-8}\sigma$); `huge_parameters_stay_finite_and_monotone` (the report's $(4.6875\times10^{22},1.40625\times10^{23})$ and $(10^{26},10^{26})$: finite, non-decreasing, within $2.2\sigma$ of the mean); `arbitrary_targets_above_the_threshold` (targets with 0, 1 and repeats).
- `python/eltq/reference.py`: `large_parameter_quantiles` and the branch at the `ppf` call.
- `secondary-uncertainty-mathematical-specification.md`: section 5 gains a paragraph "Large parameters" with the threshold, the expansion, the cumulants and the measured figures, and a sentence on why the fraction cannot serve there (cross-referring to the Narrow events paragraph of 5A); section 9 gains test 24.
- `tests/test_survival.py`: test 24 (below); the docstring header to "15 to 24".
- `CLAUDE.md`: the source; "Currently 2.3.2".
- `CHANGELOG.md`: `## [2.3.2]`, Fixed, three or four bullets. `README.md`: a milestone row. The version triple to 2.3.2; `Cargo.lock` and `uv.lock` follow.
- This plan to `dev/done/` with the execution log.

## Test 24

Large parameters under the quantiles method, both backends:

1. The reference helper against SciPy: for $\min(\alpha,\beta)$ log-spaced from $2\times10^5$ to $10^8$ and ratios 1, 3, 300 and $3\times10^4$, the expansion's 32 midpoint quantiles are within $10^{-8}\sigma$ of `scipy.special.betaincinv`. (SciPy is sound there: the engine's walk agrees with it to $4\times10^{-9}\sigma$ up to $3\times10^5$ and the expansion to $10^{-7}\sigma$ to $10^8$, two independent methods.)
2. One-row portfolios with $M=10^9$, mean share $p\in\{0.5,0.25,0.01\}$ and $\min(\alpha,\beta)\in\{10^5,2\times10^5,10^6,10^8,10^{12},10^{16},10^{20},4.7\times10^{22},10^{26}\}$ (the variance set from the parameters through `beta_params`'s inverse), bucket size $10^3$: both backends succeed; the rate is exact; the two grids agree within $10^{-12}$ of the rate; every bucket with mass lies within $\mu\pm(2.2\sigma+h)$; the reported first-moment error is below $10^{-2}\gamma_1\sigma+10^{-9}\mu$ in absolute value.
3. The report's row (exposed value $10^6$, mean $250{,}000$, standard deviation $10^{-6}$, bucket size $1000$): bucket 250 holds at least $1-10^{-8}$ of the rate on both backends.
4. The portfolio of test 22 with 300 added rows of $\min(\alpha,\beta)$ log-uniform from $10^5$ to $10^{24}$ and means from $10^3$ to $10^5$, under `method='quantiles'`: both backends succeed, the rate is exact, the grids agree within $10^{-12}$ of the total rate, and the raw grid is unchanged.

## Acceptance

- Rust gate `cargo test -p eltq-core --release` and Python gate `uv run pytest` green; golden fixtures unchanged.
- The report's repro script passes, with its mass on node 250.
- Benchmark sweep (`--survival-sweep --small-alpha --rows 1000000`) within noise, as a guard only; the quantiles route on realistic rows never reaches the new branch.
- Downstream (left for the session holding the real blocks): rerun `tieout_results()` on 2.3.2. The 32-point step completes; its results differ from 2.3.0 wherever a row had $\min(\alpha,\beta)$ above about $5\times10^5$, because 2.3.0's points for such rows were wrong, so a difference there is a correction and the rows should be listed with their rates. The deck, on the survival method, is unchanged.

## Alternatives considered and not taken

- **Raise the fraction's cap.** The iterations needed grow as $\sqrt{\min(\alpha,\beta)}$ (tens of thousands at $10^9$, times 32 quantiles, times a few evaluations each), and the exponent's rounding kills the result above $10^{16}$ regardless. Slow and still wrong.
- **An atom at the mean above a threshold.** Right where $\sigma$ is below a bucket, wrong where it is not: at $\min(\alpha,\beta)=10^6$ and $M=10^{10}$ an event has $\sigma$ near $10^6$, a thousand buckets wide at $h=10^3$. The expansion covers both cases with one formula and reduces to the atom when $\sigma$ is negligible.
- **Temme's uniform asymptotic expansion of the incomplete beta** (DiDonato and Morris, Boost): the proper fix for the function itself, a few hundred lines, outside the source set, and more than the quantiles method needs; its points are only ever deposited on a grid.
- **Normal approximation without the cumulant terms.** Error $0.35/\sqrt{\min(\alpha,\beta)}$ in $\sigma$: $10^{-3}$ at $10^5$, far worse than the walk it replaces until $10^{10}$.

## Execution log

2026-10-25, Claude, as 2.3.2 in one commit. Steve approved the plan and the source ("A&S in the sources is fine, @Abramowitz1970. Pls execute.").

- `batched.rs`: `LARGE_PARAMETER_FROM = 2e5`, `uses_expansion`, `Cumulants::of` and `Cumulants::quantile`, the branch in both kernels after the argument checks and before the fast paths (the symmetric walk is as wrong as the general one there). The cumulants are written with products of at most two parameters, so they stay finite to about $10^{150}$; beyond that the standard deviation underflows to zero and the points sit at the mean, which is the right limit. Four unit tests as planned; the SciPy table stops at $(10^8,10^8)$ because SciPy's own floor passes $10^{-8}\sigma$ from about $a=3\times10^7$ with $b=10^{10}$ (measured $8\times10^{-8}$ at $(10^8,3\times10^{10})$).
- `reference.py`: `large_parameter_quantiles`, the same formula with SciPy's `ndtri`; `_meta.py` holds the shared threshold.
- Gates: `cargo test -p eltq-core --release` 81 passed (77 before); `uv run pytest` green, one skip as before (test 24 added). Goldens unchanged.
- Benchmark guard (`--survival-sweep --small-alpha --rows 1000000`): quantiles 0.993 s against 0.961 and 1.006 s before; survival at 100 per decade 2.555 s against 2.642 and 2.743 s. Noise. The appended `results.json` record was restored.
- Measured in the test: both backends report the same first-moment error to the digit on every one-row portfolio (for example $-16.19$ on $(10^5,3\times10^5)$ at $M=10^9$, which is $6.5\times10^{-3}\gamma_1\sigma$, the midpoint rule's own bias), so the expansion changes nothing about what the method reports.

### Divergences from the plan as written

- Test 24's SciPy comparison runs at $\min(\alpha,\beta)$ of $2\times10^5$, $10^6$ and $10^7$, not to $10^8$: SciPy's inverse is not an oracle at $10^{-8}\sigma$ above about $3\times10^7$ with $b\ge10^{10}$ (see above).
- The one-row sweep uses $3\times10^5$ in place of $2\times10^5$: a row set exactly at the threshold recovers $\alpha$ from $(M,\mu,\sigma)$ on either side of it by rounding, differently in the two backends, which then take different routes and differ by $3\times10^{-10}$ in weight. The continuity at the threshold is the Rust unit test's business. Below the threshold the two backends' grids are held to $10^{-8}\sigma/h$ (the walk against SciPy's inverse), not $10^{-12}$.
- The portfolio's grids are held to $10^{-11}$ of the total rate, not $10^{-12}$: measured $4.2\times10^{-12}$, set by the base rows and the large rows below the threshold, where the two backends invert differently.
- The reported first-moment error is checked against $2\times10^{-2}\gamma_1\sigma+10^{-9}\mu$, as a sanity bound on the midpoint rule's bias, in place of the plan's $10^{-2}$.
- The A&S equation number 26.2.49 is cited from memory of the 1964 numbering and is to be checked against the copy.

### Left for Steve

- Push the commit.
- Rerun `tieout_results()` on the eight real blocks under 2.3.2. The 32-point step completes; rows with $\min(\alpha,\beta)$ above about $5\times10^5$ move their mass to the mean, a correction of 2.3.0's points. The deck is unchanged.
- Check 26.2.49 in A&S.
