# Portfolio ELT Secondary-Uncertainty Quantization

## Problem and mathematical specification

Date: 2026-08-14

## 1. Purpose

We require a utility that converts a portfolio event loss table (ELT) into two discrete, evenly spaced, rate-weighted severity distributions. The first distribution represents the original ELT, in which each event produces its conditional mean loss. The second incorporates secondary uncertainty by replacing each conditional mean with a bounded beta severity distribution and approximating that distribution by midpoint quantiles. Both versions are then mapped to the same loss grid using a linear interpolation rule.

The utility is intended for portfolio-level ELTs, potentially containing approximately one million rows. It is not intended to reconstruct secondary uncertainty from individual locations or policies. The specification describes an RMS-style mathematical transformation based on the fields supplied in a portfolio ELT; it does not claim to reproduce any proprietary model implementation whose undocumented conventions may differ.

The implementation must be callable from Python and must produce results suitable for immediate use as NumPy arrays or as arrays backed by memory-mapped files. Computational speed is a primary requirement, but the present document deliberately makes no architectural or language recommendations.

## 2. Portfolio ELT

Each input row represents a catastrophe event $j$ and contains the following quantities.

| Field | Symbol | Meaning |
| --- | ---: | --- |
| Event identifier | $e_j$ | Identifier retained for validation and audit |
| Event rate or weight | $\lambda_j$ | Annual event occurrence rate, or another nonnegative measure weight |
| Maximum loss | $M_j$ | Upper bound for the conditional event loss |
| Mean loss | $\mu_j$ | Conditional mean portfolio loss given event $j$ |
| Independent standard deviation | $\sigma_{I,j}$ | Independent component of conditional loss uncertainty |
| Correlated standard deviation | $\sigma_{C,j}$ | Correlated component of conditional loss uncertainty |

An event rate or equivalent row weight is mathematically necessary. If event rates reside in a separate event catalogue, the input preparation process must join them to the ELT before quantization.

The working interpretation is that the ELT exposure field is the maximum possible portfolio loss $M_j$ for the event. That interpretation must be confirmed against the actual file specification. If exposure is not the correct beta upper bound, the input must provide a separate maximum-loss field.

The basic admissibility conditions are
$$
\lambda_j\geq 0,\qquad M_j\geq 0,\qquad 0\leq\mu_j\leq M_j,\qquad \sigma_{I,j}\geq0,\qquad \sigma_{C,j}\geq0.
$$
All monetary fields must use the same currency and monetary scale. The design should allow a software selectable option to interpret $\mu_j$ as the loss rate, scaled by $M_j$, in which case $0\le \mu_j\le 1$.

## 3. Raw ELT severity measure

Without secondary uncertainty, event $j$ contributes an atom of rate $\lambda_j$ at loss $\mu_j$. The raw, rate-weighted severity measure is therefore
$$
\nu_{\mathrm{raw}}=\sum_j\lambda_j\delta_{\mu_j},
$$
where $\delta_x$ denotes a unit atom at $x$.

The total event rate and expected annual loss represented by the ELT are
$$
\Lambda=\sum_j\lambda_j,
\qquad
\operatorname{AAL}=\sum_j\lambda_j\mu_j.
$$
If a probability mass function is required rather than a rate-weighted severity measure, it is obtained by dividing the output weights by $\Lambda$, provided $\Lambda>0$. Normalization must be explicit; it must not silently replace the rate-weighted default.

## 4. Conditional secondary uncertainty

Let $X_j$ denote portfolio loss conditional on the occurrence of event $j$. The secondary-uncertainty model assumes
$$
0\leq X_j\leq M_j,\qquad \mathsf P(X_j)=\mu_j.
$$
(Using the physicist notation for expectation wrt $\mathsf P$.) For the one-dimensional marginal severity distribution required here, the working total conditional variance is
$$
\sigma_j^2=\sigma_{I,j}^2+\sigma_{C,j}^2.
$$
The distinction between independent and correlated uncertainty cannot be retained in a single marginal severity curve after the portfolio-level event distribution has been formed. The two components would matter separately in a model that retained common latent variables, location-level dependence or joint losses across several portfolios. If the source ELT defines a different rule for combining the supplied standard deviations at portfolio level, that documented convention must replace the working formula above.

For $M_j>0$, define the normalized loss $Y_j=X_j/M_j$, its mean $p_j$ and its variance $v_j$ by
$$
p_j=\frac{\mu_j}{M_j},
\qquad
v_j=\frac{\sigma_j^2}{M_j^2}.
$$
For a nondegenerate beta distribution $Y_j\sim\operatorname{Beta}(\alpha_j,\beta_j)$, define
$$
\kappa_j=\frac{p_j(1-p_j)}{v_j}-1,
\qquad
\alpha_j=p_j\kappa_j,
\qquad
\beta_j=(1-p_j)\kappa_j.
$$
The beta model is feasible precisely when
$$
0<p_j<1,
\qquad
0<v_j<p_j(1-p_j),
$$
or, equivalently,
$$
0<\sigma_j^2<\mu_j(M_j-\mu_j).
$$
The boundary cases require explicit treatment. If $\sigma_j=0$, the conditional distribution is the atom $X_j=\mu_j$. If $\mu_j=0$, admissibility requires $X_j=0$ almost surely; if $\mu_j=M_j$, admissibility requires $X_j=M_j$ almost surely. A row whose stated variance violates the bounded-variance condition is inconsistent with the beta model and must not be silently altered. The utility must either reject the row or apply a separately selected and fully reported repair policy.

## 5. Midpoint-quantile approximation

Let $K\geq1$ be the requested number of secondary-uncertainty points per nondegenerate event. Define midpoint probability levels
$$
u_k=\frac{k+\tfrac12}{K},\qquad k=0,\ldots,K-1.
$$
If $Q_{\alpha,\beta}$ denotes the beta quantile function, event $j$ is approximated by the equally weighted conditional points
$$
x_{jk}=M_jQ_{\alpha_j,\beta_j}(u_k),\qquad k=0,\ldots,K-1.
$$
The resulting secondary-uncertainty severity measure before loss-grid discretization is
$$
\nu_{\mathrm{SU},K}
=\sum_j\frac{\lambda_j}{K}\sum_{k=0}^{K-1}\delta_{x_{jk}}.
$$
Degenerate events contribute a single point rather than $K$ repeated points.

The midpoint rule preserves each event's total rate exactly, but for finite $K$ its conditional sample mean
$$
\bar x_{j,K}=\frac1K\sum_{k=0}^{K-1}x_{jk}
$$
need not equal $\mu_j$ exactly. The utility must report the resulting first-moment error. Any optional moment correction must be separately identified because it changes the stated quadrature rule.

The calculation involves inverse regularized incomplete-beta evaluations. Referring to them as inverse-gamma evaluations is a transcription error, although random beta sampling can alternatively be expressed using two gamma variates. The required midpoint-quantile rule is deterministic and is defined through the beta quantile function.

## 6. Even loss grid and linear discretization

Let the output grid have constant spacing $h>0$ and nodes
$$
g_i=ih,\qquad i=0,\ldots,n-1.
$$
The configured grid must cover every loss that can be generated, normally by requiring
$$
(n-1)h\geq\max_j M_j.
$$
If the grid does not cover the full input support, the utility must stop with a clear error unless the caller has explicitly selected and named an overflow policy. Silent truncation, capping or wraparound is unacceptable.

For any loss point $x\in[g_i,g_{i+1}]$, the linear rule replaces an atom of weight $w$ at $x$ by weights
$$
w_i=w\frac{g_{i+1}-x}{h},
\qquad
w_{i+1}=w\frac{x-g_i}{h}.
$$
Equivalently, with $r=x/h$ and $i=\lfloor r\rfloor$,
$$
w_i=w(i+1-r),
\qquad
w_{i+1}=w(r-i).
$$
At an exact grid point, the full weight is assigned to that point. The maximum grid point requires an explicit endpoint case so that it is not indexed beyond the array.

The linear rule preserves both mass and the first moment of every discretized point:
$$
w_i+w_{i+1}=w,
\qquad
w_ig_i+w_{i+1}g_{i+1}=wx.
$$
Consequently, the raw grid distribution must preserve the ELT's total rate and expected annual loss up to floating-point summation error. The secondary grid distribution preserves the total rate exactly and preserves the first moment of the midpoint-quantile approximation; its difference from the analytical expected annual loss arises from finite-$K$ quadrature and floating-point error, not from the linear loss-grid rule.

## 7. Required outputs

For common $h$ and $n$, the utility must return two length-$n$ arrays:

- `raw`: the linear-grid discretization of $\nu_{\mathrm{raw}}$;
- `secondary`: the linear-grid discretization of $\nu_{\mathrm{SU},K}$.

The arrays contain nonnegative rate weights by default. The loss associated with element $i$ is $ih$. The output metadata must state at least $h$, $n$, $K$, whether the arrays are normalized, the total input rate, the analytical expected annual loss, the two output masses and first moments, the finite-$K$ secondary mean error, the number of valid rows, and counts of every exceptional-row category.

The normal Python result must be directly usable as NumPy arrays without requiring a textual intermediate representation. A supported output mode must also permit the arrays to reside in, or be reopened from, a NumPy-compatible memory-mapped file.

## 8. Numerical and behavioural requirements

The transformation must be deterministic for a fixed input, grid and $K$. Reordering input rows may change the final few floating-point bits through summation order, but it must not produce a material numerical difference. Repeated execution under the same deterministic mode should produce bitwise-identical results where reasonably achievable, or else document the narrower reproducibility guarantee.

No invalid input may produce `NaN`, infinity, a negative output weight, memory corruption or an unreported row omission. Diagnostics should collect counts and representative row identifiers rather than emit a message for every bad row in a million-row file.

The utility must handle ELTs of approximately one million rows on an ordinary workstation. Performance reports must distinguish input parsing, parameter validation, beta-quantile evaluation, loss-grid accumulation and output creation. End-to-end throughput must be reported in rows per second and beta-quantile evaluations per second, together with the processor, memory, grid size, $K$ and input format used.

## 9. Correctness tests

The acceptance suite must include the following mathematical tests.

1. Zero uncertainty: with $\sigma_{I,j}=\sigma_{C,j}=0$ for every row, the secondary and raw distributions agree, subject only to the specified treatment of repeated degenerate points.
2. Exact grid points: a loss located exactly at $ih$ contributes all its weight to element $i$.
3. Linear interpolation: an off-grid atom preserves both its weight and its first moment exactly to floating-point tolerance.
4. Uniform beta: $\alpha=\beta=1$ gives midpoint points $x_k=M(k+\tfrac12)/K$.
5. Symmetry: swapping $\alpha$ and $\beta$ reflects normalized beta quantiles about $1/2$.
6. Analytical moments: beta parameters reconstructed from admissible $(M,\mu,\sigma)$ reproduce the stated mean and variance.
7. Quadrature convergence: increasing $K$ reduces the midpoint approximation error against a trusted high-accuracy beta reference over representative and difficult parameter regimes.
8. Mass conservation: both grid arrays sum to $\Lambda$ within the stated floating-point tolerance.
9. Mean conservation: the raw grid first moment equals the analytical expected annual loss; the secondary grid first moment equals the quadrature first moment.
10. Scale invariance: multiplying every loss quantity and $h$ by the same positive constant leaves grid indices and output weights unchanged.
11. Invalid beta moments: inconsistent rows follow the selected error or repair policy and appear in diagnostics.
12. Boundary means: rows with $\mu=0$ or $\mu=M$ follow the specified degenerate treatment.
13. Grid overflow: an insufficient grid produces the specified failure or explicit overflow behaviour.
14. Large-file reference: a large deterministic synthetic ELT agrees with an independent high-accuracy implementation within stated tolerances.

## 10. Performance test cases

The performance suite must include at least:

- a one-million-row well-conditioned ELT;
- a one-million-row mixture containing degenerate and nondegenerate events;
- strongly skewed beta shapes, including parameters below one;
- highly concentrated beta shapes with large parameters;
- several grid sizes and several values of $K$, including a production-scale candidate such as $K=32$;
- both direct in-memory output and memory-mapped output.

Correctness must be established before performance comparisons are accepted. Approximate numerical methods are permissible only when their errors are measured against the reference calculation and controlled relative to the requested loss-grid resolution.

## 11. Assumptions requiring confirmation

Before the utility's public interface is frozen, we must confirm:

1. whether the ELT contains $\lambda_j$ directly or requires a join to an event catalogue;
2. whether the exposure field is exactly the conditional maximum loss $M_j$;
3. whether $\sigma_j^2=\sigma_{I,j}^2+\sigma_{C,j}^2$ is the intended portfolio-marginal convention;
4. whether the desired default output is a rate measure or a normalized conditional severity probability mass function;
5. the production defaults for $h$, $n$ and $K$;
6. the required policy for inadmissible beta moments;
7. whether any finite-$K$ moment correction is desired;
8. whether an explicit overflow bin is ever preferable to requiring full support coverage.

These questions affect the mathematical contract and should be resolved identically for every competing implementation used in a head-to-head comparison.
