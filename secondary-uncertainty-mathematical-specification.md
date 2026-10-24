# Portfolio ELT Secondary-Uncertainty Quantization

## Problem and mathematical specification

Date: 2026-08-14; amended 2026-10-24 for the survival method, the Bernoulli treatment of inadmissible rows and the overflow policies (`dev/done/plan-single-unit-survival-route.md`).

## 1. Purpose

We require a utility that converts a portfolio event loss table (ELT) into two discrete, evenly spaced, rate-weighted severity distributions. The first distribution represents the original ELT, in which each event produces its conditional mean loss. The second incorporates secondary uncertainty by replacing each conditional mean with a bounded beta severity distribution. That distribution is discretized either through the portfolio exceedance rate, evaluated exactly and bucketed onto the grid (section 5A, the default), or by midpoint quantiles mapped to the grid by a linear interpolation rule (section 5). The raw distribution and every atom use the linear rule.

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
(Using the physicist notation for expectation wrt $\mathsf P$.) For the one-dimensional marginal severity distribution required here, the total conditional standard deviation is the arithmetic sum of the two components,
$$
\sigma_j=\sigma_{I,j}+\sigma_{C,j},
$$
the vendor convention (Burke, Rau-Chaplin, Varghese 2016, section III.E, step i; confirmed against the paper 2026-08-25). It is not the root sum of squares: in the underlying one-factor model the components enter the marginal sd arithmetically while the copula weights normalize in quadrature, and the two must not be conflated. `dev/secondary-uncertainty-additivity.md` sets out that model, its consequences for aggregation across units, and why unit-level structure must be retained whenever unit-level terms exist; the one-dimensional marginal here is the single-unit case. Versions before 1.6.0 combined the components in quadrature; that was incorrect and has been replaced outright.

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
The boundary cases require explicit treatment. If $\sigma_j=0$, the conditional distribution is the atom $X_j=\mu_j$. If $\mu_j=0$, admissibility requires $X_j=0$ almost surely; if $\mu_j=M_j$, admissibility requires $X_j=M_j$ almost surely.

The variance bound is a property of the support, not of the beta family. On $[0,M]$, $X^2\le MX$, so $\operatorname{Var}X\le M\mu-\mu^2=\mu(M-\mu)$, with equality only when $X^2=MX$ almost surely, that is, for the Bernoulli law on $\{0,M\}$ with $\mathsf P(X=M)=\mu/M$. A row with $\sigma_j^2\ge\mu_j(M_j-\mu_j)$ therefore asserts a variance that no distribution on $[0,M_j]$ can have. It must not be silently altered; the utility applies one selected and fully reported policy:

- `bernoulli` (the default from 1.13.0): the row becomes the variance maximizer, two atoms with rates $\lambda_j(1-p_j)$ at $0$ and $\lambda_jp_j$ at $M_j$, where $p_j=\mu_j/M_j$. Its rate and mean are exact and its variance is the admissible supremum. Reported under `variance_bernoulli`.
- `error`: the row is rejected with a typed error.
- `drop`: the row is removed from both outputs. Reported under `variance_dropped`.
- `atom`: the row contributes a single atom at $\mu_j$. Reported under `variance_atom`.
- `clamp`: the default until 1.12.0, which set $\sigma_j=0.999\sqrt{\mu_j(M_j-\mu_j)}$ and kept the row as a beta. That beta has $\alpha_j+\beta_j\approx0.002$, the Bernoulli to double precision except that about 5% of its upper mass is smeared across the interior, and the 0.999 is an arbitrary constant standing in for the maximizer. Removed from the one-row route at 2.0.0; the multi-unit route keeps it until that route adopts the Bernoulli treatment.

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

The midpoint-quantile rule is the `quantiles` method, selectable but not the default from 2.0.0. The fraction of event $j$'s $K$ points above a loss $x$ is the number of $k$ with $u_k>F_j(x)$ divided by $K$, which is $KS_j(x)$ rounded to the nearest integer and divided by $K$, where $S_j=1-F_j$ is the conditional survival function. The rule therefore sets every event's survival function to zero wherever it is below $1/(2K)$: at $K=32$ an event is cut off above its $1-1/64$ quantile. For a beta with small $\alpha_j$ most of the probability sits near zero and much of the mean sits in an upper tail of small probability, exactly where the rule puts nothing, and the loss of expected value falls only in proportion to $1/K$. Section 5A gives the default method, which has no such cut.

## 5A. Survival-function discretization

The default method (`survival`, from 2.0.0) discretizes the secondary severity measure through the portfolio exceedance rate rather than through per-event points. Split the contributing rows of section 4 into atoms (degenerate rows, rows made atoms by policy, and both atoms of a Bernoulli row) and betas. For the betas define
$$
\Lambda(x)=\sum_{j\ \text{beta}}\lambda_jS_j(x),
\qquad
\Lambda'(x)=-\sum_{j\ \text{beta}}\frac{\lambda_j}{M_j}f_j\!\left(\frac{x}{M_j}\right),
$$
where $S_j(x)=1-I_{x/M_j}(\alpha_j,\beta_j)$ is the conditional survival function of $X_j$ and $f_j$ the density of $\operatorname{Beta}(\alpha_j,\beta_j)$. $\Lambda$ is nonincreasing and adds over events, so contributions computed separately, by chunk or by a later route, combine on common nodes.

**Nodes.** $\Lambda$ is built on nodes $x_0<x_1<\dots<x_m$ spaced geometrically from a lower loss $x_0$ to $x_m=\max_jM_j$, at a density of `nodes_per_decade` nodes per decade; the default density, 100, is the smallest whose exact-edge error is within $10^{-4}$ on every benchmark portfolio (`benchmarks/report-1.16.0.md`). $x_0$ is the smallest loss the consumer uses: for the even grid of section 6 it is $e_0=h/2$. The survival value and the slope share the factor $x^{\alpha}(1-x)^{\beta}/B(\alpha,\beta)$ of the incomplete-beta continued fraction (Press et al. 1992, section 6.4), so the slope costs little beyond the value. The constant $\ln B(\alpha,\beta)$, with $a\le b$ the ordered parameters and $b\ge100$, is $\ln\Gamma(a)+a-(b-\tfrac12)\log(1+a/b)-a\ln(a+b)+\sigma(b)-\sigma(a+b)$, the two Stirling series (Press et al. 1992, section 6.1) for $\ln\Gamma(b)-\ln\Gamma(a+b)$ subtracted symbolically, $\sigma(z)=\sum_{k=1}^{6}B_{2k}/(2k(2k-1)z^{2k-1})$: the difference of three $\ln\Gamma$ values cancels to $O(a\ln b)$ from terms of size $b\ln b$, which at $\beta=10^8$ costs seven digits and reached the survival as $3\times10^{-6}$ relative on real events with $\alpha$ below 2 (test 23). Above the switch point $x=(\alpha+1)/(\alpha+\beta+2)$ the fraction evaluates $I_{1-x}(\beta,\alpha)$ and the rounding of $1-x$ leaves a residual of order $\varepsilon\beta$, $10^{-8}$ at $\beta=8\times10^7$, a property of the formulation, not of the evaluation. A node below $M_j\exp\{[\ln u+\ln\alpha_j+\ln B(\alpha_j,\beta_j)]/\alpha_j\}$, the closed-form lower-tail quantile at a probability $u$ small enough that $F_j<2^{-54}$ there, contributes exactly $\lambda_j$ to $\Lambda$ in double precision, and its contribution to $\Lambda'$, below the rounding of $\Lambda$, is dropped.

**Truncation.** Every survival function has an infinite slope in $\log x$ at its endpoint $M_j$, and the sum of such endings on shared nodes is not interpolable in $x$ wherever one event, or several ending together, carry much of $\Lambda$: measured on this project's own medium golden and benchmark portfolio, a pure interpolant missed by 1% at the 100-year level and by factors in the tail (execution log of `dev/done/plan-single-unit-survival-route.md`). Each event is therefore split at its **truncation node** $x_{k_j}$, the node $Z$ intervals below the first node at or above $M_j$ ($Z=4$, the constant `TAIL_INTERVALS`). The **curve**
$$
\tilde\Lambda(x)=\sum_{j\ \text{beta}}\lambda_j\bigl(S_j(x)-S_j(x_{k_j})\bigr)^+
$$
carries each event up to its truncation node, where its contribution vanishes continuously, and the **tail**, the mass $\lambda_jS_j(x_{k_j})$ on $(x_{k_j},M_j]$, stays with the event. $\tilde\Lambda$ and its slope $-x\tilde\Lambda'(x)$ are evaluated exactly at the nodes; the slope at a node is one-sided, an event's slope at its truncation node counting on the incoming interval only. An event whose truncation node would lie below $x_0$ is a tail from $x_0$. So $\Lambda(x)=\tilde\Lambda(x)+\sum_{j:\,x<x_{k_j}}\lambda_jS_j(x_{k_j})+\sum_{j:\,x_{k_j}\le x<M_j}\lambda_jS_j(x)$, the last sum over the few events whose tail interval contains $x$.

**Interpolation of the curve.** Between nodes $\tilde\Lambda$ is a cubic Hermite interpolant in $x$ with the exact one-sided slopes, limited by the Fritsch and Carlson conditions so that it never increases: on an interval whose secant slope is zero both end slopes are set to zero, and otherwise the two end slopes $a$ and $b$, in units of the secant slope, are scaled down together whenever $a^2+b^2>9$. Above the last node $\tilde\Lambda=0$. A log-log interpolant was planned and rejected: near a node where several truncated survival functions vanish together the logarithm magnifies the curvature, and it missed by $10^{-2}$ where the plain cubic is within $10^{-5}$. The interpolation error is a reported diagnostic: the exact $\tilde\Lambda$ at 64 log-midpoints spread over the node range against the interpolant, as the largest error relative to the exact $\Lambda$ where $\Lambda>10^{-12}\sum_j\lambda_j$. It is a sample, not a bound; the reference implementation can check every interval midpoint.

**The tails.** Near its endpoint a beta's survival is $S_j(x)=(1-y)^{\beta_j}R_j(y)$ with $R_j$ analytic and nonzero at $y=1$, so $\log S_j$ is nearly linear in $u=\log(M_j-x)$. On its tail interval each event's survival is a monotone cubic Hermite in $u$ with the exact slopes $(1-y)f_j(y)/S_j(y)$, on nodes shrinking the distance to the endpoint by $\sqrt2$ per step from $M_j-x_{k_j}$ down to $x_0/2$, and the exact power law $S_j\propto(M_j-x)^{\beta_j}$ below the last node. The first node's value is the exact $S_j(x_{k_j})$, so the tail joins the curve exactly. This costs about $2\log_2\bigl((M_j-x_{k_j})/x_0\bigr)$ exact evaluations per event plus one cubic per bucket edge walked (the rate cut below), and follows the survival to a few units in $10^{-6}$ where no interpolant in $x$ can.

**Bucketing (the nearest rule).** With the even grid of section 6 and the half-bucket edges $e_i=(i+\tfrac12)h$, the beta part of the secondary grid is
$$
p_0=\sum_{j\ \text{beta}}\lambda_j-\tilde\Lambda(e_0)-\sum_j\lambda_jS_j(x_{k_j}),
\qquad
p_i=\tilde\Lambda(e_{i-1})-\tilde\Lambda(e_i)+\sum_{j}\lambda_j\bigl(S_j(e_{i-1})-S_j(e_i)\bigr)\quad(1\le i\le n-1),
$$
the last sum over the events whose tail interval meets $(e_{i-1},e_i]$ (with $S_j$ read as $S_j(x_{k_j})$ below the tail interval and $0$ above $M_j$), and the overflow of section 6 is $\tilde\Lambda(e_{n-1})$ plus the tails beyond $e_{n-1}$, its first moment by partial expectations. Atoms are deposited by the linear rule of section 6 as before. Every $p_i\ge0$ because both interpolants are monotone, and the total rate is exact because the sums telescope. Measured on 2026-10-24 (reference implementation, 150 nodes per decade): the bucketed tail sums against the exact $\Lambda$ at every bucket edge where $\Lambda>10^{-12}\sum_j\lambda_j$ are within $4\times10^{-6}$ on the small-alpha fixture, $1.1\times10^{-5}$ on the medium golden and $1.1\times10^{-5}$ on the benchmark portfolio; with nodes alone and a log-log interpolant the same figures were 270, 0.78 and 7.0. The first moment is not exact. In the interior the bucketing error is of order $h^2$ times the slope of the density; from bucket 0, whose mass sits at loss 0, it is $\sum_j\lambda_j\int_0^{h/2}x\,dF_j\approx\sum_j\lambda_j\frac{\alpha_j}{\alpha_j+1}\frac h2F_j(h/2)$, first order in $h$ for $\alpha_j<1$. Both are reported through the secondary mean error of section 7. The linear rule, exact in the mean, was considered and rejected: it inflates the variance of every unit of mass by about $h^2/6$ against the nearest rule's $h^2/12$ (Sheppard's correction), an illusion of exactness. The nearest rule is also the rule the multi-unit route uses for beta units.

**The rate cut.** Walking a tail to its endpoint costs one cubic per bucket edge on $(x_{k_j},M_j]$, about $0.08M_j/h$ edges whatever the event's rate there, and for the low-mean events of a catastrophe ELT nearly all of those edges deposit nothing in double precision. Each tail is therefore walked in order from the first edge above $x_{k_j}$ and stopped at the first edge $e_s$ where $\lambda_jS_j(e_s)\le\tau=\varepsilon\sum_j\lambda_j$, with $\varepsilon=10^{-20}$ (`TAIL_RATE_FLOOR`), or at the top edge $e_{n-1}$; the remainder $\lambda_jS_j(e_s)$ goes in one deposit to the bucket above $e_s$, beyond the grid to the overflow with its moment by partial expectations. A tail whose whole mass $\lambda_jS_j(x_{k_j})$ is at most $\tau$ goes in one deposit to the bucket containing $x_{k_j}$. The rate stays exact. Each cut moves at most $\tau$ of rate within one event's tail interval, so the bucketed tail sums depart from the curve's $\Lambda$ by at most $N\tau$ at any edge ($10^{-14}$ of the total rate for a million events) and the first moment by at most $N\tau\max_jM_j$. The curve itself, $\Lambda(x)$ of section 7, is not cut. Measured on 2026-10-24 on a 100,000-row portfolio with endpoints to $10^9$ at $h=25$ and $2^{22}$ buckets: $2.7\times10^{10}$ edges before the cut and the window clip, $8.2\times10^8$ after, and the rate the cut moves $1.4\times10^{-16}$ of the total (`dev/done/plan-survival-tail-cost.md`).

**Narrow events.** A beta whose standard deviation is below one node interval at its mean, $\sigma_j<\mu_j(10^{1/n_d}-1)$, has a survival function that steps from $1$ to $0$ inside one interval, where the shared cubic, exact only at the nodes, cannot follow it: $\Lambda$ would be off by up to $\lambda_j$ across the interval (measured 2026-10-24: $5\times10^{-4}$ relative on a block with several hundred policies at a common limit, $1.8\times10^{-5}$ on a synthetic portfolio). Such an event, provided $\alpha_j,\beta_j>1$ (a bounded, log-concave density), is kept out of the curve. Its bulk $(s_j,t_j)$ is found by steps of $\sigma_j$ out from the mean, each side ending at the first step whose mass is at most a tenth of $\tau/\lambda_j$ of the mass accumulated so far (so the mass dropped beyond it is below the floor), or at $0$ and $M_j$; an event whose bulk does not reach that within 40 steps on either side (its mass piled at an endpoint) takes the regular route. Its survival on the bulk is evaluated by quadrature of its density: the density relative to its value at the mean, $f_j(y)/f_j(\bar y)=\exp\bigl((\alpha_j-1)\log(1+(y-\bar y)/\bar y)+(\beta_j-1)\log(1-(y-\bar y)/(1-\bar y))\bigr)$ with the logarithms of $1+\delta$ taken as such so that huge parameters lose no precision, integrated by 8-point Gauss and Legendre quadrature on pieces of at most half a standard deviation, and normalized by its integral over the bulk. Measured 2026-10-24 against SciPy: within $10^{-13}$ at $\alpha_j+\beta_j=10^4$, $10^{-12}$ at $10^6$, $10^{-11}$ at $10^8$, and within the rounding of the loss share itself beyond. The engine's own incomplete beta, a continued fraction capped at 400 iterations, is good only to about $10^{-10}$ near the mean at $\alpha_j+\beta_j=10^4$ and returns nothing usable above $10^7$ when both parameters are large (found the same day), which is why the narrow events, the one place the engine evaluates such a beta inside its bulk, do not use it; the reference uses SciPy. With one parameter small the fraction converges in tens of iterations at any $\beta$, and the error there was the constant $\ln B$ (the Nodes paragraph; `dev/done/plan-ln-beta-large-parameters.md`). Above $\alpha_j+\beta_j=10^{15}$ one unit in the last place of the loss share moves the density by more than a tenth and the event is a step at its mean. The event counts $\lambda_j$ in $\Lambda(x)$ below $s_j$, $\lambda_jS_j(x)$ on the bulk and nothing at or above $t_j$, each within $\tau$; on the grid it is walked along the edges of its bulk by the walk of the rate cut, from the bucket containing $\max(s_j,x_0)$ with the mass below $x_0$ left to bucket $0$, so the bound of the rate cut covers it. The count of narrow events is reported with the metadata (`narrow_events`).

**The curve as the primary object.** The nodes cover every loss up to $\max_jM_j$ in about `nodes_per_decade` $\times\log_{10}(\max_jM_j/x_0)$ points, whatever $h$ is. The even grid is made from the curve only when needed, at any $h\ge2x_0$ and any $n$, and the occurrence exceedance probability comes straight from it under Poisson occurrence,
$$
\operatorname{OEP}(x)=1-\exp\{-\Lambda(x)-\Lambda_{\text{atoms}}(x)\},
$$
where $\Lambda_{\text{atoms}}(x)$ is the total rate of the atoms above $x$. The utility exposes the curve with its atoms so that the OEP points need no grid.

**Determinism and cost.** Rows are processed in fixed chunks; each chunk's partial $\tilde\Lambda$ and slope vectors, and its tail deposits, are folded in chunk order with compensated sums, so results are bitwise identical for any thread count at a fixed chunk size. The cost is one incomplete beta and one density per event per active node, plus the tail work above, several times the cost of the quantiles method at $K=32$; the benchmark report records the measured figures and the chosen default density.

## 6. Even loss grid and linear discretization

Let the output grid have constant spacing $h>0$ and nodes
$$
g_i=ih,\qquad i=0,\ldots,n-1.
$$
When the grid length is derived from the input it covers every loss that can be generated,
$$
(n-1)h\geq\max_j M_j.
$$
A caller may instead ask for a window, a grid length short of that bound, for example because the exposed values run to tens of billions while the losses that matter sit far lower. The rate beyond the top bucket's upper edge $e_{n-1}=(n-\tfrac12)h$ is then the **overflow**: for the survival method $\Lambda(e_{n-1})$ plus the atoms above $e_{n-1}$, for the quantiles method and for the raw output the weight of the points above $e_{n-1}$. A point in $(g_{n-1},e_{n-1}]$ is assigned wholly to $g_{n-1}$, the linear rule having no upper node there. The overflow is handled by a named policy, applied to both outputs and both methods (from 2.0.0; before that the grid had to cover $\max_jM_j$ or the utility stopped with an error):

- `top_bucket` (default): the overflow is added to the top node $g_{n-1}$. The rate is exact, so a Poisson frequency fed from it is unchanged; the occurrence exceedance is exact for every $x\le e_{n-2}$; the first moment is understated by the overflow's displacement to $g_{n-1}$.
- `drop`: the overflow is discarded. The rate falls short by the overflow and the occurrence exceedance is low by it everywhere; the grid holds nothing it does not mean.
- `error`: a typed error whenever a contributing $M_j$ exceeds $e_{n-1}$, the condition under which the overflow can be nonzero.

Both non-error policies report, for each output, the overflow's rate and its first moment in excess of $g_{n-1}$ (`rate_above_grid`, `mean_above_grid`), so that a caller who prefers to post-process can. With the grid length derived from the input both figures are exactly zero, so the policy acts only when the caller asks for a window. Silent truncation, capping or wraparound remains unacceptable: the policy is named, and its effect is reported.

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
Consequently, the raw grid distribution must preserve the ELT's total rate and expected annual loss up to floating-point summation error and the reported effect of the overflow policy. The secondary grid distribution preserves the total rate exactly under both methods. Under the quantiles method it preserves the first moment of the midpoint-quantile approximation, so its difference from the analytical expected annual loss arises from finite-$K$ quadrature and floating-point error, not from the linear loss-grid rule; under the survival method the difference is the bucketing error of section 5A. Either way the difference is reported as the secondary mean error.

## 7. Required outputs

For common $h$ and $n$, the utility must return two length-$n$ arrays:

- `raw`: the linear-grid discretization of $\nu_{\mathrm{raw}}$;
- `secondary`: the bucketing of section 5A under the survival method (the default), or the linear-grid discretization of $\nu_{\mathrm{SU},K}$ under the quantiles method.

The arrays contain nonnegative rate weights by default. The loss associated with element $i$ is $ih$. The output metadata must state at least $h$, $n$, the method, whether the arrays are normalized, the total input rate, the analytical expected annual loss, the two output masses and first moments, the secondary mean error, the number of valid rows, and counts of every exceptional-row category. The secondary mean error is defined for both methods as the secondary grid's first moment in rate-weighted units less the analytical expected annual loss. Under the quantiles method the metadata also states $K$ and the quadrature secondary mean; under the survival method the node count, the nodes per decade and the sampled interpolation error. Under either method it states the overflow policy and, for each output, the overflow's rate and first moment above the top node (section 6).

The survival method also exposes the exceedance curve itself: the nodes with the exact $\Lambda$ at them, the atoms with their rates, $\Lambda(x)$ with the atoms included, the even grid at any $h\ge2x_0$ and $n$, and the occurrence exceedance points of section 5A under Poisson occurrence, where a step at an atom that jumps the target answers with that atom's loss.

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

Tests 4, 5 and 7 are tests of the quantiles method and select it explicitly; the others apply to both methods. The survival method adds the following, on a synthetic fixture of small-$\alpha$ events ($\alpha$ from $0.003$ to $3$, $M/\mu$ from $10^2$ to $10^5$, some atoms and inadmissible rows, and a gross-like copy with $\mu$ shrunk and $M$ kept).

15. Exceedance accuracy: the tail sums of the survival grid match the exact incomplete beta at every bucket edge within $10^{-4}$ relative wherever $\Lambda>10^{-12}\sum_j\lambda_j$; the reference's check of every interval midpoint and the sampled interpolation error are both at most $10^{-4}$.
16. Rate and mean: the secondary mass equals the total rate within $10^{-14}$ relative; with the grid length derived from the input, the secondary mean error relative to the analytical expected annual loss is at most $10^{-4}$ at the fixture's $h$ and falls at least linearly when $h$ is halved twice.
17. Nonnegativity and determinism: no negative weight; bitwise identical results across thread counts at a fixed chunk size; the two implementations agree on the curve within $10^{-12}$ relative and on the grids within $10^{-12}$ of the total rate.
18. Overflow: with a grid length short of $\max_jM_j$, for both methods and both outputs, `top_bucket` adds the overflow to the top node and reports its rate and displaced moment, `drop` leaves the mass short by exactly that and reports the same figures, and `error` raises the typed error; with the grid length derived from the input both figures are exactly zero.
19. The K-point cut: on the same fixture the quantiles method at $K=32$ has $\Lambda=0$ above each event's $1-1/64$ quantile (plus one bucket of linear spread) and a secondary mean error below $-5\%$ of the expected annual loss, while the survival method is within the tolerance of test 16. This documents the defect.
20. Bernoulli rows: an inadmissible row contributes $\lambda(1-p)$ at $0$ and $\lambda p$ at $M$ in both methods, with rate and mean exact and grid variance $\mu(M-\mu)$ to bucketing tolerance; under the survival method the exceedance curve has an exact step of $\lambda p$ at $M$; the row is reported under `variance_bernoulli`.
21. The rate cut: on the fixture the survival grid with the cut and the same grid with every tail walked to its endpoint have tail sums within $N\varepsilon\sum_j\lambda_j$ of each other at every edge, the rate is exact in both, and the two implementations still agree as in test 17.
22. Narrow events: on a synthetic portfolio with events of coefficient of variation from $10^{-6}$ to $10^{-2}$ at a common mean, both implementations report them as narrow, the exceedance curve is within $10^{-8}$ relative of the exact $\Lambda$ across $[0.97,1.03]$ times that mean, the grid's tail sums are within $10^{-7}$ of the exact $\Lambda$ at the edges there, the rate is exact, and the two grids agree as in test 17.
23. Large parameters: on one-row portfolios with $\alpha\in\{0.13,0.5,1,1.5,2.5\}$ and $\beta\in\{10^4,10^6,10^7,8\times10^7\}$, both implementations' node values from 0.3 to 30 times the mean are within $10^{-13}$ relative of the regularized incomplete beta computed in multiple precision below the switch point $(\alpha+1)/(\alpha+\beta+2)$ and within $2\times10^{-8}$ above it where $S\ge10^{-15}$; on a synthetic portfolio with 300 such events among the base rows of test 22, the node values are within $10^{-11}$ of the exact $\Lambda$, the grid's tail sums within $10^{-5}$ of it at the sampled edges where $\Lambda\ge10^{-12}\sum_j\lambda_j$, the rate exact and the two grids within $10^{-12}$ of the total rate of each other.

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
3. **resolved (1.6.0):** the marginal convention is $\sigma_j=\sigma_{I,j}+\sigma_{C,j}$ (Burke 2016, section III.E, step i; see section 4 and `dev/secondary-uncertainty-additivity.md`);
4. whether the desired default output is a rate measure or a normalized conditional severity probability mass function;
5. **resolved (2.0.0):** the default method is `survival` at the node density set by the benchmark; the quantiles method stays selectable with $K=32$ by default; $h$ and $n$ remain the caller's (sections 5A and 6);
6. **resolved (1.13.0):** inadmissible rows become Bernoulli rows by default (section 4);
7. whether any finite-$K$ moment correction is desired;
8. **resolved (2.0.0):** no overflow bin; a named overflow policy, `top_bucket` by default, with the overflow's rate and moment reported (section 6).

These questions affect the mathematical contract and should be resolved identically for every competing implementation used in a head-to-head comparison.
