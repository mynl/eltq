# Portfolio ELT Secondary-Uncertainty Quantization

## Problem and mathematical specification

Date: 2026-08-14; amended 2026-10-24 for the survival method, the Bernoulli treatment of inadmissible rows and the overflow policies (`dev/done/plan-single-unit-survival-route.md`); amended 2026-10-25 for several units in one event, section 4A (`dev/plan-multiunit-coupling.md`).

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
A row with $0<\sigma_j<2^{-48}\mu_j$ (sixteen units in the last place of the loss at the mean) is an atom at $\mu_j$ as well, reported under `atom_by_resolution` apart from the stated zero: no coordinate in double precision resolves a law that narrow, and treating it as the atom it is keeps it on the same linear deposit as its $\sigma_j=0$ twin (from 2.3.4).

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
- `clamp`: the default until 1.12.0, which set $\sigma_j=0.999\sqrt{\mu_j(M_j-\mu_j)}$ and kept the row as a beta. That beta has $\alpha_j+\beta_j\approx0.002$, the Bernoulli to double precision except that about 5% of its upper mass is smeared across the interior, and the 0.999 is an arbitrary constant standing in for the maximizer. Removed from the one-row route at 2.0.0 and from the multi-unit route by `dev/plan-multiunit-coupling.md` [bernoulli] (section 4A); between 1.13.0 and that step the multi-unit route kept it as its default.

## 4A. Several units in one event

A **multi-unit ELT** has one row per event and unit: the fields of section 2 for unit $i$ of event $j$, written $\mu_{ij}$, $\sigma_{I,ij}$, $\sigma_{C,ij}$, $M_{ij}$, with the rate $\lambda_j$ constant within the event. A unit is any part of the portfolio with its own rows (a line, a region, a program, a cedent). The rows of one event must be contiguous and the units within an event distinct; both are validated, not repaired. Each row is classified exactly as a one-row ELT row is in section 4: degenerate rows are atoms at $\mu_{ij}$, inadmissible rows ($\sigma_{ij}^2\ge\mu_{ij}(M_{ij}-\mu_{ij})$, $\sigma_{ij}=\sigma_{I,ij}+\sigma_{C,ij}$) follow the selected variance policy with `bernoulli` the default, and admissible rows are betas moment-matched to $(\mu_{ij},\sigma_{ij})$ on $[0,M_{ij}]$. The marginal distribution of each unit's loss is therefore the one-row model of section 4, whatever the coupling below. The article *Secondary Uncertainty in Catastrophe Model Event Loss Tables for Multi-Unit Portfolios* (`D:\Projects\aggregate\secondary-uncertainty\MVMELT.qmd`, the multi-unit article below) derives and illustrates everything in this section; `dev/secondary-uncertainty-additivity.md` holds the proofs of the additivity propositions.

**The model.** The units of an event are coupled by the one-factor Gaussian copula of Burke, Rau-Chaplin and Varghese 2016, section III.E. With one standard normal $Z_c$ per event occurrence and independent standard normals $Z_i$ per unit, unit $i$ has the normal score and loss
$$
V_i=w_iZ_c+\sqrt{1-w_i^2}\,Z_i,\qquad X_i=M_iF_i^{-1}(\Phi(V_i)),
$$
where $F_i$ is the distribution function of the unit's marginal (the beta, or the two-point law of a Bernoulli row, or the step at $\mu_i$ of an atom) and $w_i\in[0,1]$ is the unit's **loading** on the common factor. $\Phi(V_i)$ is uniform, so the marginals are exact for every loading. Two units of an event have normal-score correlation $w_iw_k$ and the losses have a Gaussian copula with a one-factor correlation matrix. Conditional on $Z_c=z$ the units are independent, with
$$
\mathsf P(X_i\le x\mid Z_c=z)=\Phi\!\left(\frac{y_i(x)-w_iz}{\sqrt{1-w_i^2}}\right),\qquad y_i(x)=\Phi^{-1}\bigl(F_i(x/M_i)\bigr),
$$
for $w_i<1$, and $\mathsf P(X_i\le x\mid Z_c=z)=\mathbf 1\{y_i(x)\ge z\}$ for $w_i=1$. A Bernoulli row fits this as it stands with $F_i$ its step function: given $z$ the unit loses $M_i$ with probability $1-\Phi\bigl((\Phi^{-1}(1-p_i)-w_iz)/\sqrt{1-w_i^2}\bigr)$, $p_i=\mu_i/M_i$, and nothing otherwise. An atom is the same at every $z$. The pair $(Z_c,Z_1,Z_2,\dots)$ is drawn afresh at each occurrence, so occurrences are independent and identically distributed and the compound Poisson structure of sections 3 to 5A applies to the event totals unchanged.

**Coupling rules.** The loading is the one quantity the published sources leave open, and the utility exposes it as a named `coupling`, applied to every event of a run and reported with the metadata:

- `burke` (the default): $w_i=\sigma_{C,i}/\sqrt{\sigma_{I,i}^2+\sigma_{C,i}^2}$ (Burke 2016, step iv), the correlation structure of the linear model $X_i=\mu_i+\sigma_{C,i}Z_c+\sigma_{I,i}Z_i$ whose moment rule is the roll-up below.
- `comonotone`: $w_i=1$ for every unit, one uniform per event occurrence (Mitchell-Wallace et al. 2017, section 1.10.5.9, "full correlation in uncertainty"; Burke's extreme $\sigma_I=0$).
- `independent`: $w_i=0$ for every unit (Burke's extreme $\sigma_C=0$).
- `homer`: the roll-up and pro rata allocation of Homer and Li 2017, section 3.2.2, equations 38 to 42. The event's rows are rolled into one, $\mu_R=\sum_k\mu_k$, $M_R=\sum_kM_k$, $\sigma_{I,R}=(\sum_k\sigma_{I,k}^2)^{1/2}$, $\sigma_{C,R}=\sum_k\sigma_{C,k}$ (the same rule as Mitchell-Wallace et al. 2017, section 1.10.5.10), the rolled row is classified by section 4, its loss $X_R$ is allocated $X_k=\min(X_R\mu_k/\mu_R,M_k)$, and the units' terms apply to the $X_k$. The cap at $M_k$ is this specification's (the published rule allocates up to $M_R\mu_k/\mu_R$, which exceeds $M_k$ whenever a unit's share of the mean exceeds its share of the exposure, as it does in the paper's own Table 11); the capped allocations sum to at most $X_R$, and the mean lost to the cap is reported per run (`homer_cap_mean_loss`). Under this coupling the units do not have their own marginals, the event total before terms is the rolled row's one-row distribution, and the whole route is one-dimensional: the one-row survival method of section 5A on the rolled table composed with the nondecreasing piecewise-linear map $x\mapsto\sum_k\mathrm{term}_k(\min(x\mu_k/\mu_R,M_k))$.
- `matched` (optional, last step of the plan): Burke's loadings scaled by a common factor $c\ge0$ per event, $w_i=\min(c\,\sigma_{C,i}/t_i,1)$, with $c$ the root of $\operatorname{Var}(\sum_iX_i)=(\sigma_{I,R}+\sigma_{C,R})^2$, the roll-up convention's variance of the event total; the variance is computed from the conditional moments on the grid and increases continuously in $c$ from the independent value, so the root is found by bracketing. When the target is not reached before the largest loading saturates at one, the saturated loadings are used and the shortfall is reported per event (`matched_shortfall`). This loading has no published source; it is the convention under which the copula and the roll-up agree on second moments (the multi-unit article, section on the joint distribution).

**Three routes to the portfolio total.** Given unit rows there are three ways to a portfolio distribution. Two of them compute the model and agree exactly; the third does not. The **roll-up route** (roll each event up, refit, run the one-row method) is the projection of the model onto the beta family: exact in the mean always, exact in distribution only when every $\sigma_{I,i}=0$ and all units share $(\alpha,\beta)$ (Proposition 6 of the additivity note), and otherwise off by a few percent in the standard deviation and tail quantiles of an event total in no fixed direction, for three reasons that do not shrink with any refinement: the marginal convention ($\sigma_I+\sigma_C$ against $t$), copula attenuation (the loss correlation is strictly below $w_iw_k$ by Lancaster's theorem), and the beta family not being closed under addition. The last reason makes the roll-up wrong even when every $\sigma_{C,i}=0$ and the units are independent given the event: the rolled variance is then exact, but a sum of betas is not a beta. The **conditional route**, the event total formed from the joint law of the units conditional on $Z_c$ and integrated over it, then the one-row compound Poisson calculation on the event totals, is the model, and it is the route the utility computes. The **joint aggregate route** forms the same joint law at each event, mixes the events into the joint law of the units at an occurrence, compounds it with one Poisson count for all the units into the joint annual distribution, and adds the units' annual losses. It computes $\sum_i\sum_j\sum_{n\le N_j}X_{ij}^{(n)}$ where the conditional route computes $\sum_j\sum_{n\le N_j}\sum_iX_{ij}^{(n)}$, the same finite sum in the other order, so the two annual totals are one random variable. The count must be shared: the annual losses $A_i$ of the units are dependent through it even under independence given the event, $\operatorname{Cov}(A_i,A_k)=\sum_j\lambda_j\mathsf P(X_{ij}X_{kj})\ge\sum_j\lambda_j\mu_{ij}\mu_{kj}$ for $i\ne k$, so per-unit aggregates computed one unit at a time do not determine the joint, and adding them as if independent drops these cross terms. Neither of the two routes fits a beta to a sum. The joint aggregate route needs a grid with one axis per unit and in return gives the joint annual distribution, which an allocation and any term on a unit's annual loss (an aggregate deductible or limit on one unit) need; the conditional route works on one-dimensional grids but forms the total at each occurrence, so it cannot apply annual terms by unit. For two units the utility supplies the occurrence joint (`joint_two_units`), and the compounding in two dimensions is done by `aggregate`'s bivariate aggregate. The roll-up appears only as the `homer` coupling and as a diagnostic (`roll_up_diagnostic`: per event, the standard deviation and 99th percentile of the exact ground-up total against the rolled row's beta).

**Terms and views.** Per-unit terms (a deductible and limit; occurrence reinsurance clauses) are applied to each unit's conditional distribution before the convolution across units, so that `ground_up`, `gross`, `ceded` and `net` totals are the sums of the per-unit views. Their grammar, semantics and grid alignment rules are those of `dev/done/plan-multiunit.md` (Terms grammar) and are unchanged here.

**The integral over the factor.** Every joint quantity is computed as a conditional quantity at a fixed $z$, where the units are independent (a convolution for a total, an outer product for a pair), followed by a quadrature over $z$ against the standard normal density with $Q$ nodes. The rule and $Q$ are selectable (`z_rule`, `z_nodes`) and reported. No quadrature rule reproduces a unit's marginal exactly at finite $Q$: the midpoint rule (equal weights at $z_q=\Phi^{-1}((q+\tfrac12)/Q)$) cuts the tails of $Z_c$, which at $Q=32$ moved the mean of a lone beta unit with $w=1/\sqrt2$ by $6\times10^{-4}$ and the survival of units with $w$ near 1 by factors at the $10^{-3}$ level (`dev/done/plan-single-unit-survival-route.md`, Not in this plan), while Gauss-Hermite at 32 nodes is exact in the mean of that unit to the cent but misses the marginal of a unit with $w$ near 1 by $2\times10^{-2}$ (`benchmarks/z-convergence-1.7.0.md`, case B). The requirements are therefore: (i) an event with one contributing unit never enters the mixture; its total is the one-row distribution of section 4, bucketed as section 5A buckets it; (ii) the default rule and $Q$ are set by a convergence study with tail metrics (the exceedance of the event total at the $10^{-2}$ to $10^{-5}$ levels against a $Q=4001$ reference, not the 99th percentile alone), recorded in the plan; (iii) whatever the rule, the per-view mass is exact and the view means reconcile to the analytic layer means within a tolerance stated for the chosen $(Q,h)$ and reported (`view_mean_error`). The stratified rule, in which the conditional distribution is averaged over each of $Q$ equal-probability strata of $Z_c$ and the units are treated as independent given the stratum, has exact marginals and means by construction and is the candidate the study must include; it is a different copula at finite $Q$, converging to the model as $Q$ grows, and its dependence error is what the study measures.

**Bivariate outputs.** For two named units $u$ and $w$ the utility returns, on a common two-dimensional grid of spacing $h$, the rate-weighted joint occurrence severity
$$
S_{uw}=\sum_j\lambda_jJ_{j,uw},
$$
where $J_{j,uw}$ is the joint law of the two units given event $j$ (an atom at zero on the axis of a unit not exposed to the event), each $J_{j,uw}$ built as a sum of $Q$ outer products of conditional bucket probabilities. For each unit $u$ it likewise returns the joint of the unit with the event total, $S_{u\Sigma}=\sum_j\lambda_jJ_{j,u\Sigma}$, built from the pair (unit, convolution of the other units) by the shear $(a,b)\mapsto(a,a+b)$; conditional on $z$ the unit and the rest are independent, and the convolutions of the others for all $n$ units of an event are formed in linear time by prefix and suffix products of the unit spectra. These are arrays, with their grids and masses, and nothing more: the annual bivariate aggregate under the shared Poisson count $\Lambda$, its marginals, and the conditional mean of a unit given the total, $\kappa_u(x)=\mathsf P(A_u\mid A=x)$, are computed by the aggregate package from $S/\Lambda$ as a discrete bivariate severity (`BivariateAggregate(mode="discrete")`, `exeqa_df`), and the utility does not duplicate them. The marginals of $S_{uw}$ are the two units' rate-weighted occurrence severities under the chosen rule, and the second marginal of $S_{u\Sigma}$ is the ground-up total of the same run, to the rule's error; $\sum_u\kappa_u(x)=x$.

**Metadata.** In addition to section 7: the coupling, the rule and $Q$, per view the mass, the mean, the analytic layer mean and their difference, the count of one-unit events taken outside the mixture, the Bernoulli, atom and dropped row counts by category, `homer_cap_mean_loss` and the per-event `matched_shortfall` list where those couplings are selected, and the roll-up diagnostic when requested.

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

**Large parameters.** When both parameters are large the inversion of the incomplete beta is not available in double precision: the continued fraction (section 5A, Narrow events) needs about $\sqrt{\min(\alpha_j,\beta_j)}$ iterations near the mean and stops at 400, so its quantiles drift from $\min(\alpha_j,\beta_j)$ about $5\times10^5$ ($10^{-5}$ standard deviations at $10^6$, $10^{-2}$ at $10^7$, measured 2026-10-25 against SciPy's inverse), and the front factor $\exp(\alpha_j\log y+\beta_j\log(1-y)-\ln B)$ carries an absolute rounding of about $\varepsilon(\alpha_j+\beta_j)$ in its exponent, which overflows beyond about $5\times10^{18}$; between, the bracketing solver converges to dyadic points, so such rows got wrong points silently until 2.3.2. For $\min(\alpha_j,\beta_j)\ge2\times10^5$ the quantile is therefore the Cornish-Fisher asymptotic expansion in the cumulants (Abramowitz and Stegun 1970, 26.2.49) to second order,
$$
Q_{\alpha,\beta}(u)\approx m+s\Bigl[z+\frac{\gamma_1}{6}(z^2-1)+\frac{\gamma_2}{24}(z^3-3z)-\frac{\gamma_1^2}{36}(2z^3-5z)\Bigr],\qquad z=\Phi^{-1}(u),
$$
with the beta's mean $m=\alpha/(\alpha+\beta)$, variance $s^2=\alpha\beta/((\alpha+\beta)^2(\alpha+\beta+1))$, skewness $\gamma_1=2(\beta-\alpha)\sqrt{\alpha+\beta+1}/((\alpha+\beta+2)\sqrt{\alpha\beta})$ and excess kurtosis $\gamma_2=6[(\alpha-\beta)^2(\alpha+\beta+1)-\alpha\beta(\alpha+\beta+2)]/(\alpha\beta(\alpha+\beta+2)(\alpha+\beta+3))$, which follow from its moments; $u_k=0$ and $1$ give $0$ and $M_j$. Measured against SciPy's inverse, worst over the 32 midpoint levels and $\beta/\alpha$ from $1$ to $3\times10^4$: $9\times10^{-9}$ standard deviations at $\min(\alpha,\beta)=10^5$, $4\times10^{-9}$ at $1.8\times10^5$, $2\times10^{-9}$ at $3.2\times10^5$, falling as $0.3\min(\alpha,\beta)^{-3/2}$, while the inversion is within $4\times10^{-9}$ up to $3.2\times10^5$, so the two routes meet at the threshold to that level (with the skewness term alone the expansion would be $10^{-6}$ there). The route costs one normal quantile per point. Such an event has coefficient of variation at most $2.2\times10^{-3}$, so under the survival method it is narrow and never reaches the regular routes; the multi-unit beta units share the kernel. SciPy's own inverse returns non-monotone points hundreds of standard deviations off from about $\alpha=5\times10^{20}$, so the reference implementation takes the same expansion (`dev/done/plan-quantiles-large-parameters.md`).

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

**The fine mesh.** A cubic in $x$ cannot follow a bell-shaped survival function once $S$ falls several orders of magnitude per node interval: measured on one event against SciPy at every interval midpoint (2026-10-25), the shared cubic's relative error falls as about the $2.7$th power of the event's standard deviation in node intervals, and at the base density every bell-shaped event (both parameters above 1) with a coefficient of variation below about $90\%$ misses $10^{-4}$ somewhere above $S=10^{-6}$, below $28\%$ in the band $[10^{-4},10^{-2})$ where the thousand-year losses of a portfolio of a few hundred events sit (peer review, finding A3, and `dev/done/plan-peer-review-response.md`). The test portfolios pass test 15 because they are dominated by small-$\alpha$ events, whose survival in $\log x$ is smooth, and because at any $x$ a portfolio's $\Lambda$ is dominated by the events in their bulk there. So from 2.4.0 a bell-shaped event ($\alpha_j,\beta_j>1$) asks for `nodes_per_sigma` ($T$, default 12) nodes per standard deviation, a density of $d_j=T\ln10\cdot\mu_j/\sigma_j$ nodes per decade, on $[\mu_j-3\sigma_j,\ \mu_j+7\sigma_j]\cap[x_0,M_j]$ (the upper end is where $S_j\approx10^{-12}$ for a near-normal law); one asking more than the base density, a coefficient of variation below $T(10^{1/n_d}-1)$, $0.28$ at the defaults, is a **requesting event** and lives on a second curve, the **fine curve**, over the **fine mesh**: the base nodes with each base interval split into the largest number of equal pieces in $\ln x$ asked of it, at most $T$ (an event narrower than a base interval is narrow, below). The base curve never sees the requesting events and the base events never see the fine nodes, so the small-$\alpha$ events, active across the whole range, pay nothing; a requesting event is evaluated at the fine nodes between its lower skip and its truncation node and, from the first node where $S_j<2^{-60}$ (the upper skip, which bounds the contribution of everything above by $2^{-60}\lambda_j$), at none, so it costs about $10T$ evaluations plus the base nodes in its window. $\tilde\Lambda(x)$ is the sum of the two cubics, each monotone on its own mesh; the exact rate at the base nodes adds the fine curve's value there, which is exact since the base nodes are fine nodes. On every mesh the plateau below an event's lower skip, $\lambda_j(1-S_j(x_{k_j}))$ at every node, is one difference ($+$ at node 0, $-$ at the first evaluated node) prefix-summed after the fold, not a visit per node. **The contract** (test 33): at $T=12$ the curve is within $10^{-4}$ relative of a bell-shaped event's survival where $S_j\ge10^{-4}$ (except $1.5\times10^{-4}$ on the skewed small-parameter $\operatorname{Beta}(10,7.3)$ at a coefficient of variation of $0.2$), within $10^{-3}$ where $S_j\ge10^{-8}$ and within $3\times10^{-3}$ where $S_j\ge10^{-12}$, measured over $\alpha_j$ from $1.5$ to $10^4$ and coefficients of variation from $10^{-3}$ to $0.2$ (`benchmarks/report-2.4.0.md`); $T=20$ buys another decade of $S$ at each tolerance. The cost is linear in $T$ and falls on the requesting events alone: on the mixture benchmark portfolio, whose events are $10.6\%$ requesting, the curve build took $1.8$ times as long at $T=12$ as with the fine mesh off ($T=1$), on the small-$\alpha$ and large-exposure portfolios nothing.

**Truncation.** Every survival function has an infinite slope in $\log x$ at its endpoint $M_j$, and the sum of such endings on shared nodes is not interpolable in $x$ wherever one event, or several ending together, carry much of $\Lambda$: measured on this project's own medium golden and benchmark portfolio, a pure interpolant missed by 1% at the 100-year level and by factors in the tail (execution log of `dev/done/plan-single-unit-survival-route.md`). Each event is therefore split at its **truncation node** $x_{k_j}$, the node $Z$ intervals below the first node at or above $M_j$ ($Z=4$, the constant `TAIL_INTERVALS`). The **curve**
$$
\tilde\Lambda(x)=\sum_{j\ \text{beta}}\lambda_j\bigl(S_j(x)-S_j(x_{k_j})\bigr)^+
$$
carries each event up to its truncation node, where its contribution vanishes continuously, and the **tail**, the mass $\lambda_jS_j(x_{k_j})$ on $(x_{k_j},M_j]$, stays with the event. $\tilde\Lambda$ and its slope $-x\tilde\Lambda'(x)$ are evaluated exactly at the nodes; the slope at a node is one-sided, an event's slope at its truncation node counting on the incoming interval only. An event whose truncation node would lie below $x_0$ is a tail from $x_0$. So $\Lambda(x)=\tilde\Lambda(x)+\sum_{j:\,x<x_{k_j}}\lambda_jS_j(x_{k_j})+\sum_{j:\,x_{k_j}\le x<M_j}\lambda_jS_j(x)$, the last sum over the few events whose tail interval contains $x$.

**Interpolation of the curve.** Between nodes $\tilde\Lambda$ is a cubic Hermite interpolant in $x$ with the exact one-sided slopes, limited by the Fritsch and Carlson conditions so that it never increases: on an interval whose secant slope is zero both end slopes are set to zero, and otherwise the two end slopes $a$ and $b$, in units of the secant slope, are scaled down together whenever $a^2+b^2>9$. Above the last node $\tilde\Lambda=0$. A log-log interpolant was planned and rejected: near a node where several truncated survival functions vanish together the logarithm magnifies the curvature, and it missed by $10^{-2}$ where the plain cubic is within $10^{-5}$. The interpolation error is a reported diagnostic: the exact $\tilde\Lambda$ at 64 log-midpoints spread over the node range against the interpolant, as the largest error relative to the exact $\Lambda$ where $\Lambda>10^{-12}\sum_j\lambda_j$. It is a sample, not a bound; the reference implementation can check every interval midpoint.

**The tails.** Near its endpoint a beta's survival is $S_j(x)=(1-y)^{\beta_j}R_j(y)$ with $R_j$ analytic and nonzero at $y=1$, so $\log S_j$ is nearly linear in $u=\log(M_j-x)$. On its tail interval each event's survival is a monotone cubic Hermite in $u$ with the exact slopes $(1-y)f_j(y)/S_j(y)$, on nodes shrinking the distance to the endpoint by $\sqrt2$ per step from $M_j-x_{k_j}$ down to $x_0/2$, and the exact power law $S_j\propto(M_j-x)^{\beta_j}$ below the last node. Two provisos (from 2.3.6; peer review, finding A2). The endpoint expansion is $S_j=t^{\beta_j}/(\beta_jB(\alpha_j,\beta_j))\,[1-c_jt+O(\alpha_j^2t^2)]$ with $t=(M_j-x)/M_j$ and $c_j=(\alpha_j-1)\beta_j/(\beta_j+1)$, so the power law's relative error below the last node at $t_L$ is $c_jt_L$: it is used only where $c_jt_L\le10^{-6}$, and otherwise the queried point (at most one bucket edge, since edges are $h$ apart and the last node is within $h/4$ of $M_j$) is evaluated exactly; on $\operatorname{Beta}(10^6,\tfrac12)$ at an edge $5\times10^{-7}$ below $M_j$ the pure power law gave $0.054$ for the exact $0.683$ and moved $0.63$ of the event's rate across that edge. And where $c_j>1$ and $c_jt\in[10^{-2},10^2]$, the ladder steps by $2^{1/4}$ instead of $\sqrt2$: the cubic in $u$ of $\log(1-c_jt)$ was off by $3\times10^{-4}$ at $\beta_j=5$ with $\sqrt2$ steps ($6\times10^{-6}$ at $\beta_j=0.3$), sixteen times less with the finer step. The nodes are $d_0\,2^{-k/4}$ for a shared integer sequence $k$, so both implementations place them alike. The first node's value is the exact $S_j(x_{k_j})$, so the tail joins the curve exactly. This costs about $2\log_2\bigl((M_j-x_{k_j})/x_0\bigr)$ exact evaluations per event plus one cubic per bucket edge walked (the rate cut below), and follows the survival to a few units in $10^{-6}$ where no interpolant in $x$ can.

**Bucketing (the nearest rule).** With the even grid of section 6 and the half-bucket edges $e_i=(i+\tfrac12)h$, the beta part of the secondary grid is
$$
p_0=\sum_{j\ \text{beta}}\lambda_j-\tilde\Lambda(e_0)-\sum_j\lambda_jS_j(x_{k_j}),
\qquad
p_i=\tilde\Lambda(e_{i-1})-\tilde\Lambda(e_i)+\sum_{j}\lambda_j\bigl(S_j(e_{i-1})-S_j(e_i)\bigr)\quad(1\le i\le n-1),
$$
the last sum over the events whose tail interval meets $(e_{i-1},e_i]$ (with $S_j$ read as $S_j(x_{k_j})$ below the tail interval and $0$ above $M_j$), and the overflow of section 6 is $\tilde\Lambda(e_{n-1})$ plus the tails beyond $e_{n-1}$. Its first moment about the top node $x_{n-1}$ is, for the curve part, $(e_{n-1}-x_{n-1})\tilde\Lambda(e_{n-1})+\int_{e_{n-1}}^{x_m}\tilde\Lambda(x)\,dx$ (each event's partial expectation integrated by parts and summed; the integral of the cubic is a closed form per interval), which covers every event whether or not its endpoint survival underflowed, and for the tails by partial expectations. Until 2.3.4 the curve part was summed per event over the tail records, which an event whose endpoint survival underflows does not have, and its moment was lost (peer review, finding I1; test 18). Atoms are deposited by the linear rule of section 6 as before. Every $p_i\ge0$ because both interpolants are monotone, and the total rate is exact because the sums telescope. Measured on 2026-10-24 (reference implementation, 150 nodes per decade): the bucketed tail sums against the exact $\Lambda$ at every bucket edge where $\Lambda>10^{-12}\sum_j\lambda_j$ are within $4\times10^{-6}$ on the small-alpha fixture, $1.1\times10^{-5}$ on the medium golden and $1.1\times10^{-5}$ on the benchmark portfolio; with nodes alone and a log-log interpolant the same figures were 270, 0.78 and 7.0. The first moment is not exact. In the interior the bucketing error is of order $h^2$ times the slope of the density; from bucket 0, whose mass sits at loss 0, it is $\sum_j\lambda_j\int_0^{h/2}x\,dF_j\approx\sum_j\lambda_j\frac{\alpha_j}{\alpha_j+1}\frac h2F_j(h/2)$, first order in $h$ for $\alpha_j<1$. Both are reported through the secondary mean error of section 7. The linear rule, exact in the mean, was considered and rejected: it inflates the variance of every unit of mass by about $h^2/6$ against the nearest rule's $h^2/12$ (Sheppard's correction), an illusion of exactness. The nearest rule is also the rule the multi-unit route uses for beta units.

**The rate cut.** Walking a tail to its endpoint costs one cubic per bucket edge on $(x_{k_j},M_j]$, about $0.08M_j/h$ edges whatever the event's rate there, and for the low-mean events of a catastrophe ELT nearly all of those edges deposit nothing in double precision. Each tail is therefore walked in order from the first edge above $x_{k_j}$ and stopped at the first edge $e_s$ where $\lambda_jS_j(e_s)\le\tau=\varepsilon\sum_j\lambda_j$, with $\varepsilon=10^{-20}$ (`TAIL_RATE_FLOOR`), or at the top edge $e_{n-1}$; the remainder $\lambda_jS_j(e_s)$ goes in one deposit to the bucket above $e_s$, beyond the grid to the overflow with its moment by partial expectations. A tail whose whole mass $\lambda_jS_j(x_{k_j})$ is at most $\tau$ goes in one deposit to the bucket containing $x_{k_j}$. The rate stays exact. Each cut moves at most $\tau$ of rate within one event's tail interval, so the bucketed tail sums depart from the curve's $\Lambda$ by at most $N\tau$ at any edge ($10^{-14}$ of the total rate for a million events) and the first moment by at most $N\tau\max_jM_j$. The curve itself, $\Lambda(x)$ of section 7, is not cut. Measured on 2026-10-24 on a 100,000-row portfolio with endpoints to $10^9$ at $h=25$ and $2^{22}$ buckets: $2.7\times10^{10}$ edges before the cut and the window clip, $8.2\times10^8$ after, and the rate the cut moves $1.4\times10^{-16}$ of the total (`dev/done/plan-survival-tail-cost.md`).

**Narrow events.** A beta whose standard deviation is below one node interval at its mean, $\sigma_j<\mu_j(10^{1/n_d}-1)$, has a survival function that steps from $1$ to $0$ inside one interval, where the shared cubic, exact only at the nodes, cannot follow it: $\Lambda$ would be off by up to $\lambda_j$ across the interval (measured 2026-10-24: $5\times10^{-4}$ relative on a block with several hundred policies at a common limit, $1.8\times10^{-5}$ on a synthetic portfolio). Such an event, provided $\alpha_j,\beta_j\ge2$ (a bounded, log-concave density whose derivative is finite at the endpoints: between 1 and 2 the 8-point rule was off by up to $5\times10^{-4}$ relative, measured at $\alpha_j=100$ and $\beta_j$ from $1.02$ to $1.6$ on 2026-10-26, and such an event stays on a curve, its mass against the endpoint carried by its tail ladder to $10^{-5}$), is kept out of the curve. Its bulk $(s_j,t_j)$ is found by steps of $\sigma_j$ out from the mean, each side ending at the first step whose mass is at most a tenth of $\tau/\lambda_j$ of the mass accumulated so far (so the mass dropped beyond it is below the floor), or at $0$ and $M_j$; an event whose bulk does not reach that within 40 steps on either side (its mass piled at an endpoint) takes the regular route. An event with $\min(\alpha_j,\beta_j)\ge2\times10^5$ (the threshold of section 5 from which the quantiles method takes the Cornish-Fisher expansion) is kept out of the curve whatever the node density, because the continued fraction is untrustworthy there; a continued fraction that has not converged after its 1,000 iterations (400 before this correction; at the mean it needs about $1.1\sqrt{\alpha+\beta}$ when both parameters are large, 443 at $(2\times10^5, 6\times10^7)$) is reported as a typed error, never returned as a value (test 30). Its survival on the bulk is evaluated by quadrature of its density: the density relative to its value at the mean, $f_j(y)/f_j(\bar y)=\exp\bigl((\alpha_j-1)\log(1+(y-\bar y)/\bar y)+(\beta_j-1)\log(1-(y-\bar y)/(1-\bar y))\bigr)$ with the logarithms of $1+\delta$ taken as such so that huge parameters lose no precision, integrated by 8-point Gauss and Legendre quadrature on pieces of at most half a standard deviation, and normalized by its integral over the bulk. Measured 2026-10-24 against SciPy: within $10^{-13}$ at $\alpha_j+\beta_j=10^4$, $10^{-12}$ at $10^6$, $10^{-11}$ at $10^8$, and within the rounding of the loss share itself beyond. The engine's own incomplete beta, a continued fraction capped at 1,000 iterations, is good only to about $10^{-10}$ near the mean at $\alpha_j+\beta_j=10^4$ and returns nothing usable above $10^7$ when both parameters are large (found the same day), which is why the narrow events do not use it (the reference uses SciPy), nor does the quantiles method, which takes the Cornish-Fisher expansion of section 5 there. With one parameter small the fraction converges in tens of iterations at any $\beta$, and the error there was the constant $\ln B$ (the Nodes paragraph; `dev/done/plan-ln-beta-large-parameters.md`). From $\min(\alpha_j,\beta_j)=10^{16}$ the kernel is the standardized normal law in closed form: the two logarithms of the density are each of size about $40\sqrt{\alpha_j\beta_j/(\alpha_j+\beta_j)}$ at the edge of the bulk and cancel, so the quadrature kernel's error in $\log f$ is about $4\times10^{-15}\sqrt{\min(\alpha_j,\beta_j)}$, while the normal's relative error is the skewness $2/\sqrt{\min(\alpha_j,\beta_j)}$ times a polynomial in $z$; the two cross near $10^{16}$ at about $10^{-6}$ each, and SciPy's incomplete beta is itself off by $10^{-4}$ at $10^{24}$, so the reference takes the normal there too (test 31). Until 2.3.3 an event with $\alpha_j+\beta_j>10^{15}$ was replaced by a step at its mean, on the mistaken ground that one unit in the last place of the loss share moved the density by more than a tenth; the standardized coordinate resolves the density to about $10^{-7}$ there, and a symmetric event with its mean on a bucket edge went wholly to one bucket where its $\sigma=0$ twin split half and half (peer review, finding A1). The event counts $\lambda_j$ in $\Lambda(x)$ below $s_j$, $\lambda_jS_j(x)$ on the bulk and nothing at or above $t_j$, each within $\tau$; on the grid it is walked along the edges of its bulk by the walk of the rate cut, from the bucket containing $\max(s_j,x_0)$ with the mass below $x_0$ left to bucket $0$, so the bound of the rate cut covers it. The count of narrow events is reported with the metadata (`narrow_events`).

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

For a multi-unit ELT (section 4A) the utility returns the per-view totals on the grid, as rate weights, and on request the rate-weighted joint occurrence severity of any two units and of each unit with the event total, as two-dimensional arrays on the common grid, with the metadata of section 4A. The two-unit joint of one event remains available on its own.

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
18. Overflow: with a grid length short of $\max_jM_j$, for both methods and both outputs, `top_bucket` adds the overflow to the top node and reports its rate and displaced moment, `drop` leaves the mass short by exactly that and reports the same figures, and `error` raises the typed error; with the grid length derived from the input both figures are exactly zero. A rate-one $\operatorname{Beta}(2,1000)$ on $[0,1]$ windowed at $0.00195$ (its endpoint survival underflows) reports the excess moment $5.7965\times10^{-4}$ within $10^{-4}$ relative on both implementations, and on the medium fixture windowed at a third of its largest exposure the two implementations' secondary excess moments agree within $10^{-4}$ relative.
19. The K-point cut: on the same fixture the quantiles method at $K=32$ has $\Lambda=0$ above each event's $1-1/64$ quantile (plus one bucket of linear spread) and a secondary mean error below $-5\%$ of the expected annual loss, while the survival method is within the tolerance of test 16. This documents the defect.
20. Bernoulli rows: an inadmissible row contributes $\lambda(1-p)$ at $0$ and $\lambda p$ at $M$ in both methods, with rate and mean exact and grid variance $\mu(M-\mu)$ to bucketing tolerance; under the survival method the exceedance curve has an exact step of $\lambda p$ at $M$; the row is reported under `variance_bernoulli`.
21. The rate cut: on the fixture the survival grid with the cut and the same grid with every tail walked to its endpoint have tail sums within $N\varepsilon\sum_j\lambda_j$ of each other at every edge, the rate is exact in both, and the two implementations still agree as in test 17.
22. Narrow events: on a synthetic portfolio with events of coefficient of variation from $10^{-6}$ to $10^{-2}$ at a common mean, both implementations report them as narrow, the exceedance curve is within $10^{-8}$ relative of the exact $\Lambda$ across $[0.97,1.03]$ times that mean, the grid's tail sums are within $10^{-7}$ of the exact $\Lambda$ at the edges there, the rate is exact, and the two grids agree as in test 17.
23. Large parameters: on one-row portfolios with $\alpha\in\{0.13,0.5,1,1.5,2.5\}$ and $\beta\in\{10^4,10^6,10^7,8\times10^7\}$, both implementations' node values from 0.3 to 30 times the mean are within $10^{-13}$ relative of the regularized incomplete beta computed in multiple precision below the switch point $(\alpha+1)/(\alpha+\beta+2)$ and within $2\times10^{-8}$ above it where $S\ge10^{-15}$; on a synthetic portfolio with 300 such events among the base rows of test 22, the node values are within $10^{-11}$ of the exact $\Lambda$, the grid's tail sums within $10^{-5}$ of it at the sampled edges where $\Lambda\ge10^{-12}\sum_j\lambda_j$, the rate exact and the two grids within $10^{-12}$ of the total rate of each other.
24. Large parameters under the quantiles method: the Cornish-Fisher expansion's 32 midpoint quantiles are within $10^{-8}$ standard deviations of SciPy's inverse for $\min(\alpha,\beta)$ of $2\times10^5$, $10^6$ and $10^7$ with $\beta/\alpha$ from $1$ to $3\times10^4$; on one-row portfolios with $\min(\alpha,\beta)$ from $10^5$ to $10^{26}$ and mean shares $0.5$, $0.25$ and $0.01$ both implementations succeed, the rate is exact, the two grids agree within $10^{-12}$ of the rate, every bucket with mass lies within $2.2$ standard deviations plus one bucket of the mean and the first-moment error is of the midpoint rule's order; a row with exposed value $10^6$, mean $250{,}000$ and standard deviation $10^{-6}$ puts at least $1-10^{-8}$ of its rate on the node at its mean; on the portfolio of test 22 with 300 such events added, both implementations succeed and their grids agree within $10^{-11}$ of the total rate.

The corrections of `dev/done/plan-peer-review-response.md` add the following, on both implementations.

30. Kernel validity: a symmetric beta with $\alpha=\beta=10^9$ on a two-bucket grid at 200,000 nodes per decade gives weights $(\tfrac12,\tfrac12)$ and is reported as narrow (before the correction, $(-0.45, 1.45)$ from an unconverged continued fraction); the engine's incomplete beta returns NaN, not a value, at $\alpha=\beta=10^9$, $x=\tfrac12$; one-row portfolios with $\min(\alpha,\beta)\in\{10^4,10^5,2\times10^5,10^6,10^9\}$ at 100, 1,000 and 20,000 nodes per decade have every bucket weight in $[0,1]$, the rate exact, and the two implementations within $10^{-8}$ of the total rate of each other below the threshold (the fraction's own accuracy there is about $10^{-9}$) and within $10^{-10}$ at and above it, where both take the narrow route.

31. Atoms by resolution and the normal kernel: a row with $\mu=\tfrac12$, $M=1$ and $\sigma\in\{0,10^{-8},2\times10^{-8},10^{-16}\}$ on a grid of spacing $0.2$ puts half its rate on each of the two buckets adjoining the edge at its mean, within $10^{-9}$ (the exact split for $2\times10^{-8}$ is $0.4999999998/0.5000000002$), the last row reported under `atom_by_resolution`; the row $\mu=0.55$, $\sigma=10^{-9}$ on three buckets of $0.2$ reports an overflow rate of 1 and an excess moment of $0.15$ within $10^{-9}$; symmetric and lopsided one-row portfolios with $\min(\alpha,\beta)\in\{10^{14},10^{16},10^{18},10^{24}\}$, the mean on a bucket edge and displaced from it by four units in the last place, bucket within $10^{-6}$ of the exact normal split (plus the resolution of the mean's share, a few units in the last place over $\sigma_y$, which at $10^{24}$ is $10^{-4}$ of a standard deviation) and the two implementations within $10^{-8}$ of each other (at the handover itself the two can recover $\alpha$ on opposite sides of $10^{16}$ and take different kernels).

32. Endpoint tails: $\operatorname{Beta}(10^6,\tfrac12)$ on a grid whose top edge lies $5\times10^{-7}$ below the exposed value puts $0.6827$ of its rate above that edge within $10^{-6}$ on both implementations ($0.054$ before this correction); $\operatorname{Beta}(10^6,1)$ matches $-\operatorname{expm1}(10^6\log x)$ at every bucket edge within $10^{-5}$ relative where $S\ge10^{-12}$ ($1.5\times10^{-6}$ measured, the refined ladder's residual); for $\alpha\in\{10^2,10^4,10^6\}$ and $\beta\in\{0.3,0.5,1,2,5\}$ the tail sums at every bucket edge of the tail interval where $S\ge10^{-12}$ are within $10^{-4}$ relative of the exact incomplete beta and the two implementations within $10^{-11}$ of the rate of each other (the edge below the last node is evaluated exactly on each side, the engine's fraction against SciPy).\n\nThe multi-unit route (section 4A) adds the following, on both implementations.

33. The fine mesh: on one-row portfolios of bell-shaped events with $\alpha\in\{1.5,3,10,10^2,10^3,10^4\}$ and coefficients of variation from $5\times10^{-4}$ to $1$, both implementations' curves at $\mu+z\sigma$ for $z$ from $-3$ to $7$ are within $10^{-4}$ relative of the exact survival where it is at least $10^{-4}$, within $10^{-3}$ where at least $10^{-8}$ and within $3\times10^{-3}$ where at least $10^{-12}$, and agree with each other within $10^{-8}$ relative or $10^{-16}$ absolute; the review's $\operatorname{Beta}(1800,10^6)$ at $x=0.0019446$ is within $10^{-4}$ of $3.2522\times10^{-4}$ (it was $22.8\%$ off); on a block of 300 policies at a common mean with coefficients of variation from $10^{-3}$ to $10^{-1}$ on a grid of $25$, every policy is narrow or requesting and the two implementations agree on both counts and on the fine node count, the grids within $10^{-12}$ of the total rate of each other, the tail sums at every bucket edge from $0.6$ to $1.8$ times the mean within the three tolerances above of the exact $\Lambda$ at the matching levels of $\Lambda/\sum_j\lambda_j$, and the engine bitwise identical across thread counts.

25. Coupling limits: with `independent` the event total equals the plain convolution of the unit marginals within $10^{-12}$; with `comonotone` and units sharing $(\alpha,\beta)$ the total equals the rolled row's beta to bucketing tolerance (Proposition 6), and with differing shapes it equals the bucketing of $\sum_kM_kF_k^{-1}(u)$ at the nodes; with `burke` the normal-score correlation of the two-unit joint equals $w_uw_w$ and the loss correlation lies strictly between 0 and it (Proposition 4); with `homer` the ground-up total equals the one-row survival grid of the rolled table composed with the allocation map within $10^{-12}$ of the rate, each allocation is at most $M_k$, and the reported capped mean equals the analytic shortfall; with `matched` the variance of each event total equals $(\sigma_{I,R}+\sigma_{C,R})^2$ within $10^{-6}$ relative or the event is reported with its shortfall at saturated loadings.
26. Bernoulli units: an inadmissible unit row contributes the two-point conditional law of section 4A; its marginal from the joint is $\lambda(1-p)$ at 0 and $\lambda p$ at $M$ within the rule's marginal error, the event total with a beta partner matches the multi-unit article's construction within $10^{-12}$, and the row is reported under `variance_bernoulli`; `clamp` is rejected with the typed error of the one-row route.
27. One-unit events: an event with a single contributing unit produces, under every coupling and rule, exactly the one-row survival grid of section 5A for that row, within $10^{-12}$ of the rate; on a portfolio mixing one-unit and multi-unit events the totals are the sum of the two parts.
28. The factor rule: on the two-unit cases A to D of the multi-unit article and on the fixture, the exceedance of the event total at the $10^{-2}$, $10^{-3}$, $10^{-4}$ and $10^{-5}$ levels under the default rule and $Q$ is within the tolerance the convergence study records against the $Q=4001$ reference; per-view mass is exact within $10^{-14}$ relative; view means reconcile within the stated tolerance.
29. Bivariate outputs: the marginals of $S_{uw}$ equal the two units' occurrence severities within the rule's error; the second marginal of $S_{u\Sigma}$ equals the ground-up total of the run within $10^{-12}$ of the rate; $\sum_u\kappa_u(x)=x$ at every grid total with mass, computed from the $S_{u\Sigma}$; the mixture over events of the per-event joints equals the returned array within $10^{-12}$ of the rate; and the marginal of the annual bivariate built by the aggregate package from $S_{u\Sigma}/\Lambda$ with a Poisson count $\Lambda$ equals the one-dimensional compound of the ground-up total within the aggregate package's stated grid tolerance.

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
8. **resolved (2.0.0):** no overflow bin; a named overflow policy, `top_bucket` by default, with the overflow's rate and moment reported (section 6);
9. **resolved (2026-10-25, `dev/plan-multiunit-coupling.md`):** the multi-unit coupling rules of section 4A with `burke` the default, the cap at $M_k$ on the Homer allocation, the Bernoulli treatment on the multi-unit route, and bivariate outputs as arrays only, with the annual bivariate and the conditional means left to the aggregate package; the default factor rule and $Q$ await the convergence study of that plan.

These questions affect the mathematical contract and should be resolved identically for every competing implementation used in a head-to-head comparison.
