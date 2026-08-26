# Additivity of the RMS Secondary Uncertainty Model

Date: 2026-08-26. Companion to `dev/secondary_two_units_reference.py`, which
implements the model and generated every number quoted below.

## 1. The question

An ELT row can describe an event at unit level (units $i = 1, \dots, n$, each
with mean loss $\mu_i$, independent sd $\sigma_{I,i}$, correlated sd
$\sigma_{C,i}$, maximum loss $M_i$) or at rolled-up level (one row for the sum).
The natural roll-up rule, motivated by the correlated component being driven by
a single shared factor per event, is
$$
\mu_R=\sum_i\mu_i,\qquad
M_R=\sum_i M_i,\qquad
\sigma_{C,R}=\sum_i\sigma_{C,i},\qquad
\sigma_{I,R}=\Big(\sum_i\sigma_{I,i}^2\Big)^{1/2}.
$$
Call the method **additive** if the rolled row, run through the same secondary
uncertainty recipe, reproduces the distribution of $\sum_i X_i$ implied by the
unit-level joint model. This note proves what is and is not additive.

## 2. The two descriptions

**Unit-level joint model** (Burke, Rau-Chaplin, Varghese 2016, Section III.E).
With $p_i=\mu_i/M_i$ and $s_i=(\sigma_{I,i}+\sigma_{C,i})/M_i$ (the arithmetic
total is the RMS convention), unit $i$'s damage ratio is
$\operatorname{Beta}(\alpha_i,\beta_i)$ moment-matched to $(p_i,s_i)$. With one
shared standard normal $Z_c$ per event occurrence and independent standard
normals $Z_i$ per unit,
$$
U_i=\Phi\!\left(\frac{\sigma_{C,i}Z_c+\sigma_{I,i}Z_i}{t_i}\right),
\qquad t_i=\sqrt{\sigma_{C,i}^2+\sigma_{I,i}^2},
\qquad X_i=M_iF_i^{-1}(U_i),
$$
where $F_i$ is the beta cdf. Each $U_i$ is uniform, so the marginals are exact;
the pair has a Gaussian copula with normal-score correlation
$\rho_N=\sigma_{C,1}\sigma_{C,2}/(t_1t_2)$.

**Rolled row.** The single row $(\mu_R,\sigma_{I,R},\sigma_{C,R},M_R)$ asserts
that the event loss is $M_R$ times a beta with mean $p_R=\mu_R/M_R$ and sd
$s_R=(\sigma_{I,R}+\sigma_{C,R})/M_R$.

## 3. What is additive

**Proposition 1 (means, always).** $\mathsf E\big[\sum_iX_i\big]=\sum_i\mu_i=\mu_R$,
because each marginal beta matches its mean exactly and expectation is linear.
No condition needed.

**Proposition 2 (the roll-up rule is exact for the Gaussian skeleton).** In the
linear one-factor model
$$
X_i^{\mathrm{lin}}=\mu_i+\sigma_{C,i}Z_c+\sigma_{I,i}Z_i,
$$
the sum is $\mu_R+\sigma_{C,R}Z_c+\sigma_{I,R}Z'$ with $Z'$ standard normal
independent of $Z_c$ (a linear combination of the independent $Z_i$). The
triple $(\mu,\sigma_I,\sigma_C)$ is therefore exactly additive under the rule
of Section 1, and the family is closed: the rolled row describes the sum
without error. This is the origin and the justification of the rule.

The general mechanism behind it is the conditional variance decomposition.
Because the units are independent given $Z_c$,
$$
\operatorname{Var}\Big(\sum_iX_i\Big)
=\operatorname{Var}\Big(\mathsf E\Big[\sum_iX_i\,\Big|\,Z_c\Big]\Big)
+\sum_i\mathsf E\big[\operatorname{Var}(X_i\mid Z_c)\big].
$$
The independent contributions add across units unconditionally (the second
term), and the correlated contributions are the sds of the conditional means
$g_i(Z_c)=\mathsf E[X_i\mid Z_c]$, which add arithmetically when the $g_i$ are
linearly related. In the linear model $g_i$ is exactly linear in $Z_c$ and the
conditional variances are the constants $\sigma_{I,i}^2$, so both halves of the
rule are exact.

## 4. What breaks, three separate obstructions

**Proposition 3 (the arithmetic marginal convention).** The RMS total sd
$\sigma=\sigma_I+\sigma_C$ is not the sd of a sum of independent components
(that would be $t=\sqrt{\sigma_I^2+\sigma_C^2}$), so unit and rolled rows
disagree on second moments even before any beta enters. Normal analog: give
each marginal sd $\sigma_{I,i}+\sigma_{C,i}$ and correlation $\rho_N$. With
$\sigma_{I,1}=\sigma_{I,2}=\sigma_{C,1}=\sigma_{C,2}=1$: $t_i=\sqrt2$,
$\rho_N=1/2$, marginal sds $2$, so
$\operatorname{Var}(S)=4+4+2\cdot\tfrac12\cdot4=12$, while the rolled row
asserts $(\sqrt2+2)^2=6+4\sqrt2\approx11.657$. Not equal, and no general
ordering holds (Section 5: the ratio crosses 1 between regimes).

**Proposition 4 (copula attenuation).** The loss-space correlation
$\rho_L=\operatorname{corr}(X_1,X_2)$ satisfies $\rho_L<\rho_N$ whenever
$0<\rho_N<1$ and the betas are nondegenerate. Proof: $X_i$ is a nonlinear
measurable function of the jointly normal scores
$V_i=(\sigma_{C,i}Z_c+\sigma_{I,i}Z_i)/t_i$, and by Lancaster's theorem the
maximal correlation of a bivariate normal pair over all square-integrable
transformations is $|\rho_N|$, attained only by linear maps. A bounded beta
quantile transform is not linear, so the inequality is strict. Consequently the
correlated pieces never couple at full strength: $\sigma_{C,R}=\sum\sigma_{C,i}$
implicitly assumes perfect linear coupling of the conditional means, which the
copula does not deliver. In the same vein the conditional means
$g_i(Z_c)$ are comonotone but not linearly related, so
$\operatorname{sd}(\sum g_i)\le\sum\operatorname{sd}(g_i)$ with equality only
in the common-shape case of Proposition 6.

**Proposition 5 (the beta family is not closed).** Take
$\sigma_{C,i}=0$ (independent units), where Proposition 2's variance rule is
exact. The density of $X_i$ near $0$ behaves like $x^{\alpha_i-1}$, so the
convolution behaves like $s^{\alpha_1+\alpha_2-1}$ near $0$, while the rolled
beta behaves like $s^{\alpha_R-1}$ with $\alpha_R$ set by moment matching.
Generically $\alpha_R\ne\alpha_1+\alpha_2$, so the distributions differ despite
identical mean and variance. Numerically (case C below):
$\alpha_1+\alpha_2=0.812$ versus $\alpha_R=0.969$, and the 99th percentile of
the rolled row overshoots the true convolution by 2.7 percent while the sd
matches to four figures.

## 5. Numerical illustration

Two units, bucket size 10, grids covering $M_1=1000$ and $M_2=2000$,
$n_z=4001$ quadrature nodes; exact sum from
`secondary_two_units(...).sum_distribution()`, rolled row bucketed on the same
grid. Units are $(\mu,\sigma_I,\sigma_C,M)$.

| Case | Unit 1 | Unit 2 | $\rho_N$ | $\rho_L$ | sd model | sd rolled | q99 model | q99 rolled |
|---|---|---|---|---|---|---|---|---|
| A balanced | (200,100,150,1000) | (500,300,200,2000) | 0.462 | 0.411 | 644.6 | 666.3 (+3.4%) | 2520 | 2580 (+2.4%) |
| B correlated dominant | (200,30,220,1000) | (500,60,440,2000) | 0.982 | 0.969 | 744.9 | 727.1 (\-2.4%) | 2760 | 2720 (\-1.5%) |
| C independent | (200,250,0,1000) | (500,500,0,2000) | 0 | 0 | 559.1 | 559.0 (\-0.0%) | 2220 | 2280 (+2.7%) |
| D comonotone, common shape | (200,0,250,1000) | (400,0,500,2000) | 1 | 1.000 | 750.2 | 750.1 (0.0%) | 2790 | 2790 (0.0%) |

Reading: the rolled row overstates risk in the balanced case, understates it in
the correlated-dominant case (the roll-up folds $\sigma_I$ in quadrature, while
the model's arithmetic marginals plus near-comonotone coupling add unit sds
almost linearly: $250+500\approx745$ against $67+660=727$), matches variance
but not tails in the independent case, and is exact in case D. There is no
conservative direction.

## 6. The exact case

**Proposition 6 (sufficient condition for exact additivity).** If
$\sigma_{I,i}=0$ for every unit and all units share the same normalized
moments, $p_i\equiv p$ and $s_i\equiv s$ (equivalently the same
$(\alpha,\beta)$), then the method is additive, exactly. Proof: with
$\sigma_{I,i}=0$, every $U_i=\Phi(Z_c)$, one uniform $U$ for all units, so the
losses are comonotone and quantiles add:
$$
\sum_iX_i=\sum_iM_iF^{-1}(U)=M_RF^{-1}(U),
$$
a scaled beta of the same shape. The rolled row reproduces it because
$p_R=\sum p M_i/M_R=p$ and, since $\sigma_{C,i}=sM_i$,
$s_R=\sum sM_i/M_R=s$. Degenerate rows ($\sigma_i=0$, atoms at $\mu_i$) are
trivially additive as well. $\blacksquare$

Outside this case additivity fails generically: Proposition 3 breaks the
second moment whenever both components are present, Proposition 4 whenever
$0<\rho_N<1$, and Proposition 5 breaks the distribution even when the second
moment survives. Common shape matters too: comonotone sums of different beta
shapes are not beta (their quantile sum $\sum M_iF_i^{-1}(u)$ is not a beta
quantile function), and with different $s_i$ the rolled $s_R$ is a weighted
average while the comonotone sd is the sum of the unit sds, which then
disagree.

## 7. A discretization remark, separate from the model

Bucket rounding does not commute with summation:
$\operatorname{round}(x_1/h)+\operatorname{round}(x_2/h)$ can differ from
$\operatorname{round}((x_1+x_2)/h)$ by one bucket per unit. In case D the
model-level agreement is exact (sd and quantile ratios 1.0000) while the
bucket-level pmfs differ visibly near the origin, where the
$\alpha<1$ density is steep, purely from this rounding. Any pipeline that
buckets per unit and convolves on the grid (the natural design, and what
`aggregate` itself does for compound distributions) accepts a location error of
up to $h/2$ per unit; choose $h$ accordingly and reconcile means as a
diagnostic rather than expecting bucket-level identity.

## 8. Consequences

1. **Granularity is a modeling choice, not a refactoring.** The same physical
   event coded at location, program, or portfolio level produces genuinely
   different secondary distributions. A rolled portfolio ELT and a unit-level
   ELT are different products; neither can be recovered from the other.
2. **The roll-up rule is still the right bookkeeping.** It is exactly the
   moment rule of the latent one-factor model (Proposition 2) and industry
   practice for building portfolio rows; the point is that running the beta
   recipe on the rolled triple is an approximation to the unit-level model,
   with errors of a few percent in sd and tail quantiles at realistic
   parameters (Section 5), in no fixed direction.
3. **Unit-level structure is mandatory when unit-level terms exist.** Per-risk
   limits, deductibles, and reinsurance act on $X_i$ before summation;
   no rolled row can represent them.
4. **Cheap diagnostic.** Given unit-level inputs, compute both the exact sum
   (reference implementation) and the rolled-row beta and report sd and tail
   quantile ratios per event, so the roll-up error is visible rather than
   assumed away.

## 9. Caution: consistency is unattainable at FFT precision

For the ELT consumer the non-additivity has a blunt consequence: **there is no
fully self-consistent set of RMS outputs across aggregation levels.** The ELT
format forces every row's conditional severity to be the beta determined by
$(\mu,\sigma_I,\sigma_C,M)$, and Sections 4 and 5 show that the model's own
aggregation semantics do not stay inside that family. A unit-level ELT and the
corresponding rolled total ELT are therefore never simultaneously exact
descriptions of one model: the total row is a moment-matched beta projection
of the true sum distribution. In a simulation engine the projection error (a
few percent, Section 5) sits at or below sampling noise at practical trial
counts, which is why the inconsistency is invisible in vendor practice and can
be safely ignored there. In an FFT pipeline computing exact bucketed
distributions the two routes, adding across units versus running the vendor
total ELT, converge to **different distributions**: the gap is structural and
reproducible, and no refinement of $h$, $n_z$, or anything else removes it.
Working rules:

1. The unit-level route **is** the model; the rolled route is its projection.
   When both inputs exist, prefer units and treat the total ELT as a check.
2. Reconcile to a vendor total ELT by projecting the unit-level result through
   the same roll-up (the moment rule of Section 1 plus the beta refit), which
   should agree to bucketing accuracy. Never reconcile by comparing the exact
   sum distribution to the rolled beta; those differ by design.
3. Means agree on both routes exactly (Proposition 1), so AAL agreement is no
   evidence of distributional agreement.
4. Label every output with the route that produced it.

## 10. References

- Burke, N., Rau-Chaplin, A., Varghese, B. (2016). Computing probable maximum
  loss in catastrophe reinsurance portfolios on multi-core and many-core
  architectures. Concurrency and Computation: Practice and Experience 28(3).
  Section III.E. Local copy: `192.168.4.43:9124/view/Burke2016`.
- American Academy of Actuaries (2018). Uses of Catastrophe Model Output.
  Definitional background only. Local copy: `192.168.4.43:9124/view/AAA2018`.
- Lancaster, H. O. (1957). Some properties of the bivariate normal
  distribution considered in the form of a contingency table. Biometrika
  44(1/2), 289 to 292. (Maximal correlation of a bivariate normal is $|\rho|$.)
- Meyers, G., Klinker, F., Lalonde, D. (2003). The Aggregation and Correlation
  of Insurance Exposure. CAS Forum, Spring 2003. (Burke's reference [2];
  common-shock treatment of aggregating correlated exposure.) Local copy:
  `192.168.4.43:9124/view/Meyers2003a`.
