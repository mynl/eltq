# Plan: grid-independent model moments, tail queries and windowed multi-unit views for aggregate

Status: ruled 2026-10-26 (Steve's answers to the open questions below), executing from 2026-10-26; see the execution log. Drafted 2026-10-26. Reflected half of aggregate's `dev/done/plan-elt.md` (`[DecL-Loss-Tables]`).

Drafted against the local eltq 2.10.0 source (`master` at 8a8f472). The aggregate plan is
`V:/worktrees/aggregate_REFACTOR/dev/plan-elt.md` until its first step lands and
`dev/done/plan-elt.md` after; it is cited here by its section labels, which survive the move. Its
implementation item `[Eltq-Model-Moments-And-Windows]` is this plan. A change to an interface
agreed here is mirrored in both plans.

Reviewed 2026-10-26 22:06:18, see >>> comments in Open Questions section.

## Goal

The aggregate package (`V:/worktrees/aggregate_REFACTOR`, the author's FFT library for compound
distributions) is adding event and year loss tables to its declaration language, DecL. With
secondary uncertainty, an ELT declaration builds one ordinary `Aggregate`: a Poisson count of
occurrences with severity the rate-weighted mixture of event totals, the event total discretized
by eltq on aggregate's grid. Two of aggregate's rulings of 2026-10-26 set what eltq must supply.
`[ELT-Moments-Before-Grid]`: read the tables and compute the model's moments before the grid is
chosen, so that aggregate's standard bucket and window routines can size the grid. And
`[ELT-Windowed-Support]`: a huge exposed value $M$ with a small mean must not force a grid reaching
$M$; $M$ stays in the model and the numerical window is a separate, measured choice.

eltq 2.10.0 meets neither for events with several units and only part of the first for one-unit
rows. This plan states, as requirements on eltq with its internal design left open:

1. `[Eltq-Model-Moments]`: the distinct-event rate, the rate-weighted raw moments
   $C_r=\sum_e\lambda_e\mathsf PX_e^r$ for $r=1,2,3$, the effective support, the zero mass, row
   classification counts and integration diagnostics, with no loss grid: for one-unit rows
   (`moments_arrays`) and for the event total under the declared coupling (`multiunit_moments`).
2. `[Eltq-Tail-Queries]`: the tail rate $R(t)$ and tail partial moments $D_r(t,c)$ at any
   candidate edge, with no loss grid: exact for one-unit rows, exact or rigorously bounded with
   stated error for shared events.
3. `[Eltq-Windowed-Shared-Events]`: an explicit window with a named overflow policy on
   `multiunit_views`, with bounded storage and work, overflow applied to the event total, and
   diagnostics matching the one-row route.
4. `[Eltq-Release-For-Aggregate]`: the PyPI releases that let aggregate's optional `elt` extra
   declare `eltq>=X`. Not a prerequisite for aggregate's development, which uses the local eltq
   (ruling 5).

Two more steps come out of the drafting: `[Eltq-Spec-Moments-Tails-Windows]`, the specification
amendments that govern the rest, and `[Eltq-Multiunit-Narrow-Units]`, a fix to the multi-unit
route found while verifying current behavior (below).

## Background for a reader with no context

### eltq

eltq (`V:/dev/eltq`) converts a catastrophe model event loss table (ELT) into rate-weighted loss
distributions on an even grid $g_i=ih$, $i=0,\dots,n-1$. It is a Rust engine
(`crates/eltq-core`) with PyO3 bindings (`crates/eltq-python`) and a SciPy reference
implementation (`python/eltq/reference.py`, `reference_multiunit.py`) that serves as the oracle;
every acceptance test runs on both backends (`tests/conftest.py`, the `impl` fixtures). The
mathematical contract is `secondary-uncertainty-mathematical-specification.md`; code conforms to
it, and a disagreement is a bug. Project rules (`CLAUDE.md`): Claude commits each step and
version bump, Steve pushes; a pushed `v*` tag publishes to PyPI through
`.github/workflows/wheels.yml`, behind the `pypi` environment's approval; the version is kept in
`Cargo.toml`, `pyproject.toml` and `python/eltq/__init__.py` together; speed is a product
requirement; methods rest only on the listed published sources or on derivations in the
specification.

### The model in brief

A row has rate $\lambda$, mean $\mu$, standard deviations $\sigma_I$ and $\sigma_C$, and exposed
value $M$. The total sd is $\sigma=\sigma_I+\sigma_C$ (Burke 2016, section III.E; spec section 4).
After classification a contributing row is one of three laws: an **atom** at $\mu$ (zero sd, zero
exposure, a boundary mean, $\sigma<2^{-48}\mu$, or the `atom` policy); a **beta**
$X=M\,\mathrm{Beta}(\alpha,\beta)$ moment-matched to $(\mu,\sigma)$ when
$0<\sigma^2<\mu(M-\mu)$; or, under the default `variance_policy="bernoulli"`, the **two-point**
law on $\{0,M\}$ with $\mathsf P(X=M)=\mu/M$ when the stated variance is inadmissible.

A multi-unit ELT has one row per event and unit. The units of an event share one standard normal
factor $Z_c$ per occurrence: unit $u$ has score $V_u=w_uZ_c+\sqrt{1-w_u^2}Z_u$ and loss
$X_u=M_uF_u^{-1}(\Phi(V_u))$, with Burke's loading
$w_u=\sigma_{C,u}/\sqrt{\sigma_{I,u}^2+\sigma_{C,u}^2}$ by default (spec section 4A, the
`coupling` option). Given $Z_c=z$ the units are independent. The
event total is $S_e=\sum_uX_{eu}$. Occurrences of distinct events are independent Poisson
processes, so with $\Lambda=\sum_e\lambda_e$ the annual loss is compound Poisson with severity
mixture $\Lambda^{-1}\sum_e\lambda_e\,\mathsf P(S_e\in\cdot)$ and annual cumulants
$\kappa_r=C_r=\sum_e\lambda_e\mathsf PS_e^r$.

### What aggregate does with eltq

Aggregate's `[ELT-Source-Grid]` fixes the call sequence. At construction, before any grid,
aggregate needs $\Lambda$, $C_1,C_2,C_3$, the support, the zero mass and diagnostics, to feed its
existing moment-based window selector (`Aggregate._bs_window`). During sizing it needs the tail
of the event severity at candidate edges, to accept a window shorter than the support only with
quantitative checks (`[ELT-Support-In-Sizing]`, the numerical contract). At update it calls
`quantize_arrays(method="survival", normalize=False, bucket_size=h, grid_length=n,
overflow="top_bucket")` for one-unit events and, for shared events, `multiunit_views(terms=None,
views=("ground_up",), normalize=False, coupling="burke", bucket_size=h, grid_length=n)` with a
window. Aggregate selects $h$ and $n$; eltq never does. Aggregate's grid budget, the setting
`elt_max_log2`, was deferred on 2026-10-26 to aggregate's uncertainty step and is aggregate-side;
what eltq owes it is the tail rate and tail partial moments as functions of a candidate edge, and
work and memory that scale with the window, not with $M/h$.

| aggregate step | needs from eltq |
| --- | --- |
| `[ELT-Single-Event-Uncertainty]` | `moments_arrays`, `tail_moments_arrays`, the existing windowed `quantize_arrays` |
| `[ELT-Shared-Event-Uncertainty]` | `multiunit_moments`, `multiunit_tail_moments`, windowed `multiunit_views`, `[Eltq-Multiunit-Narrow-Units]` |

### Current behavior, verified 2026-10-26

| surface | behavior at 2.10.0 | where |
| --- | --- | --- |
| one-row classification | nonfinite, negative, mean above max and zero-rate rows dropped; degenerate rows and $\sigma<2^{-48}\mu$ atoms (`atom_by_resolution`); inadmissible variance by policy, `bernoulli` default; tests in ratios | `validate.rs:47-158` (103-110, 118-120) |
| one-row moments | `Meta.total_rate` and `Meta.analytic_aal` ($C_1$); no second or third moment anywhere | `quantize.rs:258-275`, `_meta.py:107-152` |
| one-row window | caller supplies $h$; omitted $n$ covers the largest contributing $M$; a shorter $n$ is a window under `top_bucket` (default), `drop` or `error`, the last checked up front | `quantize.rs:153-182`, `__init__.py:252-255, 279-284` |
| overflow fields | `rate_above_grid`: rate beyond the top edge $(n-1/2)h$; `mean_above_grid`: its first moment in excess of the top node $(n-1)h$; per output | `_meta.py:122-125, 150-152`, `deposit.rs:7-13`, `curve.rs:1467-1482` |
| edge convention | an atom above the top edge overflows (strict); one in (top node, top edge] goes wholly to the top node and is not overflow | `deposit.rs:40-47, 63-64`, spec section 6 |
| beta partial expectation | $\lambda\mathsf P(X;X>e)=\lambda M\,\alpha/(\alpha+\beta)\,\bar I_{e/M}(\alpha+1,\beta)$, private, first order only | `curve.rs:118-125`, `reference.py:280-284` |
| exceedance curve | public $\Lambda(x)$, atoms exact, betas interpolated on geometric nodes from `min_loss`, to the tolerances of spec tests 15 and 33 | `__init__.py:495-530`, `curve.rs:1356-1392` |
| large beta parameters | engine continued fraction untrusted from $\min(\alpha,\beta)=2\times10^5$, NaN on nonconvergence; the one-row route uses the narrow kernel's quadrature (mass and first moment only) and the normal law from $10^{16}$ | `incbeta.rs:95-121`, `curve.rs:286-399` (368-390) |
| multi-unit classification | `classify_unit_row`: the same structural categories and policies, ratio-form test, but no `atom_by_resolution`; the reference's `prepare_unit` tests and forms $\kappa$ in products | `multiunit.rs:492-570` (516-528), `reference_multiunit.py:149-223` (183-184, 215-216) |
| multi-unit beta cdf | `edge_cdf` and `marginal_pmf` call `BetaFn::cdf` at every parameter size, with no regime check | `multiunit.rs:602-626, 730-757` |
| multi-unit grid | `grid_length` accepted; auto length is one plus the largest sum over an event's units of their rounded spans; a shorter explicit grid raises `GridOverflowError`; no overflow argument | `views.rs:1006-1113` (1094-1109), `reference_multiunit.py:1185-1201` |
| multi-unit storage | every live unit's conditional rows have `grid_length_for(support)` nodes per factor stratum, so full support | `views.rs:1241-1252`, `reference_multiunit.py:1260-1262` |
| multi-unit FFT | transform size from the sum of all unit row lengths | `views.rs:714-725` |
| `multiunit_views` defaults | `views` all four, `coupling="burke"`, `z_rule="stratified"`, `z_nodes=32`, `variance_policy="bernoulli"`, `grid_length=None` | `__init__.py:809-832`, `views.rs:164-181` |
| multi-unit metadata | no overflow policy or overflow fields | `_meta.py:275-326`, `views.rs:240-275` |
| stratified factor rule | 32 equal strata over $[0,3/4]$, a $\sqrt2$ ladder down to $2^{-24}$, 76 strata; units independent within a stratum; event-total tails within $7\times10^{-3}$ | `quadrature.rs:35-41`, spec section 4A |
| event-total variance | `total_variance` from bucketed conditional matrices at the caller's $h$; used by the `matched` coupling's scale | `multiunit.rs:205-225, 244` |
| occurrence outputs | always full support (`grid_length=None`) | `__init__.py:969` |
| versions | 2.10.0 | `Cargo.toml:6`, `pyproject.toml:3`, `__init__.py:40` |

### Found while drafting

Probes run 2026-10-26 on the installed 2.10.0 (read-only, scratch scripts not committed):

- **The Rust multi-unit route returns NaN for a very narrow beta unit.** A one-unit event with
  $\mu=0.5$, $\sigma_I=10^{-5}$, $M=1$ ($\alpha=\beta\approx1.25\times10^9$) at $h=0.2$ gives
  `[0, 0, nan, nan, 0, 0]` and mass NaN from `multiunit_views` on the Rust backend; with a second
  unit $(\mu,\sigma_I,\sigma_C,M)=(0.3,0,0.1,1)$ seven cells are NaN. The reference backend gives
  $0.05$ and $0.05$ at nodes 2 and 3 (and mass $0.1$ for the two-unit event), as does
  `quantize_arrays`, which classifies the row as narrow. Cause, from the code: the continued
  fraction's NaN beyond its regime reaches `edge_cdf` and `marginal_pmf` unguarded (rows above).
  This breaks spec section 8 (no NaN) and backend parity.
- **The multi-unit route does not classify $\sigma<2^{-48}\mu$ as an atom.** A unit with
  $\sigma_I=10^{-16}$, $\mu=0.5$, $M=1$ records no diagnostic on the multi-unit route and
  `atom_by_resolution` on the one-row route. Spec section 4A requires each unit row to be
  "classified exactly as a one-row ELT row is in section 4", and section 4 (2.3.4) makes such a
  row an atom.
- **The reference multi-unit admissibility test is in products**, `sigma * sigma >= mu * (m -
  mu)`, where the engine and the one-row reference use ratios (2.4.1); at monetary scales near
  $10^{154}$ the two backends can classify differently.
- **The `matched` coupling's scale depends on the grid**: `matched_scale` finds it from
  `total_variance` of bucketed conditional laws at the caller's $h$. No grid-independent moment of
  a `matched` event exists until that scale is defined without a grid.

## The mathematics

Expectation is written $\mathsf PX$, and $\mathsf P(X;A)$ is the expectation of $X$ on the event
$A$, as in spec section 4A.

### Model moments of one-unit rows

For a contributing row the effective law gives, with no special function:

| law | $\mathsf PX^r$ | support $b$ | $\mathsf P(X=0)$ |
| --- | --- | --- | --- |
| atom at $\mu$ | $\mu^r$ | $\mu$ | $\mathbf 1\{\mu=0\}$ |
| beta | $\mu M^{r-1}\prod_{k=1}^{r-1}(\alpha+k)/(\alpha+\beta+k)$ | $M$ | 0 |
| two-point | $\mu M^{r-1}$ | $M$ | $1-\mu/M$ |

Then $\Lambda=\sum\lambda_j$, $C_r=\sum\lambda_j\mathsf PX_j^r$, the zero rate
$\rho_0=\sum\lambda_j\mathsf P(X_j=0)$ (annual zero probability $\exp\{-(\Lambda-\rho_0)\}$), and
$B=\max_jb_j$. For a beta row $\mathsf PX^2=\mu^2+\sigma^2$ exactly, a check. The beta formula is
the rising-factorial moment $M^r(\alpha)_r/(\alpha+\beta)_r$ written in the ratios the
classification already uses, so it holds at every parameter size, the narrow and normal regimes of
spec section 5A included. $C_3$ overflows double precision when $\Lambda\max_jM_j^3$ does (from
about $M=10^{102}$); that is flagged, never returned silently as infinity.

### Model moments of shared events

Given $Z_c=z$ let $a_{ur}(z)=\mathsf P(X_u^r\mid Z_c=z)$. Starting from $t_1=t_2=t_3=0$, add the
event's units one at a time with simultaneous updates (aggregate `[ELT-Moment-Pass]`):

$$
t_1'=t_1+a_{u1},\qquad
t_2'=t_2+2t_1a_{u1}+a_{u2},\qquad
t_3'=t_3+3t_2a_{u1}+3t_1a_{u2}+a_{u3},
$$

so that $t_r(z)=\mathsf P(S^r\mid Z_c=z)$, linear in the number of units, with no pairs, triples
or loss grid. The marginal parts integrate in closed form, so only the cross part needs a rule:

$$
\mathsf PS^r=\sum_u\mathsf PX_u^r+\int\Bigl(t_r(z)-\sum_ua_{ur}(z)\Bigr)\varphi(z)\,dz .
$$

The first moment is therefore exact, and the same rule applied to $\sum_ua_{ur}$ gives a residual
against the closed forms, a free check of the inner integration.

Conditional moments by unit law, from the conditional law of spec section 4A:

- atom: $\mu^r$ at every $z$;
- two-point, $w<1$: $M^r\,\bar\Phi\bigl((y_0-wz)/\sqrt{1-w^2}\bigr)$ with $y_0=\Phi^{-1}(1-p)$;
  at $w=1$ the step $M^r\mathbf 1\{z>y_0\}$, integrated by splitting at $y_0$;
- beta, $w=0$: the unconditional $\mathsf PX^r$;
- beta, $w=1$: $(MF^{-1}(\Phi(z)))^r$, through the engine's beta quantiles (Cornish-Fisher from
  $\min(\alpha,\beta)=2\times10^5$);
- beta, $0<w<1$: a one-dimensional integral,

$$
a_{ur}(z)=\int g_u\bigl(w_uz+\sqrt{1-w_u^2}\,\zeta\bigr)^r\varphi(\zeta)\,d\zeta,\qquad
g_u(v)=M_uF_u^{-1}(\Phi(v)),
$$

  or the same as
  $M^r\int_0^1ry^{r-1}\bar\Phi\bigl((\Phi^{-1}(I_y(\alpha,\beta))-wz)/\sqrt{1-w^2}\bigr)dy$.
  For small $\alpha$ most of $\mathsf PX^r$ sits in the far upper tail (the second regression
  example below has 67% of its mean above probability $7\times10^{-5}$), so both the inner rule
  and the factor rule must resolve the upper tail deeply.

By coupling: `independent` needs no integration (every $a_{ur}$ constant); `homer` is the rolled
row's one-row law before terms (spec section 4A), so its closed forms apply; `comonotone` is
$\mathsf PS^r=\int_0^1\bigl(\sum_uq_u(v)\bigr)^rdv$ with $q_u$ the unit quantile functions, one
integral with breakpoints at two-point steps; `burke` needs both integrals; `matched` needs a
grid-free scale first (open question 6). The zero mass of an event is
$\int\prod_u\mathsf P(X_u=0\mid z)\varphi(z)\,dz$, zero whenever a unit is a beta or a positive
atom. The support is $B_e=\sum_ub_{eu}$ ($b=\mu$ for atoms, $M$ otherwise, as `UnitPlan::support`,
`multiunit.rs:125-127`), under `homer` raised to the rolled row's support as `views.rs:1076-1077`
does.

The target is the declared coupling's law, not the stratified rule's approximation of it. The
stratified rule treats units as independent within a stratum, which keeps every marginal exact but
not the cross moments; the quantized grid of `multiunit_views` realizes that approximation, and
its gap from the model is a measured number (acceptance below), kept apart from bucketing error,
since aggregate checks moment integration, bucket spacing and factor strata separately
(`[ELT-Shared-Event-Uncertainty]`).

### Tail measures

For an edge $t\ge0$ and a node $0\le c\le t$ (on a grid, $c=(n-1)h$ and $t=(n-1/2)h$):

$$
R(t)=\sum_e\lambda_e\,\mathsf P(X_e>t),\qquad
T_r(t)=\sum_e\lambda_e\,\mathsf P(X_e^r;X_e>t),\qquad
D_r(t,c)=T_r(t)-c^rR(t)=\sum_e\lambda_e\,\mathsf P(X_e^r-c^r;X_e>t),
$$

for $r=1,2,3$. Under `top_bucket` the rate is unchanged and the annual cumulants fall by exactly
$D_r$; under `drop` the rate falls by $R$ and the cumulants by $T_r$; $1-\exp(-R(t))$ is the
probability of at least one overflowing occurrence in a year. $R$ and $D_1$ at a grid's top edge
are the existing `rate_above_grid` and `mean_above_grid`, here as functions of any candidate edge
without a grid. The inequality is strict, as in `deposit.rs:40`: a loss exactly at $t$ stays on
the grid. Atoms in $(c,t]$ stay on the grid but move to $c$ (spec section 6); that displacement is
bucketing, not overflow, and is returned separately so that aggregate can test the final half
bucket on its own. $R$ and $T_r$ are nonincreasing in $t$, so a caller can bisect for an edge.

**One-unit rows**, exactly. A beta row has $\mathsf P(X>t)=\bar I_{t/M}(\alpha,\beta)$ and

$$
\mathsf P(X^r;X>t)=\mathsf PX^r\;\bar I_{t/M}(\alpha+r,\beta),
$$

since $y^rf_{\alpha,\beta}(y)=\bigl(B(\alpha+r,\beta)/B(\alpha,\beta)\bigr)f_{\alpha+r,\beta}(y)$
and $B(\alpha+r,\beta)/B(\alpha,\beta)=(\alpha)_r/(\alpha+\beta)_r$; the case $r=1$ is the
private `tail_moment` today. An atom at $x$ contributes $\lambda\mathbf 1\{x>t\}$ and
$\lambda x^r\mathbf 1\{x>t\}$; a two-point row is its atom at $M$ with rate $\lambda p$. Where the
engine's continued fraction is untrusted ($\min(\alpha,\beta)\ge2\times10^5$) the query takes
the one-row route's narrow kernel, extended from the first moment to the second and third, and
the normal law's closed-form partial moments from $10^{16}$. $D_r$ is computed so that a short tail beyond $t$
(near an endpoint, $\mathsf P(X\mid X>t)$ close to $c$) does not lose its digits to the
subtraction $T_r-c^rR$.

**Regression examples** (aggregate `[ELT-Support-In-Sizing]`), recomputed 2026-10-26 with SciPy
1.18.0 (`betaincc`) and, independently, mpmath 1.4.1 at 60 digits; the two agree to ten
significant digits. The columns are $\mathsf P(X^r;X>t)/\mathsf PX^r$ at $t=\mu+40\sigma$, an
upper bound on $D_r(t,c)/\mathsf PX^r$; the column $r=0$ is $\mathsf P(X>t)$.

| row | $\alpha$ | $\beta$ | $t$ | $r=0$ | $r=1$ | $r=2$ | $r=3$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\mu=20{,}000$, $\sigma=60{,}000$, $M=10^9$ | 0.1110889 | 5554.333 | 2,420,000 | $1.568651\times10^{-8}$ | $2.031163\times10^{-6}$ | $2.641404\times10^{-5}$ | $1.816930\times10^{-4}$ |
| $\mu=100$, $\sigma=10{,}000$, $M=10^9$ | $9.989999\times10^{-5}$ | 998.9998 | 400,100 | $7.024077\times10^{-5}$ | 0.6705151 | 0.9384651 | 0.9920658 |

Both of aggregate's figures reproduce: the second row has probability $7.024\times10^{-5}$ and
67.05% of its mean above $\mu+40\sigma$. The smallest edges at which all four ratios are at most
$10^{-5}$, and at most $10^{-12}$, are 3,018,875 and 6,166,119 for the first row (0.30% and 0.62%
of $M$) and 16,417,518 and 33,479,497 for the second (1.6% and 3.3% of $M$). A window far below
$M$ suffices in both, and no fixed multiple of $\sigma$ is safe: at $\mu+40\sigma$ the first row
still holds $1.8\times10^{-4}$ of its third moment and the second most of its mean.

### Shared-event tails: a rigorous factor bracket and two conditional evaluators

**The factor bracket.** Under every coupling with nonnegative loadings (all but `homer`, whose
total is one row and takes the one-row formulas), $\mathsf P(X_u\le x\mid Z_c=z)=
\Phi\bigl((y_u(x)-w_uz)/\sqrt{1-w_u^2}\bigr)$ is nonincreasing in $z$ (the indicator
$\mathbf 1\{y_u(x)\ge z\}$ at $w_u=1$, a constant for atoms), so each unit is stochastically
nondecreasing in $z$; given $z$ the units are independent, so
$g(z)=\mathsf P(\psi(X_1,\dots,X_n)\mid Z_c=z)$ is nondecreasing in $z$ for every nondecreasing
$\psi$. The functions $\mathbf 1\{S>t\}$,
$S^r\mathbf 1\{S>t\}$ and $(S^r-c^r)\mathbf 1\{S>t\}$ with $c\le t$ are nondecreasing in every
$X_u$. For any partition of the factor into cells $(z_{k-1},z_k]$ of probability $\pi_k$,

$$
\sum_{k=1}^K\pi_k\,g(z_{k-1})\;\le\;\int g(z)\varphi(z)\,dz\;\le\;\sum_{k=1}^K\pi_k\,g(z_k),
$$

with $g(\pm\infty)$ the limits (at $+\infty$ every unit with $w_u>0$ at its maximum). The width is
computable and shrinks under refinement, so the factor integration is certified rather than
estimated. The production rule cannot do this: its tail error is up to $7\times10^{-3}$ at the
$10^{-4}$ level and its ladder stops at $2^{-24}$, far above a twelve-nines target; the tail
partition must reach much deeper, which costs only cells, not grid.

**Evaluation at fixed $z$**, where the units are independent; the implementation may use either
or both:

- *A union bound*, no convolution. For thresholds $t_u\ge0$ with $\sum_ut_u\le t$,
  $\{S>t\}\subseteq\bigcup_u\{X_u>t_u\}$, so with $S_{-u}$ the other units,

$$
\mathsf P(S>t\mid z)\le\sum_u\mathsf P(X_u>t_u\mid z),\qquad
\mathsf P(S^r;S>t\mid z)\le\sum_u\sum_{k=0}^r\binom rk\mathsf P(X_u^k;X_u>t_u\mid z)\,
\mathsf P(S_{-u}^{r-k}\mid z),
$$

  the second from conditional independence, with $\mathsf P(S_{-u}^{j}\mid z)$ from the moment
  recursion by prefix and suffix products (as `occurrence_unit_first_moment` forms its spectra).
  Below, $\mathsf P(S>t\mid z)\ge1-\prod_u\mathsf P(X_u\le t\mid z)$, so
  $D_r\le T_r^{\mathrm{upper}}-c^rR^{\mathrm{lower}}$. Cheap and valid; loose when several units
  share the tail.
- *A lattice bracket.* Rounding each unit down and up to a lattice of spacing $\delta$, set by
  the implementation and independent of the caller's $h$, gives
  $S^{\downarrow}\le S\le S^{\uparrow}$ and so
  $\mathsf P(S^{\downarrow}>s)\le\mathsf P(S>s)\le\mathsf P(S^{\uparrow}>s)$. Losses are
  nonnegative, so $\mathsf P(S\le s)$ for $s\le t$ needs each unit's law on $[0,t]$ only: a prefix
  convolution of length $t/\delta$, never $M/\delta$. The partial moments then follow from

$$
\mathsf P(S^r;S>t)=\mathsf PS^r-\int_0^t rs^{r-1}\,\mathsf P(S>s)\,ds+t^r\,\mathsf P(S>t),
$$

  conditionally on $z$, with $\mathsf P(S^r\mid z)=t_r(z)$ from the moment recursion and the two
  survival brackets placed for an upper and a lower bound. The subtraction costs an absolute error
  of order $\varepsilon\,\mathsf P(S^r\mid z)$, stated, harmless for relative targets far above
  $10^{-16}$.

An event with $B_e\le t$ contributes exactly zero, a one-unit event takes the one-row formulas, and
an event whose union bound is already below its share of the target needs no lattice. "Rigorous"
here means valid to floating rounding and to the stated error of the special functions and of the
conditional moments $a_{ur}$; an estimate is never returned as a bound.

### Windows on the multi-unit route

**Contract.** A windowed run equals the full-support run with every index at or above $n$ moved
to $n-1$ (`top_bucket`), removed (`drop`), or refused (`error`, today's behavior), up to the
stated rate floor of the cut rows below. The overflow is
therefore applied to the event total of the bucketed units: the sum is formed before the cap,
which aggregate requires (`[Eltq-Windowed-Shared-Events]` in its plan); capping each unit and then
summing gives the wrong upper tail. Because the multi-unit route buckets the units and then sums,
its overflow is the bucketed total's mass beyond the window, while the one-row survival route's is
the model's mass above $t$ from the curve; both are exact accounts of what their grids hold. The
moved rate is `rate_above_grid`, and the moment effects come from the model tail query at
$(t,c)$, as the one-row `mean_above_grid` comes from the curve.

**Why a window alone is not enough (raised loudly).** Today an event costs up to strata times
units times row length, and the row length is $M_u/h$. Cutting rows at $n$ removes the $M/h$
allocation but not the cost: at $n=2^{22}$ a unit row is 32 MiB per stratum, 2.4 GiB over 76
strata, and a portfolio with $10^5$ large-exposure multi-unit events would need on the order of
$10^{14}$ operations. The work must follow the mass, as the one-row route's does (its tail walk stops at a
rate floor, spec section 5A, the rate cut): each unit row is cut at the smaller of the window and
its effective span at a stated rate floor, the cut mass kept (rate exact) and bounded in the
metadata, and the strata streamed so that working memory per event is of order units times row
length, not strata times units times row length. This step is the one most likely to prove
disproportionately hard; it starts with a measurement, and if the measured cost is out of
proportion it stops for a decision rather than a workaround.

## Rulings that bind this plan

1. Aggregate, 2026-10-26: secondary uncertainty is computed when present, with eltq's
   `variance_policy="bernoulli"`, `coupling="burke"`, `z_rule="stratified"` and `z_nodes=32`
   initially (`[ELT-Secondary-Uncertainty]`, `[ELT-Shared-Events]`); moments before the grid
   (`[ELT-Moments-Before-Grid]`); the window is distinct from $M$, which stays in the model
   (`[ELT-Windowed-Support]`); the eltq half is a reflected plan in this repository; `elt_max_log2`
   is aggregate's and is designed in its uncertainty step.
2. The content above is the requirement. Names, signatures and dataclass fields are proposals that
   eltq may change at execution, mirrored in aggregate's plan. `moments_arrays` and
   `multiunit_moments` were checked unused by aggregate's `[Loss-Table-Names]`; on 2026-10-26 a
   search of the Python, Rust and Markdown sources found no use of those two, nor of
   `tail_moments_arrays`, `multiunit_tail_moments`, `ModelMoments`, `TailMoments` or
   `moments_above_grid`.
3. eltq computes no aggregate and selects no grid (as decision 3 of
   `dev/done/plan-multiunit-coupling.md`); every new output is numbers or arrays.
4. Every method here is derived in the specification from the model of sections 4 and 4A
   (elementary probability: rising factorials, conditional independence, monotone coupling, the
   layer identity), within the project's published-sources rule.
5. Author, 2026-10-26: aggregate builds against the current local eltq, not a PyPI release. The
   releases in `[Eltq-Release-For-Aggregate]` follow in due course and gate nothing in aggregate;
   each aggregate uncertainty step waits only on the eltq code it calls.

## Steps

Each code step is one bump from the version current when it executes (a feature a minor, a fix a
patch) and one commit, staged by path; Claude commits, Steve pushes. Versions are assigned at
execution and recorded in the execution log and the CHANGELOG. Each step lands in Rust and the
reference together, tested on both. The plan moves to `dev/done/` in the commit that lands its
first step.

### [Eltq-Spec-Moments-Tails-Windows] no bump: the specification

- New section 4B, *Model moments and tail measures*: the one-row closed forms, the shared-event
  recursion with the marginal-exact split, the conditional moments by unit law and coupling, the
  definitions of $R$, $T_r$, $D_r$ with the strict edge, the beta partial-moment identity, the
  factor bracket and the two conditional evaluators, each derived.
- Section 6: the multi-unit window paragraph (the contract above, the policies, the fields, the
  rate floor of the cut rows).
- Section 7: the new outputs.
- Section 9: tests 34 to 39, as in the steps below (33 is the last today).
- Section 11: an item recording the decisions taken on the open questions.

### [Eltq-Multiunit-Narrow-Units] patch: classification parity and large parameters on the multi-unit route

- `classify_unit_row` and `prepare_unit` (Rust) and `prepare_unit` (reference):
  `atom_by_resolution` for $\sigma<2^{-48}\mu$, as `validate.rs:103-110`; the reference's
  admissibility test and $\kappa$ in ratios, as the engine's.
- Beta units with $\min(\alpha,\beta)\ge2\times10^5$: `edge_cdf`, `edge_scores`, `marginal_pmf`,
  the stratified rows and `LayerMeans` take the one-row route's narrow kernel and normal law
  instead of the unguarded continued fraction; anything still unevaluable raises the typed kernel
  error, never NaN.
- Test 34, both backends: the probe's one-unit event equals $(0.05,0.05)$ at nodes 2 and 3, and
  its two-unit event matches the reference within $10^{-12}$ of the rate (today NaN); a sweep of
  $\min(\alpha,\beta)\in\{10^4,2\times10^5,2\times10^6,10^9,10^{16},10^{24}\}$ under every
  coupling and rule has no NaN, exact mass, and the backends within $10^{-10}$ of the rate; the
  $\sigma=10^{-16}$ unit is `atom_by_resolution` on both; test 10's scales $10^{160}$ and
  $10^{-200}$ classify alike on both multi-unit backends.
- Goldens regenerate only if a fixture holds such a row (check first; none expected).

### [Eltq-Model-Moments] two minor bumps: one-unit rows, then shared events

**One-unit rows.** `moments_arrays(rate, max_loss, mean_loss, sd_indep, sd_corr, *,
mean_is_loss_rate=False, variance_policy="bernoulli", event_ids=None, threads=None)`, the columns
in `quantize_arrays` order, returning `ModelMoments` (proposed, in `_meta.py`): `total_rate`
($\Lambda$), `raw_moments` ($C_1,C_2,C_3$, rate-weighted) and `severity_moments` ($C_r/\Lambda$),
`analytic_aal`, `rate_at_zero` ($\rho_0$), `max_support` ($B$) and `support` (per-row $b$, so
aggregate need not repeat the classification), `input_rows`, `valid_rows`, counts by effective law
(atom, beta, two-point, dropped), `diagnostics` (`RowIssues` by category, as `Meta`), and the
integration fields (`method="closed_form"`, error zero, `converged=True`).

- Classification is `validate::classify_row` as `quantize::classify_rows` applies it, so
  categories and counts equal `quantize_arrays` on the same input by construction. Today
  `classify_rows` first validates a whole `Config`, which requires a positive bucket size
  (`quantize.rs:106`, `config.rs:204-206`); the classification options are separated from the
  grid options so that no bucket size enters.
- No `bucket_size` or `grid_length` argument exists, so no output can depend on aggregate's
  future `bs` or `log2`; memory is the classification arrays.
- Test 35, both backends: aggregate's oracle rows (`[ELT-Probability-Oracles]`): rate $0.1$,
  mean 50, max 100, sds 10 and 10, beta $(2.625,2.625)$, $C_1=5$, $C_2=290$, $C_3=18{,}500$; the
  same row as an atom, $C_2=250$; sds summing to 60, two-point, $C_2=500$ and $\rho_0=0.05$; a
  positive-rate zero-mean row counted in $\Lambda$ and $\rho_0$; an all-zero table. The two
  regression rows against mpmath within $10^{-14}$ relative. `total_rate` and `analytic_aal`
  equal `quantize_arrays`'s within $10^{-15}$ relative, diagnostics identical. Scale $10^{100}$
  and $10^{-100}$ moments scale as $s^r$ within $10^{-14}$. Determinism across threads bitwise.
  As $h$ halves, the survival grid's moments approach $C_r$ at the rate of section 5A (a
  consistency check, not the oracle).

**Shared events.** `multiunit_moments(event_id, unit_id, rate, mean_loss, sd_indep, sd_corr,
max_loss, *, coupling="burke", variance_policy="bernoulli", rtol=None, threads=None)`, the columns
in `multiunit_views` order, returning the same `ModelMoments` with `events`, `valid_events`,
`one_unit_events`, `coupling`, per-event `support` ($B_e$), and integration diagnostics: the
method, the requested `rtol` and the achieved relative error of $C_2$ and $C_3$ from a refinement
comparison (at least two levels of both the factor and the inner rule), the marginal residual, the
evaluation count, `converged`, and bounded examples of unconverged events.

- Event structure validated as `multiunit_views` pass 1 (`views.rs:989-1053`), shared rather than
  copied; classification after `[Eltq-Multiunit-Narrow-Units]`.
- Closed forms for one-unit events (equal to `moments_arrays`), `independent` and `homer`;
  integration for `burke` and `comonotone`; `matched` per open question 6.
- Inadequate convergence is flagged and returned, never replaced by a rolled beta or a fitted
  skew.
- Test 36, both backends: an ELT of one-unit events equals `moments_arrays` within $10^{-15}$
  relative; `independent` equals the closed form within $10^{-14}$; `homer` equals the rolled row's
  closed form; two two-point units under `burke` against the exact cross moment
  $M_uM_w\Phi_2(-y_u,-y_w;w_uw_w)$ and three of them against a one-dimensional mpmath integral;
  two beta units under `burke` against Hoeffding's identity
  $\operatorname{Cov}=\iint(\Phi_2(-y_u(x),-y_w(x');w_uw_w)-\bar F_u(x)\bar F_w(x'))\,dx\,dx'$
  (the reference's Owen's $T$) to $10^{-10}$; `comonotone` against the quantile integral in
  mpmath; Homer and Li's Table 11 events against the same references; the achieved error at most
  `rtol` on every fixture, and a forced shallow rule flagged as unconverged. The gap between
  `multiunit_views` moments at $h$, $h/2$, $h/4$ and the model is recorded as two numbers, the
  bucketing term ($O(h^2)$) and the stratified rule's limit.
- Benchmark: the 0.91M unit-row portfolio of `benchmarks/run_multiunit.py`, time and peak memory,
  in `benchmarks/report-<version>.md` (budget, open question 4).

### [Eltq-Tail-Queries] two minor bumps: one-unit rows, then shared events

**One-unit rows.** `tail_moments_arrays(rate, max_loss, mean_loss, sd_indep, sd_corr, *, edge,
top_node=None, mean_is_loss_rate=False, variance_policy="bernoulli", threads=None)`, vectorized over
candidate edges $t$ (array) with nodes $c$ (array, default $c=t$), returning `TailMoments`
(proposed): `edge`, `top_node`, `rate_above` ($R$), `moments_above` ($T_1,T_2,T_3$ per edge),
`excess_above` ($D_1,D_2,D_3$ per edge), `top_half_atoms` (rate and $\mathsf P(X^r-c^r;c<X\le t)$
of the atoms in the final half bucket), `total_rate`, and the method per regime.

- Exact per row: the incomplete beta below $2\times10^5$, the narrow kernel extended to second and
  third moments above, the normal law from $10^{16}$; atoms and two-point rows exactly with the
  strict inequality.
- Test 37, both backends: beta partial moments of orders 0 to 3 against mpmath over $\alpha$ from
  $10^{-4}$ to $10^4$ and $\beta$ from $0.5$ to $10^7$ within $10^{-12}$ relative, and in the
  narrow and normal regimes within the accuracy section 5A states for them; both regression rows
  as in the table above; $\operatorname{Beta}(10^6,1/2)$ with $t$ just below $M$ and $c=t-h/2$,
  $D_1$ within $10^{-8}$ relative of mpmath; an atom exactly at $t$ not above and at the next float
  above; a two-point row with $M=t$ excluded and with $M$ one ulp above included; atoms in $(c,t]$
  in `top_half_atoms` only. At $t=(n-1/2)h$, $c=(n-1)h$ the query equals `quantize_arrays`'s
  `rate_above_grid` and `mean_above_grid` on the same window within spec test 18's $10^{-4}$
  relative (the curve's interpolation), atoms exactly. Monotone in $t$.
- Benchmark: $10^6$ rows by 64 edges.

**Shared events.** `multiunit_tail_moments(event_id, unit_id, rate, mean_loss, sd_indep, sd_corr,
max_loss, *, edge, top_node=None, coupling="burke", variance_policy="bernoulli", rtol=None,
threads=None)`, returning `TailMoments` with, for each quantity, a point value, a rigorous
`upper` and a `lower` bound, the method (union bound, lattice bracket), the achieved bracket width
and `converged` against the requested tolerance.

- The factor bracket with a tail partition deep enough for the target; the union bound first, the
  lattice bracket where the bound fails the target (open question 10 for the first release).
- Test 38, both backends: one-unit events equal the one-row query; two two-point units against
  the exact four-outcome law; two beta units against the exact conditional formula
  $\mathsf P(X_u+X_w>t)=\int\bar\Phi\bigl((y_w(t-x)-\rho y_u(x))/\sqrt{1-\rho^2}\bigr)dF_u(x)$,
  $\rho=w_uw_w$, and its partial-moment analogues, in mpmath; lower $\le$ exact $\le$ upper
  everywhere; the width shrinks under refinement; a two-unit event of two copies of the second
  regression row under `burke`; an event with $B_e\le t$ returns zero with no work.
- Benchmark: the 0.91M portfolio at a handful of edges.

### [Eltq-Windowed-Shared-Events] minor bump

- `overflow=` on `multiunit_views`, `multiunit_views_into`, `MultiConfig` and the reference, values
  `top_bucket`, `drop`, `error` (default, open question 5). With `grid_length=None`, or at least
  the auto length, outputs are bitwise those of today.
- Each unit row cut at $\min(n,\text{its effective span at a stated rate floor})$, its mass beyond
  kept per stratum and computed directly, not as one minus a sum; strata streamed so that per-event
  working memory is units times row length; event totals by a linear convolution whose retained
  prefix has no circular wraparound (padding or pairwise truncation, eltq's choice); one-unit
  events, `comonotone` and `homer` (rolled row) windowed likewise. No array of length $M/h$
  anywhere, pass 1's arithmetic excepted.
- A memory preflight against a stated budget (the role `Config::wave_memory_bytes` plays on the
  one-row route, CHANGELOG 2.4.2), failing before allocation with the grid, the bytes and the
  operation.
- `MultiMeta` gains `overflow`, `rate_above_grid` (per view, the rate moved or dropped),
  `mean_above_grid` and `moments_above_grid` (ground_up from the shared tail query at $(t,c)$,
  other views `None`, open question 7), `window_tail` (that query's bounds), and the rate cut by
  the floor.
- With a window: `pair`, `first_moments` and `roll_up_diagnostic` refused with a clear message;
  `matched` per open question 6.
- Test 39, both backends: indices $0$ to $n-2$ equal the full-support run within $10^{-12}$ of
  the rate, index $n-1$ plus the moved rate under `top_bucket`, short by it under `drop`, and the
  typed error under `error`; every coupling but `matched`, atoms, two-point and beta units, units at
  loading one, off-grid sums; a two-unit event each of whose units fits the window while the total
  does not (sum before cap: the overflow is the total's, and differs from capping each unit);
  direct and FFT paths agree when the unit rows sum to far more than $n$; units with $M=10^{12}$ at
  $h=1$ and $n=2^{12}$ complete in bounded memory (full support would need $10^{12}$ cells); the
  backends within $10^{-12}$ of the rate; bitwise across thread counts; existing goldens hold.
- First, a measurement: the 0.91M portfolio with exposures raised to $10^9$, at windows of $2^{20}$
  and $2^{22}$, time and peak memory, before the design is committed (the raised flag above).

### [Eltq-Release-For-Aggregate] two releases

PyPI holds 1.12.0 only (`v1.12.0`, 2026-10-22, the only tag). The remote `mynl/master` is at 2.3.2,
pushed and untagged; local `master` is 2.10.0, 19 commits ahead and untagged. The 2.x line has
never been published, so PyPI users meet 2.0.0's default changes (the survival method, the
clamp's removal from the one-row route) in the first release; the CHANGELOG already records them.

- **First release**, after `[Eltq-Multiunit-Narrow-Units]` and the one-unit parts of
  `[Eltq-Model-Moments]` and `[Eltq-Tail-Queries]`: Steve pushes `master`, CI passes, Steve tags
  `v` plus that version (the workflow checks the tag against `pyproject.toml`) and approves the
  `pypi` environment. Aggregate's `elt` extra then declares `eltq>=` that version for
  `[ELT-Single-Event-Uncertainty]`.
- **Second release**, after the shared parts and `[Eltq-Windowed-Shared-Events]`, the same way;
  aggregate raises its lower bound for `[ELT-Shared-Event-Uncertainty]`.
- The upper bound is aggregate's choice. eltq has announced one breaking change, the removal of the
  deprecated multi-unit `clamp` at 3.0.0 (spec section 4), which aggregate does not use.
- Acceptance: CI green on the tag; wheels for Windows, Linux and macOS on PyPI; in a clean
  environment `pip install eltq==<version>` imports the new functions on the Rust backend.

## Order of work

| order | step | bump | unblocks in aggregate |
| --- | --- | --- | --- |
| 1 | `[Eltq-Spec-Moments-Tails-Windows]` | none | |
| 2 | `[Eltq-Multiunit-Narrow-Units]` | patch | correctness of the shared route |
| 3 | `[Eltq-Model-Moments]`, one-unit rows | minor | |
| 4 | `[Eltq-Tail-Queries]`, one-unit rows | minor | `[ELT-Single-Event-Uncertainty]`, against the local eltq |
| 5 | `[Eltq-Release-For-Aggregate]`, first release | none, a tag | the `elt` extra's lower bound |
| 6 | `[Eltq-Model-Moments]`, shared events | minor | |
| 7 | `[Eltq-Tail-Queries]`, shared events | minor | |
| 8 | `[Eltq-Windowed-Shared-Events]` | minor | `[ELT-Shared-Event-Uncertainty]`, against the local eltq |
| 9 | `[Eltq-Release-For-Aggregate]`, second release | none, a tag | the `elt` extra's lower bound |

Steps 3 and 4 can proceed while aggregate builds its deterministic step, which needs no eltq.
Step 7 needs the conditional moments of step 6; step 8 needs the shared tail query of step 7 for
its moment fields.

## Files

- `secondary-uncertainty-mathematical-specification.md`: sections 4B (new), 6, 7, 9, 11.
- `crates/eltq-core/src/`: a new `moments.rs` (closed forms, the recursion, the factor and inner
  rules) and `tails.rs` (one-row tails, the factor bracket, the evaluators); `multiunit.rs`
  (classification, large parameters, conditional moments by unit law), `views.rs` (pass 1 shared,
  windows, cut rows, streamed strata, overflow accounting), `curve.rs` (the narrow kernel's higher
  moments), `quadrature.rs` (the deep tail partition), `config.rs`, `diagnostics.rs`, `error.rs`,
  `lib.rs`.
- `crates/eltq-python/src/lib.rs`: the four functions and the `overflow` keyword.
- `python/eltq/__init__.py`, `_meta.py` (`ModelMoments`, `TailMoments`, `MultiMeta` fields),
  `reference.py`, `reference_multiunit.py`.
- `tests/test_moments.py` and `tests/test_tails.py` (new; tests 35 to 38), `tests/test_multiunit.py`
  (tests 34 and 39), `tests/conftest.py` if the registries need entries.
- `benchmarks/run_moments.py` (new) or an extension of `run_multiunit.py`;
  `benchmarks/report-<version>.md`.
- `README.md` (milestone rows, API block), `CHANGELOG.md`, the version triple.

## Acceptance

- `cargo test -p eltq-core` at the plain test profile and `uv run pytest` green at every step, both
  backends.
- Spec tests 34 to 39 implemented and passing, with the figures they leave open recorded in the
  specification.
- The new moment and tail functions take no `bucket_size` or `grid_length`, so their outputs
  cannot depend on aggregate's `bs` or `log2`; their memory and time are independent of $M$ (the
  same rows at $M$ and $10^{12}M$ cost the same).
- Without a window every existing output is bitwise unchanged; existing goldens hold except where
  `[Eltq-Multiunit-Narrow-Units]` records a regeneration.
- Determinism across thread counts bitwise on every new path; no NaN, infinity or negative weight
  from any input (spec section 8).
- Benchmarks recorded with time and peak memory; no regression on unchanged routes.

## Alternatives considered

- *Moments from a fine hidden grid.* Rejected by aggregate's `[ELT-Moment-Pass]`: bucketing error
  and the stratified rule's approximation would enter the theoretical moments, and the grid's
  size would follow $M/h$.
- *Moments under the stratified rule's within-stratum independence.* Cheaper and equal to what the
  views route discretizes, but not the model; the gap is measured instead (step 6).
- *A beta fitted to the event total.* The roll-up route of spec section 4A, not the model.
- *Tail queries as methods of `ExceedanceCurve`.* The curve interpolates the betas (tolerances of
  spec tests 15 and 33) and needs a `min_loss`; exact per-row closed forms cost about the same per
  edge. The curve stays the fast approximate $R(t)$.
- *The shared tail from the windowed grid itself.* Available only after quantization and tied to
  $h$; kept as the measured overflow, not the sizing query.
- *Monte Carlo for shared tails.* Neither rigorous nor fast at twelve nines.
- *Lancaster or Hermite expansions of the cross moments.* Slow to converge for small $\alpha$,
  whose $g_u$ is nearly a step in the score.
- *Capping units before summing.* The wrong upper tail (aggregate's plan).
- *A model object returned by `moments_arrays` that evaluates later queries.* Saves one
  classification pass per call, which is $O(N)$ and small beside the queries; possible later.

## Not in this plan

- Moments and tails of the gross, ceded and net views (unit terms); aggregate uses ground_up only.
- Windows for `occurrence_pair_joint`, `occurrence_unit_first_moment`, `joint_two_units` and the
  roll-up diagnostic.
- `matched` beyond the decision in open question 6.
- Aggregate's sizing, `elt_max_log2`, YLTs (no eltq involvement), and DecL exposure of couplings.
- A survival-curve formulation of the multi-unit total.

## Open questions for review

>>>Steve comments 2026-10-26 22:02:24

1. **Return types.** Dataclasses in `_meta.py` (`ModelMoments`, `TailMoments`), as every other
   eltq result (recommended), or plain dicts. >>>Dataclasses
2. **Placement of the tail queries.** Free functions vectorized over edges (recommended), methods of
   `ExceedanceCurve`, or methods of a model object returned by the moment functions. >>>Free functions
3. **Default tolerances.** `rtol` for shared moments (aggregate proposes $10^{-6}$ relative in
   $C_2$ and $C_3$, its `elt_moment_rtol`), and the default bracket tolerance and partition floor
   of the shared tail query; set from the convergence study. >>>Agree
4. **Benchmark budgets.** Proposed: one-unit moments and tails on $10^6$ rows by 64 edges well
   under a second; `multiunit_moments` within one default `multiunit_views` run of the 0.91M
   portfolio (8.4 s, `benchmarks/report-2.10.0.md`); windowed views at $2^{22}$ from the
   measurement step. >>>Seems reasonable; full quantization on 250K is mostly < 1 sec
5. **Default `overflow` on `multiunit_views`** with a short `grid_length`: `top_bucket`, as
   `quantize_arrays` (recommended; only calls that raise today change), or `error`, today's
   behavior. >>>agree recommendation
6. **The `matched` coupling.** Refuse it in `multiunit_moments`, `multiunit_tail_moments` and
   windowed views for now (recommended), or define its scale without a grid, which moves its
   current outputs. >>>Agree - that's the next phase
7. **Window moment fields for gross, ceded and net**: `None` until view tails exist (recommended),
   or required now. >>>agree - also next phase.
8. **Release cadence.** Two releases (recommended) or one at the end; and whether to publish
   2.10.0 on its own first. >>>Two releases
9. **`[Eltq-Multiunit-Narrow-Units]` as its own patch release** ahead of the rest, since the NaN
   reaches any current user of the multi-unit route with narrow rows. >>>Yes, patch first.
10. **The lattice bracket in the first shared release**, or the union bound with the factor
    bracket until aggregate's fixtures show a loose bound failing a window that is in fact
    adequate. >>>Union bound to start with and see if material.

## Execution log

Versions, measured figures and divergences, as in `dev/done/plan-multiunit-coupling.md`. Started
2026-10-26 from `master` at 8a8f472 (2.10.0), the tree clean but for this plan. Baseline gates
green: `uv run pytest` (3 skipped, the aggregate-package tests) and `cargo test -p eltq-core`.

### [Eltq-Spec-Moments-Tails-Windows], no bump

Section 4B (model moments, tail measures, the factor bracket and the union bound), the multi-unit
window paragraph of section 6, the outputs of section 7, tests 34 to 39 of section 9 and item 10 of
section 11. Committed without a version, the plan moving to `dev/done/` in the same commit.
Divergences:

- The centred tail moments $K_k$ and their recurrence from the beta's differential equation are the
  chosen way to form $D_r$ without the subtraction $T_r-c^rR$ (the plan asked for the property,
  not a method). At $\hat c=0$ the recurrence is the partial-moment identity, so one incomplete beta
  gives all four orders on the regular route.
- The union bound's lower bounds are spelled out: $1-\prod_u\mathsf P(X_u\le t\mid z)$ for $R$, and
  the larger of $t^r\mathsf P(S>t\mid z)$ and the units' own tails for $T_r$ (with $t^r-c^r$ for
  $D_r$). The lattice bracket is described as waiting on evidence (ruling 10), not specified.
- Test 34's agreement was first written as $10^{-10}$ below $\min(\alpha,\beta)=10^{14}$ and
  test 31's $10^{-8}$ from there, fearing the kernels' $10^{-6}$ near the handover; the
  measurement at 2.10.1 met the plan's single $10^{-10}$ throughout, and the spec says so.
- Tidying in the same commit: the header's path to the coupling plan now points into `dev/done/`,
  and a literal `\n\n` in test 32 that had put the sentence introducing the multi-unit tests ahead
  of test 33 is replaced, the sentence now preceding test 25.

Planning checks, 2026-10-26: read aggregate's plan (rulings, `[ELT-Row-Laws]`,
`[ELT-Shared-Events]`, `[ELT-Moment-Pass]`, `[ELT-Source-Grid]`, `[ELT-Support-In-Sizing]`,
`[Eltq-Windowed-Shared-Events]`, `[ELT-Dependency]`, the implementation sequence,
`[ELT-Grid-And-Window-Tests]`, the remaining choices) and verified every statement of current
behavior above against the source at the cited lines. Recomputed the two regression rows with SciPy
and mpmath. Probed the multi-unit route on both backends for the narrow and sub-resolution rows.
Searched the proposed names. No code, specification or other repository was changed.

### [Eltq-Multiunit-Narrow-Units], 2.10.1

The probe reproduced on the engine (NaN at nodes 2 and 3, mass NaN) before the change; after it
both backends give 0.05 and 0.05, and the two-unit event agrees to $5.5\times10^{-18}$. Test 34 on
both backends; the sweep over $\min(\alpha,\beta)\in\{10^4,2\times10^5,2\times10^6,10^9,10^{14},
10^{16},10^{24}\}$ at mean shares $\tfrac12$ and $\tfrac14$, under `burke`, `comonotone`,
`independent` and `homer` and the three factor rules, has the backends within $1.4\times10^{-12}$
of the rate at $10^4$, $6\times10^{-14}$ at $2\times10^5$ and $2\times10^6$, $1.3\times10^{-13}$ at
$10^9$, $5.8\times10^{-11}$ at $10^{14}$, $6\times10^{-14}$ at $10^{16}$ and $1.1\times10^{-13}$ at
$10^{24}$, the rate exact to $10^{-14}$ everywhere. No golden held such a row; all hold.
Divergences:

- A new module `crates/eltq-core/src/law.rs` (`BetaLaw`, `NarrowLaw`) carries the law of a beta
  share at every parameter size, the incomplete beta below $2\times10^5$ bit for bit and the
  narrow kernel normalized on its bulk (relative floor `TAIL_RATE_FLOOR`) above; the plan's file
  list did not name it. `narrow_bulk` became `pub(crate)` and `NarrowKernel` gained `mean` and
  `sd`. The multi-unit functions `marginal_pmf`, `edge_cdf`, `edge_scores` and
  `LayerMeans::new` now return `Result`, the typed kernel error replacing what would have been a
  NaN.
- On the reference, units past $2\times10^5$ take the one-row reference's law (SciPy's incomplete
  beta with the normal law where SciPy fails, and the normal law from $10^{16}$), and the
  comonotone node path takes `large_parameter_quantiles`, which the engine's `beta_quantiles`
  already used; the plan named the classification only. The reference's clamp is also in ratios.
- `atom_by_resolution` applies in `prepare_unit` too, so the two-unit joint and the rolled row of
  the `homer` coupling and of the roll-up diagnostic classify the same way.

### [Eltq-Model-Moments], one-row ELTs, 2.11.0

`moments_arrays` on both backends; `ModelMoments` in `_meta.py`; `crates/eltq-core/src/moments.rs`
(`RowLaw`, `model_moments`); `ClassifyOptions` and `classify_rows_with` on the core, through which
`quantize` now classifies. Test 35 in `tests/test_moments.py`, 22 cases on both backends, and the
fallback subprocess calls the function. A million rows take 20 ms. Divergences:

- No `threads` keyword: the pass is the serial classification `quantize` already makes plus a few
  multiplications per row (20 ms per million rows), so parallel chunks would buy nothing and the
  result is trivially a function of the input. The plan's determinism check is therefore moot.
- Field names: `moment_overflow` (a flag per order; the moment itself is infinite) for the plan's
  "flagged", `achieved_error` for the integration error of $C_2$ and $C_3$, and `rtol` None for a
  closed form. `support` is NaN on dropped rows, so that the array lines up with the input.
- The 60-digit values of the tests are embedded, computed with mpmath 1.4.1: CI installs
  `[reference]`, `pytest` and `pandas` but not mpmath. Found on the way, not changed: test 23 in
  `tests/test_survival.py` imports mpmath unguarded, so CI's wheel tests would fail on it unless
  something else installs mpmath (open item for Steve).
- The plan's consistency check (grid moments approaching $C_r$ as $h$ halves) is in the test with
  the measured figures in the spec.

### [Eltq-Tail-Queries], one-row ELTs, 2.12.0

`tail_moments_arrays` on both backends; `TailMoments` in `_meta.py`; `crates/eltq-core/src/tails.rs`
(`tail_moments`, `beta_tail`), `NarrowKernel::centred_integrals` and `NarrowLaw::centred_tail`.
Test 37 in `tests/test_tails.py` (both backends, plus the engine against 60-digit arithmetic where
mpmath is installed); `benchmarks/run_moments.py` and `benchmarks/report-2.12.0.md`: a million rows
by 64 edges in 0.16 s to 0.29 s on 16 cores, 3.2 s to 6.1 s on one, bitwise across thread counts.
Divergences:

- The plan's `top_half_atoms` is two arrays, `top_half_rate` and `top_half_excess`; `regimes`
  counts the rows by evaluation (`incomplete_beta`, `narrow_kernel`, `normal_law`, `atoms`,
  `two_point`); `method` is `exact`; `moment_overflow` flags an order that leaves double
  precision, as `ModelMoments` does.
- The plan's "within $10^{-12}$ relative over $\beta$ from $\tfrac12$ to $10^7$" does not hold
  everywhere: the engine's continued fraction carries the residuals spec section 5A already
  documents, measured here at up to $2.6\times10^{-10}$ relative above the switch point at
  $\beta=10^7$ (of order $\varepsilon\beta$), an absolute $\varepsilon$ below it, and a few units
  in $10^{-12}$ near the mean when both parameters are large; the excess moments about a node a
  thousandth below the edge are within $1.5\times10^{-9}$ there. Test 37 in the spec states these
  figures. Removing them would mean a different incomplete beta, not a change to the tail query.
- The reference forms the centred moments by adaptive quadrature (`scipy.integrate.quad`) where
  both parameters reach $10^3$: its recurrence needs the density value, and SciPy's `betaln` is
  off by $10^{-9}$ at $(10^6,10^6)$ and $7\times10^{-6}$ at $(10^9,10^9)$ absolute, which put the
  reference's $D_1$ at $(10^9,10^9)$ $4.6\times10^{-5}$ off before the change. $T_r$ on the
  reference is the partial-moment identity through SciPy, so the two backends do not share a
  formula for it.
- The 60-digit oracle at $\beta=10^7$ with $\alpha=10^4$ is a quadrature of the density (mpmath's
  incomplete beta did not finish there in ten minutes); the sweep runs only where mpmath is
  installed, as CI does not install it.
- A beta row stops at the first edge, in ascending order, where its whole contribution is zero;
  atoms never stop early, since a later edge's node can still put them in its half bucket.

### [Eltq-Model-Moments], shared events, 2.13.0

`multiunit_moments` on both backends: `crates/eltq-core/src/event_moments.rs` (`ScoreTable`,
`multiunit_moments`), the reference's `multiunit_moments_reference`, `ModelMoments`'s multi-unit
fields; the shared pass 1 `views::event_pass`. Test 36 in `tests/test_moments.py` (both backends)
against oracles from `fixtures/generate_moment_oracles.py`; `benchmarks/report-2.13.0.md`.

The numerical design was settled by prototype before the engine was written, each version against a
brute-force reference (nested adaptive quadrature over the factor and the unit's own score, with
SciPy's beta quantile) on four cases (moderate shapes, $\alpha=10^{-4}$, loadings near one, the
first regression row):

- Per unit, $\ln y$ against the score $v$ on nodes placed by forward evaluations only, cubic
  Hermite with exact slopes from the density: the tables were accurate to about three times their
  tolerance in every case. Seeding at one node per seven decades left a table ending at $10^{-8}$;
  seeding per decade fixed it.
- The inner integral by Gauss-Hermite in the unit's own score failed for small $lpha$ ($10^{-4}$
  relative at 16 nodes, $5\times10^{-8}$ at 48), the loss switching on abruptly far in the tail;
  Gauss-Legendre over the nonzero range was worse. Pieces split at the table's breakpoints in the
  kernel's standardized variable reached the table's accuracy everywhere.
- Breakpoints by the rise of $\ln y$ missed the knee where a small-$\beta$ share approaches one
  (Homer and Li's event 3, $\alpha=0.0028$, $\beta=0.054$: $-5.7\times10^{-4}$ at the first
  level); the logit of $y$ in its place resolves both ends.
- Cost: the first engine took 91 s on the 0.91M portfolio, 6.7e10 evaluations, because units with
  loadings above about 0.9 are steep in the factor and every unit of their event was evaluated at
  the dense nodes they need. Tuned on 29 sampled events against a tight run of the same scheme:
  evaluating each beta unit at its own factor nodes and interpolating the logarithm onto the
  event's nodes, 8-point inner pieces 2.5 wide, breakpoints every 3 in the logit, a reach of 8,
  took the sample from 823k to 122k evaluations per event at an unchanged worst error of
  $1.4\times10^{-8}$; a coarse level 1.5 times the fine level's widths is the cheapest that still
  confirms convergence at the first comparison (at 2.5, half the events refined). The engine then
  took 9.5 s.
- A tolerance below about $10^{-10}$ ran the tables out of nodes, and depth-first refinement left
  the right end of a table unrefined ($8\times10^{-5}$ off at `rtol=1e-10`); the tables are now
  refined breadth-first with a node budget of 16,384 and a floor of $10^{-12}$ on their tolerance.

Divergences:

- The plan's budget (within one default `multiunit_views` run, 8.4 s) is missed by 13%: 9.5 s, at
  40 times the requested accuracy. Steep units, inherent to Burke's loading wherever
  $\sigma_I\ll\sigma_C$, set the cost. Open item for Steve: accept, or ask for a further pass.
- The plan's Hoeffding check "to $10^{-10}$" holds at `rtol=1e-8` ($2\times10^{-12}$ measured); at
  the default `rtol` the two betas are within $10^{-8}$, so test 36 states both. The three-unit
  two-point check uses SciPy's quadrature over the factor rather than 60-digit arithmetic, and
  the comonotone and Hoeffding oracles are SciPy computations converged by doubling, all embedded
  with their generator; Homer's event 3 oracle is good to $10^{-7}$ in $C_3$ and the test says so.
- `max_refinements` is a public keyword (default 3), so that a test can force an unconverged event;
  the plan named only `rtol`. `ModelMoments` gained `unconverged_events` (a count) beside the
  bounded `unconverged` examples.
- The plan's "the gap between the grids' moments and the model recorded as two numbers" is
  recorded as measured: the mean converges at first order in $h$ (the bucket-zero term), not the
  plan's $O(h^2)$, and the second and third moments to the stratified rule's limits,
  $-5.2\times10^{-4}$ and $-1.0\times10^{-3}$ on Homer and Li's three events.
- `multiunit_views`'s pass 1 moved into `views::event_pass`, shared with the moments as the plan
  asked; the views' outputs are unchanged.

