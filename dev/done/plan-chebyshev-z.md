# Plan: chebyshev-z — cutting the shared-factor loop in the multi-unit engine

Date: 2026-08-26
Status: closed 2026-10-25 without starting (Steve: "move cheby"). Superseded by [z-rule] of `dev/plan-multiunit-coupling.md`, which runs the same study with tail metrics and adds the stratified rule; [hybrid-rule] and [chebyshev-z] are reopened only if that study's node count leaves the speed question open.

## Context

`benchmarks/report-1.11.0.md`: on the 0.91M unit-row portfolio, ~85% of the
multi-unit engine's time scales with the number of shared-factor nodes Q.
Every event repeats the same work Q = 32 times: transform each correlated
unit's conditional row through its view map, convolve across units, add
w_q x result. Q = 8 runs 2.8x faster but the convergence study
(`benchmarks/z-convergence-1.7.0.md`) shows the midpoint rule at Q = 8 has
~1e-2 relative error in the 99th percentile on the balanced case, so Q
cannot simply be lowered. The question is whether Q = 32 accuracy is
reachable with far fewer convolutions.

The idea is not part of `dev/done/plan-multiunit.md`'s deliverables; it was
listed there as optional and profiling-justified. Profiling now justifies
looking at it.

## The observation

Everything z-dependent inside an event is a smooth function of z. For a
beta unit the conditional row at node z is
$$
p_k(z)=\Phi\!\left(\frac{y_{k+1}-w_C z}{w_I}\right)-\Phi\!\left(\frac{y_k-w_C z}{w_I}\right),
$$
analytic in z with z entering linearly. View maps are linear, convolution is
a product in the frequency domain, so every spectral coefficient of the
event pmf is a product of analytic functions of z, and the quantity wanted
is the integral
$$
\widehat{\mathrm{pmf}}_\omega=\int\Big(\prod_{i}\widehat{r}_{i,\omega}(z)\Big)\varphi(z)\,dz ,
$$
one smooth integrand per spectral bin. The engine only needs that integral,
not the integrand at 32 arbitrary nodes. Two ways to exploit this, in order
of ambition:

1. **[hybrid-rule]** Keep the node-mixture structure but choose the rule per
   event: Gauss-Hermite with small Q (8-12) where every correlated unit's
   conditional is smooth in z, midpoint at 32 where any unit is step-like.
   Positive weights throughout, so mass and nonnegativity stay exact; no new
   machinery beyond a per-event choice of `(z, w)`.
2. **[chebyshev-z]** Represent each unit's spectrum as a Chebyshev
   interpolant in z on a truncated range, multiply interpolants across units
   bin by bin, and integrate against phi(z) with a precomputed weighted
   Clenshaw-Curtis rule. Fewer nodes for the same polynomial exactness, at
   the price of possibly negative weights and Gibbs ringing on step-like
   units.

The smoothness that both rely on is controlled by the ratio $w_C/w_I$ per
unit: the conditional cdf in z is a smoothed step of width $w_I/w_C$.
Balanced units (case A of the study) are gentle; correlation-dominant units
(case B, $w_I \ll w_C$) approach a step; comonotone units ($w_I = 0$) are a
discontinuity and no polynomial rule helps (the engine already handles the
pure-comonotone event without a z loop of convolutions, but a comonotone
unit inside a mixed event forces its Q rows through the chain).

## What could be gained

If m = 10 nodes replaced Q = 32 at equal accuracy on the typical portfolio,
the z-scaling 85% shrinks by ~3x: 22.5 s -> ~9 s on the laptop, inside the
original single-digit target. On correlation-dominant books the gain shrinks
toward zero because m must approach Q there. The gain is portfolio-dependent
and must be measured on the mixed synthetic portfolio, not asserted.

## Risks, stated up front

- **Step-like units.** Gauss-Hermite at Q = 32 already shows 1.7e-2 pmf
  artifacts on case B and 1.4e-1 on case D (study tables). Any rule with
  clustered nodes or polynomial assumptions inherits this. The per-event
  classifier must route such events to the robust rule; misrouting is the
  main correctness risk.
- **Degree growth under products.** With U correlated units the product of
  degree-(m-1) interpolants has degree U(m-1). Evaluating the product at m
  nodes aliases. Either the product turns out smoother than its formal
  degree (products of smoothed steps often are) and aliasing is tolerable,
  or the product stage needs more nodes than it saves. Only measurement
  settles it.
- **Sign and mass.** The current scheme is a positive mixture. A
  Chebyshev-derived rule can carry small negative weights and interpolation
  error is not sign-controlled; the pmf may need clipping and
  renormalization, a modeling compromise the package has avoided so far and
  which must be reported in `MultiMeta` if introduced.
- **Determinism** is unaffected in either variant (fixed nodes and weights
  per event, same chunk fold), but results change versus 1.11.0 whenever the
  rule differs, so goldens must be regenerated under a stated tolerance
  rather than bitwise.

## [z-study] Proof of concept (Python, no engine changes)

Extend `benchmarks/z_convergence.py` (or a sibling `z_rules_study.py`) with
the two candidate rules and re-run the study grid. Everything here uses the
packaged SciPy reference, so nothing in Rust moves until the numbers say so.

1. **Rules under test.** For each of the four regimes A-D and Q in
   {6, 8, 10, 12, 16}:
   - midpoint (baseline, plus Q = 32 as the production comparator);
   - Gauss-Hermite (already implemented);
   - Chebyshev-z: nodes $z_j$ = Chebyshev points of the second kind on
     $[-L, L]$ with $L$ in {5, 6, 7}, weights $c_j$ from the
     Clenshaw-Curtis rule with the normal density folded in (compute once by
     integrating each Lagrange basis polynomial against phi numerically at
     high resolution; verify $\sum c_j = 1$ and the first four moments).
   Truncation mass outside $[-L, L]$ is reported (2e-9 at L = 6).
2. **Metrics** against the midpoint Q = 4001 reference, as in the existing
   study: maximum marginal cell error against direct beta bucketing, relative
   error of the sum distribution's mean, sd and q99, maximum sum-pmf cell
   difference, plus two new ones: the most negative pmf cell (sign control)
   and |mass - 1|.
3. **Product-degree probe.** A three-unit and a ten-unit mixed event (draw
   from the benchmark's synthetic generator, seeds fixed) to see whether
   aliasing at m nodes is tolerable when U(m-1) exceeds m. Compare the
   full-product integral at m nodes with the same at 4m nodes.
4. **Classifier probe.** Sweep $w_C/w_I$ from 0.25 to 8 for a single unit and
   record, per rule and Q, the smallest Q reaching 1e-4 marginal error. This
   gives the threshold for the per-event routing in [hybrid-rule].
5. **Report** `benchmarks/z-rules-study-<version>.md` with the tables and a
   one-paragraph verdict per rule: the Q at which each rule matches midpoint
   Q = 32 per regime, and the estimated engine speedup on the benchmark
   portfolio's regime mix (68% mixed, 10% independent, 5% comonotone, ...).

Go/no-go: proceed to [hybrid-rule] if Gauss-Hermite at Q <= 12 matches
midpoint Q = 32 on the mixed regime within the study's tolerances; proceed
to [chebyshev-z] only if the Chebyshev rule beats Gauss-Hermite at equal Q
on the mixed regime *and* the product-degree probe shows tolerable aliasing.
Otherwise close the plan with the report as the record.

## [hybrid-rule] Development sketch (if the study clears it)

- `quadrature.rs`: no change to the rules; `views.rs` gains a per-event
  choice. `MultiConfig` gets `z_rule = "adaptive"` alongside the existing
  two, with `z_nodes` meaning the smooth-event node count and a second
  field `z_nodes_robust` (default 32) for step-like events; the routing
  threshold on $\max_i w_{C,i}/w_{I,i}$ over the event's correlated units is
  a config constant taken from the study (documented, not user-tunable at
  first).
- Per event: classify, pick `(z, w)` from two precomputed pairs, everything
  downstream unchanged (the engine already takes `z` and `w` as slices).
  Comonotone units force the robust rule.
- `MultiMeta.engine` reports counts of events per rule.
- Tests: fast-path equivalence (adaptive versus midpoint Q = 32 within the
  study's stated tolerance on the fixture), determinism sweep unchanged,
  golden regenerated with tolerance, and a routing test that a
  correlation-dominant event takes the robust rule.
- Benchmark: the harness adds an `adaptive` run; the report records the
  all-four delta and the speedup on the mixed portfolio.
- Version 1.12.0.

## [chebyshev-z] Development sketch (only if justified by the study)

- `quadrature.rs`: `ZRule::Chebyshev { range, nodes }` producing the nodes
  and the phi-weighted Clenshaw-Curtis weights (closed form via the DCT of
  the moments of phi against Chebyshev polynomials, or the numerical
  integration used in the study, cached).
- `views.rs`: the node loop is unchanged in shape; what changes is that the
  per-event product may need evaluation at a denser node set than the
  per-unit rows were built on. Two options to decide from the study:
  (a) build unit rows directly at the dense product nodes (cost: more erfc
  work, still O(rows) not O(convolutions)); (b) interpolate unit spectra
  from m to the dense set with a precomputed Chebyshev-to-Chebyshev matrix
  (a small GEMM per unit per bin block). (a) is simpler and the erfc work is
  cheap after 1.11.0.
- Sign control: report the most negative cell per event in `MultiMeta`;
  decide on clipping only with numbers in hand (a clip changes mass, which
  the reconciliation would expose).
- Tests as for [hybrid-rule] plus a proptest that the rule integrates
  polynomials of degree up to the design order exactly against phi.
- Version 1.13.0.

## Non-goals

Changing the default rule (midpoint Q = 32 remains the default until a
release note says otherwise); any change to the one-unit `quantize` route;
GPU.

## Verification (both development steps)

- `uv run pytest` and `cargo test -p eltq-core` green, goldens regenerated
  under stated tolerances.
- `uv run python benchmarks/run_multiunit.py` before and after, same machine,
  same day; report appended to `benchmarks/results.json` and a
  `benchmarks/report-<version>.md`.
- Determinism thread sweep bitwise, as today.
