# Plan: survival-tail-cost, negligible tail work and three defects found after 2.0.0

Date: 2026-10-24. Status: proposed, for an independent check before any implementation.
Author of the analysis: the downstream tie-out session (Claude), at Steve's request.

## What is asked of the reader

This plan records four findings about the survival method as it stands in the 2.0.0 working
tree, with synthetic reproductions, and proposes changes. **Check the findings independently
before building anything**: rerun the reproductions, read the code at the lines cited, and confirm
or refute each root cause and each estimate. Where the analysis is wrong or a better remedy
exists, say so and revise the plan; it is a proposal, not a ruling. The checks requested are
listed under "Independent check" at the end of each finding and collected in "Checks before
implementation" below.

Steve's targets (2026-10-24): the survival grid on a real ELT block at $2^{22}$ buckets should
take seconds, not minutes; 7 seconds is acceptable, 70 is not. Recommendation: land at least
[tail-cut] and [oep-pml] before the 2.0.0 tag, because 2.0.0 makes the survival method the
default and the default is too slow on real ELTs without [tail-cut].

## Background for a reader with no context

`eltq.quantize_arrays(..., method="survival")` (the default from 2.0.0) builds the portfolio
exceedance curve $\Lambda(x)=\sum_j\lambda_jS_j(x)$ of the beta events (spec section 5A) and
buckets it by the nearest rule at the half-bucket edges $e_i=(i+\tfrac12)h$. The pieces that
matter here:

- **Nodes.** Geometric from $x_0=h/2$ to $\max_jM_j$ at 100 per decade
  (`curve.rs::geometric_nodes`, lines 51-65).
- **Truncation and tails.** Each beta event is split at its truncation node $x_{k_j}$, four node
  intervals below the first node at or above $M_j$ (`TAIL_INTERVALS`). The shared curve carries
  $\lambda_j(S_j(x)-S_j(x_{k_j}))^+$; the **tail**, mass $\lambda_jS_j(x_{k_j})$ on
  $(x_{k_j},M_j]$, stays with the event. The build keeps a `TailEvent` only when
  $S_j(x_{k_j})>0$ (`curve.rs` line 427), so tails whose survival underflows to exactly zero cost
  nothing later.
- **Tail bucketing.** `Curve::grid` (`curve.rs` lines 635-756) buckets every kept tail per event,
  in chunks: for each tail it builds every edge from the first above $x_{k_j}$ to the last below
  $M_j$ (lines 716-723), evaluates the tail interpolant at each (`tail_survival`, lines 119-165:
  a `partition_point`, a `ln`, `limit_slopes`, `hermite` and an `exp` per edge), and deposits the
  differences (lines 725-729). The SciPy reference does the same (`reference.py`, the curve's
  `grid`, from line 565, and `_tail_survival`, line 206).
- Because the truncation node is four intervals of ratio $10^{0.01}$ below the node at or above
  $M_j$, the tail interval is 7% to 9% of $M_j$ wide.

## Finding 1: tail bucketing costs about $0.08\sum_jM_j/h$ edge evaluations, nearly all wasted

**Current behavior.** The per-event tail work is proportional to the width of the tail interval in
buckets, $(M_j-x_{k_j})/h\approx0.08M_j/h$, for every event whose tail survival does not
underflow to zero. Two things make most of that work worthless:

1. The edges are evaluated whatever the event's exceedance rate there. For the low-mean,
   high-exposed-value events typical of catastrophe ELTs ($\mu_j/M_j$ of $10^{-4}$, $\alpha_j$
   well below 1, $\beta_j$ in the hundreds or thousands), $S_j$ on the top 8% of $[0,M_j]$ is
   positive in double precision but far below anything that matters, so millions of edges deposit
   numbers like $10^{-100}$ of the total rate.
2. The edges are evaluated beyond the window. `last` (line 717) is not clipped to $n-1$, so a
   window shorter than $\max_jM_j$ still pays for every edge up to $M_j$, and `put` then sends
   each of them to the overflow one at a time.

The benchmark did not show this because its exposed values are at most $10^6$ with buckets of 100
or more, so a tail is at most about a thousand edges (`benchmarks/report-1.16.0.md`).

**Evidence, synthetic** (reproduction 1 in the appendix; 100,000 rows, $M$ log-uniform on
$[10^6,10^9]$, 98% low-mean events and 2% with mean damage ratio 0.5 to 0.9, $h=25$,
$n=2^{22}$; this machine, all threads):

| | seconds |
|---|---|
| quantiles method, $K=32$ | 0.91 |
| survival method | 213 |
| `exceedance_curve` alone | 0.37 |

Tail edges evaluated: $2.7\times10^{10}$. Edges inside the window where the event's own exceedance
rate $\lambda_jS_j$ is above $10^{-20}$ of the total rate: $8.2\times10^8$, a ratio of 33. The
remaining edges belong mostly to the 2% of events with real mass near $M_j$, which need them.
(Both methods report a mean error of about $-0.57$ on this portfolio. That is the `top_bucket`
window: the top-heavy events put mass above $(n-\tfrac12)h$, and it is not related to this
finding.)

**Evidence, real ELTs** (a downstream tie-out of eight vendor ELT blocks, ground-up and gross,
four perils; the data is kept outside this repository and only ratios are given here). At
$2^{22}$ buckets the survival grid took 3 to 80 times as long as the quantiles method, up to about
70 seconds against about 1. The ratio of tail edges evaluated to edges carrying more than
$10^{-20}$ of the total rate was between 34 and 29,000 by block, and infinite in two blocks, where
no edge needed evaluating. The block with the smallest ratio has many policies at limits, events
with real mass at $M_j$.

**Proposed change, [tail-cut].** Stop each tail where its contribution is negligible, and stop it
at the window.

- A constant $\varepsilon=10^{-20}$ (`TAIL_RATE_FLOOR`, shared by `_meta.py` and `curve.rs`) and
  the threshold $\tau=\varepsilon\Lambda(0)$, with $\Lambda(0)$ the total rate of the beta
  events.
- A tail whose whole mass $\lambda_jS_j(x_{k_j})$ is at most $\tau$ is not bucketed edge by edge.
  Its mass goes, in one deposit, to the bucket containing $x_{k_j}$ (or by the linear rule at its
  conditional mean, which costs one partial expectation; either keeps the rate exact).
- A kept tail is walked from $x_{k_j}$ up to the first edge $e_s$ where $\lambda_jS_j(e_s)\le\tau$,
  and its remaining mass $\lambda_jS_j(e_s)$ goes to the bucket above $e_s$. The tail's own nodes
  descend in survival, so the stop can be found on them before any edge is built.
- The walk also stops at $e_{n-1}$: the remainder $\lambda_jS_j(e_{n-1})$ goes to the overflow in
  one step through the policy, with its moment from `tail_moment` as lines 731-735 already
  compute.
- `exceedance(x)` and `exceedance_rate` must stay consistent with the grid: a lumped tail is a step
  at its deposit point, exactly as an atom is.

**Error bound.** Each cut moves at most $\tau$ of rate, and only upward in loss within one event's
tail interval, so $\Lambda$ changes by at most $N\tau=N\varepsilon\Lambda(0)$ at any $x$, where
$N$ is the number of beta events: $10^{-14}$ of the total rate for a million rows, below the
$10^{-12}$ floor of spec test 15. The rate is exact; the first moment changes by at most
$N\tau\max_jM_j$, negligible against the reported bucketing error.

**Projection.** At today's per-edge wall cost on the real blocks (4 to 16 nanoseconds an edge,
all threads) and a fixed cost of about 0.6 seconds per grid at $2^{22}$ buckets, [tail-cut]
brings the real blocks to about 1 to 3 seconds. On reproduction 1 it brings 213 seconds to about
6.5, which is why [edge-cost] is likely to be needed as well. These are projections from edge
counts, to be confirmed by measurement.

**Independent check.** (a) Confirm from the code that kept tails are evaluated at every edge from
`first` to `last` with no window clip and no rate floor. (b) Rerun reproduction 1 and the edge
accounting. (c) Confirm or correct the error bound, and say whether lumping at $x_{k_j}$ or at the
conditional mean is preferable. (d) Consider whether a smaller `TAIL_INTERVALS`, or a tail
interval defined in buckets rather than node intervals, would be a better remedy than the cut
(I think not: the tail interval exists to resolve the endpoint corner on the shared nodes, and its
width is set by the node spacing, not by $h$).

## Finding 2: per-edge cost and fixed cost of the grid

**Current behavior.** Each tail edge costs a binary search over the tail nodes, a logarithm, an
exponential and a limited Hermite evaluation, and the edges are independent calls. Work is split
by chunks of 4,096 events, so when a few events hold most of the tail edges, one or two chunks
hold most of the work. Separately, the grid has a fixed cost of about 0.6 seconds at $2^{22}$
buckets even with no tail work (measured on a real block whose tails all underflow): the curve is
interpolated at every edge by an independent `interpolate` call with its own binary search
(lines 667-674, serial), and every tail chunk allocates and folds a grid-length histogram.

**Proposed change, [edge-cost], conditional.** Measure after [tail-cut]. If reproduction 1 is
still above five times the quantiles method, then, in order of expected gain per line of code:
walk the edges of each tail sub-interval in order instead of searching for each; evaluate the
curve at the edges by merging the ascending edges with the ascending nodes rather than by a search
per edge, and in parallel; split very long tails into edge ranges as separate work items, keeping
the fold order fixed for determinism. Replacing the cubic in $\log(M_j-x)$ by a cubic in $x$ would
remove the logarithm and exponential, but on a sub-interval of distance ratio $\sqrt2$ a cubic in
$x$ through $S\propto(M-x)^\beta$ has a relative error of order $10^{-4}$ against the present
$10^{-6}$, so it is not recommended without a measurement.

**Independent check.** Profile the grid after [tail-cut] before choosing any of these.

## Finding 3: `ExceedanceCurve.oep_pml` returns the largest atom when the answer lies above every atom

**Current behavior.** `python/eltq/__init__.py` lines 553-557:

```python
i = np.minimum(np.searchsorted(atoms, x_lo, side="left"), atoms.size - 1)
inside = atoms[i] <= x_hi
```

When the bisection bracket $[x_{lo},x_{hi}]$ lies above every atom, `searchsorted` returns
`atoms.size`, the clip makes it the last atom, and `atoms[-1] <= x_hi` is true, so the largest
atom's loss replaces the answer. Any curve with atoms is wrong at every return period whose PML
lies above its largest atom.

**Evidence.** Reproduction 3: one beta event and one atom at 100,000. `oep_pml` returns 100,000
at 10 and 100 years; bisection on `curve.exceedance` gives 111,222 and 2,318,848. On the real
ELTs, the one block with atoms was wrong from the 100-year point on.

**Proposed change, [oep-pml].**

```python
j = np.searchsorted(atoms, x_lo, side="left")
inside = j < atoms.size
inside[inside] = atoms[j[inside]] <= x_hi[inside]
found[inside] = atoms[j[inside]]
```

with a test that a curve with an atom below the answer returns the bisection answer, alongside the
existing test that an atom whose step jumps the target returns that atom.

**Independent check.** Confirm the diagnosis and that the existing atom tests did not cover an
answer above every atom.

## Finding 4: betas narrower than a node interval are smeared by the shared interpolant

**Current behavior.** A beta with a very small coefficient of variation has a survival function
that steps from 1 to 0 over a few standard deviations around its mean. When that width is small
against the node spacing (2.3% of the loss at 100 nodes per decade), the shared cubic between two
nodes cannot follow the step, and $\Lambda$ is wrong by up to the events' rate across the
interval. Nothing in the build treats such events separately; the tails handle only the corner at
$M_j$.

**Evidence.** Reproduction 2: a 20,000-row portfolio, exact to $4.5\times10^{-9}$ in $\Lambda$
alone, plus 300 events with mean 250,000 and coefficients of variation from $10^{-6}$ to
$10^{-2}$, total rate $7.5\times10^{-5}$. With them, `curve.exceedance` is off by
$2.9\times10^{-5}$ near 249,800. On the real ELTs, one block had several hundred events at a
common policy limit with coefficients of variation below 1%, some with $\alpha$ and $\beta$ near
$10^{22}$. $\Lambda$ was off by up to $5\times10^{-4}$ relative across one node interval, and no
OEP or AEP point moved, because the interval lay between the return periods reported.

**Proposed change, [narrow-events].** Give each beta whose bulk lies within a few node intervals a
per-event treatment, by the same machinery as the tails: remove it from the shared curve and bucket
its survival exactly at the edges where both $\lambda_jF_j$ and $\lambda_jS_j$ exceed $\tau$ (the
same floor as [tail-cut]), which for such an event is a few standard deviations wide. An event
whose bulk lies within $[0,x_0]$ or within one bucket of width $2x_0$ could instead become an atom
at $\mu_j$ by the linear rule, mean exact. The criterion should depend on the curve's nodes and
$x_0$, not on $h$, so that one curve still serves every $h\ge2x_0$.

**Independent check.** Confirm the mechanism (the error appears only with the narrow events
present, and is confined to one or two node intervals), and choose the criterion. The "Not in this
plan" list of `dev/done/plan-single-unit-survival-route.md` anticipated a related case at the
endpoint (admissible near-Bernoulli rows with a step at $M_j$); this is the interior counterpart.

## Finding 5: the benchmark has no large-exposed-value, fine-bucket portfolio

The [benchmark] gate of the survival plan passed on portfolios whose tails are at most about a
thousand edges. **Proposed change, [benchmark].** Add reproduction 1 (or a seeded variant) to
`benchmarks/run_end_to_end.py` as a third survival-sweep portfolio, record it before and after
[tail-cut], and apply the five-times gate to it.

## Steps

Each step one bump and one commit, made by Steve. Versions are assigned at execution.

1. [oep-pml] the fix and its test.
2. [tail-cut] the rate floor and the window clip in both backends, the spec section 5A text, a test
   that the cut changes $\Lambda$ by at most $N\varepsilon\Lambda(0)$ on a small-alpha, large-$M$
   fixture against `exact_edges`, and the benchmark record.
3. [edge-cost] only if the measurement after step 2 requires it.
4. [narrow-events] the per-event treatment, the spec text and a test built on reproduction 2.
5. [benchmark] folded into step 2's record, or its own step if the portfolio is added separately.

## Acceptance

- Reproduction 1: the survival method within five times the quantiles method at the same grid
  (today 234 times).
- Reproduction 2: `curve.exceedance` within $10^{-8}$ of the exact $\Lambda$ across
  $[0.97L,1.03L]$.
- Reproduction 3: `oep_pml` equal to the bisection answer at every return period.
- Spec tests 1 to 20 pass on both backends; the survival goldens change by at most
  $N\varepsilon\Lambda(0)$ per element where the cut applies, and are regenerated if they compare
  bitwise.
- Downstream: Steve's tie-out rerun on the real ELTs takes at most 7 seconds per block for the
  survival grid at $2^{22}$ buckets, with its OEP and AEP points unchanged.

## Checks before implementation

1. Finding 1: the code path, reproduction 1, the edge accounting, the error bound, the lump
   location, and whether a different tail-interval definition is better.
2. Finding 2: a profile after [tail-cut].
3. Finding 3: the diagnosis and the test gap.
4. Finding 4: the mechanism and the criterion.
5. The whole: whether any of the four is better solved another way, and whether the 2.0.0 tag
   should wait (recommended: yes, for [tail-cut] and [oep-pml]).

## Appendix: reproductions

Synthetic data only. Needs scipy; runs against the installed `eltq`. Measured 2026-10-24 on the
2.0.0 working tree, Rust backend.

```python
"""Synthetic reproductions of three eltq 2.0.0 survival-method findings."""
import time

import numpy as np
from scipy.special import betaincc, betainccinv

import eltq


def beta_parameters(mean, sd, max_loss):
    p = mean / max_loss
    kappa = mean * (max_loss - mean) / sd ** 2 - 1
    return p * kappa, (1 - p) * kappa


def exact_lambda(x, rate, max_loss, alpha, beta, chunk=2000):
    out = np.zeros(len(x))
    for s in range(0, len(rate), chunk):
        sl = slice(s, s + chunk)
        y = np.clip(x[None, :] / max_loss[sl, None], 0, 1)
        out += rate[sl] @ betaincc(alpha[sl, None], beta[sl, None], y)
    return out


# 1. Tail cost
rng = np.random.default_rng(20261024)
N = 100_000
rate = 10 ** rng.uniform(-6, -2, N)
M = 10 ** rng.uniform(6, 9, N)
top_heavy = rng.random(N) < 0.02          # a few policies with mass at the limit
p = np.where(top_heavy, rng.uniform(0.5, 0.9, N), 10 ** rng.uniform(-5, -2, N))
mean = p * M
cv = np.where(top_heavy, rng.uniform(0.3, 0.6, N), 10 ** rng.uniform(0, np.log10(20), N))
sd = np.minimum(cv * mean, 0.95 * np.sqrt(mean * (M - mean)))
alpha, beta = beta_parameters(mean, sd, M)
cols = (rate, M, mean, 0.6 * sd, 0.4 * sd)
h, n = 25.0, 2 ** 22
for method in ['quantiles', 'survival']:
    t0 = time.perf_counter()
    _, sec, meta = eltq.quantize_arrays(*cols, bucket_size=h, grid_length=n, method=method)
    print(method, f'{time.perf_counter() - t0:.2f} s')
# Edge accounting, replicating Curve::grid.
x0 = h / 2
intervals = int(np.ceil(100 * np.log10(M.max() / x0)))
nodes = np.exp(np.linspace(np.log(x0), np.log(M.max()), intervals + 1))
k = np.searchsorted(nodes, M, side='left') - 4
xk = np.where(k < 0, nodes[0], nodes[np.maximum(k, 0)])
s = betaincc(alpha, beta, np.clip(xk / M, 0, 1))
first = np.floor(xk / h - 0.5) + 1
last = np.ceil(M / h - 0.5) - 1
edges = np.maximum(last - first + 1, 0)
evaluated = edges[s > 0].sum()
target = 1e-20 * rate.sum() / rate
x_eps = M * betainccinv(alpha, beta, np.minimum(target, 1.0))
x_eps = np.where(np.isfinite(x_eps), x_eps, M)
stop = np.minimum(np.minimum(np.ceil(x_eps / h - 0.5) - 1, last), n - 1)
needed = np.where(s > target, np.maximum(stop - first + 1, 0), 0).sum()
print(f'edges evaluated {evaluated:.3g}, needed {needed:.3g}')

# 2. Narrow betas
L = 250_000.0
r2 = 10 ** rng.uniform(-5, -2, 20_000)
M2 = 10 ** rng.uniform(5, 8, 20_000)
mu2 = 10 ** rng.uniform(-4, -1, 20_000) * M2
sd2 = np.minimum(10 ** rng.uniform(0, 1, 20_000) * mu2, 0.95 * np.sqrt(mu2 * (M2 - mu2)))
r_n = np.full(300, 2.5e-7)
M_n = 10 ** rng.uniform(np.log10(3e5), 7, 300)
mu_n = np.full(300, L)
sd_n = L * 10 ** rng.uniform(-6, -2, 300)
R, MM = np.concatenate([r2, r_n]), np.concatenate([M2, M_n])
MU, SD = np.concatenate([mu2, mu_n]), np.concatenate([sd2, sd_n])
curve2 = eltq.exceedance_curve(R, MM, MU, SD, np.zeros_like(SD), min_loss=12.5)
x = np.linspace(0.97 * L, 1.03 * L, 241)
err = curve2.exceedance(x) - exact_lambda(x, R, MM, *beta_parameters(MU, SD, MM))
print(f'narrow: largest |curve - exact| {np.abs(err).max():.2e}')

# 3. oep_pml with an atom below the answer
curve3 = eltq.exceedance_curve(np.array([0.5, 1e-3]), np.array([1e7, 1e5]),
                               np.array([2e5, 1e5]), np.array([6e5, 0.0]), np.zeros(2),
                               min_loss=50.0)
rps = np.array([10.0, 100.0])
theta = -np.log1p(-1 / rps)
lo, hi = np.full(2, 50.0), np.full(2, 1e7)
for _ in range(200):
    mid = np.sqrt(lo * hi)
    above = curve3.exceedance(mid) > theta
    lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
print('oep_pml', curve3.oep_pml(rps), 'bisection', hi)
```

Output on 2026-10-24, two runs: quantiles 0.90 and 0.91 s, survival 199 and 213 s; edges
evaluated $2.67\times10^{10}$,
needed $8.18\times10^8$; narrow $2.92\times10^{-5}$ (without the 300 narrow events,
$4.5\times10^{-9}$); `oep_pml` [100,000, 100,000] against bisection [111,222, 2,318,848].

## Execution log

Approved by Steve on 2026-10-24 with the corrections of the independent check and the version sequence: [oep-pml] 2.0.1, [tail-cut] 2.1.0, the publishing tag v2.1.0 at the earliest. The plan moved to `dev/done/` at the first step. Steve makes every commit; the files per step are listed here.

### [oep-pml] 2.0.1

- `python/eltq/__init__.py`: the atom search in `oep_pml` as proposed; `__version__`.
- `tests/test_survival.py`: `test_curve_oep_pml_above_every_atom`, reproduction 3 as a test, plus a check that the atom still wins where its step jumps the target.
- `Cargo.toml`, `pyproject.toml` (`Cargo.lock`, `uv.lock` on build), `CHANGELOG.md`, this plan moved to `dev/done/`.
- Commit subject: `2.0.1 fix oep_pml returning the largest atom above every atom`.

### [tail-cut] 2.1.0

- `crates/eltq-core/src/curve.rs`: `TAIL_RATE_FLOOR`; `tail_survival` becomes `TailInterpolant` (built once per event, limited slopes per interval, `at` by search and `at_ascending` by cursor), `tail_survival` kept as a wrapper for the unit test; `Curve::grid` walks each tail from the first edge above `x_k` to the smaller of the last edge below `M` and the top edge, breaks at the first edge where `lam S <= threshold`, lumps the remainder in the next bucket, and adds the overflow moment only when the remainder went beyond the grid; the per-event edge vectors are gone; unit test `tail_walk_with_cursor_matches_search`. `lib.rs` exports the constant.
- `python/eltq/_meta.py`: `TAIL_RATE_FLOOR`. `python/eltq/reference.py`: the same walk in blocks of `TAIL_BLOCK` edges in `SurvivalCurve.bucket`, with a reference-only `tail_rate_floor` argument; the overflow moment loop merged into the walk.
- `secondary-uncertainty-mathematical-specification.md`: section 5A "The rate cut" paragraph, the cost sentence of "The tails", test 21.
- `benchmarks/run_end_to_end.py`: `large_m_elt` and `--large-m` (default density only, one repeat, no single-thread determinism run); `benchmarks/results.json`: two `large_m` records, before and after.
- `tests/test_survival.py`: test 21. Version lines, `CHANGELOG.md`.
- Measurements (this machine, all threads): before, quantiles 0.895 s and survival 240.7 s (269 times); after, 0.894 s and 5.05 s (5.66 times). The plan projected about 7.4 s from edge counts. Both backends pass tests 15 to 21.
- Downstream tie-out (2026-10-24, the tie-out session, eight real blocks at 2**22 buckets, editable install rebuilt from this tree): survival grid 0.54 to 1.60 s per block (2.0.0: 0.6 to 70 s), 0.8 to 2.5 times K=32; OEP and AEP points unchanged at every return period; `oep_pml` within 4.1e-7 of the exact OEP in every block; the whole tie-out 2 min 17 s against about 7 min. The 7 s target is met by the cut alone.
- Divergence: the plan's [tail-cut] text had the stop found on the tail's nodes and the lumped tail reflected in `exceedance(x)`; as corrected in the independent check, the walk breaks in order and the curve stays exact.
- Commit subject: `2.1.0 cut survival tail walks at the rate floor and the grid window`.

## Independent check, 2026-10-24

By the survival-route session (Claude), at the tie-out session's request through Steve. Code read at the cited lines; the appendix script rerun unchanged on the 2.0.0 working tree, Rust backend, all threads; extra accounting and a coefficient-of-variation sweep run alongside. Nothing implemented.

**Reproductions.** All three reproduce: quantiles 0.90 s, survival 222.7 s; edges evaluated $2.67\times10^{10}$, needed $8.18\times10^8$; narrow $2.92\times10^{-5}$; `oep_pml` [100,000, 100,000] against bisection [111,222, 2,318,848].

**Finding 1, confirmed, with two corrections.**

- Code path confirmed: `Curve::grid` builds every edge from `first` to `last` with `last` from $M_j$ and no clip at $n-1$, evaluates `tail_survival` at each, and `put` sends each edge beyond the grid to the overflow accumulator one at a time; the reference does the same in `SurvivalCurve.grid` and `_put`. No rate floor anywhere.
- Correction: the window, not the rate floor, holds most of the waste on reproduction 1. Of the $2.67\times10^{10}$ edges, $2.88\times10^9$ lie inside the window of $2^{22}$ buckets; the clip alone removes 89% of the work (a factor 9.3), the rate floor the remaining factor 3.5. Both are needed for the 33.
- Correction: the sentence "the remaining edges belong mostly to the 2% of events with real mass near $M_j$" is wrong. The top-heavy events hold $9.3\times10^8$ of the evaluated edges and need $9.7\times10^7$; the low-mean events hold $2.57\times10^{10}$ and need $7.2\times10^8$. Most of the edges that survive the cut belong to low-mean events of large $M_j$ whose tail survival is small but above $10^{-20}$ of the total.
- Error bound confirmed, with the direction loosened: the lumped remainder is at most $\tau$ per event, and lumping it at the bucket above the stop edge moves it down in loss, so $\Lambda$ moves by at most $\tau$ per event in either sign and $N\tau$ in all. Measured on reproduction 1 the rate the cut moves is $1.4\times10^{-16}$ of the total against the bound $10^{-15}$. The tail widths are 6.7% to 8.8% of $M_j$ (the plan says 7 to 9%).
- Lump location: the bucket above the stop edge (and the bucket containing $x_{k_j}$ for a tail cut whole), not the conditional mean. The moment the lump displaces is at most $\tau M_j$, invisible against the reported bucketing error, and the conditional mean costs an incomplete beta per event for nothing.
- Simplification: no stop search on the tail's nodes is needed. Walk the edges from `first` in order and break at the first edge where $\lambda_jS_j\le\tau$ or at $e_{n-1}$; the cost is the needed edges plus one. This also removes the per-edge `partition_point`, which is the first item of [edge-cost].
- Consistency of `exceedance(x)` with the grid: recommended against. The curve stays the exact object and the grid departs from it by at most $N\tau$; that bound goes into the spec section 5A rather than a step into `exceedance_betas`. Changing the curve to match a grid-side economy inverts the design.
- A smaller `TAIL_INTERVALS` or a tail interval in buckets: agreed, no. The width is set by the node spacing for the endpoint corner (measured in the survival route's execution log), and halving it would gain a factor 2 against the 33.

**Finding 2, plausible, not measured.** At 222.7 s for $2.67\times10^{10}$ edges the wall cost is 8.3 ns an edge, all threads. After [tail-cut] reproduction 1 projects to about 6.8 s of edges plus the fixed cost, about 7.4 s, 8 times the quantiles method, so [edge-cost] is needed for reproduction 1 under the five-times gate. The serial curve interpolation at every edge (lines 667 to 674) is confirmed from the code. The per-thread cost (about 130 ns an edge on 16 threads) is far above a search, a logarithm and an exponential, which points at memory (the `edges` and `tail` vectors of up to $3\times10^6$ doubles per event, the grid-length histogram per chunk) and at load balance across chunks of 4,096 events. Profile before choosing, as the plan says; the in-order walk from finding 1 comes free.

**Finding 3, confirmed.** The clipped `searchsorted` makes `inside` true whenever the bracket lies above every atom. The existing test `test_curve_oep_pml` has its only atom at the curve's largest loss (the Bernoulli at $M=100$), so no answer can lie above it and the gap was not covered. The proposed four lines are correct.

**Finding 4, confirmed, with the criterion measured.** With the 300 narrow events the error is $2.92\times10^{-5}$ at 249,812 where $\Lambda=1.63$, relative $1.8\times10^{-5}$, under the $10^{-4}$ of spec test 15 here but the real-block figure of $5\times10^{-4}$ is over it. The 64-sample diagnostic reports $8.6\times10^{-7}$ and misses it, as the plan expects. One narrow event of rate $10^{-6}$ at mean 250,000 on the 20,000-row portfolio, error as a fraction of its own rate after the portfolio's own $4.5\times10^{-9}$:

| CV | error / rate |
|---|---|
| $10^{-4}$ | 0.54 |
| $10^{-3}$ | 0.41 |
| $3\times10^{-3}$ | 0.21 |
| $10^{-2}$ | $6\times10^{-3}$ |
| $2\times10^{-2}$ | $1.5\times10^{-3}$ |
| $3\times10^{-2}$ and above | below $5\times10^{-4}$ |

The node interval at the mean is 2.33% of the loss at 100 per decade. Recommended criterion: an event is narrow when $\sigma_j<\mu_j(10^{1/n_d}-1)$, one node interval at its mean, defined on the curve's density and not on $h$. Per-event bucketing for such events must evaluate the exact survival at the edges (a few hundred to a few thousand per event at $h=25$), not the endpoint interpolant in $\log(M_j-x)$, which is built for the corner at $M_j$ and has no business in the interior. An event whose bulk lies within one bucket is an atom at $\mu_j$ by the linear rule, as the plan says.

**Finding 5, agreed.** The added portfolio should keep $M_j$ beyond the window at $2^{22}$ buckets, as reproduction 1 does, because that is the real use.

**The tag.** Agreed that v2.0.0 should not be pushed as the tree stands. Since nothing is committed yet, the simplest sequence is: Steve commits the seven survival-route steps as listed, then [oep-pml] lands as 2.0.1 and [tail-cut] as 2.1.0, and the publishing tag is v2.1.0. Folding the fixes into the 2.0.0 commit would break one step, one bump.
