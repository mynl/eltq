"""Tail measures with no loss grid (spec section 4B, test 37; shared events,
test 38, from the second half of ``dev/done/plan-model-moments-and-windowed-views.md``).

The one-row query runs on both backends through ``TAIL_IMPLS``: the SciPy
reference (``eltq.reference.tail_moments_reference``: the partial-moment
identity through SciPy's incomplete beta, exact sums) and the package API
(the Rust engine: the identity in recursive form from one continued
fraction). The 60-digit values are embedded, computed with mpmath 1.4.1;
the sweep against mpmath itself runs where mpmath is installed (the dev
group) and is skipped otherwise.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, optimize, stats

import eltq
from eltq import reference
from eltq import reference_multiunit as rm

from conftest import FIXTURES_DIR

COLUMNS = ("rate", "max_loss", "mean_loss", "sd_indep", "sd_corr")

TAIL_IMPLS = [("reference", reference.tail_moments_reference), ("eltq", eltq.tail_moments_arrays)]


@pytest.fixture(params=TAIL_IMPLS, ids=[name for name, _ in TAIL_IMPLS])
def tails(request):
    return request.param[1]


def run(tails, rows, **options):
    cols = np.asarray(rows, dtype=np.float64).reshape(-1, 5).T
    return tails(*cols, **options)


def beta_row(alpha, beta, m=1.0, rate=1.0):
    """The row ``(rate, M, mu, sigma, 0)`` whose beta has these parameters."""
    p = alpha / (alpha + beta)
    sd = math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0)))
    return (rate, m, p * m, sd * m, 0.0)


# [Eltq-Tail-Queries] Spec test 37: the aggregate package's regression rows
# at t = mu + 40 sigma, the ratios P(X^r; X > t) / P(X^r) to seven figures
# (SciPy and 60-digit arithmetic agree to ten).
REGRESSION = [
    ((20_000.0, 60_000.0), (1.568651e-8, 2.031163e-6, 2.641404e-5, 1.816930e-4)),
    ((100.0, 10_000.0), (7.024077e-5, 0.6705151, 0.9384651, 0.9920658)),
]


@pytest.mark.parametrize("row, want", REGRESSION)
def test_regression_rows(tails, row, want):
    mu, sd = row
    cols = (1.0, 1e9, mu, sd, 0.0)
    t = mu + 40.0 * sd
    out = run(tails, [cols], edge=[t])
    moments = eltq.moments_arrays(*(np.array([v]) for v in cols)).raw_moments
    got = (out.rate_above[0], *(out.moments_above[0] / np.asarray(moments)))
    assert got == pytest.approx(want, rel=1e-6)
    assert out.regimes["incomplete_beta"] == 1 and out.method == "exact"


def test_beta_window_figures(tails):
    # Test 18's Beta(2, 1000) window: top edge 0.00195, node 0.0019; the
    # overflow rate and the excess moment about the node (SciPy), which the
    # grid route reaches only to its curve's 1e-4.
    out = run(tails, [beta_row(2.0, 1000.0)], edge=[0.00195], top_node=[0.0019])
    assert math.isclose(out.rate_above[0], 0.41891026155683486, rel_tol=1e-13)
    assert math.isclose(out.excess_above[0, 0], 5.796480662291177e-4, rel_tol=1e-12)


# Beta(1e6, 1/2) with t just below M and c = t - h/2 (test 32's edge,
# 5e-7 below the exposed value, h = 1e-6): the tail is short and close to c,
# where T_1 - c R would lose its digits. 60-digit values at the parameters
# the row reconstructs.
ENDPOINT_ROW = beta_row(1e6, 0.5)
ENDPOINT_T, ENDPOINT_C = 1.0 - 5e-7, 1.0 - 1e-6
ENDPOINT_RATE = 0.6826894920786666
ENDPOINT_D = (5.833154597462061e-7, 1.1666302648785945e-6, 1.749944415397838e-6)


def test_endpoint_excess_keeps_its_digits(tails):
    out = run(tails, [ENDPOINT_ROW], edge=[ENDPOINT_T], top_node=[ENDPOINT_C])
    assert math.isclose(out.rate_above[0], ENDPOINT_RATE, rel_tol=1e-9)
    for r in range(3):
        assert math.isclose(out.excess_above[0, r], ENDPOINT_D[r], rel_tol=1e-8), r


def test_atoms_with_the_strict_inequality(tails):
    t, c = 10.0, 9.5
    above = float(np.nextafter(t, np.inf))
    at = run(tails, [(0.5, 20.0, t, 0.0, 0.0)], edge=[t], top_node=[c])
    assert (at.rate_above[0], at.top_half_rate[0]) == (0.0, 0.5)
    assert at.top_half_excess[0] == pytest.approx(0.5 * np.array([t - c, t * t - c * c, t ** 3 - c ** 3]), rel=1e-15)
    nxt = run(tails, [(0.5, 20.0, above, 0.0, 0.0)], edge=[t], top_node=[c])
    assert (nxt.rate_above[0], nxt.top_half_rate[0]) == (0.5, 0.0)
    assert nxt.excess_above[0, 0] == pytest.approx(0.5 * (above - c), rel=1e-15)
    # A two-point row (mu 4, M = 10, its sds summing to 9): its atom at M
    # with rate 0.4 is excluded at t = M and included one ulp below.
    below = float(np.nextafter(t, 0.0))
    two = run(tails, [(1.0, t, 4.0, 9.0, 0.0)], edge=[t, below])
    assert two.regimes["two_point"] == 1
    assert two.rate_above[0] == 0.0 and math.isclose(two.rate_above[1], 0.4, rel_tol=1e-15)
    # An atom in the half bucket is in no tail measure.
    half = run(tails, [(0.5, 20.0, 9.75, 0.0, 0.0)], edge=[t], top_node=[c])
    assert half.rate_above[0] == 0.0 and half.moments_above[0].max() == 0.0
    assert half.top_half_rate[0] == 0.5


def test_window_equals_the_grid_route(tails):
    # At t = (n - 1/2) h and c = (n - 1) h the query is the grid's overflow:
    # the medium fixture windowed at a third of its largest exposure, the
    # betas to the curve's interpolation (test 18's 1e-4), atoms exactly.
    df = pd.read_csv(FIXTURES_DIR / "medium_elt.csv", float_precision="round_trip")
    cols = [df[c].to_numpy(dtype=np.float64) for c in COLUMNS]
    h = 1000.0
    n = int(df["max_loss"].max() / 3.0 / h)
    t, c = (n - 0.5) * h, (n - 1.0) * h
    out = tails(*cols, edge=[t], top_node=[c])
    _, _, meta = eltq.quantize_arrays(*cols, bucket_size=h, grid_length=n)
    assert math.isclose(out.rate_above[0], meta.rate_above_grid["secondary"], rel_tol=1e-4)
    assert math.isclose(out.excess_above[0, 0], meta.mean_above_grid["secondary"], rel_tol=1e-4)
    assert math.isclose(out.total_rate, meta.total_rate, rel_tol=1e-15)
    # Atoms alone: exact.
    atoms = [col.copy() for col in cols]
    atoms[3][:] = 0.0
    atoms[4][:] = 0.0
    out = tails(*atoms, edge=[t], top_node=[c])
    raw, _, meta = eltq.quantize_arrays(*atoms, bucket_size=h, grid_length=n)
    assert math.isclose(out.rate_above[0], meta.rate_above_grid["raw"], rel_tol=1e-14)
    assert math.isclose(out.excess_above[0, 0], meta.mean_above_grid["raw"], rel_tol=1e-14)


def test_monotone_in_the_edge_and_backends_agree():
    df = pd.read_csv(FIXTURES_DIR / "medium_elt.csv", float_precision="round_trip")
    cols = [df[c].to_numpy(dtype=np.float64) for c in COLUMNS]
    top = df["max_loss"].max()
    edges = np.geomspace(1e-3 * top, top, 40)
    engine = eltq.tail_moments_arrays(*cols, edge=edges)
    ref = reference.tail_moments_reference(*cols, edge=edges)
    assert (np.diff(engine.rate_above) <= 0.0).all()
    assert (np.diff(engine.moments_above, axis=0) <= 0.0).all()
    positive = ref.rate_above > 1e-12 * ref.total_rate
    np.testing.assert_allclose(engine.rate_above[positive], ref.rate_above[positive], rtol=1e-9)
    np.testing.assert_allclose(engine.moments_above[positive], ref.moments_above[positive], rtol=1e-9)
    np.testing.assert_allclose(engine.excess_above[positive], ref.excess_above[positive], rtol=1e-9)
    assert engine.regimes == ref.regimes


def test_bitwise_across_threads():
    df = pd.read_csv(FIXTURES_DIR / "medium_elt.csv", float_precision="round_trip")
    cols = [df[c].to_numpy(dtype=np.float64) for c in COLUMNS]
    edges = np.geomspace(1e2, df["max_loss"].max(), 16)
    runs = [eltq.tail_moments_arrays(*cols, edge=edges, threads=k) for k in (1, 2, None)]
    for other in runs[1:]:
        for field in ("rate_above", "moments_above", "excess_above", "top_half_rate", "top_half_excess"):
            assert np.array_equal(getattr(runs[0], field), getattr(other, field)), field


def test_large_parameter_regimes(tails):
    # Past the continued fraction's regime the narrow kernel (engine) and
    # quadrature (reference), and from 1e16 the normal law on both:
    # symmetric rows at the mean and two standard deviations above, against
    # the normal law at the z the double edge actually has (at 1e24 the
    # share resolves 3e-4 of a standard deviation). The beta's own excess
    # kurtosis, -3 / alpha, moves the tail by 6e-7 at 1e6 and 2.4e-6 in the
    # excess; measured agreement 2026-10-26.
    from scipy.stats import norm

    for smallest, regime, rtol in ((1e6, "narrow_kernel", 5e-6), (1e9, "narrow_kernel", 1e-8),
                                   (1e16, "normal_law", 1e-10), (1e24, "normal_law", 1e-10)):
        row = beta_row(smallest, smallest)
        mean, sd = row[2], row[3]
        edge = mean + 2.0 * sd
        out = run(tails, [row], edge=[mean, edge])
        assert out.regimes[regime] == 1, smallest
        assert math.isclose(out.rate_above[0], 0.5, rel_tol=1e-9)
        z = (edge - mean) / sd
        assert math.isclose(out.rate_above[1], norm.sf(z), rel_tol=rtol), smallest
        stop_loss = sd * (norm.pdf(z) - z * norm.sf(z))
        assert math.isclose(out.excess_above[1, 0], stop_loss, rel_tol=rtol), smallest


def test_bad_edges(tails):
    with pytest.raises(Exception, match="top_node"):
        run(tails, [beta_row(2.0, 5.0)], edge=[0.5], top_node=[0.6])
    with pytest.raises(Exception, match="top_node"):
        run(tails, [beta_row(2.0, 5.0)], edge=[-1.0])


# The sweep against 60-digit arithmetic (spec test 37): the partial moments
# of orders 0 to 3 and the excess moments about a node a thousandth below
# the edge, over alpha from 1e-4 to 1e4 and beta from 1/2 to 1e7 at survival
# levels 1/2, 1e-4 and 1e-10. The engine's continued fraction carries the
# residuals spec section 5A documents: below its switch point an absolute
# error of order eps (relative eps / S), above it a relative one of order
# eps * beta (2.6e-10 at beta = 1e7, measured 2026-10-26), and near the
# mean with both parameters large a few units in 1e-12; elsewhere within
# 1e-12. The excess moments about a node this close to the edge carry the
# same residuals amplified a few times by the centred recurrence (1.5e-9 at
# beta = 1e7). Where both parameters are at least 1e3 the oracle is a quadrature
# of the density (mpmath's incomplete beta is slow there).
SWEEP_ALPHA = (1e-4, 1e-2, 0.5, 3.0, 100.0, 1e4)
SWEEP_BETA = (0.5, 2.0, 100.0, 1e5, 1e7)


@pytest.mark.parametrize("alpha", SWEEP_ALPHA)
def test_partial_moments_against_60_digits(alpha):
    mp = pytest.importorskip("mpmath")
    from scipy.special import betainccinv

    mp.mp.dps = 30
    for b0 in SWEEP_BETA:
        row = beta_row(alpha, b0)
        a, b = reference.beta_params(1.0, row[2], row[3])
        edges = [float(betainccinv(a, b, s)) for s in (0.5, 1e-4, 1e-10)]
        edges = [y for y in edges if 0.0 < y < 1.0]
        nodes = [y * (1.0 - 1e-3) for y in edges]
        out = eltq.tail_moments_arrays(*(np.array([v]) for v in row), edge=edges, top_node=nodes)
        big_a, big_b = mp.mpf(a), mp.mpf(b)
        full = [mp.rf(big_a, r) / mp.rf(big_a + big_b, r) for r in range(4)]
        if min(a, b) >= 1e3:
            ln_b = mp.log(mp.beta(big_a, big_b))
            ybar = big_a / (big_a + big_b)
            sd = mp.sqrt(ybar * (1 - ybar) / (big_a + big_b + 1))

            def partial(r, y):
                f = lambda u: u ** r * mp.exp((big_a - 1) * mp.log(u) + (big_b - 1) * mp.log1p(-u) - ln_b)
                points = sorted({mp.mpf(y), mp.mpf(1)} | {ybar + k * sd for k in range(-30, 60, 3) if y < ybar + k * sd < 1})
                return mp.quad(f, points)
        else:
            def partial(r, y):
                return full[r] * mp.betainc(big_a + r, big_b, mp.mpf(y), 1, regularized=True)

        switch = (a + 1.0) / (a + b + 2.0)
        for e, (y, c) in enumerate(zip(edges, nodes)):
            exact = [partial(r, y) for r in range(4)]
            got = [out.rate_above[0 + e], *out.moments_above[e]]
            # Near the mean with both parameters large the fraction itself is
            # good to a few units in 1e-12 (spec section 5A).
            base = 1e-11 if min(a, b) >= 1e3 else 1e-12
            for r in range(4):
                tolerance = (base + 4e-17 * b) * float(exact[r])
                if y < switch:
                    tolerance += 5e-15 * float(full[r])
                assert abs(got[r] - float(exact[r])) <= tolerance, (alpha, b0, y, r, got[r], float(exact[r]))
            chat = mp.mpf(c)
            excess = [exact[r] - chat ** r * exact[0] for r in (1, 2, 3)]
            for r in range(3):
                want = float(excess[r])
                # The centred recurrence amplifies the fraction's residual a
                # few times where the node is this close to the edge.
                assert abs(out.excess_above[e, r] - want) <= (1e-9 + 2e-16 * b) * want + 5e-15 * float(full[r + 1]), (
                    alpha, b0, y, r + 1, out.excess_above[e, r], want)


# [Eltq-Tail-Queries] Spec test 38: tail measures of shared events, both
# backends through SHARED_TAIL_IMPLS (the reference evaluates every unit at
# the event's factor nodes, the engine at each unit's own nodes, interpolated).
# The two-beta and regression oracles are the exact conditional formula on
# dense score grids, `fixtures/generate_tail_oracles.py` at 8001 by 80001
# nodes: the two betas agree with the 4001 grid to 2e-6, the regression pair
# to 2e-3 at its two shallower edges and 1e-6 at the deepest.
SHARED_TAIL_IMPLS = [("reference", rm.multiunit_tail_moments_reference), ("eltq", eltq.multiunit_tail_moments)]


@pytest.fixture(params=SHARED_TAIL_IMPLS, ids=[name for name, _ in SHARED_TAIL_IMPLS])
def shared_tails(request):
    return request.param[1]


def run_shared(shared_tails, events, **options):
    """``events``: ``(rate, [(mean_loss, sd_indep, sd_corr, max_loss), ...])``."""
    rows = [(f"e{k}", f"u{i}", lam, *unit) for k, (lam, units) in enumerate(events) for i, unit in enumerate(units)]
    event_id, unit_id, *cols = zip(*rows)
    return shared_tails(list(event_id), list(unit_id), *(np.asarray(c, dtype=np.float64) for c in cols), **options)


def measures(out, which=None):
    """``(R, T_1..3, D_1..3)`` per edge: the point values, or a bound."""
    d = (dict(rate_above=out.rate_above, moments_above=out.moments_above, excess_above=out.excess_above)
         if which is None else getattr(out, which))
    return np.column_stack([d["rate_above"], d["moments_above"], d["excess_above"]])


def assert_bracketed(out, exact, rtol):
    """``lower <= exact <= upper`` within the oracle's accuracy ``rtol``, and
    ``lower <= point <= upper``; ``exact`` has as many columns as it checks."""
    k = exact.shape[1]
    lower, upper, point = measures(out, "lower"), measures(out, "upper"), measures(out)
    assert np.all(lower[:, :k] <= exact * (1.0 + rtol))
    assert np.all(upper[:, :k] >= exact * (1.0 - rtol))
    assert np.all(lower <= point) and np.all(point <= upper)


def assert_close(got, want, rtol):
    """Elementwise within ``rtol``, each column's zeros within ``rtol`` of
    that column's largest magnitude."""
    scale = np.abs(want).max(axis=0, keepdims=True)
    assert np.all(np.abs(got - want) <= rtol * (np.abs(want) + scale)), np.abs(got - want).max()


def loading(unit):
    """Burke's loading of a unit's normal score on the factor (spec section
    4A): ``sigma_C / sqrt(sigma_I^2 + sigma_C^2)``."""
    return unit[2] / math.hypot(unit[1], unit[2])


SHARED_BETAS = [(200.0, 100.0, 150.0, 1000.0), (500.0, 300.0, 200.0, 2000.0)]
SHARED_BETA_EDGES = [1000.0, 2000.0, 2900.0]
SHARED_BETA_EXACT = np.array([
    [2.804116339474e-01, 4.421167756854e+02, 7.508615027027e+05, 1.371978102312e+09],
    [5.057087579334e-02, 1.168616204052e+02, 2.727490341697e+05, 6.431830813235e+08],
    [4.367115278975e-04, 1.281255519024e+00, 3.759287456240e+03, 1.103072729395e+07],
])
# Two copies of the second regression row of test 37 (mu 100, sigma 10,000,
# M 1e9) split evenly between independent and correlated parts.
SHARED_REGRESSION = [(100.0, 5000.0, 5000.0, 1e9)] * 2
SHARED_REGRESSION_EDGES = [4e5, 4e6, 4e7]
SHARED_REGRESSION_EXACT = np.array([
    [1.397611748017e-04, 1.352004413586e+02, 1.941053840963e+08, 4.280316724611e+14],
    [8.768874246696e-07, 4.309142585069e+00, 2.198487316946e+07, 1.177708451102e+14],
    [4.810066734313e-19, 1.991754248263e-11, 8.256993952891e-04, 3.427219970837e+04],
])


def test_shared_one_unit_events_equal_the_one_row_query(shared_tails):
    units = [(0.5, (200.0, 100.0, 150.0, 1000.0)), (0.2, (300.0, 400.0, 300.0, 1000.0)),
             (0.3, (50.0, 0.0, 0.0, 100.0))]
    edges, nodes = [40.0, 100.0, 300.0, 900.0], [20.0, 60.0, 300.0, 600.0]
    out = run_shared(shared_tails, [(lam, [u]) for lam, u in units], edge=edges, top_node=nodes, rtol=1e-3)
    rate = np.array([lam for lam, _ in units])
    cols = np.array([u for _, u in units]).T
    one = eltq.tail_moments_arrays(rate, cols[3], cols[0], cols[1], cols[2], edge=edges, top_node=nodes)
    want = measures(one)
    assert_close(measures(out), want, 1e-13)
    np.testing.assert_allclose(out.top_half_rate, one.top_half_rate, rtol=1e-13, atol=0)
    np.testing.assert_array_equal(measures(out, "lower"), measures(out, "upper"))
    assert out.regimes == {"exact_events": 3, "bounded_events": 0, "one_unit_events": 3}
    assert out.method == "exact" and out.converged is True and np.all(out.width == 0.0)


def test_shared_two_point_pair_against_the_four_outcome_law(shared_tails):
    units = [(300.0, 400.0, 300.0, 1000.0), (500.0, 600.0, 500.0, 1500.0)]
    p = [u[0] / u[3] for u in units]
    rho = loading(units[0]) * loading(units[1])
    y = [stats.norm.ppf(1.0 - q) for q in p]
    both = stats.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]]).cdf([-y[0], -y[1]])
    outcomes = [(0.0, 1.0 - p[0] - p[1] + both), (1000.0, p[0] - both), (1500.0, p[1] - both), (2500.0, both)]
    edges, nodes = [400.0, 1200.0, 2000.0], [300.0, 1000.0, 1800.0]
    exact = np.array([[math.fsum(q * s ** r for s, q in outcomes if s > t) for r in range(4)]
                      + [math.fsum(q * (s ** r - c ** r) for s, q in outcomes if s > t) for r in (1, 2, 3)]
                      for t, c in zip(edges, nodes)])
    out = run_shared(shared_tails, [(1.0, units)], edge=edges, top_node=nodes)
    assert_bracketed(out, exact, 1e-9)
    # Every outcome at these edges either passes both or neither of the lower
    # bound's events, so its tail rate is the exact rate less the margin.
    np.testing.assert_allclose(out.lower["rate_above"], exact[:, 0], rtol=2e-6)
    assert out.regimes["bounded_events"] == 1 and out.method == "union_bound"


def test_shared_two_betas_against_the_conditional_formula(shared_tails):
    nodes = [0.9 * t for t in SHARED_BETA_EDGES]
    out = run_shared(shared_tails, [(1.0, SHARED_BETAS)], edge=SHARED_BETA_EDGES, top_node=nodes)
    c = np.array(nodes)[:, None] ** np.arange(1, 4)
    excess = SHARED_BETA_EXACT[:, 1:] - c * SHARED_BETA_EXACT[:, :1]
    assert_bracketed(out, np.column_stack([SHARED_BETA_EXACT, excess]), 1e-5)


def test_shared_regression_pair_is_bracketed(shared_tails):
    out = run_shared(shared_tails, [(1.0, SHARED_REGRESSION)], edge=SHARED_REGRESSION_EDGES)
    assert_bracketed(out, SHARED_REGRESSION_EXACT, 1e-2)
    assert np.all(np.isfinite(measures(out, "upper")))


def test_shared_comonotone_events_are_exact(shared_tails):
    # S = sum M F^-1(U): P(S > t) = 1 - u* and P(S^r; S > t) = int_u*^1 S^r.
    edges = [600.0, 1500.0, 2500.0]
    out = run_shared(shared_tails, [(1.0, SHARED_BETAS)], edge=edges, coupling="comonotone", rtol=1e-3)
    laws = [(stats.beta(*reference.beta_params(u[3], u[0], u[1] + u[2])), u[3]) for u in SHARED_BETAS]

    def total(q):
        return math.fsum(m * law.ppf(q) for law, m in laws)

    for e, t in enumerate(edges):
        u_star = optimize.brentq(lambda q: total(q) - t, 1e-12, 1.0 - 1e-15, xtol=1e-15, rtol=1e-15)
        assert out.rate_above[e] == pytest.approx(1.0 - u_star, rel=1e-7)
        for r in (1, 2, 3):
            want = integrate.quad(lambda q: total(q) ** r, u_star, 1.0, epsabs=0, epsrel=1e-11, limit=200)[0]
            assert out.moments_above[e, r - 1] == pytest.approx(want, rel=1e-7)
    assert out.regimes == {"exact_events": 1, "bounded_events": 0, "one_unit_events": 0}
    assert out.method == "exact" and out.converged is True
    np.testing.assert_array_less(measures(out, "lower"), measures(out, "upper") + 1e-300)


def test_shared_atoms_and_homer_are_exact(shared_tails):
    atoms = [(50.0, 0.0, 0.0, 100.0), (70.0, 0.0, 0.0, 200.0)]
    out = run_shared(shared_tails, [(0.25, atoms)], edge=[100.0, 119.0, 120.0], top_node=[100.0, 110.0, 110.0])
    np.testing.assert_allclose(out.rate_above, [0.25, 0.25, 0.0], rtol=1e-15)
    np.testing.assert_allclose(out.moments_above[:, 1], [0.25 * 120.0 ** 2, 0.25 * 120.0 ** 2, 0.0], rtol=1e-15)
    np.testing.assert_allclose(out.top_half_rate, [0.0, 0.0, 0.25], rtol=1e-15)
    # homer rolls the event up (Homer and Li 2017, section 3.2.2): means,
    # exposures and correlated sds added, independent sds in quadrature.
    edges, nodes = [1000.0, 2000.0], [800.0, 2000.0]
    out = run_shared(shared_tails, [(0.5, SHARED_BETAS)], edge=edges, top_node=nodes, coupling="homer")
    mu, sd_i, sd_c, m = (math.fsum(u[0] for u in SHARED_BETAS), math.hypot(*(u[1] for u in SHARED_BETAS)),
                         math.fsum(u[2] for u in SHARED_BETAS), math.fsum(u[3] for u in SHARED_BETAS))
    one = eltq.tail_moments_arrays([0.5], [m], [mu], [sd_i], [sd_c], edge=edges, top_node=nodes)
    np.testing.assert_allclose(measures(out), measures(one), rtol=1e-13)
    np.testing.assert_array_equal(measures(out, "lower"), measures(out, "upper"))


def test_shared_unreachable_edges_return_zero_with_no_work(shared_tails):
    events = [(1.0, SHARED_BETAS), (0.5, [(300.0, 400.0, 300.0, 1000.0), (500.0, 600.0, 500.0, 1500.0)])]
    out = run_shared(shared_tails, events, edge=[3000.0, 5000.0])
    assert np.all(measures(out) == 0.0) and np.all(measures(out, "lower") == 0.0)
    assert np.all(measures(out, "upper") == 0.0) and np.all(out.width == 0.0)
    assert out.evaluations == 0 and out.regimes["bounded_events"] == 2
    # A reachable edge beside them does the work for the events that reach it.
    out = run_shared(shared_tails, events, edge=[2900.0, 5000.0])
    assert out.evaluations > 0 and out.upper["rate_above"][0] > 0.0 and out.upper["rate_above"][1] == 0.0


def test_shared_converged_against_rtol(shared_tails):
    out = run_shared(shared_tails, [(1.0, SHARED_BETAS)], edge=SHARED_BETA_EDGES)
    assert out.converged is None and out.rtol is None
    out = run_shared(shared_tails, [(1.0, SHARED_BETAS)], edge=SHARED_BETA_EDGES, rtol=0.5)
    assert out.converged is False and np.all(out.width > 0.5)
    assert out.coupling == "burke" and out.events == 1 and out.valid_events == 1


def test_shared_tails_refusals(shared_tails):
    events = [(1.0, SHARED_BETAS)]
    with pytest.raises(Exception, match="matched"):
        run_shared(shared_tails, events, edge=[1000.0], coupling="matched")
    for rtol in (0.0, 1.5):
        with pytest.raises(Exception, match="rtol"):
            run_shared(shared_tails, events, edge=[1000.0], rtol=rtol)
    for edge, node in (([1000.0], [1100.0]), ([-1.0], [-1.0]), ([math.nan], [0.0])):
        with pytest.raises(Exception, match="edge"):
            run_shared(shared_tails, events, edge=edge, top_node=node)
    with pytest.raises(Exception, match="top_node"):
        run_shared(shared_tails, events, edge=[1000.0, 2000.0], top_node=[500.0])


def shared_portfolio(n_events, seed):
    """Seeded events of two to five units: correlated betas at loadings over
    (0, 1), steep ones near one, two-point units, atoms and lone units."""
    rng = np.random.default_rng(seed)
    events = []
    for k in range(n_events):
        units = []
        for _ in range(int(rng.integers(1, 6))):
            m = float(rng.choice([1000.0, 5000.0, 20000.0]))
            mu = m * float(rng.uniform(0.02, 0.3))
            kind = rng.uniform()
            if kind < 0.1:
                units.append((mu, 0.0, 0.0, m))
            elif kind < 0.2:
                units.append((mu, 2.0 * mu, 2.0 * mu, m))
            else:
                sd = mu * float(rng.uniform(0.3, 1.2))
                w = float(rng.choice([rng.uniform(0.05, 0.9), rng.uniform(0.95, 0.999)]))
                units.append((mu, sd * (1.0 - w), sd * w, m))
        events.append((float(rng.uniform(0.001, 0.05)), units))
    return events


def test_shared_tail_rule_has_converged(monkeypatch):
    # The reference's outer width and transition step divided by four move
    # the bounds by far less than their margin, on the seeded portfolio's
    # events with the steepest units and the two-point units.
    events = [shared_portfolio(24, 38)[k] for k in (1, 6, 8, 12, 14, 18)]
    edges = [2000.0, 8000.0, 20000.0, 40000.0]
    nodes = [0.95 * t for t in edges]
    default = run_shared(rm.multiunit_tail_moments_reference, events, edge=edges, top_node=nodes)
    monkeypatch.setattr(rm, "_TAIL_REFINE", 4.0)
    refined = run_shared(rm.multiunit_tail_moments_reference, events, edge=edges, top_node=nodes)
    for which in ("lower", "upper"):
        assert_close(measures(default, which), measures(refined, which), 1e-8)


def test_shared_tails_backends_agree_and_are_bitwise():
    events = shared_portfolio(24, 38)
    edges = [2000.0, 8000.0, 20000.0, 40000.0]
    nodes = [0.95 * t for t in edges]
    ref = run_shared(rm.multiunit_tail_moments_reference, events, edge=edges, top_node=nodes)
    rust = run_shared(eltq.multiunit_tail_moments, events, edge=edges, top_node=nodes)
    for which in (None, "lower", "upper"):
        a, b = measures(rust, which), measures(ref, which)
        assert_close(a, b, 1e-6)
    np.testing.assert_allclose(rust.top_half_rate, ref.top_half_rate, rtol=1e-13)
    assert rust.regimes == ref.regimes and rust.valid_rows == ref.valid_rows
    # Fixed chunks folded in order: bitwise identical for any thread count.
    one = run_shared(eltq.multiunit_tail_moments, events, edge=edges, top_node=nodes, threads=1, chunk_rows=3)
    many = run_shared(eltq.multiunit_tail_moments, events, edge=edges, top_node=nodes, threads=4, chunk_rows=3)
    for which in (None, "lower", "upper"):
        np.testing.assert_array_equal(measures(one, which), measures(many, which))
    np.testing.assert_array_equal(one.width, many.width)
