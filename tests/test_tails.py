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

import eltq
from eltq import reference

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
