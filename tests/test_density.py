"""The event-loss density with no loss grid (spec section 4B, Density; test
40, from ``dev/done/plan-density-arrays.md``).

Runs on both backends through ``DENSITY_IMPLS``: the SciPy reference
(``eltq.reference.density_reference``: SciPy's beta density, by reflection
above one half, exact sums) and the package API (the Rust engine: the beta
density in a Stirling form from ``min(alpha, beta) = 100``, the narrow
kernel from 2e5, the normal law from 1e16). The comparisons near the poles
run against mpmath where it is installed (the dev group) and are skipped
otherwise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

import eltq
from eltq import reference

DENSITY_IMPLS = [("reference", reference.density_reference), ("eltq", eltq.density_arrays)]
TAIL_IMPLS = {"reference": reference.tail_moments_reference, "eltq": eltq.tail_moments_arrays}


@pytest.fixture(params=DENSITY_IMPLS, ids=[name for name, _ in DENSITY_IMPLS])
def backend(request):
    return request.param


def run(density, rows, x, **options):
    cols = np.asarray(rows, dtype=np.float64).reshape(-1, 5).T
    return density(*cols, x=np.asarray(x, dtype=np.float64), **options)


def beta_row(alpha, beta, m=1.0, rate=1.0):
    """The row ``(rate, M, mu, sigma, 0)`` whose beta has these parameters."""
    p = alpha / (alpha + beta)
    sd = math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0)))
    return (rate, m, p * m, sd * m, 0.0)


def shares(alpha, beta, ks=(-4.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0)):
    """Points about the mean in standard deviations, inside ``(0, 1)``, with
    the parameters the classification recovers from the row."""
    a, b = reference.beta_params(*beta_row(alpha, beta)[1:4])
    mean = a / (a + b)
    sd = math.sqrt(mean * (1.0 - mean) / (a + b + 1.0))
    ys = [mean + k * sd for k in ks]
    return a, b, np.array([y for y in ys if 0.0 < y < 1.0])


# [Eltq-Density-Arrays] Test 40, clause 1: against SciPy's density on beta
# rows from alpha, beta of 1e-3 to the narrow threshold 2e5, below one
# included, relative 1e-12 away from the poles (both backends are within 5e-13
# of 50-digit arithmetic there, measured 2026-11-06).
RNG_SHAPES = [(float(a), float(b)) for a, b in np.exp(
    np.random.default_rng(20261106).uniform(math.log(1e-3), math.log(1.9e5), size=(40, 2)))]
FIXED_SHAPES = [(0.3, 7.0), (0.5, 0.5), (2.5, 2.5), (1e-3, 900.0), (40.0, 0.7), (5.0, 1e6), (99.9, 99.9),
                (100.0, 100.0), (1e3, 1e3), (1e4, 3e4), (1e5, 1e5), (1.99e5, 6e5)]


@pytest.mark.parametrize("alpha, beta", FIXED_SHAPES + RNG_SHAPES)
def test_against_scipy(backend, alpha, beta):
    _, density = backend
    a, b, ys = shares(alpha, beta)
    got = run(density, [beta_row(alpha, beta, rate=0.25)], ys)
    want = 0.25 * stats.beta.pdf(ys, a, b)
    assert np.all(want > 0.0)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


# Clause 1, at a monetary scale: M = 1e6 and the loss x = M y give f(y) / M,
# within the rounding of x / M (the density's conditioning in y).
@pytest.mark.parametrize("alpha, beta", [(0.3, 7.0), (2.5, 2.5), (40.0, 0.7), (1e3, 1e3)])
def test_monetary_scale(backend, alpha, beta):
    _, density = backend
    m = 1e6
    a, b, ys = shares(alpha, beta)
    got = run(density, [beta_row(alpha, beta, m=m)], m * ys)
    np.testing.assert_allclose(got, stats.beta.pdf(ys, a, b) / m, rtol=1e-11, atol=0.0)


# Near the poles, where differencing the exceedance curve failed: alpha < 1
# at shares down to 1e-200 and beta < 1 within 2^-40 and one unit in the last
# place of M, against 50-digit arithmetic at the exact share of the loss.
@pytest.mark.parametrize("alpha, beta", [(0.2, 3.0), (3.0, 0.3), (0.5, 0.5), (1e-3, 0.9)])
def test_poles(backend, alpha, beta):
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    _, density = backend
    m = 1e6
    a, b = reference.beta_params(*beta_row(alpha, beta, m=m)[1:4])
    xs = []
    if a < 1.0:
        xs += [m * 1e-200, m * 1e-30, m * 1e-8]
    if b < 1.0:
        xs += [m - m * 2.0 ** -40, m - m * 2.0 ** -30, math.nextafter(m, 0.0)]
    got = run(density, [beta_row(alpha, beta, m=m)], xs)
    for x, v in zip(xs, got):
        y = mp.mpf(x) / m
        want = float(mp.exp((a - 1) * mp.log(y) + (b - 1) * mp.log1p(-y) - mp.log(mp.beta(a, b))) / m)
        assert math.isfinite(v) and abs(v / want - 1.0) < 1e-12, (x, v, want)


# Clause 2: the density integrates to the tail rate, int_a^b nu_c = R(a) -
# R(b) - (rate of the atoms in (a, b]), by adaptive quadrature split at every
# endpoint inside (a, b), relative 1e-9; on a table with poles at both ends,
# atoms, Bernoulli rows and a narrow row.
MIXED = [
    beta_row(0.4, 6.0, m=100.0, rate=0.2),
    beta_row(3.0, 0.5, m=60.0, rate=0.1),
    beta_row(2.0, 5.0, m=150.0, rate=0.05),
    beta_row(8.0, 2.0, m=80.0, rate=0.3),
    beta_row(3e5, 6e5, m=90.0, rate=0.01),
    (0.4, 100.0, 30.0, 0.0, 0.0),        # an atom at 30
    (0.2, 50.0, 20.0, 40.0, 0.0),        # a Bernoulli row: atoms at 0 and 50
]


@pytest.mark.parametrize("lo, hi", [(0.0, 200.0), (10.0, 55.0), (25.0, 61.0), (59.0, 100.0), (5.0, 35.0)])
def test_integrates_to_the_tail_rate(backend, lo, hi):
    name, density = backend
    rows = np.asarray(MIXED, dtype=np.float64)
    tails = TAIL_IMPLS[name](*rows.T, edge=np.array([lo, hi]))
    atoms = sum(rate for rate, m, mu, s, _ in MIXED[5:6] if lo < mu <= hi)
    atoms += sum(rate * mu / m for rate, m, mu, s, _ in MIXED[6:] if lo < m <= hi)
    want = tails.rate_above[0] - tails.rate_above[1] - atoms
    # Split at every endpoint and every mean inside (a, b), so that the poles
    # sit at subinterval ends and the narrow row's spike is found.
    breaks = sorted({v for r in MIXED[:5] for v in (r[1], r[2]) if lo < v < hi} | {lo, hi})
    total = 0.0
    for left, right in zip(breaks, breaks[1:]):
        value, _ = integrate.quad(lambda v: float(run(density, MIXED, [v])[0]), left, right,
                                  epsabs=0.0, epsrel=1e-11, limit=500)
        total += value
    assert abs(total - want) <= 1e-9 * want, (total, want)


# Clause 3: atoms and Bernoulli rows contribute exactly zero, a table of them
# alone is zero everywhere, and appending them leaves the betas' density
# bitwise unchanged.
def test_atoms_contribute_nothing(backend):
    _, density = backend
    atoms = [MIXED[5], MIXED[6], (0.3, 10.0, 0.0, 1.0, 0.0), (0.1, 10.0, 10.0, 1.0, 0.0)]
    x = np.linspace(-5.0, 160.0, 67)
    assert np.all(run(density, atoms, x) == 0.0)
    betas = run(density, MIXED[:5], x)
    assert np.array_equal(run(density, MIXED[:5] + atoms, x), betas)
    assert np.any(betas > 0.0)


# Clause 4: the regimes at their thresholds. Either side of min(alpha, beta)
# = 2e5 (the continued fraction's beta and the narrow kernel) against SciPy's
# beta density within 1e-11; from 1e16 the normal density within 1e-12, and
# just below it the narrow kernel within the beta's skewness of it, 1e-6.
@pytest.mark.parametrize("alpha, beta", [(2e5 * (1 - 1e-9), 6e5), (2e5 * (1 + 1e-9), 6e5),
                                         (2e5 * (1 + 1e-9), 2e5 * (1 + 1e-9))])
def test_narrow_threshold(backend, alpha, beta):
    _, density = backend
    a, b, ys = shares(alpha, beta)
    np.testing.assert_allclose(run(density, [beta_row(alpha, beta)], ys), stats.beta.pdf(ys, a, b),
                               rtol=1e-11, atol=0.0)


@pytest.mark.parametrize("scale, rtol", [(1.0 + 1e-6, 1e-12), (1.0 - 1e-6, 1e-6)])
def test_normal_threshold(backend, scale, rtol):
    _, density = backend
    a, b, ys = shares(1e16 * scale, 3e16 * scale)
    mean = a / (a + b)
    sd = math.sqrt(mean * (1.0 - mean) / (a + b + 1.0))
    np.testing.assert_allclose(run(density, [beta_row(1e16 * scale, 3e16 * scale)], ys),
                               stats.norm.pdf(ys, mean, sd), rtol=rtol, atol=0.0)


def seeded_table(rows, seed=7, min_cv=1e-6):
    """A seeded one-row table across the regimes, with atoms and Bernoulli
    rows: coefficients of variation from ``min_cv`` (1e-6 is narrow, to
    alpha + beta of about 1e13; from 3e-3 no row reaches 2e5) to 3
    (Bernoulli)."""
    rng = np.random.default_rng(seed)
    m = np.exp(rng.uniform(math.log(1e3), math.log(1e6), size=rows))
    p = rng.uniform(0.01, 0.9, size=rows)
    cv = np.exp(rng.uniform(math.log(min_cv), math.log(3.0), size=rows))
    sigma = cv * p * m
    sigma[rng.random(rows) < 0.05] = 0.0
    split = rng.uniform(0.0, 1.0, size=rows)
    rate = np.exp(rng.uniform(math.log(1e-5), math.log(1e-1), size=rows))
    return np.column_stack([rate, m, p * m, split * sigma, (1.0 - split) * sigma])


# Clause 5: bitwise identical results for threads 1, 2 and all cores on a
# table of several chunks; the two backends agree wherever the value is at
# least 1e-12 of the largest (the narrow kernel is zero outside its bulk,
# where SciPy's density is below the floor): within 1e-11 relative with no
# narrow row, and within the rounding of the share the narrow kernel is
# centred on, about eps sqrt(alpha + beta) per standard score, with narrow
# rows to alpha + beta of 1e13 (7.5e-10 measured 2026-11-06).
def test_bitwise_across_threads():
    table = seeded_table(20_000)
    x = np.geomspace(1e2, 1.2e6, 64)
    every = eltq.density_arrays(*table.T, x=x)
    assert np.any(every > 0.0)
    for threads in (1, 2):
        assert np.array_equal(eltq.density_arrays(*table.T, x=x, threads=threads), every)


@pytest.mark.parametrize("min_cv, rtol", [(3e-3, 1e-11), (1e-6, 1e-9)])
def test_backends_agree(min_cv, rtol):
    table = seeded_table(300, min_cv=min_cv)
    x = np.concatenate([np.geomspace(1e1, 1.2e6, 200), table[:40, 2]])
    engine = eltq.density_arrays(*table.T, x=x)
    oracle = reference.density_reference(*table.T, x=x)
    floor = 1e-12 * oracle.max()
    big = oracle >= floor
    np.testing.assert_allclose(engine[big], oracle[big], rtol=rtol, atol=0.0)
    assert np.all(np.abs(engine[~big] - oracle[~big]) <= floor)


# Clause 6: below zero and from the largest exposed value up the density is
# zero; any order of the points is kept; a non-finite point raises.
def test_outside_support_and_bad_points(backend):
    _, density = backend
    x = np.array([120.0, -1.0, 0.0, 1e300, 150.0, 75.0, -1e300, 200.0])
    got = run(density, MIXED, x)
    assert np.all(got[[1, 2, 3, 4, 6, 7]] == 0.0)
    assert got[0] > 0.0 and got[5] > 0.0
    np.testing.assert_array_equal(got[[5, 0]], run(density, MIXED, [75.0, 120.0]))
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(eltq.EltqError if density is eltq.density_arrays else reference.EltqError,
                           match="density point"):
            run(density, MIXED, [10.0, bad])


# The classification is the grid route's: an inadmissible row is no density
# under bernoulli, atom and drop alike, and error raises the variance error.
def test_variance_policies(backend):
    _, density = backend
    row = (0.2, 50.0, 20.0, 40.0, 0.0)
    for policy in ("bernoulli", "atom", "drop"):
        assert np.all(run(density, [row, MIXED[0]], [5.0, 30.0], variance_policy=policy)
                      == run(density, [MIXED[0]], [5.0, 30.0]))
    error = eltq.VarianceError if density is eltq.density_arrays else reference.VarianceError
    with pytest.raises(error):
        run(density, [row], [5.0], variance_policy="error")
