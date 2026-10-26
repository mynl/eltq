"""Acceptance suite: the fourteen mathematical tests of specification section 9.

Each test runs against every available implementation (SciPy reference now,
Rust ``eltq`` from 1.2.0) through the unified ``Impl`` interface in conftest.
Supplementary contract tests (validation, policies, modes) live in
``test_contract.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eltq import reference

from conftest import diag_count, elt, grid_of, load_fixture, load_golden, run


# [1] Zero uncertainty: secondary equals raw when every sigma is zero.
def test_01_zero_uncertainty(impl):
    rows = [(0.02, 100.0, 40.0, 0.0, 0.0),
            (0.01, 80.0, 17.3, 0.0, 0.0),
            (0.05, 60.0, 60.0, 0.0, 0.0),
            (0.03, 50.0, 0.0, 0.0, 0.0)]
    raw, secondary, meta = run(impl, elt(rows), bucket_size=0.5)
    assert np.array_equal(raw, secondary)
    assert diag_count(meta, "degenerate_atom") == 4


# [2] Exact grid points: a loss at node i h contributes all weight to element i.
def test_02_exact_grid_points(impl):
    h = 0.25  # dyadic so mu / h is exactly representable
    mus = [0.0, 0.25, 1.0, 2.5, 10.0]
    rates = [0.01, 0.02, 0.03, 0.04, 0.05]
    rows = [(lam, 10.0, mu, 0.0, 0.0) for lam, mu in zip(rates, mus)]
    raw, secondary, meta = run(impl, elt(rows), bucket_size=h)
    expected = np.zeros(meta.grid_length)
    for lam, mu in zip(rates, mus):
        expected[int(round(mu / h))] += lam
    assert np.array_equal(raw, expected)
    assert np.array_equal(secondary, expected)


# [3] Linear interpolation: off-grid atom preserves weight and first moment.
def test_03_linear_rule_mass_and_moment(impl):
    lam, x = 0.037, 3.3567
    raw, secondary, meta = run(impl, elt([(lam, 10.0, x, 0.0, 0.0)]),
                               bucket_size=1.0, grid_length=12)
    i = int(x)  # h = 1
    assert np.count_nonzero(raw) == 2 and raw[i] > 0 and raw[i + 1] > 0
    assert math.isclose(raw.sum(), lam, rel_tol=1e-15)
    assert math.isclose(float(raw @ grid_of(meta)), lam * x, rel_tol=1e-13)


# Tests 4, 5, 7 and 9 are tests of the quantiles method and select it; the
# survival method (the default from 2.0.0) is tested in test_survival.py.
QUANTILES = {"method": "quantiles"}


# [4] Uniform beta (alpha = beta = 1): midpoint points x_k = M (k + 1/2) / K.
def test_04_uniform_beta(impl):
    lam, m, k_pts = 0.008, 96.0, 8
    rows = [(lam, m, m / 2.0, m / math.sqrt(12.0), 0.0)]
    raw, secondary, meta = run(impl, elt(rows), bucket_size=1.0,
                               quadrature_points=k_pts, **QUANTILES)
    expected = np.zeros(meta.grid_length)
    for k in range(k_pts):
        expected[int(m * (k + 0.5) / k_pts)] += lam / k_pts  # nodes 6, 18, ...
    assert np.allclose(secondary, expected, rtol=0.0, atol=1e-9 * lam)


# [5] Symmetry: swapping alpha and beta reflects quantiles about 1/2.
def test_05_symmetry(impl):
    m, h, n = 10.0, 0.5, 21
    sigma = math.sqrt(0.3 * 3.0 * (m - 3.0))  # admissible for mu = 3 and mu = 7
    _, sec_lo, _ = run(impl, elt([(0.01, m, 3.0, sigma, 0.0)]),
                       bucket_size=h, grid_length=n, **QUANTILES)
    _, sec_hi, _ = run(impl, elt([(0.01, m, 7.0, sigma, 0.0)]),
                       bucket_size=h, grid_length=n, **QUANTILES)
    assert np.allclose(sec_lo, sec_hi[::-1], rtol=0.0, atol=1e-12)


# [6] Analytical moments: reconstructed beta parameters reproduce mean and
# variance. Unit-level test of the parameterization (Rust twin in eltq-core).
@pytest.mark.parametrize("m,mu,frac", [
    (100.0, 40.0, 0.3), (100.0, 2.0, 0.7), (100.0, 99.0, 0.5),
    (1e6, 3e5, 0.05), (0.001, 0.0004, 0.9), (50.0, 25.0, 0.999),
])
def test_06_beta_parameterization_moments(m, mu, frac):
    var = frac * mu * (m - mu)
    a, b = reference.beta_params(m, mu, math.sqrt(var))
    assert a > 0 and b > 0
    mean_back = m * a / (a + b)
    var_back = m * m * a * b / ((a + b) ** 2 * (a + b + 1.0))
    assert math.isclose(mean_back, mu, rel_tol=1e-12)
    assert math.isclose(var_back, var, rel_tol=1e-11)


# [7] Quadrature convergence: increasing K reduces the midpoint mean error.
@pytest.mark.parametrize("mu,frac", [(5.0, 0.7), (30.0, 0.4)])
def test_07_k_convergence(impl, mu, frac):
    m = 100.0
    sigma = math.sqrt(frac * mu * (m - mu))
    errors = {}
    for k_pts in (8, 32, 512):
        _, _, meta = run(impl, elt([(0.01, m, mu, sigma, 0.0)]),
                         bucket_size=0.01, quadrature_points=k_pts, **QUANTILES)
        errors[k_pts] = abs(meta.secondary_mean_error)
    floor = 1e-13 * abs(meta.analytic_aal)
    assert errors[512] <= max(errors[8] / 4.0, floor)
    assert errors[512] <= max(errors[32], floor)


# [8] Mass conservation: both arrays sum to the total contributing rate.
def test_08_mass_conservation(impl):
    df = load_fixture("medium")
    raw, secondary, meta = run(impl, df, bucket_size=5000.0)
    assert math.isclose(raw.sum(), meta.total_rate, rel_tol=1e-12)
    assert math.isclose(secondary.sum(), meta.total_rate, rel_tol=1e-12)


# [9] Mean conservation: raw first moment equals the analytic AAL; secondary
# first moment equals the quadrature first moment.
def test_09_mean_conservation(impl):
    df = load_fixture("medium")
    raw, secondary, meta = run(impl, df, bucket_size=5000.0, **QUANTILES)
    g = grid_of(meta)
    assert math.isclose(float(raw @ g), meta.analytic_aal, rel_tol=1e-10)
    assert math.isclose(float(secondary @ g), meta.quadrature_secondary_mean,
                        rel_tol=1e-10)


# [10] Scale invariance: scaling all loss quantities and h leaves weights
# unchanged (bitwise for a dyadic factor under the quantiles method; the
# survival method's geometric nodes are not dyadic in the loss, so there the
# invariance holds to rounding). At factors 1e160 and 1e-200 the squared
# standard deviation and mu (M - mu) overflow and underflow while every
# field stays finite; the admissibility test and the shape parameters are
# formed in ratios, so the classification and the grids match the unit
# scale's (peer review, finding I3; until 2.4.0 the admissible row
# (M, mu, sigma) = (1, 0.4, 0.1) was a Bernoulli row at 1e160 and an atom at
# 1e-200).
def test_10_scale_invariance(impl):
    df = load_fixture("small")
    for method, exact_dyadic in (("quantiles", True), ("survival", False)):
        base_raw, base_sec, base_meta = run(impl, df, bucket_size=0.5, quadrature_points=8,
                                            method=method)
        for c, exact in ((64.0, exact_dyadic), (3.0, False), (1e160, False), (1e-200, False)):
            scaled = df.copy()
            for col in ("max_loss", "mean_loss", "sd_indep", "sd_corr"):
                scaled[col] *= c
            raw, sec, meta = run(impl, scaled, bucket_size=0.5 * c, quadrature_points=8,
                                 method=method)
            assert meta.valid_rows == base_meta.valid_rows
            assert {k: v.count for k, v in meta.diagnostics.items()} == \
                {k: v.count for k, v in base_meta.diagnostics.items()}
            if exact:
                assert np.array_equal(raw, base_raw)
                assert np.array_equal(sec, base_sec)
            else:
                assert np.allclose(raw, base_raw, rtol=1e-9, atol=1e-15)
                assert np.allclose(sec, base_sec, rtol=1e-9, atol=1e-12)
    # The review's row at the three scales, both methods: a beta each time.
    for method in ("quantiles", "survival"):
        for c in (1.0, 1e160, 1e-200):
            _, sec, meta = run(impl, elt([(1.0, c, 0.4 * c, 0.1 * c, 0.0)]),
                               bucket_size=0.1 * c, grid_length=11, method=method)
            assert meta.valid_rows == 1 and not meta.diagnostics, (method, c)
            assert sec[0] < 1e-3 and sec[10] < 1e-3 and sec.sum() == pytest.approx(1.0, rel=1e-14)


# [11] Invalid beta moments follow the selected policy and appear in
# diagnostics. sigma^2 = 2500 > mu (M - mu) = 2100 for the bad row.
BAD = (0.005, 100.0, 30.0, 50.0, 0.0)
GOOD = (0.010, 100.0, 40.0, 15.0, 10.0)


def test_11_variance_policy_bernoulli(impl):
    # Default policy: the bad row becomes rate (1 - p) at 0 and rate p at M,
    # p = 0.3, with its rate and mean exact.
    raw, secondary, meta = run(impl, elt([BAD, GOOD]), bucket_size=0.5)
    assert diag_count(meta, "variance_bernoulli") == 1
    assert meta.valid_rows == 2
    assert math.isclose(secondary.sum(), 0.015, rel_tol=1e-12)
    # The pair's residual mean error is the good row's K-point quadrature.
    assert math.isclose(float(secondary @ grid_of(meta)), meta.analytic_aal,
                        rel_tol=5e-2)
    _, bad_only, bad_meta = run(impl, elt([BAD]), bucket_size=0.5)
    assert bad_only[0] == BAD[0] * 0.7
    assert bad_only[200] == BAD[0] * 0.3
    assert np.count_nonzero(bad_only) == 2
    assert bad_meta.secondary_mean_error == 0.0


def test_11_variance_policy_clamp_removed(impl):
    # The clamp of versions before 2.0.0 is refused on the one-row route.
    with pytest.raises(impl.base_error):
        run(impl, elt([BAD, GOOD]), bucket_size=0.5, variance_policy="clamp")


def test_11_variance_policy_error(impl):
    with pytest.raises(impl.variance_error):
        run(impl, elt([BAD, GOOD]), bucket_size=0.5, variance_policy="error")


def test_11_variance_policy_drop(impl):
    raw, secondary, meta = run(impl, elt([BAD, GOOD]), bucket_size=0.5,
                               variance_policy="drop")
    assert diag_count(meta, "variance_dropped") == 1
    assert meta.valid_rows == 1
    assert math.isclose(raw.sum(), GOOD[0], rel_tol=1e-12)
    assert math.isclose(meta.total_rate, GOOD[0], rel_tol=1e-15)


def test_11_variance_policy_atom(impl):
    raw, secondary, meta = run(impl, elt([BAD, GOOD]), bucket_size=0.5,
                               variance_policy="atom")
    assert diag_count(meta, "variance_atom") == 1
    assert meta.valid_rows == 2
    assert math.isclose(secondary.sum(), 0.015, rel_tol=1e-12)


# [12] Boundary means: mu = 0 and mu = M are degenerate atoms even with
# sigma > 0.
def test_12_boundary_means(impl):
    m, h = 80.0, 0.5
    rows = [(0.02, m, 0.0, 5.0, 0.0), (0.03, m, m, 4.0, 3.0)]
    raw, secondary, meta = run(impl, elt(rows), bucket_size=h)
    assert diag_count(meta, "degenerate_atom") == 2
    top = int(round(m / h))
    assert secondary[0] == 0.02 and secondary[top] == 0.03
    assert np.array_equal(raw, secondary)


# [13] Grid overflow: an insufficient grid is a hard, typed error under the
# error policy (the named policies are test 18).
def test_13_grid_overflow(impl):
    rows = [(0.01, 100.0, 40.0, 15.0, 0.0)]
    with pytest.raises(impl.overflow_error):
        run(impl, elt(rows), bucket_size=1.0, grid_length=50, overflow="error")
    # Auto-derived length covers max M exactly: n = ceil(M / h) + 1.
    _, _, meta = run(impl, elt(rows), bucket_size=1.0)
    assert meta.grid_length == 101


# [14] Golden-fixture agreement with the independent reference, under the
# quantiles method (small, medium) and the survival method (the rest).
@pytest.mark.parametrize("name", ["small", "medium", "small_survival", "medium_survival"])
def test_14_golden_fixtures(impl, name):
    g_raw, g_sec, g_meta = load_golden(name)
    df = load_fixture(g_meta.get("fixture", name))
    raw, secondary, meta = run(impl, df, **g_meta["params"])
    scale = g_meta["total_rate"]
    assert meta.grid_length == g_meta["grid_length"]
    assert np.allclose(raw, g_raw, rtol=1e-9, atol=1e-13 * scale)
    assert np.allclose(secondary, g_sec, rtol=1e-9, atol=1e-13 * scale)
    assert math.isclose(meta.total_rate, g_meta["total_rate"], rel_tol=1e-12)
    assert math.isclose(meta.analytic_aal, g_meta["analytic_aal"], rel_tol=1e-12)
    if g_meta["quadrature_secondary_mean"] is not None:
        assert math.isclose(meta.quadrature_secondary_mean,
                            g_meta["quadrature_secondary_mean"], rel_tol=1e-9)
    else:
        assert math.isclose(float(secondary @ grid_of(meta)) - meta.analytic_aal,
                            g_meta["secondary_mean_error"], rel_tol=1e-6,
                            abs_tol=1e-12 * meta.analytic_aal)
    for category, issues in g_meta["diagnostics"].items():
        assert diag_count(meta, category) == issues["count"], category
    assert meta.valid_rows == g_meta["valid_rows"]
