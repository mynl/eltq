"""Model moments with no loss grid (spec section 4B, test 35; shared events,
test 36, from the second half of ``dev/done/plan-model-moments-and-windowed-views.md``).

Every test runs on both backends through ``MOMENT_IMPLS``: the SciPy
reference (``eltq.reference.moments_reference``, exact sums) and the package
API (the Rust engine). The 60-digit values below were computed with mpmath
1.4.1 at ``mp.dps = 60`` from the rising-factorial moment ``M^r (alpha)_r /
(alpha + beta)_r``, the parameters formed from the double inputs; they are
embedded so that the suite needs no mpmath at run time.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import eltq
from eltq import reference

from conftest import FIXTURES_DIR, diag_count

COLUMNS = ("rate", "max_loss", "mean_loss", "sd_indep", "sd_corr")

MOMENT_IMPLS = [("reference", reference.moments_reference), ("eltq", eltq.moments_arrays)]


@pytest.fixture(params=MOMENT_IMPLS, ids=[name for name, _ in MOMENT_IMPLS])
def moments(request):
    return request.param[1]


def run(moments, rows, **options):
    cols = np.asarray(rows, dtype=np.float64).reshape(-1, 5).T
    return moments(*cols, **options)


# [Eltq-Model-Moments] Spec test 35: the aggregate package's oracle rows
# (its [ELT-Probability-Oracles]): rate 0.1, mean 50, maximum 100.
@pytest.mark.parametrize("sds, law, want, zero", [
    ((10.0, 10.0), "beta", (5.0, 290.0, 18_500.0), 0.0),
    ((0.0, 0.0), "atom", (5.0, 250.0, 12_500.0), 0.0),
    ((30.0, 30.0), "two_point", (5.0, 500.0, 50_000.0), 0.05),
])
def test_oracle_rows(moments, sds, law, want, zero):
    m = run(moments, [(0.1, 100.0, 50.0, *sds)])
    assert m.raw_moments == pytest.approx(want, rel=1e-14)
    assert m.severity_moments == pytest.approx(tuple(c / 0.1 for c in want), rel=1e-14)
    assert m.analytic_aal == m.raw_moments[0]
    assert math.isclose(m.rate_at_zero, zero, abs_tol=1e-17)
    assert (m.atom_rows, m.beta_rows, m.two_point_rows) == (
        int(law == "atom"), int(law == "beta"), int(law == "two_point"))
    assert m.max_support == (50.0 if law == "atom" else 100.0)
    assert (m.method, m.achieved_error, m.converged, m.moment_overflow) == (
        "closed_form", (0.0, 0.0), True, (False, False, False))
    if law == "two_point":
        assert diag_count(m, "variance_bernoulli") == 1


def test_zero_mean_rows_count_in_the_rate_and_at_zero(moments):
    # A positive-rate row with mean zero is an atom at zero: its rate counts
    # in the total and in the zero rate, and it moves no moment.
    m = run(moments, [(0.1, 100.0, 50.0, 10.0, 10.0), (0.2, 100.0, 0.0, 5.0, 0.0)])
    assert math.isclose(m.total_rate, 0.3, rel_tol=1e-15)
    assert math.isclose(m.rate_at_zero, 0.2, rel_tol=1e-15)
    assert m.raw_moments == pytest.approx((5.0, 290.0, 18_500.0), rel=1e-14)
    # A table all at zero: a positive rate, every moment zero, every year
    # without loss.
    m = run(moments, [(0.1, 100.0, 0.0, 0.0, 0.0), (0.2, 0.0, 0.0, 0.0, 0.0)])
    assert math.isclose(m.total_rate, 0.3, rel_tol=1e-15)
    assert m.rate_at_zero == m.total_rate
    assert m.raw_moments == (0.0, 0.0, 0.0) and m.severity_moments == (0.0, 0.0, 0.0)
    assert m.max_support == 0.0
    # No contributing row at all: zero rate, NaN severity moments.
    m = run(moments, [(0.0, 100.0, 50.0, 10.0, 10.0)])
    assert m.total_rate == 0.0 and m.valid_rows == 0 and m.dropped_rows == 1
    assert all(math.isnan(c) for c in m.severity_moments)
    assert np.isnan(m.support).all()


# The aggregate package's two regression rows (spec section 4B, the tail
# table): 60-digit raw moments; P X^2 = mu^2 + sigma^2 exactly.
REGRESSION_ROWS = [
    ((20_000.0, 60_000.0, 1e9), (20_000.0, 4.0e9, 1.5197408414653655415e15)),
    ((100.0, 10_000.0, 1e9), (100.0, 1.0001e8, 1.9983018080021776226e14)),
]


@pytest.mark.parametrize("row, want", REGRESSION_ROWS)
def test_regression_rows_against_60_digits(moments, row, want):
    mu, sd, m_max = row
    m = run(moments, [(1.0, m_max, mu, sd, 0.0)])
    assert m.raw_moments == pytest.approx(want, rel=1e-14)


def test_classification_equals_the_grid_route(moments):
    # The medium fixture's rows (atoms, inadmissible rows, dropped rows) give
    # the grid route's rate, expected annual loss, valid rows and
    # diagnostics, examples included.
    df = pd.read_csv(FIXTURES_DIR / "medium_elt.csv", float_precision="round_trip")
    cols = [df[c].to_numpy(dtype=np.float64) for c in COLUMNS]
    ids = list(df["event_id"])
    m = moments(*cols, event_ids=ids)
    _, _, meta = eltq.quantize_arrays(*cols, bucket_size=1000.0, event_ids=ids)
    assert math.isclose(m.total_rate, meta.total_rate, rel_tol=1e-15)
    assert math.isclose(m.analytic_aal, meta.analytic_aal, rel_tol=1e-15)
    assert m.valid_rows == meta.valid_rows and m.input_rows == meta.input_rows
    assert m.atom_rows + m.beta_rows + m.two_point_rows == m.valid_rows
    assert m.valid_rows + m.dropped_rows == m.input_rows
    assert {k: (v.count, v.examples) for k, v in m.diagnostics.items()} == {
        k: (v.count, v.examples) for k, v in meta.diagnostics.items()}
    contributing = ~np.isnan(m.support)
    assert contributing.sum() == m.valid_rows
    assert m.max_support == np.nanmax(m.support)


def test_backends_agree_and_repeat_bitwise():
    df = pd.read_csv(FIXTURES_DIR / "medium_elt.csv", float_precision="round_trip")
    cols = [df[c].to_numpy(dtype=np.float64) for c in COLUMNS]
    ref = reference.moments_reference(*cols)
    first, second = eltq.moments_arrays(*cols), eltq.moments_arrays(*cols)
    assert first.raw_moments == second.raw_moments and first.rate_at_zero == second.rate_at_zero
    assert first.raw_moments == pytest.approx(ref.raw_moments, rel=1e-14)
    assert math.isclose(first.rate_at_zero, ref.rate_at_zero, rel_tol=1e-14)
    np.testing.assert_array_equal(first.support, ref.support)


@pytest.mark.parametrize("scale", [1e100, 1e-100])
def test_moments_scale_as_the_power(moments, scale):
    rows = [(0.1, 100.0, 40.0, 10.0, 15.0), (0.2, 50.0, 30.0, 20.0, 15.0), (0.3, 80.0, 20.0, 0.0, 0.0)]
    base = run(moments, rows)
    scaled = run(moments, [(lam, m * scale, mu * scale, si * scale, sc * scale) for lam, m, mu, si, sc in rows])
    for r in range(3):
        assert math.isclose(scaled.raw_moments[r], base.raw_moments[r] * scale ** (r + 1), rel_tol=1e-14)
    assert scaled.diagnostics.keys() == base.diagnostics.keys()
    assert scaled.moment_overflow == (False, False, False)


def test_overflow_is_flagged(moments):
    # At a monetary scale of 1e105 the third moment leaves double precision:
    # infinite and flagged, never a silent value; the lower orders stand.
    m = run(moments, [(0.1, 100e105, 40e105, 10e105, 15e105)])
    assert m.moment_overflow == (False, False, True)
    assert math.isfinite(m.raw_moments[1]) and m.raw_moments[2] == math.inf


def test_grid_moments_approach_the_model():
    # A consistency check, not the oracle (spec section 5A): as h halves, the
    # survival grid's first moment approaches C_1 at first order (the mass
    # of bucket 0 sits at zero) and its second and third at about second
    # order (the nearest rule's h^2 / 12 per unit of mass). Measured
    # 2026-10-26 on the small fixture: -1.5e-4, -9.5e-5, -4.4e-5, -2.1e-5 in
    # the mean, 1.2e-3 to 1.1e-5 and 2.3e-3 to 1.8e-5 in the higher orders,
    # at h = 4, 2, 1, 0.5.
    df = pd.read_csv(FIXTURES_DIR / "small_elt.csv", float_precision="round_trip")
    cols = [df[c].to_numpy(dtype=np.float64) for c in COLUMNS]
    model = eltq.moments_arrays(*cols)
    errors = []
    for h in (4.0, 2.0, 1.0, 0.5):
        _, secondary, _ = eltq.quantize_arrays(*cols, bucket_size=h)
        x = h * np.arange(secondary.size)
        errors.append([abs(secondary @ x ** r / model.raw_moments[r - 1] - 1.0) for r in (1, 2, 3)])
    errors = np.asarray(errors)
    assert (errors[1:, 0] < errors[:-1, 0] / 1.4).all()
    assert (errors[1:, 1:] < errors[:-1, 1:] / 3.0).all()
    assert errors[-1].max() < 3e-5
