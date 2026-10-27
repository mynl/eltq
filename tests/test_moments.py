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


# ---------------------------------------------------------------------------
# [Eltq-Model-Moments] spec test 36: shared events. Both backends through
# MULTI_MOMENT_IMPLS. The oracles come from `fixtures/generate_moment_oracles.py`,
# by methods that share nothing with the score tables and rules: the
# bivariate normal for two-point pairs, a factor quadrature for a triple,
# Hoeffding's identity in score coordinates for two betas, one-dimensional
# bivariate-normal integrals for a beta beside a two-point unit, and the
# quantile integral under the comonotone coupling.

from eltq import reference_multiunit as rm  # noqa: E402

MULTI_MOMENT_IMPLS = [("reference", rm.multiunit_moments_reference), ("eltq", eltq.multiunit_moments)]


@pytest.fixture(params=MULTI_MOMENT_IMPLS, ids=[name for name, _ in MULTI_MOMENT_IMPLS])
def shared(request):
    return request.param[1]


def run_shared(shared, events, coupling="burke", **options):
    """``events``: list of ``(event_id, rate, [units])``, units as
    ``(mean, sd_indep, sd_corr, M)``."""
    rows = [(e, f"u{i}", lam, *u) for e, lam, units in events for i, u in enumerate(units)]
    cols = list(zip(*rows))
    return shared(list(cols[0]), list(cols[1]), *[np.array(c, dtype=np.float64) for c in cols[2:]],
                  coupling=coupling, **options)


TWO_POINT_UNITS = [(30.0, 50.0, 30.0, 100.0), (20.0, 40.0, 60.0, 200.0), (50.0, 100.0, 100.0, 400.0)]
BETA_UNITS = [(200.0, 100.0, 150.0, 1000.0), (500.0, 300.0, 200.0, 2000.0)]
HOMER = {
    1: ((300.0, 400.0, 300.0, 3000.0), (200.0, 300.0, 200.0, 1000.0)),
    2: ((100.0, 371.0, 267.0, 1000.0), (200.0, 150.0, 533.0, 4000.0)),
    3: ((100.0, 224.0, 200.0, 2000.0), (100.0, 200.0, 200.0, 2000.0)),
}
HOMER_RATES = {1: 0.1, 2: 0.1, 3: 0.5}
#: (P S, P S^2, P S^3) of each oracle event's total (generate_moment_oracles.py).
ORACLES = {
    "two betas": (BETA_UNITS, "burke", (700.0, 905258.6872798419, 1481501182.8264463)),
    "homer 1": (HOMER[1], "burke", (500.0, 1006943.8322303265, 2603109099.502201)),
    "homer 2": (HOMER[2], "burke", (300.0, 772782.7257790992, 2624114036.203572)),
    # Event 3's personal unit has alpha = 0.0028: its oracle is good to 1e-7 in P S^3.
    "homer 3": (HOMER[3], "burke", (200.0, 449413.60260062444, 1180735793.8039532)),
    "comonotone betas": (BETA_UNITS, "comonotone", (700.0, 1050231.3565742385, 1989712990.401895)),
}


@pytest.mark.parametrize("name", list(ORACLES))
def test_shared_moments_against_the_oracles(shared, name):
    units, coupling, want = ORACLES[name]
    m = run_shared(shared, [("e", 1.0, units)], coupling)
    assert m.raw_moments == pytest.approx(want, rel=1e-7), name
    assert m.method == "factor_integration" and m.converged
    assert max(m.achieved_error) <= m.rtol == 1e-6
    assert m.marginal_residual < 1e-6


def test_tight_tolerance_meets_hoeffding(shared):
    # At rtol 1e-8 the two betas are within 1e-10 of Hoeffding's identity
    # (2e-12 measured 2026-10-26).
    units, coupling, want = ORACLES["two betas"]
    m = run_shared(shared, [("e", 1.0, units)], coupling, rtol=1e-8)
    assert m.raw_moments == pytest.approx(want, rel=1e-10)


def test_two_point_units_are_exact(shared):
    # Two two-point units: P(X_u^a X_w^b) = M_u^a M_w^b P(V_u > y_u, V_w > y_w)
    # at correlation w_u w_w; three: the triple probability by a factor
    # quadrature (235350.2376887846 = M_1 M_2 M_3 P(all three)).
    preps = [rm.couple(rm.prepare_unit(rm.Unit(*u), "u"), "burke") for u in TWO_POINT_UNITS]
    y = [float(rm.ndtri(1.0 - p.two_point_p)) for p in preps]
    m_ = [p.max_loss for p in preps]
    both = {(i, j): float(rm.bivariate_normal_cdf(-y[i], -y[j], preps[i].w_corr * preps[j].w_corr))
            for i in range(3) for j in range(3) if i != j}
    marginal = [(p.mean_loss, p.mean_loss * p.max_loss, p.mean_loss * p.max_loss ** 2) for p in preps]
    two = run_shared(shared, [("e", 1.0, TWO_POINT_UNITS[:2])])
    want2 = (50.0, marginal[0][1] + marginal[1][1] + 2 * m_[0] * m_[1] * both[(0, 1)],
             marginal[0][2] + marginal[1][2] + 3 * (m_[0] ** 2 * m_[1] + m_[0] * m_[1] ** 2) * both[(0, 1)])
    assert two.raw_moments == pytest.approx(want2, rel=1e-12)
    three = run_shared(shared, [("e", 1.0, TWO_POINT_UNITS)])
    second = sum(mm[1] for mm in marginal) + 2 * sum(
        m_[i] * m_[j] * both[(i, j)] for i in range(3) for j in range(i + 1, 3))
    third = (sum(mm[2] for mm in marginal)
             + 3 * sum(m_[i] ** 2 * m_[j] * both[(i, j)] for i in range(3) for j in range(3) if i != j)
             + 6 * 235350.2376887846)
    assert three.raw_moments == pytest.approx((100.0, second, third), rel=1e-10)


def test_closed_form_couplings(shared):
    # One-unit events are the one-row closed forms; independent units the
    # moments of a sum of independent laws; homer the rolled row's.
    events = [("a", 0.1, [BETA_UNITS[0]]), ("b", 0.2, [HOMER[1][1]]), ("c", 0.3, [(50.0, 0.0, 0.0, 100.0)])]
    one = run_shared(shared, events)
    cols = np.array([(lam, u[3], u[0], u[1], u[2]) for _, lam, us in events for u in us]).T
    row = eltq.moments_arrays(*cols)
    assert one.raw_moments == pytest.approx(row.raw_moments, rel=1e-15)
    assert (one.method, one.one_unit_events, one.valid_events) == ("closed_form", 3, 3)
    assert math.isclose(one.rate_at_zero, row.rate_at_zero, rel_tol=1e-15)
    ind = run_shared(shared, [("e", 1.0, BETA_UNITS)], "independent")
    m = [eltq.moments_arrays(*np.array([(1.0, u[3], u[0], u[1], u[2])]).T).raw_moments for u in BETA_UNITS]
    want = (m[0][0] + m[1][0], m[0][1] + m[1][1] + 2 * m[0][0] * m[1][0],
            m[0][2] + m[1][2] + 3 * m[0][1] * m[1][0] + 3 * m[0][0] * m[1][1])
    assert ind.raw_moments == pytest.approx(want, rel=1e-14) and ind.method == "closed_form"
    homer = run_shared(shared, [("e", 1.0, HOMER[1])], "homer")
    rolled = rm.rolled_unit([rm.Unit(*u) for u in HOMER[1]])
    want = eltq.moments_arrays(*np.array([(1.0, rolled.max_loss, rolled.mean_loss, rolled.sd_indep,
                                           rolled.sd_corr)]).T).raw_moments
    assert homer.raw_moments == pytest.approx(want, rel=1e-14)
    assert homer.support[0] == rolled.max_loss


def test_unconverged_and_refused(shared):
    # A tolerance no comparison reaches, with no refinement allowed, is
    # reported; the moments stay those of the fine level.
    m = run_shared(shared, [("e", 1.0, BETA_UNITS)], rtol=1e-13, max_refinements=0)
    assert not m.converged and m.unconverged_events == 1
    assert m.unconverged[0]["event_id"] == "e"
    assert m.raw_moments == pytest.approx(ORACLES["two betas"][2], rel=1e-7)
    with pytest.raises(Exception, match="matched"):
        run_shared(shared, [("e", 1.0, BETA_UNITS)], "matched")
    with pytest.raises(Exception, match="rtol"):
        run_shared(shared, [("e", 1.0, BETA_UNITS)], rtol=2.0)


def test_shared_moments_backends_agree_and_are_bitwise():
    df = pd.read_csv(FIXTURES_DIR / "multiunit_small_elt.csv", float_precision="round_trip")
    cols = [list(df["event_id"].astype(str)), list(df["unit_id"].astype(str))] + [
        df[c].to_numpy(dtype=np.float64) for c in ("rate", "mean_loss", "sd_indep", "sd_corr", "max_loss")]
    ref = rm.multiunit_moments_reference(*cols)
    runs = [eltq.multiunit_moments(*cols, threads=k) for k in (1, 3, None)]
    assert runs[0].raw_moments == pytest.approx(ref.raw_moments, rel=1e-7)
    assert runs[0].diagnostics.keys() == ref.diagnostics.keys()
    assert (runs[0].valid_events, runs[0].one_unit_events) == (ref.valid_events, ref.one_unit_events)
    for other in runs[1:]:
        assert other.raw_moments == runs[0].raw_moments
    np.testing.assert_array_equal(runs[0].support, ref.support)


def test_views_grids_approach_the_model_up_to_the_stratified_rule():
    # The multiunit_views grids realize the stratified rule's approximation
    # of the coupling, its units independent within a stratum; their first
    # moment approaches the model's at first order in h (the mass of bucket
    # zero sits at zero), their second and third settle at the rule's own
    # limit, below the model by the within-stratum association it ignores
    # (Homer and Li's Table 11, measured 2026-10-26: -7.2e-4, -3.6e-4,
    # -1.8e-4, -9.0e-5, -4.4e-5 in the mean at h = 40, 20, 10, 5, 2.5; the
    # second moment to -5.2e-4, the third to -1.0e-3).
    rows = [(f"e{e}", HOMER_RATES[e], list(HOMER[e])) for e in HOMER]
    model = run_shared(eltq.multiunit_moments, rows)
    flat = [(e, f"u{i}", lam, *u) for e, lam, units in rows for i, u in enumerate(units)]
    cols = list(zip(*flat))
    args = (list(cols[0]), list(cols[1]), *[np.array(c, dtype=np.float64) for c in cols[2:]])
    gaps = []
    for h in (10.0, 5.0, 2.5):
        views, _ = eltq.multiunit_views(*args, bucket_size=h, views=("ground_up",))
        grid = views["ground_up"]
        x = h * np.arange(grid.size)
        gaps.append([(grid @ x ** r) / model.raw_moments[r - 1] - 1.0 for r in (1, 2, 3)])
    gaps = np.asarray(gaps)
    assert np.allclose(gaps[1:, 0] / gaps[:-1, 0], 0.5, atol=0.02)
    assert (gaps[:, 1:] < 0.0).all() and (np.abs(gaps[:, 1:]) < 1.5e-3).all()
    assert abs(gaps[-1, 1] - gaps[-2, 1]) < 3e-5 and abs(gaps[-1, 2] - gaps[-2, 2]) < 6e-5
