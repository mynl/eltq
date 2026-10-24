"""Spec tests 15 to 20: the survival method, the overflow policies and the
Bernoulli treatment of inadmissible rows (`dev/done/plan-single-unit-survival-route.md`).

Every test runs on both backends through the ``impl`` fixture in conftest;
the oracle checks (``exact_edges``, ``check_every_interval``) call the
reference directly. The fixture is ``fixtures/small_alpha_elt.csv``: 2,000
rows with alpha from 0.003 to 3 and M / mu from 1e2 to 1e5, ground-up and
gross-like, with a few atoms, inadmissible and zero-rate rows.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eltq import reference

from conftest import diag_count, elt, grid_of, load_fixture, run

# The fixture's h: about 1e-4 of a typical mean, with max M = 1e6 (n = 1e6
# nodes).
FIXTURE_H = 1.0

# One inadmissible row: sigma^2 = 2500 >= mu (M - mu) = 2100, p = 0.3.
BAD = (0.005, 100.0, 30.0, 50.0, 0.0)


def columns(df):
    return tuple(df[c].to_numpy(dtype=np.float64)
                 for c in ("rate", "max_loss", "mean_loss", "sd_indep", "sd_corr"))


def tail_sums(weights):
    """Mass above each bucket edge e_k = (k + 1/2) h, k = 0 .. n - 2."""
    return np.cumsum(weights[::-1])[::-1][1:]


# [15] Exceedance accuracy: the survival grid's tail sums match the exact
# incomplete beta at every bucket edge; the interpolation error is small.
def test_15_exceedance_accuracy(impl):
    df = load_fixture("small_alpha")
    h = 500.0  # 2,001 edges: the exact oracle at every edge is affordable
    _, sec, meta = run(impl, df, bucket_size=h, method="survival")
    _, sec_exact, meta_exact = reference.quantize_reference(
        *columns(df), bucket_size=h, method="survival", exact_edges=True,
        check_every_interval=True)
    assert meta.method == "survival" and meta.nodes > 100
    assert meta.quadrature_points is None and meta.quadrature_secondary_mean is None
    got, want = tail_sums(sec), tail_sums(sec_exact)
    keep = want > 1e-12 * meta.total_rate
    assert np.max(np.abs(got[keep] - want[keep]) / want[keep]) <= 1e-4
    assert meta_exact.interpolation_error <= 1e-4   # every interval midpoint
    assert meta.interpolation_error <= 1e-4         # the 64-point sample


# [16] Rate and mean: the mass is the total rate; the mean error is small
# and falls at least linearly in h (first order from bucket 0 at alpha < 1).
def test_16_rate_and_mean(impl):
    df = load_fixture("small_alpha")
    errors = {}
    for h in (FIXTURE_H, FIXTURE_H / 2.0, FIXTURE_H / 4.0):
        _, sec, meta = run(impl, df, bucket_size=h, method="survival")
        assert math.isclose(sec.sum(), meta.total_rate, rel_tol=1e-14)
        assert meta.rate_above_grid == {"raw": 0.0, "secondary": 0.0}
        errors[h] = abs(meta.secondary_mean_error) / meta.analytic_aal
    assert errors[FIXTURE_H] <= 1e-4
    assert errors[FIXTURE_H / 2.0] <= 0.55 * errors[FIXTURE_H]
    assert errors[FIXTURE_H / 4.0] <= 0.55 * errors[FIXTURE_H / 2.0]


# [17] Nonnegativity and determinism: no negative weight; bitwise identical
# across thread counts at a fixed chunk size; the backends agree.
def test_17_nonnegative_and_deterministic(impl):
    df = load_fixture("small_alpha")
    h = 50.0
    # The reference is single-threaded and takes no thread keywords.
    parallel = {"chunk_rows": 256} if impl.name == "eltq" else {}
    raw, sec, meta = run(impl, df, bucket_size=h, method="survival", **parallel)
    assert (sec >= 0.0).all() and np.isfinite(sec).all()
    for threads in (1, 2, 3, 7) if impl.name == "eltq" else ():
        raw_t, sec_t, meta_t = run(impl, df, bucket_size=h, method="survival",
                                   chunk_rows=256, threads=threads)
        assert np.array_equal(sec_t, sec) and np.array_equal(raw_t, raw)
        assert meta_t.secondary_mean_error == meta.secondary_mean_error
    ref_raw, ref_sec, ref_meta = reference.quantize_reference(
        *columns(df), bucket_size=h, method="survival")
    assert np.allclose(sec, ref_sec, rtol=1e-9, atol=1e-12 * meta.total_rate)
    # The raw grids differ only by the chunked fold's summation order.
    assert np.allclose(raw, ref_raw, rtol=1e-12, atol=0.0)
    got, want = tail_sums(sec), tail_sums(ref_sec)
    keep = want > 1e-12 * meta.total_rate
    assert np.max(np.abs(got[keep] - want[keep]) / want[keep]) <= 1e-12


# [18] Overflow: with a window short of max M, the named policy applies to
# both methods and both outputs and reports the overflow.
WINDOW_ROWS = [
    (0.010, 100.0, 40.0, 15.0, 10.0),   # beta reaching M = 100
    (0.020, 100.0, 90.0, 0.0, 0.0),     # atom at 90, above the window
    (0.005, 100.0, 30.0, 50.0, 0.0),    # Bernoulli: atoms at 0 and at 100
    (0.030, 20.0, 5.0, 2.0, 1.0),       # beta wholly inside the window
]


@pytest.mark.parametrize("method", ["quantiles", "survival"])
def test_18_overflow_policies(impl, method):
    h, n = 1.0, 50                      # top node 49, top edge 49.5
    rows = elt(WINDOW_ROWS)
    total = sum(r[0] for r in WINDOW_ROWS)
    aal = sum(r[0] * r[2] for r in WINDOW_ROWS)
    # Atoms above the window: the raw atoms at 90 and the Bernoulli's 100.
    raw_over = 0.020
    raw_moment = 0.020 * (90.0 - 49.0)

    raw_top, sec_top, meta_top = run(impl, rows, bucket_size=h, grid_length=n,
                                     method=method)
    assert meta_top.overflow == "top_bucket"
    assert math.isclose(raw_top.sum(), total, rel_tol=1e-14)
    assert math.isclose(sec_top.sum(), total, rel_tol=1e-12)
    assert math.isclose(meta_top.rate_above_grid["raw"], raw_over, rel_tol=1e-14)
    assert math.isclose(meta_top.mean_above_grid["raw"], raw_moment, rel_tol=1e-14)
    assert meta_top.rate_above_grid["secondary"] > 0.020 + 0.005 * 0.3
    assert meta_top.mean_above_grid["secondary"] > raw_moment
    # The moment lost is exactly the reported displacement.
    g = grid_of(meta_top)
    assert math.isclose(aal - float(raw_top @ g), raw_moment, rel_tol=1e-12)
    # Against the full grid, the window loses the reported secondary moment
    # (to the bucketing error of the tail under survival).
    _, sec_full, meta_full_grid = run(impl, rows, bucket_size=h, method=method)
    lost = float(sec_full @ grid_of(meta_full_grid)) - float(sec_top @ g)
    assert math.isclose(lost, meta_top.mean_above_grid["secondary"],
                        rel_tol=1e-3 if method == "survival" else 1e-12)

    raw_drop, sec_drop, meta_drop = run(impl, rows, bucket_size=h, grid_length=n,
                                        method=method, overflow="drop")
    assert meta_drop.overflow == "drop"
    assert math.isclose(raw_drop.sum(), total - raw_over, rel_tol=1e-14)
    assert math.isclose(sec_drop.sum(), total - meta_drop.rate_above_grid["secondary"],
                        rel_tol=1e-12)
    assert meta_drop.rate_above_grid == meta_top.rate_above_grid
    assert meta_drop.mean_above_grid == meta_top.mean_above_grid
    assert np.array_equal(raw_drop[:-1], raw_top[:-1])
    assert np.array_equal(sec_drop[:-1], sec_top[:-1])
    assert raw_top[-1] == raw_drop[-1] + raw_over

    with pytest.raises(impl.overflow_error):
        run(impl, rows, bucket_size=h, grid_length=n, method=method,
            overflow="error")
    # A window that still reaches every M is no overflow.
    _, _, meta_full = run(impl, rows, bucket_size=h, grid_length=101,
                          method=method, overflow="error")
    assert meta_full.rate_above_grid == {"raw": 0.0, "secondary": 0.0}
    assert meta_full.mean_above_grid == {"raw": 0.0, "secondary": 0.0}
    _, _, meta_auto = run(impl, rows, bucket_size=h, method=method)
    assert meta_auto.rate_above_grid == {"raw": 0.0, "secondary": 0.0}


# [19] The K-point cut: at K = 32 the quantiles method has no mass above each
# event's 1 - 1/64 quantile (plus one bucket of linear spread) and loses
# more than 5% of the expected annual loss on the fixture; survival does not.
def test_19_k_point_cut(impl):
    df = load_fixture("small_alpha")
    _, sec_q, meta_q = run(impl, df, bucket_size=FIXTURE_H, method="quantiles",
                           quadrature_points=32)
    assert meta_q.secondary_mean_error / meta_q.analytic_aal < -0.05
    _, _, meta_s = run(impl, df, bucket_size=FIXTURE_H, method="survival")
    assert abs(meta_s.secondary_mean_error) / meta_s.analytic_aal <= 1e-4
    # Single small-alpha events: nothing above the 1 - 1/64 quantile.
    picked = df[df["event_id"].isin(["gu0000", "gu0001", "gu0002"])]
    for _, row in picked.iterrows():
        lam, m, mu = row["rate"], row["max_loss"], row["mean_loss"]
        sigma = row["sd_indep"] + row["sd_corr"]
        if lam == 0.0 or sigma == 0.0 or sigma * sigma >= mu * (m - mu):
            continue
        a, b = reference.beta_params(m, mu, sigma * sigma)
        cut = m * reference.beta_dist.ppf(1.0 - 1.0 / 64.0, a, b)
        _, sec_one, meta_one = run(
            impl, elt([(lam, m, mu, row["sd_indep"], row["sd_corr"])]),
            bucket_size=FIXTURE_H, method="quantiles", quadrature_points=32)
        above = grid_of(meta_one) > cut + FIXTURE_H
        assert sec_one[above].sum() == 0.0


# [20] Bernoulli rows: an inadmissible row contributes rate (1 - p) at 0 and
# rate p at M under both methods, with rate and mean exact and grid variance
# mu (M - mu); the exceedance curve has an exact step of rate p at M.
@pytest.mark.parametrize("method", ["quantiles", "survival"])
def test_20_bernoulli_row(impl, method):
    lam, m, mu = BAD[0], BAD[1], BAD[2]
    raw, secondary, meta = run(impl, elt([BAD]), bucket_size=0.5, method=method)
    assert diag_count(meta, "variance_bernoulli") == 1
    g = grid_of(meta)
    assert secondary[0] == lam * (1.0 - mu / m)
    assert secondary[int(round(m / 0.5))] == lam * mu / m
    assert np.count_nonzero(secondary) == 2
    assert math.isclose(secondary.sum(), lam, rel_tol=1e-15)
    mean = float(secondary @ g) / lam
    variance = float(secondary @ (g * g)) / lam - mean * mean
    assert math.isclose(mean, mu, rel_tol=1e-15)
    assert math.isclose(variance, mu * (m - mu), rel_tol=1e-13)
    assert meta.secondary_mean_error == 0.0
    # The raw output is unaffected: the row's atom at mu.
    assert raw[int(round(mu / 0.5))] == lam
    # The exceedance curve steps by exactly rate p at M.
    curve = reference.build_survival_curve(
        [], [(0.0, lam * (1.0 - mu / m)), (m, lam * mu / m)], 0.25, 150)
    step = curve.exceedance(np.array([m * (1.0 - 1e-12), m, m * (1.0 + 1e-12)]))
    assert step[0] == lam * mu / m and step[1] == 0.0 and step[2] == 0.0


# [curve] The public exceedance curve: the grid it makes is the survival
# method's grid, its exceedance matches the exact incomplete beta, and the
# OEP points come straight from it.
def test_curve_matches_quantize_and_exact(impl):
    import eltq

    if impl.name == "reference":
        pytest.skip("the package-level curve runs on the loaded backend")
    df = load_fixture("small_alpha")
    cols = columns(df)
    h = 50.0
    curve = eltq.exceedance_curve(*cols, min_loss=0.5 * h, event_ids=list(df["event_id"]))
    assert curve.meta.min_loss == 0.5 * h and curve.meta.nodes > 100
    assert curve.meta.interpolation_error <= 1e-4
    assert diag_count(curve.meta, "variance_bernoulli") > 0
    raw, sec, meta = eltq.quantize_arrays(*cols, bucket_size=h, method="survival")
    grid, gmeta = curve.grid(h)
    assert np.array_equal(grid, sec)
    assert gmeta.grid_length == meta.grid_length
    assert math.isclose(gmeta.secondary_mass, meta.total_rate, rel_tol=1e-14)
    assert gmeta.secondary_mean_error == meta.secondary_mean_error
    # A coarser grid from the same curve carries the same mass.
    coarse, cmeta = curve.grid(4.0 * h)
    assert math.isclose(coarse.sum(), meta.total_rate, rel_tol=1e-14)
    assert cmeta.grid_length == int(math.ceil(curve.meta.max_loss / (4.0 * h))) + 1
    with pytest.raises(eltq.EltqError):
        curve.grid(0.5 * h)
    # Exceedance against the exact incomplete beta plus the atoms.
    ref_curve, _ = reference.exceedance_curve_reference(*cols, min_loss=0.5 * h)
    x = np.exp(np.linspace(np.log(0.5 * h), np.log(curve.meta.max_loss), 300))
    x[0] = 0.5 * h   # exp(log(x)) can round a hair below the lowest node
    exact = ref_curve.exact(x)
    atoms = np.array([ref_curve.atom_rate[ref_curve.atom_loss > v].sum() for v in x])
    got = curve.exceedance(x)
    keep = exact > 1e-12 * meta.total_rate
    assert np.max(np.abs(got[keep] - (exact + atoms)[keep]) / (exact + atoms)[keep]) <= 1e-4
    assert np.allclose(curve.exceedance_rate, ref_curve.exceedance_rate, rtol=1e-10, atol=0.0)
    assert np.isnan(curve.exceedance(np.array([0.1 * h]))[0])
    assert curve.exceedance(np.array([2.0 * curve.meta.max_loss]))[0] == 0.0
    # The nodes carry the exact exceedance rate of the betas.
    assert np.allclose(curve.exceedance_rate, ref_curve.exact(curve.loss), rtol=1e-10, atol=0.0)


def test_curve_oep_pml():
    import eltq

    # Two betas and a Bernoulli row: a smooth curve with one step at M = 100.
    rows = [(0.5, 100.0, 40.0, 15.0, 10.0), (0.3, 80.0, 20.0, 8.0, 6.0),
            (0.2, 100.0, 30.0, 50.0, 0.0)]
    cols = elt(rows)
    curve = eltq.exceedance_curve(*(cols[c] for c in ("rate", "max_loss", "mean_loss",
                                                     "sd_indep", "sd_corr")),
                                  min_loss=0.25)
    # Total rate 1.0, so a 1.5-year period (target rate 1.1) is out of range.
    periods = np.array([1.0, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0, 1e4])
    pml = curve.oep_pml(periods)
    assert pml[0] == 0.0                                  # T <= 1: no exceedance
    assert np.isnan(pml[1])
    assert np.isfinite(pml[2:]).all()
    assert np.all(np.diff(pml[2:]) >= 0.0)               # longer periods, larger losses
    # Each solved point satisfies Lambda(x) = -log(1 - 1/T) to bisection precision,
    # or sits on the Bernoulli atom at 100 where the step jumps the target.
    for t, x in zip(periods[2:], pml[2:]):
        target = -math.log1p(-1.0 / t)
        if x == 100.0:
            below = curve.exceedance(np.array([100.0 * (1.0 - 1e-9)]))[0]
            assert below >= target >= curve.exceedance(np.array([100.0]))[0]
        else:
            assert math.isclose(curve.exceedance(np.array([x]))[0], target, rel_tol=1e-9)
    # Below the curve's range: the target exceeds Lambda(min_loss).
    assert np.isnan(curve.oep_pml(np.array([1.0 + 1e-9]))[0])


def test_curve_oep_pml_above_every_atom():
    """An answer above the largest atom is the bisection answer, not that
    atom (2.0.1; `dev/done/plan-survival-tail-cost.md`, finding 3)."""
    import eltq

    # One wide beta and one atom at 1e5, below every solved point.
    curve = eltq.exceedance_curve(np.array([0.5, 1e-3]), np.array([1e7, 1e5]),
                                  np.array([2e5, 1e5]), np.array([6e5, 0.0]),
                                  np.zeros(2), min_loss=50.0)
    assert curve.atom_loss.tolist() == [1e5]
    periods = np.array([10.0, 100.0, 1000.0])
    pml = curve.oep_pml(periods)
    assert np.all(pml > 1e5)
    for t, x in zip(periods, pml):
        target = -math.log1p(-1.0 / t)
        assert math.isclose(curve.exceedance(np.array([x]))[0], target, rel_tol=1e-9)
    # The atom's step still wins where it jumps the target: Lambda just
    # below the atom is 0.501 + the beta, just above it loses the 1e-3.
    below = curve.exceedance(np.array([1e5 * (1.0 - 1e-9)]))[0]
    above = curve.exceedance(np.array([1e5]))[0]
    t_atom = 1.0 / -math.expm1(-0.5 * (below + above))
    assert curve.oep_pml(np.array([t_atom]))[0] == 1e5
