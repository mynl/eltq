"""Spec tests 15 to 24: the survival method, the overflow policies, the
Bernoulli treatment of inadmissible rows (`dev/done/plan-single-unit-survival-route.md`),
the tail rate cut and the narrow events (`dev/done/plan-survival-tail-cost.md`),
the incomplete beta at large parameters (`dev/done/plan-ln-beta-large-parameters.md`)
and the quantiles method at large parameters (`dev/done/plan-quantiles-large-parameters.md`).

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
# [21] The rate cut: against the same grid with every tail walked to its
# endpoint, the tail sums move by at most N eps Lambda(0); the rate is exact.
def test_21_tail_rate_cut():
    from eltq._meta import TAIL_RATE_FLOOR

    df = load_fixture("small_alpha")
    h = FIXTURE_H
    curve, _ = reference.exceedance_curve_reference(*columns(df), min_loss=0.5 * h)
    n = int(math.ceil(curve.max_loss / h)) + 1
    cut, over_cut, _ = curve.bucket(h, n)
    full, over_full, _ = curve.bucket(h, n, tail_rate_floor=0.0)
    assert over_cut == 0.0 and over_full == 0.0
    assert math.isclose(cut.sum(), curve.beta_rate, rel_tol=1e-14)
    assert math.isclose(full.sum(), curve.beta_rate, rel_tol=1e-14)
    # The cut fires on this fixture (its deep events have long tails).
    assert not np.array_equal(cut, full)
    bound = len(df) * TAIL_RATE_FLOOR * curve.beta_rate
    assert np.max(np.abs(tail_sums(cut) - tail_sums(full))) <= bound


# [22] Narrow events: betas narrower than a node interval are kept out of
# the shared curve and walked exactly, so the curve is exact across their
# bulk and the grid's tail sums match the exact Lambda there.
def test_22_narrow_events():
    import eltq

    rng = np.random.default_rng(22)
    center, n_base, n_narrow = 250_000.0, 2_000, 100
    rate = 10.0 ** rng.uniform(-5, -2, n_base)
    max_loss = 10.0 ** rng.uniform(5, 7, n_base)
    mean = 10.0 ** rng.uniform(-4, -1, n_base) * max_loss
    sd = np.minimum(10.0 ** rng.uniform(0, 1, n_base) * mean,
                    0.95 * np.sqrt(mean * (max_loss - mean)))
    # Coefficients of variation from 1e-6 to 1e-2 at a common mean.
    cols = (np.concatenate([rate, np.full(n_narrow, 2.5e-7)]),
            np.concatenate([max_loss, 10.0 ** rng.uniform(np.log10(3e5), 7, n_narrow)]),
            np.concatenate([mean, np.full(n_narrow, center)]),
            np.concatenate([sd, center * 10.0 ** rng.uniform(-6, -2, n_narrow)]),
            np.zeros(n_base + n_narrow))
    h = 25.0
    ref_curve, ref_meta = reference.exceedance_curve_reference(*cols, min_loss=0.5 * h)
    curve = eltq.exceedance_curve(*cols, min_loss=0.5 * h)
    assert ref_meta.narrow_events == n_narrow
    assert curve.meta.narrow_events == n_narrow
    x = np.linspace(0.97 * center, 1.03 * center, 241)
    exact = ref_curve.exact(x)
    for got in (ref_curve.exceedance_betas(x), curve.exceedance(x)):
        assert np.max(np.abs(got - exact) / exact) <= 1e-8
    # The grids: the full rate, the backends within 1e-12 of the total, and
    # the tail sums at the edges around the narrow events exact to 1e-7.
    grid, gmeta = curve.grid(h)
    ref_grid, _ = ref_curve.grid(h)
    total = curve.meta.total_rate
    assert math.isclose(grid.sum(), total, rel_tol=1e-14)
    assert np.max(np.abs(grid - ref_grid)) <= 1e-12 * total
    sums = tail_sums(grid)
    edges = (np.arange(sums.size) + 0.5) * h
    near = (edges >= 0.95 * center) & (edges <= 1.05 * center)
    want = ref_curve.exact(edges[near])
    assert np.max(np.abs(sums[near] - want) / want) <= 1e-7
    # Under quantize the count is reported too.
    _, _, meta = eltq.quantize_arrays(*cols, bucket_size=h)
    assert meta.narrow_events == n_narrow


# [23] Large parameters: a J-shaped beta with beta up to 8e7 (a real ELT's
# limit at 1e7 to 3.5e8 times the mean) has its survival at the nodes to
# double precision below the continued fraction's switch point and to the
# formulation's eps * beta ceiling above it, on both backends, against mpmath
# (SciPy's betaincc is off by 5e-9 at an integer alpha of 2 or more with
# beta = 8e7, so it is not the oracle here); then a portfolio with 300 such
# events buckets as the method promises and the backends agree.
def test_23_large_beta_parameters():
    import mpmath as mp
    import eltq

    mp.mp.dps = 30
    big_m = 1e6
    for alpha in (0.13, 0.5, 1.0, 1.5, 2.5):
        for beta in (1e4, 1e6, 1e7, 8e7):
            mu = big_m * alpha / (alpha + beta)
            sd = big_m * math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0)))
            cols = tuple(np.array([v]) for v in (1.0, big_m, mu, sd, 0.0))
            # The parameters both backends recover from (M, mu, sd).
            a, b = reference.beta_params(big_m, mu, sd * sd)
            switch = (a + 1.0) / (a + b + 2.0)
            ref_curve, _ = reference.exceedance_curve_reference(*cols, min_loss=0.25 * mu)
            curve = eltq.exceedance_curve(*cols, min_loss=0.25 * mu)
            for loss, got in ((ref_curve.loss, ref_curve.exceedance_rate),
                              (curve.loss, curve.exceedance_rate)):
                # Every tenth node from 0.3 to 30 times the mean: about 20 points.
                sel = np.flatnonzero((loss >= 0.3 * mu) & (loss <= 30.0 * mu))[::10]
                y = loss[sel] / big_m
                exact = np.array([float(mp.betainc(a, b, yi, 1, regularized=True)) for yi in y])
                ok = exact >= 1e-15
                rel = np.abs(got[sel] - exact) / exact
                lower = y < switch
                assert np.all(rel[lower & ok] <= 1e-13), (alpha, beta, rel[lower & ok].max())
                assert np.all(rel[~lower & ok] <= 2e-8), (alpha, beta, rel[~lower & ok].max())
    # The portfolio: the base recipe of test 22 plus 300 large-parameter
    # events with means from 1e3 to 1e5 (M from 1e10 to 6e13), on a windowed
    # grid with the overflow in the top bucket.
    rng = np.random.default_rng(23)
    n_base, n_large = 2_000, 300
    rate = 10.0 ** rng.uniform(-5, -2, n_base)
    max_loss = 10.0 ** rng.uniform(5, 7, n_base)
    mean = 10.0 ** rng.uniform(-4, -1, n_base) * max_loss
    sd = np.minimum(10.0 ** rng.uniform(0, 1, n_base) * mean,
                    0.95 * np.sqrt(mean * (max_loss - mean)))
    a = 10.0 ** rng.uniform(np.log10(0.13), np.log10(2.0), n_large)
    b = 10.0 ** rng.uniform(4, np.log10(8e7), n_large)
    mean_large = 10.0 ** rng.uniform(3, 5, n_large)
    m_large = mean_large * (a + b) / a
    sd_large = m_large * np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))
    cols = (np.concatenate([rate, 10.0 ** rng.uniform(-5, -3, n_large)]),
            np.concatenate([max_loss, m_large]),
            np.concatenate([mean, mean_large]),
            np.concatenate([sd, sd_large]),
            np.zeros(n_base + n_large))
    h, n = 100.0, 2 ** 18
    ref_curve, _ = reference.exceedance_curve_reference(*cols, min_loss=0.5 * h)
    curve = eltq.exceedance_curve(*cols, min_loss=0.5 * h)
    total = curve.meta.total_rate
    # The node values are exact (the beta part only: there are no atoms).
    assert curve.atom_loss.size == 0
    exact = ref_curve.exact(curve.loss)
    ok = exact >= 1e-12 * total
    assert np.max(np.abs(curve.exceedance_rate[ok] - exact[ok]) / exact[ok]) <= 1e-11
    grid, _ = curve.grid(h, n)
    ref_grid, _ = ref_curve.grid(h, n)
    assert math.isclose(grid.sum(), total, rel_tol=1e-14)
    assert np.max(np.abs(grid - ref_grid)) <= 1e-12 * total
    # The tail sums against the exact Lambda at every 32nd edge where it is
    # above 1e-12 of the rate: the shared cubic between nodes sets the
    # figure (6e-6 measured, the same on both backends, against the 1e-4 the
    # method promises in test 15).
    sums = tail_sums(grid)
    edges = (np.arange(sums.size) + 0.5) * h
    sel = np.arange(0, edges.size, 32)
    sel = sel[sums[sel] >= 1e-12 * total]
    want = ref_curve.exact(edges[sel])
    assert np.max(np.abs(sums[sel] - want) / want) <= 1e-5


# [24] Large parameters under the quantiles method: from min(alpha, beta) =
# 2e5 both backends take the Cornish-Fisher expansion (spec section 5),
# where the inversion of the incomplete beta drifts and then fails (and
# SciPy's ppf with it, from about 1e20). The expansion against SciPy where
# SciPy is sound; one-row portfolios across the range on both backends; the
# row of the 2026-10-25 report; a portfolio.
def test_24_large_parameters_quantiles():
    from scipy.special import betaincinv
    import eltq
    from eltq._meta import LARGE_PARAMETER_FROM

    u = (np.arange(32) + 0.5) / 32
    for a in (LARGE_PARAMETER_FROM, 1e6, 1e7):
        for ratio in (1.0, 3.0, 300.0, 3e4):
            b = a * ratio
            sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))
            got = reference.large_parameter_quantiles(a, b, u)
            assert np.max(np.abs(got - betaincinv(a, b, u))) <= 1e-8 * sd, (a, b)
    backends = (eltq.quantize_arrays, reference.quantize_reference)
    # One-row portfolios: the walk at 1e5, the expansion from 3e5 up to
    # parameters where the beta is a normal to double precision.
    big_m, h = 1e9, 1e3
    for p in (0.5, 0.25, 0.01):
        for small in (1e5, 3e5, 1e6, 1e8, 1e12, 1e16, 1e20, 4.7e22, 1e26):
            s = small / min(p, 1.0 - p)
            mu = p * big_m
            sd = big_m * math.sqrt(p * (1.0 - p) / (s + 1.0))
            skew = 2.0 * (1.0 - 2.0 * p) * math.sqrt(s + 1.0) / ((s + 2.0) * math.sqrt(p * (1.0 - p)))
            cols = tuple(np.array([v]) for v in (1.0, big_m, mu, sd, 0.0))
            grids = []
            for backend in backends:
                _, sec, meta = backend(*cols, bucket_size=h, method="quantiles")
                assert math.isclose(sec.sum(), 1.0, rel_tol=1e-14)
                idx = np.flatnonzero(sec)
                assert np.all(np.abs(idx * h - mu) <= 2.2 * sd + h), (p, small, backend)
                # The midpoint rule's first-moment error, of order skew * sd.
                assert abs(meta.secondary_mean_error) <= 2e-2 * skew * sd + 1e-9 * mu, (p, small, backend)
                grids.append(sec)
            # Above the threshold the backends share the formula; below it
            # the walk and SciPy's inverse differ by their 1e-9 sigma errors,
            # which the linear rule turns into weight of order sd / h.
            tol = 1e-12 if small >= LARGE_PARAMETER_FROM else 1e-8 * sd / h
            assert np.max(np.abs(grids[0] - grids[1])) <= tol, (p, small)
    # The report's row: exposed value 1e6, mean 250,000, sd 1e-6 (alpha =
    # 4.7e22), bucket size 1000; the whole rate on node 250.
    cols = tuple(np.array([v]) for v in (1.0, 1e6, 250_000.0, 1e-6, 0.0))
    for backend in backends:
        _, sec, _ = backend(*cols, bucket_size=1000.0, method="quantiles")
        assert sec[250] >= 1.0 - 1e-8
    # The portfolio: the base recipe of test 22 plus 300 rows with
    # min(alpha, beta) from 1e5 to 1e24, on a windowed grid.
    rng = np.random.default_rng(24)
    n_base, n_large = 2_000, 300
    rate = 10.0 ** rng.uniform(-5, -2, n_base)
    max_loss = 10.0 ** rng.uniform(5, 7, n_base)
    mean = 10.0 ** rng.uniform(-4, -1, n_base) * max_loss
    sd = np.minimum(10.0 ** rng.uniform(0, 1, n_base) * mean,
                    0.95 * np.sqrt(mean * (max_loss - mean)))
    small = 10.0 ** rng.uniform(5, 24, n_large)
    p = 10.0 ** rng.uniform(-3, np.log10(0.5), n_large)
    s = small / p
    mean_large = 10.0 ** rng.uniform(3, 5, n_large)
    m_large = mean_large / p
    sd_large = m_large * np.sqrt(p * (1.0 - p) / (s + 1.0))
    cols = (np.concatenate([rate, 10.0 ** rng.uniform(-5, -3, n_large)]),
            np.concatenate([max_loss, m_large]),
            np.concatenate([mean, mean_large]),
            np.concatenate([sd, sd_large]),
            np.zeros(n_base + n_large))
    h, n = 100.0, 2 ** 18
    results = [backend(*cols, bucket_size=h, grid_length=n, method="quantiles") for backend in backends]
    total = results[0][2].total_rate
    for raw, sec, meta in results:
        assert math.isclose(meta.total_rate, total, rel_tol=0.0)
        assert math.isclose(sec.sum(), total, rel_tol=1e-14)
    assert np.array_equal(results[0][0], results[1][0])
    # 1e-11 of the total: the base rows and the large rows below the
    # threshold set the walk against SciPy's inverse (4e-12 measured).
    assert np.max(np.abs(results[0][1] - results[1][1])) <= 1e-11 * total


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
