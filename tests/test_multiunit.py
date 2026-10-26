"""Multi-unit acceptance suite: the two-unit joint and the per-view totals.

Runs against every registered multi-unit implementation (the SciPy reference
and the package API, which the Rust engine will serve from 1.10.0) through
the ``MultiImpl`` interface in conftest. Tolerances trace to the z-convergence
study recorded in ``benchmarks/z-convergence-1.7.0.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eltq import reference_multiunit as rm
from eltq.terms import parse_terms_line

from conftest import (
    MULTI_COLUMNS,
    MULTI_IMPLS,
    diag_count,
    load_joint_golden,
    load_multi_fixture,
    load_multi_golden,
    multi_rows,
    run_multi,
)

U1 = rm.Unit(200.0, 100.0, 150.0, 1000.0)
U2 = rm.Unit(500.0, 300.0, 200.0, 2000.0)
H, N1, N2 = 10.0, 101, 201


def direct_marginal(unit, n, h=H):
    return rm.marginal_pmf(rm.prepare_unit(unit, "u"), h, n)


def moments(xs, ps):
    mean = ps @ xs
    sd = math.sqrt(max(ps @ xs**2 - mean * mean, 0.0))
    cdf = np.cumsum(ps)
    q99 = xs[min(int(np.searchsorted(cdf, 0.99 * cdf[-1])), xs.size - 1)]
    return mean, sd, q99


# ---------------------------------------------------------------------------
# [joint] two-unit bivariate pmf.

def test_joint_mass_and_marginals(multi):
    pmf, x1, x2, meta = multi.joint(U1, U2, bucket_size=H, n1=N1, n2=N2, n_z=1001)
    assert pmf.shape == (N1, N2)
    assert math.isclose(pmf.sum(), 1.0, abs_tol=1e-12)
    assert np.array_equal(x1, H * np.arange(N1)) and np.array_equal(x2, H * np.arange(N2))
    # Midpoint quadrature error is O(Q^-2) with a visible constant: 4e-5 at Q = 1001.
    assert np.abs(pmf.sum(axis=1) - direct_marginal(U1, N1)).max() < 1e-4
    assert np.abs(pmf.sum(axis=0) - direct_marginal(U2, N2)).max() < 1e-4
    assert math.isclose(meta.grid_mean[0], U1.mean_loss, rel_tol=5e-3)
    assert math.isclose(meta.copula_normal_correlation,
                        150.0 * 200.0 / (math.hypot(100.0, 150.0) * math.hypot(300.0, 200.0)))
    assert 0.0 < meta.grid_correlation < meta.copula_normal_correlation  # Proposition 4


def test_joint_gauss_hermite_is_spectral_on_smooth_case(multi):
    pmf, _, _, meta = multi.joint(U1, U2, bucket_size=H, n1=N1, n2=N2, n_z=64,
                                  z_rule="gauss_hermite")
    assert np.abs(pmf.sum(axis=1) - direct_marginal(U1, N1)).max() < 1e-12
    assert np.abs(pmf.sum(axis=0) - direct_marginal(U2, N2)).max() < 1e-12
    assert meta.z_rule == "gauss_hermite"


def test_joint_independent_units_are_an_outer_product(multi):
    a, b = U1._replace(sd_corr=0.0), U2._replace(sd_corr=0.0)
    pmf, _, _, meta = multi.joint(a, b, bucket_size=H, n1=N1, n2=N2, n_z=8)
    outer = np.outer(pmf.sum(axis=1), pmf.sum(axis=0))
    assert np.abs(pmf - outer).max() < 1e-15
    assert meta.copula_normal_correlation == 0.0
    # Rust's own incomplete beta agrees with SciPy's to ~1e-13.
    assert np.abs(pmf.sum(axis=1) - direct_marginal(a, N1)).max() < 5e-13


def test_joint_comonotone_units(multi):
    a, b = U1._replace(sd_indep=0.0), U2._replace(sd_indep=0.0)
    pmf, _, _, meta = multi.joint(a, b, bucket_size=H, n1=N1, n2=N2, n_z=2001)
    assert math.isclose(pmf.sum(), 1.0, abs_tol=1e-12)
    assert meta.copula_normal_correlation == 1.0
    # Comonotone but not linearly related (different beta shapes), so the
    # loss correlation sits below 1 (Proposition 4); 0.988 on this grid.
    assert 0.95 < meta.grid_correlation < 1.0
    # Means are exact for point deposits up to quadrature.
    assert math.isclose(meta.grid_mean[0], a.mean_loss, rel_tol=2e-3)


def test_joint_atom_unit(multi):
    atom = rm.Unit(333.0, 0.0, 0.0, 1000.0)
    pmf, x1, _, meta = multi.joint(atom, U2, bucket_size=H, n1=N1, n2=N2, n_z=16)
    m1 = pmf.sum(axis=1)
    assert np.count_nonzero(m1) == 2 and m1[33] > 0 and m1[34] > 0  # linear deposit
    assert math.isclose(m1 @ x1, 333.0, rel_tol=1e-12)
    assert meta.notes[0] == ("degenerate_atom",)


def test_joint_errors(multi):
    with pytest.raises(multi.overflow_error):
        multi.joint(U1, U2, bucket_size=H, n1=50, n2=N2)
    with pytest.raises(multi.base_error):
        multi.joint(U1, U2, bucket_size=H, n1=N1, n2=N2, n_z=0)
    with pytest.raises(multi.base_error):
        multi.joint(U1, U2, bucket_size=-1.0, n1=N1, n2=N2)
    with pytest.raises(multi.base_error):
        multi.joint(U1, U2, bucket_size=H, n1=N1, n2=N2, z_rule="simpson")
    with pytest.raises(multi.variance_error):  # 80^2 = 6400 >= 30 * 70
        multi.joint(rm.Unit(30.0, 50.0, 30.0, 100.0), U2, bucket_size=H, n1=N1,
                    n2=N2, variance_policy="error")
    with pytest.raises(multi.base_error):
        multi.joint(rm.Unit(300.0, math.nan, 30.0, 1000.0), U2, bucket_size=H, n1=N1, n2=N2)


def test_joint_golden(multi):
    g_pmf, g_meta = load_joint_golden()
    p = g_meta["params"]
    pmf, _, _, meta = multi.joint(tuple(p["units"][0]), tuple(p["units"][1]),
                                  bucket_size=p["bucket_size"], n1=p["n1"], n2=p["n2"],
                                  n_z=p["z_nodes"], z_rule=p["z_rule"])
    assert np.allclose(pmf, g_pmf, rtol=1e-9, atol=1e-15)
    assert math.isclose(meta.total_mass, g_meta["total_mass"], rel_tol=1e-12)
    assert math.isclose(meta.grid_correlation, g_meta["grid_correlation"], rel_tol=1e-9)
    assert tuple(meta.alpha) == tuple(g_meta["alpha"])


# ---------------------------------------------------------------------------
# [views] per-view totals for a multi-unit ELT.

TOWER = "unit a 750 xs 250 occurrence net of 50% po 400 xs 100 and 500 xs 500"


def test_view_semantics_on_atoms(multi):
    # Atom at 1000 -> gross 750, ceded 200 + 250 = 450, net 300 (all on grid).
    lam = 0.02
    views, meta = run_multi(multi, [("e", "a", lam, 1000.0, 0.0, 0.0, 1000.0)],
                            terms=TOWER, bucket_size=10.0)
    for view, loss in (("ground_up", 1000), ("gross", 750), ("ceded", 450), ("net", 300)):
        expected = np.zeros(meta.grid_length)
        expected[loss // 10] = lam
        assert np.array_equal(views[view], expected), view
    assert meta.terms_echo == {"a": TOWER}


def test_amount_share_and_overlap_and_off_grid(multi):
    rows = [
        ("e1", "b", 0.01, 600.0, 0.0, 0.0, 1000.0),   # 25% of 400 xs 100 on gross 600
        ("e2", "o", 0.01, 500.0, 0.0, 0.0, 1000.0),   # overlapping half-share layers
        ("e3", "f", 0.01, 250.0, 0.0, 0.0, 1000.0),   # 50% of 150 = 75, off grid
    ]
    terms = ["unit b 1000 xs 0 occurrence net of 100 po 400 xs 100",
             "unit o occurrence net of 50% po 300 xs 100 and 50% po 200 xs 200",
             "unit f occurrence net of 50% po 300 xs 100"]
    views, meta = run_multi(multi, rows, terms=terms, bucket_size=10.0)
    g = 10.0 * np.arange(meta.grid_length)
    # e1: ceded 100, net 500. e2: ceded 150 + 100 = 250, net 250. e3: ceded
    # 75 (between nodes 70 and 80, equal weights), net 175.
    assert views["ceded"][10] == pytest.approx(0.01)
    assert views["net"][50] == pytest.approx(0.01)
    assert views["ceded"][25] == pytest.approx(0.01) and views["net"][25] == pytest.approx(0.01)
    assert views["ceded"][7] == pytest.approx(0.005) and views["ceded"][8] == pytest.approx(0.005)
    assert math.isclose(float(views["ceded"] @ g), 0.01 * (100 + 250 + 75), rel_tol=1e-12)
    assert math.isclose(float(views["net"] @ g), 0.01 * (500 + 250 + 175), rel_tol=1e-12)
    # Overlap beyond 100% of a band is rejected: ceded would exceed gross.
    with pytest.raises(multi.base_error):
        run_multi(multi, rows[:1], terms="unit b occurrence net of 300 xs 100 and 200 xs 200",
                  bucket_size=10.0)


def test_units_without_terms_and_structured_terms(multi):
    rows = [("e", "a", 0.01, 300.0, 40.0, 30.0, 1000.0), ("e", "c", 0.01, 200.0, 20.0, 10.0, 500.0)]
    structured = {"a": {"limit": 750, "ded": 250, "clauses": [(0.5, 400, 100), (1.0, 500, 500)]}}
    v_text, m_text = run_multi(multi, rows, terms=TOWER, bucket_size=10.0)
    v_struct, m_struct = run_multi(multi, rows, terms=structured, bucket_size=10.0)
    for view in m_text.views:
        assert np.array_equal(v_text[view], v_struct[view])
    # Unit c has no terms: its gross equals ground up, so the only difference
    # between the views comes from unit a.
    assert m_text.unit_ids == ("a", "c")
    assert m_text.unused_terms == ()
    _, m_spare = run_multi(multi, rows, terms=[TOWER, "unit spare 10 xs 0"], bucket_size=10.0)
    assert m_spare.unused_terms == ("spare",)


def test_independent_event_is_plain_convolution(multi):
    a = rm.Unit(200.0, 250.0, 0.0, 1000.0)
    b = rm.Unit(500.0, 500.0, 0.0, 2000.0)
    lam = 0.03
    rows = [("e", "a", lam, *a), ("e", "b", lam, *b)]
    views, meta = run_multi(multi, rows, bucket_size=H, views=("ground_up",), z_nodes=4)
    expected = lam * np.convolve(direct_marginal(a, N1), direct_marginal(b, N2))
    assert meta.grid_length == N1 + N2 - 1
    assert np.abs(views["ground_up"] - expected).max() < 1e-14 * lam


def test_comonotone_common_shape_reproduces_rolled_beta(multi):
    # Proposition 6: sd_indep = 0 and equal normalized moments -> the total is
    # the rolled row's beta exactly (bucket rounding aside).
    h = 5.0
    rows = [("e", "p", 0.01, 200.0, 0.0, 250.0, 1000.0),
            ("e", "q", 0.01, 400.0, 0.0, 500.0, 2000.0)]
    views, meta = run_multi(multi, rows, bucket_size=h, views=("ground_up",), z_nodes=1001,
                            roll_up_diagnostic=True)
    total = views["ground_up"] / meta.total_rate
    xs = h * np.arange(total.size)
    rolled = direct_marginal(rm.Unit(600.0, 0.0, 750.0, 3000.0), total.size, h)
    mean_t, sd_t, q_t = moments(xs, total)
    mean_r, sd_r, q_r = moments(xs, rolled)
    assert math.isclose(mean_t, 600.0, rel_tol=1e-3) and math.isclose(mean_r, 600.0, rel_tol=1e-3)
    assert math.isclose(sd_t, sd_r, rel_tol=1e-3)
    assert math.isclose(q_t, q_r, rel_tol=5e-3)
    assert len(meta.roll_up) == 1
    assert math.isclose(meta.roll_up[0]["sd_ratio"], 1.0, rel_tol=1e-3)


def test_mass_is_rate_per_view_and_means_reconcile(multi):
    df, terms = load_multi_fixture()
    for z_nodes, tol in ((32, 2e-2), (128, 6e-3)):
        views, meta = run_multi(multi, df, terms=terms, bucket_size=10.0, z_nodes=z_nodes)
        for view in meta.views:
            assert math.isclose(meta.view_mass[view], meta.total_rate, rel_tol=1e-12), view
            assert math.isclose(views[view].sum(), meta.total_rate, rel_tol=1e-12), view
            # View means reconcile to the analytic beta layer means (LEV
            # identity) within the midpoint z-quadrature error at this Q.
            assert abs(meta.view_mean_error[view]) <= tol * abs(meta.analytic_view_mean[view]), view
    assert math.isclose(meta.analytic_view_mean["ground_up"], meta.analytic_aal, rel_tol=1e-12)
    assert math.isclose(meta.analytic_view_mean["net"],
                        meta.analytic_view_mean["gross"] - meta.analytic_view_mean["ceded"],
                        rel_tol=1e-12)


def test_gauss_hermite_reconciles_tightly_on_smooth_units(multi):
    rows = [("e", "a", 0.01, 200.0, 100.0, 150.0, 1000.0),
            ("e", "b", 0.01, 500.0, 300.0, 200.0, 2000.0)]
    _, meta = run_multi(multi, rows, terms=TOWER, bucket_size=10.0, z_nodes=32,
                        z_rule="gauss_hermite")
    for view in meta.views:
        assert abs(meta.view_mean_error[view]) <= 1e-3 * abs(meta.analytic_view_mean[view]), view


def test_gauss_hermite_with_comonotone_units_at_large_q(multi):
    # Gauss-Hermite nodes reach |z| ~ 10 at Q = 64, where Phi(w_c z) rounds
    # to exactly 0 or 1: the comonotone quantile walk must accept repeated
    # endpoint targets (1.11.0 regression). From Q in the hundreds two
    # adjacent nodes in [8.14, 8.29] both give Phi = 1 - 2^-53, a repeated
    # interior target, which must be accepted too (1.11.4).
    df, terms = load_multi_fixture()
    for z_nodes in (64, 1001):
        views, meta = run_multi(multi, df, terms=terms, bucket_size=10.0, z_nodes=z_nodes,
                                z_rule="gauss_hermite")
        for view in meta.views:
            assert math.isclose(views[view].sum(), meta.total_rate, rel_tol=1e-12), \
                (view, z_nodes)
    como = rm.Unit(574.0, 0.0, 232.0, 1000.0)
    pmf, _, _, _ = multi.joint(como, U2, bucket_size=H, n1=N1, n2=N2, n_z=64,
                               z_rule="gauss_hermite")
    assert math.isclose(pmf.sum(), 1.0, abs_tol=1e-12)
    for z_nodes in (700, 1001):
        pmf, _, _, _ = multi.joint((200.0, 0.0, 250.0, 1000.0), (500.0, 300.0, 200.0, 2000.0),
                                   bucket_size=H, n1=N1, n2=N2, n_z=z_nodes,
                                   z_rule="gauss_hermite")
        assert math.isclose(pmf.sum(), 1.0, abs_tol=1e-12), z_nodes


def test_normalize(multi):
    rows = [("e1", "a", 0.02, 300.0, 40.0, 30.0, 1000.0), ("e2", "a", 0.03, 200.0, 20.0, 10.0, 1000.0)]
    views, meta = run_multi(multi, rows, bucket_size=10.0, normalize=True)
    assert meta.normalized and math.isclose(meta.total_rate, 0.05, rel_tol=1e-15)
    for view in meta.views:
        assert math.isclose(views[view].sum(), 1.0, rel_tol=1e-12)
    with pytest.raises(multi.base_error):
        run_multi(multi, [("e1", "a", 0.0, 300.0, 40.0, 30.0, 1000.0)], bucket_size=10.0,
                  normalize=True)


def test_row_policies_and_diagnostics(multi):
    good = ("e", "a", 0.01, 300.0, 40.0, 30.0, 1000.0)
    bad = ("e", "b", 0.01, 300.0, 50.0, 30.0, 1000.0)      # 80^2 = 6400 >= 300 * 700? no: 210000
    bad = ("e", "b", 0.01, 30.0, 50.0, 30.0, 100.0)        # 6400 >= 30 * 70 = 2100
    nan_row = ("e", "n", 0.01, math.nan, 1.0, 1.0, 100.0)
    over = ("e", "o", 0.01, 150.0, 1.0, 1.0, 100.0)
    views, meta = run_multi(multi, [good, bad, nan_row, over], bucket_size=1.0)
    assert diag_count(meta, "variance_bernoulli") == 1  # the default from 2.5.0
    assert diag_count(meta, "sigma_clamped") == 0
    assert diag_count(meta, "nonfinite") == 1 and diag_count(meta, "mean_exceeds_max") == 1
    assert meta.valid_rows == 2 and meta.valid_events == 1
    assert meta.diagnostics["nonfinite"].examples == ["e/n"]
    assert meta.grid_length == 1101  # supports 1000 + 100
    _, m_drop = run_multi(multi, [good, bad], bucket_size=1.0, variance_policy="drop")
    assert diag_count(m_drop, "variance_dropped") == 1 and m_drop.grid_length == 1001
    _, m_atom = run_multi(multi, [good, bad], bucket_size=1.0, variance_policy="atom")
    assert diag_count(m_atom, "variance_atom") == 1 and m_atom.grid_length == 1031
    with pytest.raises(multi.variance_error):
        run_multi(multi, [good, bad], bucket_size=1.0, variance_policy="error")
    # The clamp stays selectable until 3.0.0, with a deprecation warning.
    with pytest.warns(DeprecationWarning, match="deprecated"):
        _, m_clamp = run_multi(multi, [good, bad], bucket_size=1.0, variance_policy="clamp")
    assert diag_count(m_clamp, "sigma_clamped") == 1 and m_clamp.grid_length == 1101
    _, m_zero = run_multi(multi, [("z", "a", 0.0, 300.0, 40.0, 30.0, 1000.0), good],
                          bucket_size=1.0)
    assert diag_count(m_zero, "zero_rate") == 1 and m_zero.valid_events == 1
    assert m_zero.events == 2


def test_structural_errors(multi):
    a = (0.01, 300.0, 40.0, 30.0, 1000.0)
    with pytest.raises(multi.structure_error):  # regrouped event
        run_multi(multi, [("e1", "a", *a), ("e2", "a", *a), ("e1", "b", *a)], bucket_size=10.0)
    with pytest.raises(multi.structure_error):  # rate varies within event
        run_multi(multi, [("e1", "a", *a), ("e1", "b", 0.02, *a[1:])], bucket_size=10.0)
    with pytest.raises(multi.structure_error):  # duplicate unit
        run_multi(multi, [("e1", "a", *a), ("e1", "a", *a)], bucket_size=10.0)
    with pytest.raises(multi.overflow_error) as info:
        run_multi(multi, [("e1", "a", *a), ("e1", "b", *a)], bucket_size=10.0, grid_length=150)
    assert info.value.required_length == 201
    with pytest.raises(multi.base_error):  # misaligned breakpoint
        run_multi(multi, [("e1", "a", *a)], terms="unit a 755 xs 0", bucket_size=10.0)
    with pytest.raises(multi.base_error):
        run_multi(multi, [("e1", "a", *a)], bucket_size=10.0, views=("bogus",))
    with pytest.raises(multi.base_error):
        run_multi(multi, [("e1", "a", *a)], bucket_size=10.0, z_rule="simpson")


def test_golden_fixture(multi):
    df, terms = load_multi_fixture()
    g_views, g_meta = load_multi_golden()
    views, meta = run_multi(multi, df, terms=terms, **g_meta["params"])
    scale = g_meta["total_rate"]
    assert meta.grid_length == g_meta["grid_length"]
    assert tuple(meta.views) == tuple(g_meta["views"])
    for view in meta.views:
        assert np.allclose(views[view], g_views[view], rtol=1e-9, atol=1e-13 * scale), view
        assert math.isclose(meta.analytic_view_mean[view], g_meta["analytic_view_mean"][view],
                            rel_tol=1e-12), view
    assert math.isclose(meta.total_rate, g_meta["total_rate"], rel_tol=1e-12)
    assert math.isclose(meta.analytic_aal, g_meta["analytic_aal"], rel_tol=1e-12)
    for category, issues in g_meta["diagnostics"].items():
        assert diag_count(meta, category) == issues["count"], category
    assert (meta.valid_rows, meta.events, meta.valid_events) == (
        g_meta["valid_rows"], g_meta["events"], g_meta["valid_events"])
    assert tuple(meta.unit_ids) == tuple(g_meta["unit_ids"])
    assert meta.terms_echo == g_meta["terms_echo"]
    assert tuple(meta.unused_terms) == tuple(g_meta["unused_terms"])


def test_dataframe_convenience():
    import eltq

    df, terms = load_multi_fixture()
    v_df, m_df = eltq.multiunit_views_dataframe(df, terms=terms, bucket_size=10.0)
    v_arr, m_arr = eltq.multiunit_views(*(df[c] for c in df.columns), terms=terms,
                                        bucket_size=10.0)
    for view in m_df.views:
        assert np.array_equal(v_df[view], v_arr[view])
    assert m_df == m_arr


def test_z_quadrature_rules():
    for rule in ("midpoint", "gauss_hermite"):
        z, w = rm.z_quadrature(64, rule)
        assert math.isclose(w.sum(), 1.0, abs_tol=1e-14)
        assert abs(w @ z) < 1e-12
        assert np.all(np.diff(z) > 0)
    z, w = rm.z_quadrature(1001, "gauss_hermite")   # stable at large Q
    assert math.isclose(w @ z**2, 1.0, rel_tol=1e-10) and math.isclose(w @ z**4, 3.0, rel_tol=1e-9)
    z, w = rm.z_quadrature(1, "gauss_hermite")
    assert z[0] == 0.0 and w[0] == 1.0
    with pytest.raises(rm.EltqError):
        rm.z_quadrature(0)
    # Q = 4 body strata over [0, 3/4], the ladder 1/4 / sqrt(2)^k for k = 1
    # to 43 (2^-24 is the floor) and the top stratum: 48 strata whose widths
    # are the weights.
    strata = rm.FactorRule.new(4, "stratified")
    assert strata.stratified and len(strata) == 48 and strata.nodes.size == 0
    assert np.allclose(strata.edges_u[:5], [0.0, 0.1875, 0.375, 0.5625, 0.75], rtol=0, atol=0)
    assert math.isclose(strata.edges_u[5], 1.0 - 0.25 / math.sqrt(2.0), abs_tol=1e-15)
    assert math.isclose(1.0 - strata.edges_u[-2], 0.25 * 2.0 ** -21.5, abs_tol=1e-15)
    assert strata.edges_u[-1] == 1.0 and np.all(np.diff(strata.edges_u) > 0)
    assert strata.edges_z[0] == -np.inf and strata.edges_z[-1] == np.inf
    assert math.isclose(strata.weights.sum(), 1.0, abs_tol=1e-15)
    assert len(rm.FactorRule.new(32, "stratified")) == 76
    equal = rm.FactorRule.from_edges(np.arange(5) / 4)
    assert equal.edges_z[2] == 0.0 and np.all(equal.weights == 0.25)
    with pytest.raises(rm.EltqError):
        rm.z_quadrature(4, "stratified")


def test_bivariate_normal_cdf():
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(20261026)
    for _ in range(200):
        h, k = rng.normal(size=2) * 2.5
        rho = rng.uniform(-0.999, 0.999)
        want = multivariate_normal.cdf([h, k], mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
        assert abs(float(rm.bivariate_normal_cdf(h, k, rho)) - want) < 5e-16
    # The zero arguments, the infinities and rho = +-1 are exact limits.
    assert math.isclose(float(rm.bivariate_normal_cdf(0.0, 0.0, 0.3)), 0.25 + math.asin(0.3) / (2 * math.pi))
    assert math.isclose(float(rm.bivariate_normal_cdf(0.0, 1.0, 0.5)), 0.4687429526451681, abs_tol=1e-15)
    assert float(rm.bivariate_normal_cdf(np.inf, 0.4, 0.5)) == rm.norm.cdf(0.4)
    assert float(rm.bivariate_normal_cdf(0.4, -np.inf, 0.5)) == 0.0
    assert float(rm.bivariate_normal_cdf(0.4, 1.1, 1.0)) == rm.norm.cdf(0.4)
    y = np.linspace(-3, 3, 7)
    assert np.allclose(rm.bivariate_normal_cdf(y, np.inf, 0.7), rm.norm.cdf(y), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# [z-rule] spec test 28: the stratified rule and the exact references (2.6.0).

def test_stratified_marginals_are_exact(multi):
    # Every unit's marginal from the joint equals its direct bucketing to
    # rounding at any Q, loading one and two-point units included; the node
    # rules leave a unit at loading one as Q atoms.
    como = rm.Unit(200.0, 0.0, 250.0, 1000.0)
    two_point = rm.Unit(30.0, 50.0, 30.0, 100.0)
    step = rm.Unit(30.0, 0.0, 80.0, 100.0)
    for q in (1, 8, 32):
        for a, b, n1 in ((U1, U2, N1), (como, U2, N1), (two_point, U2, 11), (step, U2, 11),
                         (U1._replace(sd_corr=0.0), U2, N1)):
            pmf, _, _, meta = multi.joint(a, b, bucket_size=H, n1=n1, n2=N2, n_z=q,
                                          z_rule="stratified")
            assert meta.z_rule == "stratified"
            assert math.isclose(pmf.sum(), 1.0, abs_tol=1e-12)
            assert np.abs(pmf.sum(axis=1) - direct_marginal(a, n1)).max() < 1e-13, (q, a)
            assert np.abs(pmf.sum(axis=0) - direct_marginal(b, N2)).max() < 1e-13, (q, a)
            assert pmf.min() >= 0.0
    pmf, _, _, _ = multi.joint(como, U2, bucket_size=H, n1=N1, n2=N2, n_z=8, z_rule="midpoint")
    assert np.count_nonzero(pmf.sum(axis=1)) <= 16


def exceedance_errors(got, exact, levels=(1e-2, 1e-3, 1e-4, 1e-5)):
    """Relative error of the exceedance probability at the losses where the
    exact exceedance is nearest each level (the study's tail metric); NaN
    where the exact total never reaches the level."""
    s_exact, s_got = 1.0 - np.cumsum(exact), 1.0 - np.cumsum(got)
    out = []
    for level in levels:
        at = np.nonzero(s_exact >= level)[0]
        if at.size == 0 or s_exact[at[-1]] > 10 * level:
            out.append(math.nan)
        else:
            out.append((s_got[at[-1]] - s_exact[at[-1]]) / s_exact[at[-1]])
    return np.array(out)


def test_stratified_tails_against_the_exact_joint(multi):
    # Spec test 28: cases A to D of the multi-unit article and the three
    # Homer events against the bivariate normal rectangles. The tail ladder
    # holds the event total's exceedance at the 1e-2 to 1e-5 levels within
    # 1e-2 relative at the default (6.7e-3 measured, case B at 1e-4;
    # `benchmarks/z-convergence-2.6.0.md`); case D, both loadings one, is
    # exact beyond 1e-2 and within 2e-2 there, where the body meets the
    # ladder; the independent case C is exact.
    cases = {
        "A": (U1, U2),
        "B": (rm.Unit(200.0, 30.0, 220.0, 1000.0), rm.Unit(500.0, 60.0, 440.0, 2000.0)),
        "C": (rm.Unit(200.0, 250.0, 0.0, 1000.0), rm.Unit(500.0, 500.0, 0.0, 2000.0)),
        "D": (rm.Unit(200.0, 0.0, 250.0, 1000.0), rm.Unit(400.0, 0.0, 500.0, 2000.0)),
        "H1": (rm.Unit(*HOMER[1][0]), rm.Unit(*HOMER[1][1])),
        "H2": (rm.Unit(*HOMER[2][0]), rm.Unit(*HOMER[2][1])),
        "H3": (rm.Unit(*HOMER[3][0]), rm.Unit(*HOMER[3][1])),
    }
    for name, (a, b) in cases.items():
        n1, n2 = int(a.max_loss / H) + 1, int(b.max_loss / H) + 1
        idx = np.add.outer(np.arange(n1), np.arange(n2)).ravel()

        def total(pmf):
            return np.bincount(idx, weights=pmf.ravel(), minlength=n1 + n2 - 1)

        exact = rm.exact_two_unit_joint([a, b], H, n1, n2)
        assert math.isclose(exact.sum(), 1.0, abs_tol=1e-12)
        assert np.abs(exact.sum(axis=1) - direct_marginal(a, n1)).max() < 1e-13
        pmf, _, _, meta = multi.joint(a, b, bucket_size=H, n1=n1, n2=n2)
        assert (meta.z_rule, meta.z_nodes, meta.z_strata) == ("stratified", 32, 76)
        errors = exceedance_errors(total(pmf), total(exact))
        measured = errors[~np.isnan(errors)]
        if name == "C":
            assert np.abs(measured).max() < 1e-9, name  # rounding at the 1e-5 level
        elif name == "D":
            assert abs(errors[0]) < 2e-2 and np.abs(measured[1:]).max() < 1e-10, (name, errors)
        else:
            assert np.abs(measured).max() < 1e-2, (name, errors)
        # The ladder, not the body count, sets the tail: Q = 8 does as well.
        pmf8, _, _, _ = multi.joint(a, b, bucket_size=H, n1=n1, n2=n2, n_z=8)
        errors8 = exceedance_errors(total(pmf8), total(exact))
        assert np.nanmax(np.abs(errors8 - errors)[1:]) < 1e-3, (name, errors8)


def test_stratified_views_agree_across_backends():
    reference, package = MULTI_IMPLS
    df, terms = load_multi_fixture()
    want, m_want = run_multi(reference, df, terms=terms, bucket_size=10.0, z_nodes=16,
                             z_rule="stratified")
    got, m_got = run_multi(package, df, terms=terms, bucket_size=10.0, z_nodes=16,
                           z_rule="stratified")
    for view in m_want.views:
        assert np.allclose(got[view], want[view], rtol=0.0, atol=1e-12 * m_want.total_rate), view
        assert math.isclose(m_got.view_mass[view], m_want.total_rate, rel_tol=1e-12), view
        assert got[view].min() >= 0.0


def test_terms_line_is_reusable_in_engine(multi):
    # A parsed UnitTerms object is accepted directly.
    terms = {"a": parse_terms_line(TOWER)}
    views, meta = run_multi(multi, [("e", "a", 0.01, 1000.0, 0.0, 0.0, 1000.0)], terms=terms,
                            bucket_size=10.0)
    assert views["net"][30] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# [bernoulli] spec test 26: inadmissible rows are two-point units (2.5.0).

# Homer and Li 2017, Table 11 (the multi-unit article, `tbl-homer`): the
# personal (P) and commercial (C) rows of the three events as
# (mean, sd_indep, sd_corr, M). Event 1's commercial row and event 2's
# personal row are two-point rows.
HOMER = {
    1: ((300.0, 400.0, 300.0, 3000.0), (200.0, 300.0, 200.0, 1000.0)),
    2: ((100.0, 371.0, 267.0, 1000.0), (200.0, 150.0, 533.0, 4000.0)),
    3: ((100.0, 224.0, 200.0, 2000.0), (100.0, 200.0, 200.0, 2000.0)),
}
HOMER_RATES = {1: 0.1, 2: 0.1, 3: 0.5}


def article_unit_cdf(unit, x):
    """The multi-unit article's ``Unit.cdf`` (chunk `setup`): the beta, or the
    two-point step when ``sigma^2 >= mu (M - mu)``."""
    from scipy import stats

    mu, sdi, sdc, m = unit
    sigma = sdi + sdc
    if sigma * sigma >= mu * (m - mu):
        p = mu / m
        return np.where(x < 0, 0.0, np.where(x < m, 1 - p, 1.0))
    kappa = (mu / m) * (1 - mu / m) / (sigma / m) ** 2 - 1
    return stats.beta.cdf(x / m, (mu / m) * kappa, (1 - mu / m) * kappa)


def article_conditional_pmf(unit, z, n, h):
    """The article's ``Unit.conditional_pmf``: bucket probabilities given
    ``Z_c = z`` by differences of the conditional distribution function at
    the half-bucket edges, one row per node."""
    from scipy.special import ndtr, ndtri

    mu, sdi, sdc, m = unit
    w = sdc / math.hypot(sdi, sdc)
    edges = (np.arange(n + 1) - 0.5) * h
    cdf = article_unit_cdf(unit, edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = ndtri(np.clip(cdf, 0, 1))
        if w < 1:
            cond = ndtr((y[None, :] - w * z[:, None]) / math.sqrt(1 - w**2))
        else:
            cond = (y[None, :] >= z[:, None]).astype(float)
    cond[:, 0] = 0.0
    cond[:, -1] = 1.0
    return np.diff(cond, axis=1)


def article_joint(u1, u2, n1, n2, h, q=64):
    """The article's ``joint``: the weighted sum over Gauss-Hermite nodes of
    the outer products of the conditional bucket probabilities."""
    z, w = np.polynomial.hermite_e.hermegauss(q)
    w = w / w.sum()
    c1, c2 = article_conditional_pmf(u1, z, n1, h), article_conditional_pmf(u2, z, n2, h)
    return (c1 * w[:, None]).T @ c2


def test_two_point_unit_in_the_joint(multi):
    # 80^2 = 6400 >= 30 * 70: the two-point law with p = 0.3, Burke's loading
    # kept. Over Gauss-Hermite nodes the mixture of the smooth conditional
    # law is the marginal to rounding; the atoms sit at 0 and M only.
    bad = rm.Unit(30.0, 50.0, 30.0, 100.0)
    pmf, x1, _, meta = multi.joint(bad, U2, bucket_size=H, n1=11, n2=N2, n_z=64,
                                   z_rule="gauss_hermite")
    marginal = pmf.sum(axis=1)
    assert math.isclose(marginal[0], 0.7, abs_tol=1e-12)
    assert math.isclose(marginal[10], 0.3, abs_tol=1e-12)
    assert np.all(marginal[1:10] == 0.0)
    assert math.isclose(pmf.sum(), 1.0, abs_tol=1e-12)
    assert meta.notes[0] == ("variance_bernoulli",)
    assert math.isclose(meta.sigma_used[0], math.sqrt(30.0 * 70.0), rel_tol=1e-12)
    assert math.isclose(meta.copula_normal_correlation,
                        (30.0 / math.hypot(50.0, 30.0)) * (200.0 / math.hypot(300.0, 200.0)))
    assert 0.0 < meta.grid_correlation < meta.copula_normal_correlation
    # Independent of the factor when sd_corr = 0: an outer product.
    pmf_ind, _, _, _ = multi.joint(bad._replace(sd_corr=0.0, sd_indep=80.0), U2, bucket_size=H,
                                   n1=11, n2=N2, n_z=8)
    outer = np.outer(pmf_ind.sum(axis=1), pmf_ind.sum(axis=0))
    assert np.abs(pmf_ind - outer).max() < 1e-15


def test_two_point_event_matches_the_article(multi):
    # Spec test 26: Homer's event 2 (personal two-point, commercial beta)
    # against the multi-unit article's construction at 64 Gauss-Hermite nodes.
    h = 10.0
    for event, (personal, commercial) in HOMER.items():
        n1, n2 = int(personal[3] / h) + 1, int(commercial[3] / h) + 1
        pmf, _, _, _ = multi.joint(personal, commercial, bucket_size=h, n1=n1, n2=n2, n_z=64,
                                   z_rule="gauss_hermite")
        want = article_joint(personal, commercial, n1, n2, h)
        assert np.abs(pmf - want).max() < 1e-12, event
    # The same joint through the views route: the event total of the joint
    # equals the ground-up total, and the two-point rows are reported.
    rows = []
    for event, (personal, commercial) in HOMER.items():
        rows.append((f"e{event}", "P", HOMER_RATES[event], *personal))
        rows.append((f"e{event}", "C", HOMER_RATES[event], *commercial))
    views, meta = run_multi(multi, rows, bucket_size=h, views=("ground_up",), z_nodes=64,
                            z_rule="gauss_hermite")
    assert diag_count(meta, "variance_bernoulli") == 2
    assert set(meta.diagnostics["variance_bernoulli"].examples) == {"e1/C", "e2/P"}
    total = np.zeros(meta.grid_length)
    for event, (personal, commercial) in HOMER.items():
        n1, n2 = int(personal[3] / h) + 1, int(commercial[3] / h) + 1
        joint = article_joint(personal, commercial, n1, n2, h)
        idx = np.add.outer(np.arange(n1), np.arange(n2)).ravel()
        total[:n1 + n2 - 1] += HOMER_RATES[event] * np.bincount(idx, weights=joint.ravel(),
                                                               minlength=n1 + n2 - 1)
    assert np.abs(views["ground_up"] - total).max() < 1e-12 * meta.total_rate
    assert math.isclose(meta.view_mass["ground_up"], meta.total_rate, rel_tol=1e-12)
    # Gauss-Hermite at 64 nodes misses the marginal of a beta row with a
    # loading near one (event 2's commercial row has w = 0.96) at the 1e-4
    # level; the rule's error, not the two-point rows'.
    assert abs(meta.view_mean_error["ground_up"]) < 1e-3 * meta.analytic_aal
    # A two-point row beside an atom reconciles exactly whatever the rule.
    _, m_exact = run_multi(multi, [("e", "P", 0.1, *HOMER[2][0]), ("e", "A", 0.1, 150.0, 0.0, 0.0, 500.0)],
                           bucket_size=h, views=("ground_up",), z_nodes=64,
                           z_rule="gauss_hermite")
    assert abs(m_exact.view_mean_error["ground_up"]) < 1e-12 * m_exact.analytic_aal


# ---------------------------------------------------------------------------
# [coupling] spec test 25: the coupling rules (2.8.0).

def comonotone_by_inverse_cdf(marginals, steps=1_000_000):
    """The comonotone sum of bucketed marginals by inverse cdf at the midpoints
    of a fine uniform grid: an independent check of the merged breakpoints."""
    u = (np.arange(steps) + 0.5) / steps
    total = np.zeros(sum(m.size - 1 for m in marginals) + 1)
    index = sum(np.searchsorted(np.cumsum(m), u) for m in marginals)
    np.add.at(total, np.minimum(index, total.size - 1), 1.0 / steps)
    return total


def homer_rows(terms=False):
    rows = []
    for event, (personal, commercial) in HOMER.items():
        rows.append((f"e{event}", "P", HOMER_RATES[event], *personal))
        rows.append((f"e{event}", "C", HOMER_RATES[event], *commercial))
    return rows


HOMER_TERMS = ["unit P 2000 xs 100 occurrence net of 50% po 1000 xs 500", "unit C 1500 xs 0"]


def test_coupling_independent_is_the_convolution(multi):
    lam = 0.03
    rows = [("e", "a", lam, *U1), ("e", "b", lam, *U2)]
    views, meta = run_multi(multi, rows, bucket_size=H, views=("ground_up",), coupling="independent")
    expected = lam * np.convolve(direct_marginal(U1, N1), direct_marginal(U2, N2))
    assert meta.coupling == "independent"
    assert np.abs(views["ground_up"] - expected).max() < 1e-12 * lam
    pmf, _, _, j_meta = multi.joint(U1, U2, bucket_size=H, n1=N1, n2=N2, coupling="independent")
    assert j_meta.coupling == "independent" and j_meta.copula_normal_correlation == 0.0
    assert np.abs(pmf - np.outer(pmf.sum(axis=1), pmf.sum(axis=0))).max() < 1e-15
    assert np.abs(pmf.sum(axis=1) - direct_marginal(U1, N1)).max() < 1e-13


def test_coupling_comonotone_is_exact(multi):
    # The two-unit joint: marginals exact, total the comonotone sum of the
    # bucketed marginals, checked by inverse cdf on a fine uniform grid.
    pmf, _, _, meta = multi.joint(U1, U2, bucket_size=H, n1=N1, n2=N2, coupling="comonotone")
    assert meta.coupling == "comonotone" and meta.copula_normal_correlation == 1.0
    m1, m2 = direct_marginal(U1, N1), direct_marginal(U2, N2)
    assert np.abs(pmf.sum(axis=1) - m1).max() < 1e-12
    assert np.abs(pmf.sum(axis=0) - m2).max() < 1e-12
    idx = np.add.outer(np.arange(N1), np.arange(N2)).ravel()
    total = np.bincount(idx, weights=pmf.ravel(), minlength=N1 + N2 - 1)
    assert np.abs(np.cumsum(total) - np.cumsum(comonotone_by_inverse_cdf([m1, m2]))).max() < 2e-6
    assert math.isclose(total.sum(), 1.0, abs_tol=1e-12)
    # The joint sits on the comonotone curve: every cell with mass lies where
    # the two cumulative probabilities overlap, so the mass-weighted count of
    # cells per row stays near one per breakpoint.
    assert np.count_nonzero(pmf) < N1 + N2
    # The views route agrees with the joint's total, and with a two-point and
    # a comonotone unit in the mix stays exact in mass and marginals.
    lam = 0.02
    views, v_meta = run_multi(multi, [("e", "a", lam, *U1), ("e", "b", lam, *U2)], bucket_size=H,
                              views=("ground_up",), coupling="comonotone")
    assert np.abs(views["ground_up"] - lam * total).max() < 1e-12 * lam
    assert v_meta.coupling == "comonotone"
    rows = [("e", "a", lam, 30.0, 50.0, 30.0, 100.0), ("e", "b", lam, *U2),
            ("e", "c", lam, 200.0, 0.0, 250.0, 1000.0)]
    views, v_meta = run_multi(multi, rows, terms=TOWER, bucket_size=H, coupling="comonotone")
    for view in v_meta.views:
        assert math.isclose(views[view].sum(), lam, rel_tol=1e-12), view
        # The comonotone sum keeps the bucketed marginals' means exactly;
        # those differ from the analytic means by the bucketing of the beta
        # at h = 10 (cdf differences are exact in probability, not in mean).
        assert abs(v_meta.view_mean_error[view]) < 1e-3 * abs(v_meta.analytic_view_mean[view]) + 1e-12, view
    marginal_mean = sum(lam * (direct_marginal(rm.Unit(*r[3:]), rm.grid_length_for(r[6], H)) @ (H * np.arange(rm.grid_length_for(r[6], H))))
                        for r in rows)
    assert math.isclose(views["ground_up"] @ (H * np.arange(v_meta.grid_length)), marginal_mean, rel_tol=1e-12)


def test_coupling_comonotone_common_shape_is_the_rolled_beta(multi):
    # Proposition 6, by the exact path: equal normalized moments and loading
    # one make the total the rolled row's beta (bucket rounding aside).
    h = 5.0
    rows = [("e", "p", 0.01, 200.0, 0.0, 250.0, 1000.0), ("e", "q", 0.01, 400.0, 0.0, 500.0, 2000.0)]
    views, meta = run_multi(multi, rows, bucket_size=h, views=("ground_up",), coupling="comonotone")
    total = views["ground_up"] / meta.total_rate
    xs = h * np.arange(total.size)
    rolled = direct_marginal(rm.Unit(600.0, 0.0, 750.0, 3000.0), total.size, h)
    mean_t, sd_t, q_t = moments(xs, total)
    mean_r, sd_r, q_r = moments(xs, rolled)
    assert math.isclose(mean_t, 600.0, rel_tol=1e-3)  # the betas' bucketing at h = 5
    assert math.isclose(sd_t, sd_r, rel_tol=1e-3) and math.isclose(q_t, q_r, rel_tol=5e-3)
    # The comonotone sum of the bucketed marginals is coarser than the
    # bucketing of the rolled beta where both U-shaped units pile mass at
    # the origin (both in bucket 0 lands at 0, not spread over 0 and 1): a
    # bucketing effect of order h in the first cells, 1.6e-2 here.
    assert np.abs(np.cumsum(total) - np.cumsum(rolled)).max() < 2e-2


def test_coupling_comonotone_matches_the_article(multi):
    # The multi-unit article's `tbl-couplings`: case A under loading one,
    # built there on 4,096 midpoint nodes with each unit's whole sd on the
    # factor; the exact path agrees to that rule's error.
    a1, a2 = rm.Unit(200.0, 0.0, 250.0, 1000.0), rm.Unit(500.0, 0.0, 500.0, 2000.0)
    article = rm.secondary_two_units([a1, a2], H, N1, N2, n_z=4096, z_rule="midpoint")
    pmf, _, _, _ = multi.joint(U1, U2, bucket_size=H, n1=N1, n2=N2, coupling="comonotone")
    xs = H * np.arange(N1 + N2 - 1)
    idx = np.add.outer(np.arange(N1), np.arange(N2)).ravel()
    total = np.bincount(idx, weights=pmf.ravel(), minlength=N1 + N2 - 1)
    mean_a, sd_a, q_a = moments(*article.sum_distribution())
    mean_t, sd_t, q_t = moments(xs, total)
    assert math.isclose(mean_t, mean_a, rel_tol=1e-3)
    assert math.isclose(sd_t, sd_a, rel_tol=1e-3) and math.isclose(q_t, q_a, rel_tol=5e-3)


def test_coupling_homer_is_the_rolled_table(multi):
    rows = homer_rows()
    views, meta = run_multi(multi, rows, terms=HOMER_TERMS, bucket_size=H, coupling="homer")
    assert meta.coupling == "homer" and meta.homer_excess_mean is not None
    expected = np.zeros(meta.grid_length)
    gross = np.zeros(meta.grid_length)
    excess = 0.0
    terms = {t.unit_id: t for t in map(parse_terms_line, HOMER_TERMS)}
    for event, (personal, commercial) in HOMER.items():
        lam = HOMER_RATES[event]
        rolled = rm.rolled_unit([rm.Unit(*personal), rm.Unit(*commercial)])
        base = rm.marginal_pmf(rm.prepare_unit(rolled, "r"), H, rm.grid_length_for(rolled.max_loss, H))
        expected[:base.size] += lam * base
        shares = [personal[0] / rolled.mean_loss, commercial[0] / rolled.mean_loss]
        for i, p in enumerate(base):
            x = H * i
            value = sum(float(terms[u].view("gross", s * x)) for u, s in zip("PC", shares))
            rm._deposit(gross, value, lam * p, 1.0 / H)
            excess += lam * p * sum(max(s * x - m, 0.0) for s, m in zip(shares, (personal[3], commercial[3])))
    assert np.abs(views["ground_up"] - expected).max() < 1e-12 * meta.total_rate
    assert np.abs(views["gross"] - gross).max() < 1e-12 * meta.total_rate
    # The allocations sum to the rolled loss, so the ground-up mean is the
    # rolled row's to the bucketing of its beta by cdf differences (3e-4 at
    # h = 10); the other views reconcile to the scaled layer means of the
    # rolled row to the same bucketing tolerance.
    assert abs(meta.view_mean_error["ground_up"]) < 1e-3 * meta.analytic_aal
    for view in ("gross", "ceded", "net"):
        assert abs(meta.view_mean_error[view]) < 2e-2 * abs(meta.analytic_view_mean[view]), view
    # The published rule allocates beyond the exposures on this table (event
    # 1's commercial line holds 40% of the mean and 25% of the exposure).
    assert meta.homer_excess_mean > 0.0
    assert math.isclose(meta.homer_excess_mean, excess, rel_tol=2e-2)
    # The one-row survival grid of the rolled table agrees to its
    # interpolation tolerance.
    import eltq

    rolled_rows = [rm.rolled_unit([rm.Unit(*p), rm.Unit(*c)]) for p, c in HOMER.values()]
    _, secondary, _ = eltq.quantize_arrays(
        np.array(list(HOMER_RATES.values())), np.array([u.max_loss for u in rolled_rows]),
        np.array([u.mean_loss for u in rolled_rows]), np.array([u.sd_indep for u in rolled_rows]),
        np.array([u.sd_corr for u in rolled_rows]), bucket_size=H, grid_length=meta.grid_length)
    assert np.abs(views["ground_up"] - secondary).max() < 1e-6 * meta.total_rate
    # The two-unit joint under homer: the marginal of unit k is the rolled
    # row scaled by s_k, and a grid short of s_k M_R is refused.
    p1, c1 = HOMER[1]
    rolled = rm.rolled_unit([rm.Unit(*p1), rm.Unit(*c1)])
    n_c = rm.grid_length_for(c1[0] / rolled.mean_loss * rolled.max_loss, H)
    with pytest.raises(multi.overflow_error):
        multi.joint(p1, c1, bucket_size=H, n1=301, n2=101, coupling="homer")
    pmf, x1, x2, j_meta = multi.joint(p1, c1, bucket_size=H, n1=301, n2=n_c, coupling="homer")
    assert j_meta.coupling == "homer" and math.isclose(pmf.sum(), 1.0, abs_tol=1e-12)
    # Means to the bucketing of the rolled row (its allocation is deposited
    # mean-exactly, the rolled row itself by cdf differences).
    assert math.isclose(pmf.sum(axis=1) @ x1, p1[0], rel_tol=1e-3)
    assert math.isclose(pmf.sum(axis=0) @ x2, c1[0], rel_tol=1e-3)
    assert x2[np.nonzero(pmf.sum(axis=0))[0][-1]] > c1[3]  # allocated beyond the exposure


def article_total_sd(u1, u2, c, h=H):
    """The article's ``sd_at`` (chunk `tbl-matched`): the sd of the event total
    at Burke's loadings scaled by ``c`` (capped at one), on 64 Gauss-Hermite
    nodes."""
    from scipy.special import ndtr, ndtri

    def cond(unit, w, n, z):
        edges = (np.arange(n + 1) - 0.5) * h
        cdf = article_unit_cdf(unit, edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = ndtri(np.clip(cdf, 0, 1))
            out = ndtr((y[None, :] - w * z[:, None]) / math.sqrt(1 - w**2)) if w < 1 \
                else (y[None, :] >= z[:, None]).astype(float)
        out[:, 0], out[:, -1] = 0.0, 1.0
        return np.diff(out, axis=1)

    z, w = np.polynomial.hermite_e.hermegauss(64)
    w = w / w.sum()
    loads = [min(c * u[2] / math.hypot(u[1], u[2]), 1.0) for u in (u1, u2)]
    n1, n2 = int(u1[3] / h) + 1, int(u2[3] / h) + 1
    joint = (cond(u1, loads[0], n1, z) * w[:, None]).T @ cond(u2, loads[1], n2, z)
    idx = np.add.outer(np.arange(n1), np.arange(n2)).ravel()
    total = np.bincount(idx, weights=joint.ravel(), minlength=n1 + n2 - 1)
    xs = h * np.arange(total.size)
    mean = total @ xs
    return math.sqrt(max(total @ xs**2 - mean**2, 0.0))


def test_coupling_matched(multi):
    # Spec test 25, `matched`: the variance of each event total equals the
    # roll-up convention's (sd_I,R + sd_C,R)^2 within 1e-6 relative, or the
    # event is reported with its shortfall at saturated loadings; the
    # multi-unit article's `tbl-matched` (case A and Homer's events) within
    # the difference of its 64 Gauss-Hermite nodes from the stratified rule.
    from scipy import optimize

    cases = {"A": (U1, U2), **{f"H{e}": (rm.Unit(*p), rm.Unit(*c)) for e, (p, c) in HOMER.items()}}
    for name, (a, b) in cases.items():
        n1, n2 = int(a.max_loss / H) + 1, int(b.max_loss / H) + 1
        pmf, x1, x2, meta = multi.joint(a, b, bucket_size=H, n1=n1, n2=n2, coupling="matched")
        assert meta.coupling == "matched" and meta.matched is not None
        target = math.hypot(a.sd_indep, b.sd_indep) + a.sd_corr + b.sd_corr
        idx = np.add.outer(np.arange(n1), np.arange(n2)).ravel()
        total = np.bincount(idx, weights=pmf.ravel(), minlength=n1 + n2 - 1)
        _, sd, _ = moments(H * np.arange(total.size), total)
        assert math.isclose(meta.matched["sd_target"], target, rel_tol=1e-12)
        assert math.isclose(meta.matched["sd_achieved"], sd, rel_tol=1e-9), name
        w_max = max(a.sd_corr / math.hypot(a.sd_indep, a.sd_corr), b.sd_corr / math.hypot(b.sd_indep, b.sd_corr))
        if name == "H2":
            # Event 2: the personal line is a two-point row and the commercial
            # line nearly comonotone with the factor already; the target is
            # out of reach and the loadings saturate.
            assert meta.matched["saturated"] and sd < target
            assert math.isclose(meta.matched["scale"], 1.0 / w_max, rel_tol=1e-12)
        else:
            assert not meta.matched["saturated"]
            assert math.isclose(sd * sd, target * target, rel_tol=1e-6), name
            # The article's root by its own construction.
            c_article = optimize.brentq(lambda c: article_total_sd(tuple(a), tuple(b), c) - target,
                                        0.0, 1.0 / w_max, xtol=1e-6)
            assert math.isclose(meta.matched["scale"], c_article, rel_tol=5e-3), (name, c_article)
    # The views route: the shortfall list names event 2 only, the other
    # events reconcile in variance, and the per-event scale agrees with the
    # joint's.
    rows = homer_rows()
    views, v_meta = run_multi(multi, rows, bucket_size=H, views=("ground_up",), coupling="matched")
    assert v_meta.coupling == "matched"
    assert [e["event_id"] for e in v_meta.matched_shortfall] == ["e2"]
    assert math.isclose(views["ground_up"].sum(), v_meta.total_rate, rel_tol=1e-12)
    _, m_burke = run_multi(multi, rows, bucket_size=H, views=("ground_up",))
    assert m_burke.matched_shortfall is None


def test_coupling_errors_and_echo(multi):
    rows = [("e", "a", 0.01, *U1), ("e", "b", 0.01, *U2)]
    with pytest.raises(multi.base_error):
        run_multi(multi, rows, bucket_size=H, coupling="gumbel")
    with pytest.raises(multi.base_error):
        multi.joint(U1, U2, bucket_size=H, n1=N1, n2=N2, coupling="gumbel")
    _, meta = run_multi(multi, rows, bucket_size=H)
    assert meta.coupling == "burke" and meta.homer_excess_mean is None


# ---------------------------------------------------------------------------
# [joint] spec test 29: the occurrence outputs (2.9.0).

def run_pair(impl, elt, units, **params):
    """The pair joint on either backend: ``(pmf, x_u, x_w, mass, events)``."""
    cols = multi_rows(elt) if isinstance(elt, list) else {
        c: (elt[c].to_numpy(dtype=np.float64) if c not in ("event_id", "unit_id")
            else list(elt[c].astype(str))) for c in MULTI_COLUMNS}
    if impl.name == "reference":
        view = params.pop("view", "ground_up")
        _, _, extras = rm.multiunit_views_reference(*(cols[c] for c in MULTI_COLUMNS), views=(view,),
                                                     pair=units, **params)
        return extras["pair"]
    import eltq

    pmf, x_u, x_w, meta = eltq.occurrence_pair_joint(*(cols[c] for c in MULTI_COLUMNS), units=units,
                                                     **params)
    return pmf, x_u, x_w, meta.mass, meta.events


def run_moments(impl, elt, **params):
    """The first-moment vectors on either backend: ``(dict, x, identity_error)``."""
    cols = multi_rows(elt) if isinstance(elt, list) else {
        c: (elt[c].to_numpy(dtype=np.float64) if c not in ("event_id", "unit_id")
            else list(elt[c].astype(str))) for c in MULTI_COLUMNS}
    if impl.name == "reference":
        view = params.pop("view", "ground_up")
        _, meta, extras = rm.multiunit_views_reference(*(cols[c] for c in MULTI_COLUMNS), views=(view,),
                                                        first_moments=True, **params)
        moments, identity = extras["first_moments"]
        return moments, params.get("bucket_size") * np.arange(meta.grid_length), identity
    import eltq

    moments, x, meta = eltq.occurrence_unit_first_moment(*(cols[c] for c in MULTI_COLUMNS), **params)
    return moments, x, meta.identity_error


def antidiagonal_sums(pmf):
    n1, n2 = pmf.shape
    idx = np.add.outer(np.arange(n1), np.arange(n2)).ravel()
    return np.bincount(idx, weights=pmf.ravel(), minlength=n1 + n2 - 1)


@pytest.mark.parametrize("coupling", ["burke", "comonotone", "independent", "homer"])
def test_pair_joint_marginals_and_antidiagonals(multi, coupling):
    # A two-unit portfolio (Homer's table): the marginals of S_uw are each
    # unit's own occurrence severity and the antidiagonal sums the total of
    # the same run; under homer the units are the rolled row's allocations,
    # so the first identity is checked against the pair's own marginals.
    rows = homer_rows()
    pmf, x_u, x_w, mass, events = run_pair(multi, rows, ("P", "C"), bucket_size=H, coupling=coupling)
    assert events == 3 and math.isclose(mass, 0.7, rel_tol=1e-12)
    total, meta = run_multi(multi, rows, bucket_size=H, views=("ground_up",), coupling=coupling)
    anti = antidiagonal_sums(pmf)
    n = min(anti.size, meta.grid_length)
    x_t = H * np.arange(anti.size)
    if coupling == "homer":
        # The two allocations of a rolled bucket are deposited on their own
        # axes, so their index sum spreads the total by up to a bucket; the
        # mass and the mean agree.
        assert math.isclose(anti.sum(), total["ground_up"].sum(), rel_tol=1e-12)
        assert math.isclose(anti @ x_t, meta.view_mean["ground_up"], rel_tol=1e-10)
        assert np.abs(np.cumsum(anti[:n]) - np.cumsum(total["ground_up"][:n])).max() < 2e-2 * mass
    else:
        assert np.abs(anti[:n] - total["ground_up"][:n]).max() < 1e-12 * mass
        assert abs(anti[n:].sum()) + abs(total["ground_up"][n:].sum()) < 1e-12 * mass
    if coupling != "homer":
        for unit, axis, x in (("P", 1, x_u), ("C", 0, x_w)):
            alone, m_alone = run_multi(multi, [r for r in rows if r[1] == unit], bucket_size=H,
                                       views=("ground_up",), coupling=coupling)
            marginal = pmf.sum(axis=axis)
            k = min(marginal.size, m_alone.grid_length)
            assert np.abs(marginal[:k] - alone["ground_up"][:k]).max() < 1e-12 * mass, unit


def test_pair_joint_with_terms_and_absent_units(multi):
    # Three units, the pair's view after terms; e2 exposes a but not b (b at
    # zero), e3 exposes neither (nothing), so the mass is the rate of e1
    # and e2.
    rows = [("e1", "a", 0.01, *U1), ("e1", "b", 0.01, *U2), ("e1", "c", 0.01, 300.0, 50.0, 100.0, 800.0),
            ("e2", "a", 0.02, 250.0, 0.0, 200.0, 1000.0), ("e2", "c", 0.02, 100.0, 120.0, 0.0, 500.0),
            ("e3", "c", 0.005, 400.0, 80.0, 60.0, 1200.0)]
    terms = [TOWER, "unit b 1000 xs 0 occurrence net of 100 po 400 xs 100"]
    pmf, x_u, x_w, mass, events = run_pair(multi, rows, ("a", "b"), view="gross", terms=terms,
                                           bucket_size=H)
    assert events == 2 and math.isclose(mass, 0.03, rel_tol=1e-12)
    assert math.isclose(pmf.sum(), mass, rel_tol=1e-12)
    # e2's contribution sits in column 0 (b absent) and is a's gross view,
    # the marginal of a's conditional rows (unit c marginalizes out), which
    # a alone in e2 gives directly.
    only_a, m_a = run_multi(multi, rows[3:4], terms=terms, bucket_size=H, views=("gross",))
    pmf_e1, _, _, _, _ = run_pair(multi, rows[:3], ("a", "b"), view="gross", terms=terms, bucket_size=H)
    assert pmf_e1.shape == pmf.shape
    column0 = pmf[:, 0] - pmf_e1[:, 0]
    k = min(column0.size, m_a.grid_length)
    assert np.abs(column0[:k] - only_a["gross"][:k]).max() < 1e-12 * mass
    # The gross view of a never exceeds its limit of 750 and b's its 1000.
    assert x_u[np.nonzero(pmf.sum(axis=1))[0][-1]] <= 750.0
    assert x_w[np.nonzero(pmf.sum(axis=0))[0][-1]] <= 1000.0


@pytest.mark.parametrize("coupling", ["burke", "comonotone", "homer"])
def test_first_moments_identity_and_backends(multi, coupling):
    df, terms = load_multi_fixture()
    for view in ("ground_up", "net"):
        moments, x, identity = run_moments(multi, df, view=view, terms=terms, bucket_size=H,
                                           coupling=coupling)
        total, meta = run_multi(multi, df, terms=terms, bucket_size=H, views=(view,), coupling=coupling)
        s = total[view]
        summed = sum(moments.values())
        assert set(moments) == set(meta.unit_ids)
        assert np.abs(summed - x * s).max() < 1e-10 * meta.analytic_view_mean[view], (view, coupling)
        assert identity < 1e-10 * meta.analytic_view_mean[view]
        assert math.isclose(summed.sum(), meta.view_mean[view], rel_tol=1e-10)
        assert all(m.min() >= -1e-15 * meta.analytic_view_mean[view] for m in moments.values())


def test_occurrence_outputs_agree_across_backends():
    reference, package = MULTI_IMPLS
    df, terms = load_multi_fixture()
    for coupling in ("burke", "comonotone", "homer"):
        want = run_pair(reference, df, ("a", "b"), view="gross", terms=terms, bucket_size=H, coupling=coupling)
        got = run_pair(package, df, ("a", "b"), view="gross", terms=terms, bucket_size=H, coupling=coupling)
        assert got[0].shape == want[0].shape
        assert np.abs(got[0] - want[0]).max() < 1e-12 * want[3], coupling
        assert math.isclose(got[3], want[3], rel_tol=1e-12) and got[4] == want[4]
        m_want, _, _ = run_moments(reference, df, terms=terms, bucket_size=H, coupling=coupling)
        m_got, _, _ = run_moments(package, df, terms=terms, bucket_size=H, coupling=coupling)
        for unit in m_want:
            assert np.abs(m_got[unit] - m_want[unit]).max() < 1e-12 * sum(m.sum() for m in m_want.values()), (unit, coupling)


def poisson_compound(severity, h, size):
    """The annual density of a compound Poisson with rate-weighted severity
    ``severity`` on the grid, by FFT with zero padding to ``size``."""
    rate = severity.sum()
    spectrum = np.fft.rfft(severity, size)
    return np.fft.irfft(np.exp(spectrum - rate), size)


def test_annual_conditional_mean_sums_to_the_total(multi):
    # The Mecke identity: kappa_u from the first-moment vectors and a numpy
    # compound Poisson of the view's severity; the kappa sum to the total
    # wherever the annual density is at least 1e-12.
    rows = homer_rows()
    import eltq

    moments, x, identity = run_moments(multi, rows, bucket_size=H)
    total, meta = run_multi(multi, rows, bucket_size=H, views=("ground_up",))
    # A 2^16 grid: the compound of a severity reaching 17,000 wraps on a
    # smaller one, and the identity cannot hold against a wrapped density.
    f_a = poisson_compound(total["ground_up"], H, 1 << 16)
    kappa = eltq.annual_conditional_mean(moments, f_a)
    s = H * np.arange(f_a.size)
    # Dividing by f_A amplifies the convolution's absolute rounding (about
    # 1e-16 of the total mean), so the identity is read where f_A >= 1e-7.
    live = f_a >= 1e-7
    assert np.abs(sum(kappa.values())[live] - s[live]).max() < 1e-8 * s[live].max()
    assert np.all(np.isnan(kappa["P"][~(f_a > 0.0)])) if (~(f_a > 0.0)).any() else True
    # The personal line's share of a small total: event 1's commercial line
    # at its two-point atom of 1,000 means the personal share near 1,000 is
    # well below one half (the article's reading of its figure).
    near = int(1000 / H)
    assert kappa["P"][near] < 0.5 * s[near]
    # One unit returns one array.
    one = eltq.annual_conditional_mean(moments["P"], f_a)
    assert np.array_equal(np.nan_to_num(one), np.nan_to_num(kappa["P"]))


def _aggregate():
    try:
        import aggregate  # noqa: F401
        from aggregate.bivariate import BivariateAggregate
    except ImportError:
        pytest.skip("the handoff tests need the aggregate package")
    return BivariateAggregate


def test_handoff_route_two_against_route_three():
    # Route 2 (the two-dimensional compound of S_uw under the pair's own
    # Poisson count) summed along its antidiagonals equals route 3 (the
    # one-dimensional compound of the total), within the aggregate package's
    # grid tolerance (the article reaches 1e-10).
    BivariateAggregate = _aggregate()
    import eltq

    rows = homer_rows()
    cols = multi_rows(rows)
    pmf, x_u, x_w, meta = eltq.occurrence_pair_joint(*(cols[c] for c in MULTI_COLUMNS), units=("P", "C"),
                                                     bucket_size=H)
    total, _ = eltq.multiunit_views(*(cols[c] for c in MULTI_COLUMNS), bucket_size=H, views=("ground_up",))
    bv = BivariateAggregate("route 2", mode="discrete", dbv_xs=x_u, dbv_ys=x_w, dbv_S=pmf / meta.mass,
                            freq_name="poisson", exp_en=meta.mass)
    bv.update(log2=(11, 11), bs=(H, H))
    annual_joint = np.asarray(bv.density)
    route2 = antidiagonal_sums(annual_joint)
    route3 = poisson_compound(total["ground_up"], H, 4096)
    n = min(route2.size, route3.size, 2048)
    assert np.abs(np.cumsum(route2[:n]) - np.cumsum(route3[:n])).max() < 1e-9


def test_annual_conditional_mean_against_exeqa():
    # kappa_u by the Mecke identity against exeqa_df of the two-dimensional
    # compound of the sheared S_uw, where the two-dimensional grid does not
    # wrap.
    BivariateAggregate = _aggregate()
    import eltq

    rows = homer_rows()
    cols = multi_rows(rows)
    pmf, x_u, x_w, meta = eltq.occurrence_pair_joint(*(cols[c] for c in MULTI_COLUMNS), units=("P", "C"),
                                                     bucket_size=H)
    n_u, n_w = pmf.shape
    sheared = np.zeros((n_u, n_u + n_w - 1))
    for a in range(n_u):
        sheared[a, a:a + n_w] = pmf[a]
    x_t = H * np.arange(n_u + n_w - 1)
    bv = BivariateAggregate("sheared", mode="discrete", dbv_xs=x_u, dbv_ys=x_t, dbv_S=sheared / meta.mass,
                            freq_name="poisson", exp_en=meta.mass)
    bv.update(log2=(10, 11), bs=(H, H))
    kappa_df = bv.exeqa_df(axis=1)
    moments, x, _ = run_moments(MULTI_IMPLS[1], rows, bucket_size=H)
    total, _ = eltq.multiunit_views(*(cols[c] for c in MULTI_COLUMNS), bucket_size=H, views=("ground_up",))
    f_a = poisson_compound(total["ground_up"], H, 2048)
    kappa = eltq.annual_conditional_mean(moments["P"], f_a)
    kd = kappa_df[(kappa_df.p > 1e-9) & (kappa_df.index < 4000)]
    index = (kd.index.to_numpy() / H).round().astype(int)
    assert np.abs(kappa[index] - kd.exeqa_X.to_numpy()).max() < 1e-6 * 4000


# ---------------------------------------------------------------------------
# [one-unit] spec test 27: one-unit events leave the mixture (2.7.0).

ONE_UNIT_ROWS = {
    "beta": ("e3", "b", 0.005, 400.0, 80.0, 60.0, 1200.0),   # the fixture's e3
    "two_point": ("t", "a", 0.01, 30.0, 50.0, 30.0, 100.0),
    "atom": ("a", "c", 0.02, 333.0, 0.0, 0.0, 1000.0),
    "comonotone": ("k", "d", 0.01, 200.0, 0.0, 250.0, 1000.0),
}


@pytest.mark.parametrize("rule", ["stratified", "midpoint", "gauss_hermite"])
def test_one_unit_event_is_the_direct_bucketing(multi, rule):
    for kind, row in ONE_UNIT_ROWS.items():
        views, meta = run_multi(multi, [row], terms=TOWER if row[1] == "a" else None,
                                bucket_size=10.0, z_nodes=8, z_rule=rule)
        lam, unit = row[2], rm.Unit(*row[3:])
        n = rm.grid_length_for(rm.prepare_unit(unit, "u").support, 10.0)
        direct = lam * direct_marginal(unit, n)
        assert meta.one_unit_events == 1 and meta.valid_events == 1, (kind, rule)
        assert np.abs(views["ground_up"][:n] - direct).max() < 1e-12 * lam, (kind, rule)
        assert math.isclose(meta.view_mass["ground_up"], lam, rel_tol=1e-12)
        if row[1] == "a":
            # The compiled view maps of that row, no z loop: the gross view is
            # the direct bucketing pushed through the tower.
            gross = np.zeros(meta.grid_length)
            for k, p in enumerate(direct):
                rm._deposit(gross, parse_terms_line(TOWER).view("gross", 10.0 * k), p, 0.1)
            assert np.abs(views["gross"] - gross).max() < 1e-12 * lam, (kind, rule)


def test_one_unit_event_against_the_survival_grid(multi):
    # The one-row survival grid of section 5A on the same row, built on its
    # own nodes, agrees to its interpolation tolerance (6e-8 of the rate
    # measured on the fixture's e3 at h = 10, spec test 27).
    import eltq

    row = ONE_UNIT_ROWS["beta"]
    views, meta = run_multi(multi, [row], bucket_size=10.0)
    raw, secondary, q_meta = eltq.quantize_arrays(
        np.array([row[2]]), np.array([row[6]]), np.array([row[3]]), np.array([row[4]]),
        np.array([row[5]]), bucket_size=10.0, grid_length=meta.grid_length)
    assert np.abs(views["ground_up"] - secondary).max() < 1e-6 * row[2]
    assert math.isclose(secondary.sum(), views["ground_up"].sum(), rel_tol=1e-12)


def test_mixed_portfolio_is_the_sum_of_its_parts(multi):
    df, terms = load_multi_fixture()
    one = df[df.event_id == "e3"]
    rest = df[df.event_id != "e3"]
    whole, m_whole = run_multi(multi, df, terms=terms, bucket_size=10.0)
    part_one, m_one = run_multi(multi, one, terms=terms, bucket_size=10.0,
                                grid_length=m_whole.grid_length)
    part_rest, m_rest = run_multi(multi, rest, terms=terms, bucket_size=10.0,
                                  grid_length=m_whole.grid_length)
    # e3 has one unit; e6 keeps one after its dropped row (mean above max).
    assert (m_whole.one_unit_events, m_one.one_unit_events, m_rest.one_unit_events) == (2, 1, 1)
    for view in m_whole.views:
        assert np.abs(whole[view] - part_one[view] - part_rest[view]).max() < 1e-12 * m_whole.total_rate


# ---------------------------------------------------------------------------
# [tail-support] the grid covers each unit's support rounded up to the grid.

TWO_ATOMS = [("e", "a", 1.0, 13.0, 0.0, 0.0, 20.0), ("e", "b", 1.0, 13.0, 0.0, 0.0, 20.0)]


def test_off_grid_atoms_keep_the_tail(multi):
    # Each atom at 13 deposits [0, 0.7, 0.3] on h = 10, so the total reaches
    # 40; sizing by 13 + 13 = 26 gave 4 nodes and dropped the 0.09 at 40.
    views, meta = run_multi(multi, TWO_ATOMS, bucket_size=10.0, views=("ground_up",))
    assert meta.grid_length == 5
    assert np.allclose(views["ground_up"], [0.0, 0.0, 0.49, 0.42, 0.09], rtol=0.0, atol=1e-15)
    assert math.isclose(meta.view_mass["ground_up"], 1.0, rel_tol=1e-15)
    assert math.isclose(meta.view_mean["ground_up"], 26.0, rel_tol=1e-14)
    # An explicit grid that covers 26 but not the span of 4 buckets is refused.
    with pytest.raises(multi.overflow_error) as info:
        run_multi(multi, TWO_ATOMS, bucket_size=10.0, views=("ground_up",), grid_length=4)
    assert info.value.required_length == 5


def test_off_grid_beta_supports_keep_full_mass(multi):
    # Three betas on [0, 16] at h = 10 each span 2 buckets: 7 nodes, not the
    # 6 that a support of 48 suggests. The mean sits high so the top bucket
    # carries visible mass.
    rows = [("e", unit, 1.0, 12.0, 2.0, 1.0, 16.0) for unit in "abc"]
    views, meta = run_multi(multi, rows, bucket_size=10.0)
    assert meta.grid_length == 7
    for view in meta.views:
        assert math.isclose(views[view].sum(), 1.0, abs_tol=1e-12), view
    assert views["ground_up"][6] > 1e-6


# ---------------------------------------------------------------------------
# [fft-ceded] convolution paths agree when bare units sit beside a termed unit.

BARE_BLOCK_ROWS = [
    ("e", "a", 0.01, 300.0, 40.0, 30.0, 1000.0),   # termed
    ("e", "b", 0.01, 200.0, 50.0, 60.0, 800.0),    # bare beta
    ("e", "c", 0.01, 250.0, 0.0, 120.0, 900.0),    # bare comonotone (sd_indep = 0)
]
BARE_BLOCK_TERMS = "unit a 1000 xs 0 occurrence net of 50% po 500 xs 0"


@pytest.mark.parametrize("views", [("ground_up",), ("gross",), ("ceded",), ("net",),
                                   ("ceded", "net")])
def test_convolution_paths_agree_with_bare_block(views):
    # Ceded never uses the shared bare block; a ceded-only FFT run built the
    # block from spectra it never computed and panicked (1.11.2).
    import eltq

    reference, package = MULTI_IMPLS
    params = dict(terms=BARE_BLOCK_TERMS, bucket_size=10.0, views=views, z_nodes=16)
    want, _ = run_multi(reference, BARE_BLOCK_ROWS, **params)
    paths = {c: run_multi(package, BARE_BLOCK_ROWS, convolution=c, **params)[0]
             for c in ("direct", "fft", "auto")}
    for view in views:
        for convolution in ("fft", "auto"):
            assert np.allclose(paths[convolution][view], paths["direct"][view],
                               rtol=0.0, atol=1e-13), (view, convolution)
        for convolution, got in paths.items():
            assert np.allclose(got[view], want[view], rtol=1e-9, atol=1e-13 * 0.01), \
                (view, convolution)
    if eltq.BACKEND == "rust":
        _, meta = run_multi(package, BARE_BLOCK_ROWS, convolution="fft", **params)
        assert meta.engine["fft_paths"] == 1


# ---------------------------------------------------------------------------
# [fft-nonnegative] the FFT path returns nonnegative weights.

def concentrated_portfolio(events=200, seed=20261021):
    """Seeded ELT of ``events`` events x units a, b, d, e of the fixture terms
    (policy and reinsurance) with tight betas: most bins inside an event's
    FFT range are exactly zero, which is where inverse-transform round-off
    used to come out negative."""
    rng = np.random.default_rng(seed)
    units = np.array(list("abde"), dtype=object)
    n = events * units.size
    m = rng.choice([800.0, 1000.0, 1200.0, 2000.0], size=n)
    mu = m * rng.uniform(0.1, 0.5, size=n)
    sd = rng.uniform(0.02, 0.1, size=n) * np.sqrt(mu * (m - mu))
    corr_share = rng.uniform(0.0, 1.0, size=n)
    return {
        "event_id": list(np.repeat(np.arange(events), units.size)),
        "unit_id": list(np.tile(units, events)),
        "rate": np.repeat(rng.uniform(1e-3, 1e-2, size=events), units.size),
        "mean_loss": mu,
        "sd_indep": sd * (1.0 - corr_share),
        "sd_corr": sd * corr_share,
        "max_loss": m,
    }


def test_fft_weights_are_nonnegative():
    # 1.11.2 left ~1e-18 negatives in up to 171 bins per view on the
    # concentrated portfolio; the direct path never produced any.
    import eltq

    if eltq.BACKEND != "rust":
        pytest.skip("the FFT path is a property of the Rust engine")
    _, package = MULTI_IMPLS
    df, terms = load_multi_fixture()
    for elt in (df, concentrated_portfolio()):
        views, meta = run_multi(package, elt, terms=terms, bucket_size=10.0, convolution="fft")
        assert meta.engine["direct_paths"] == 0
        for view in meta.views:
            assert views[view].min() >= 0.0, view
            assert math.isclose(views[view].sum(), meta.total_rate, rel_tol=1e-12), view


# ---------------------------------------------------------------------------
# [Eltq-Multiunit-Narrow-Units] spec test 34: large parameters and atoms by
# resolution on the multi-unit route (`dev/done/plan-model-moments-and-
# windowed-views.md`).

#: mu = 1/2, sigma = 1e-5 on [0, 1]: alpha = beta of about 1.25e9, beyond the
#: continued fraction's regime, which gave NaN on the engine before 2.10.1.
NARROW_UNIT = ("e", "a", 0.1, 0.5, 1e-5, 0.0, 1.0)
PARTNER = ("e", "b", 0.1, 0.3, 0.0, 0.1, 1.0)


def test_narrow_unit_is_half_on_each_side_of_its_mean(multi):
    views, meta = run_multi(multi, [NARROW_UNIT], bucket_size=0.2, views=("ground_up",))
    grid = views["ground_up"]
    assert np.isfinite(grid).all()
    assert np.abs(grid[[2, 3]] - 0.05).max() < 1e-12 * 0.1
    assert grid[[0, 1, 4, 5]].max() == 0.0
    assert math.isclose(meta.view_mass["ground_up"], 0.1, rel_tol=1e-14)


def test_narrow_unit_beside_a_partner_agrees_across_backends():
    reference, package = MULTI_IMPLS
    out = [run_multi(impl, [NARROW_UNIT, PARTNER], bucket_size=0.2, views=("ground_up",))[0]["ground_up"]
           for impl in (reference, package)]
    assert all(np.isfinite(grid).all() for grid in out)
    assert np.abs(out[0] - out[1]).max() < 1e-12 * 0.1
    assert math.isclose(out[1].sum(), 0.1, rel_tol=1e-14)


def _unit_with_min_parameter(smallest, share):
    """``(mu, sigma)`` on ``M = 1`` of the beta with mean ``share`` and
    ``min(alpha, beta) = smallest``."""
    kappa = smallest / min(share, 1.0 - share)
    return share, math.sqrt(share * (1.0 - share) / (kappa + 1.0))


@pytest.mark.parametrize("coupling", ["burke", "comonotone", "independent", "homer"])
def test_large_parameter_sweep_agrees_across_backends(coupling):
    # Measured 2026-10-26: within 1.4e-12 of the rate at 1e4 (the engine's
    # fraction against SciPy), 6e-14 from 2e5 to 2e6, 1.3e-13 at 1e9,
    # 5.8e-11 at 1e14 (the quadrature kernel nearest its handover), 6e-14 at
    # 1e16 and 1.1e-13 at 1e24 (both the normal law).
    reference, package = MULTI_IMPLS
    for smallest in (1e4, 2e5, 2e6, 1e9, 1e14, 1e16, 1e24):
        for share in (0.5, 0.25):
            mu, sd = _unit_with_min_parameter(smallest, share)
            rows = [("e", "a", 0.1, mu, sd / 2.0, sd / 2.0, 1.0), ("e", "b", 0.1, 0.3, 0.05, 0.05, 1.0)]
            for rule in ("stratified", "midpoint", "gauss_hermite"):
                options = dict(bucket_size=0.2, views=("ground_up",), coupling=coupling, z_rule=rule)
                out = [run_multi(impl, rows, **options)[0]["ground_up"] for impl in (reference, package)]
                for grid in out:
                    assert np.isfinite(grid).all() and grid.min() >= 0.0, (smallest, share, rule)
                    assert math.isclose(grid.sum(), 0.1, rel_tol=1e-14), (smallest, share, rule)
                assert np.abs(out[0] - out[1]).max() < 1e-10 * 0.1, (smallest, share, rule)


def test_unit_below_the_resolution_is_an_atom(multi):
    tiny = ("e", "a", 0.1, 0.5, 1e-16, 0.0, 1.0)
    views, meta = run_multi(multi, [tiny, PARTNER], bucket_size=0.2, views=("ground_up",))
    assert diag_count(meta, "atom_by_resolution") == 1
    # The same unit stated as an atom, sigma zero, gives the same grid.
    exact = ("e", "a", 0.1, 0.5, 0.0, 0.0, 1.0)
    atom, atom_meta = run_multi(multi, [exact, PARTNER], bucket_size=0.2, views=("ground_up",))
    assert diag_count(atom_meta, "degenerate_atom") == 1
    assert np.abs(views["ground_up"] - atom["ground_up"]).max() == 0.0


@pytest.mark.parametrize("scale", [1e160, 1e-200])
def test_scaled_units_classify_as_at_unit_scale(multi, scale):
    # Spec test 10's row (M, mu, sigma) = (1, 0.4, 0.1) beside a partner: at
    # these scales sigma^2 and mu (M - mu) overflow or underflow, and the
    # reference's product-form test made the row a Bernoulli row (1e160) or
    # an atom (1e-200) before 2.10.1.
    rows = [("e", "a", 1.0, 0.4, 0.05, 0.05, 1.0), ("e", "b", 1.0, 0.3, 0.05, 0.05, 1.0)]
    scaled = [(e, u, lam, mu * scale, si * scale, sc * scale, m * scale) for e, u, lam, mu, si, sc, m in rows]
    base, base_meta = run_multi(multi, rows, bucket_size=0.1, views=("ground_up",))
    views, meta = run_multi(multi, scaled, bucket_size=0.1 * scale, views=("ground_up",))
    assert meta.diagnostics == {} and base_meta.diagnostics == {}
    assert meta.grid_length == base_meta.grid_length
    assert np.abs(views["ground_up"] - base["ground_up"]).max() < 1e-12
