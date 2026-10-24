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
    MULTI_IMPLS,
    diag_count,
    load_joint_golden,
    load_multi_fixture,
    load_multi_golden,
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
    assert diag_count(meta, "sigma_clamped") == 1
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
    # The Bernoulli treatment (1.13.0) belongs to the one-row route for now.
    with pytest.raises(multi.base_error):
        run_multi(multi, [good, bad], bucket_size=1.0, variance_policy="bernoulli")
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


def test_terms_line_is_reusable_in_engine(multi):
    # A parsed UnitTerms object is accepted directly.
    terms = {"a": parse_terms_line(TOWER)}
    views, meta = run_multi(multi, [("e", "a", 0.01, 1000.0, 0.0, 0.0, 1000.0)], terms=terms,
                            bucket_size=10.0)
    assert views["net"][30] == pytest.approx(0.01)


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
