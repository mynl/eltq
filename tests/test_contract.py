"""Supplementary contract tests: validation, diagnostics, modes, edge cases.

Complements the numbered acceptance suite in ``test_acceptance.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import diag_count, elt, grid_of, run

GOOD = (0.010, 100.0, 40.0, 15.0, 10.0)


def test_invalid_rows_dropped_and_reported(impl):
    rows = [
        GOOD,
        (float("nan"), 100.0, 40.0, 15.0, 10.0),   # nonfinite rate
        (0.01, float("inf"), 40.0, 15.0, 10.0),    # nonfinite max_loss
        (0.01, 100.0, -1.0, 15.0, 10.0),           # negative mean
        (-0.01, 100.0, 40.0, 15.0, 10.0),          # negative rate
        (0.01, 100.0, 140.0, 15.0, 10.0),          # mean > max
    ]
    ids = ["good", "nan_rate", "inf_max", "neg_mean", "neg_rate", "mu_gt_m"]
    raw, secondary, meta = run(impl, {**elt(rows), "event_id": ids},
                               bucket_size=0.5)
    assert meta.input_rows == 6
    assert meta.valid_rows == 1
    assert diag_count(meta, "nonfinite") == 2
    assert diag_count(meta, "negative_field") == 2
    assert diag_count(meta, "mean_exceeds_max") == 1
    assert math.isclose(meta.total_rate, GOOD[0], rel_tol=1e-15)
    assert math.isclose(raw.sum(), GOOD[0], rel_tol=1e-12)
    # Representative examples carry the offending event ids.
    examples = meta.diagnostics["nonfinite"].examples
    if not isinstance(examples, list):
        examples = list(examples)
    assert examples == ["nan_rate", "inf_max"]


def test_zero_rate_rows_skipped(impl):
    rows = [GOOD, (0.0, 500.0, 100.0, 20.0, 0.0)]
    raw, secondary, meta = run(impl, elt(rows), bucket_size=0.5)
    assert diag_count(meta, "zero_rate") == 1
    assert meta.valid_rows == 1
    # A zero-rate row generates no deposit, so it does not force grid coverage.
    assert meta.grid_length == 201


def test_duplicate_event_ids_are_independent_rows(impl):
    rows = [(0.002, 30.0, 10.0, 4.0, 2.0), (0.002, 30.0, 12.0, 4.0, 2.0)]
    raw, secondary, meta = run(impl, {**elt(rows), "event_id": ["dup", "dup"]},
                               bucket_size=0.5)
    assert meta.valid_rows == 2
    assert math.isclose(raw.sum(), 0.004, rel_tol=1e-12)


def test_mean_is_loss_rate_mode(impl):
    m, p, sigma = 200.0, 0.3, 40.0
    direct = elt([(0.01, m, p * m, sigma, 0.0)])
    as_rate = elt([(0.01, m, p, sigma, 0.0)])
    raw_d, sec_d, meta_d = run(impl, direct, bucket_size=1.0)
    raw_r, sec_r, meta_r = run(impl, as_rate, bucket_size=1.0,
                               mean_is_loss_rate=True)
    assert np.allclose(raw_r, raw_d, rtol=1e-12, atol=0.0)
    assert np.allclose(sec_r, sec_d, rtol=1e-9, atol=1e-15)
    # In loss-rate mode a mean above one exceeds M and is dropped.
    _, _, meta = run(impl, elt([(0.01, m, 1.5, sigma, 0.0)]),
                     bucket_size=1.0, mean_is_loss_rate=True)
    assert diag_count(meta, "mean_exceeds_max") == 1


def test_normalize(impl):
    rows = [GOOD, (0.03, 50.0, 20.0, 5.0, 0.0)]
    raw, secondary, meta = run(impl, elt(rows), bucket_size=0.5, normalize=True)
    assert meta.normalized
    assert math.isclose(raw.sum(), 1.0, rel_tol=1e-12)
    assert math.isclose(secondary.sum(), 1.0, rel_tol=1e-12)
    # total_rate stays rate-weighted even when the arrays are normalized.
    assert math.isclose(meta.total_rate, 0.04, rel_tol=1e-15)
    with pytest.raises(impl.base_error):
        run(impl, elt([(0.0, 10.0, 5.0, 0.0, 0.0)]), bucket_size=0.5,
            normalize=True)


def test_k_equals_one_uses_median(impl):
    raw, secondary, meta = run(impl, elt([GOOD]), bucket_size=0.5,
                               quadrature_points=1, method="quantiles")
    assert math.isclose(secondary.sum(), GOOD[0], rel_tol=1e-12)
    # Single quadrature point is the conditional median, not the mean.
    median = float(secondary @ grid_of(meta)) / GOOD[0]
    assert 0.0 < median < GOOD[1]
    assert math.isclose(median, meta.quadrature_secondary_mean / GOOD[0],
                        rel_tol=1e-9)


def test_empty_input(impl):
    raw, secondary, meta = run(
        impl,
        {k: np.array([]) for k in
         ("rate", "max_loss", "mean_loss", "sd_indep", "sd_corr")},
        bucket_size=0.5,
    )
    assert meta.grid_length == 1
    assert meta.valid_rows == 0
    assert raw.sum() == 0.0 and secondary.sum() == 0.0


def test_far_inadmissible_rows_are_bernoulli(impl):
    # sigma far beyond admissible: the Bernoulli pair carries the rate.
    rows = [(0.01, 1000.0, 300.0, 900.0, 0.0)]
    raw, secondary, meta = run(impl, elt(rows), bucket_size=1.0)
    assert diag_count(meta, "variance_bernoulli") == 1
    assert np.isfinite(secondary).all()
    assert (secondary >= 0.0).all()
    assert math.isclose(secondary.sum(), 0.01, rel_tol=1e-12)
    assert math.isclose(secondary[0], 0.01 * 0.7, rel_tol=1e-15)
    assert math.isclose(secondary[1000], 0.01 * 0.3, rel_tol=1e-15)


def test_bad_config_rejected(impl):
    good = elt([GOOD])
    with pytest.raises(impl.base_error):
        run(impl, good, bucket_size=0.0)
    with pytest.raises(impl.base_error):
        run(impl, good, bucket_size=1.0, quadrature_points=0)
    with pytest.raises(impl.base_error):
        run(impl, good, bucket_size=1.0, variance_policy="nonsense")
    with pytest.raises(impl.base_error):
        run(impl, good, bucket_size=1.0, grid_length=0)
    with pytest.raises(impl.base_error):
        run(impl, good, bucket_size=1.0, method="nonsense")
    with pytest.raises(impl.base_error):
        run(impl, good, bucket_size=1.0, overflow="nonsense")
    with pytest.raises(impl.base_error):
        run(impl, good, bucket_size=1.0, method="survival", nodes_per_decade=0)
