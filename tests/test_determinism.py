"""Deterministic-mode guarantee: bitwise-identical results across runs and
thread counts (fixed input, config, and chunk size)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import load_fixture

eltq = pytest.importorskip("eltq")

PARAMS = {"bucket_size": 5000.0, "quadrature_points": 32}


def columns():
    df = load_fixture("medium")
    return tuple(
        df[c].to_numpy(dtype=np.float64)
        for c in ("rate", "max_loss", "mean_loss", "sd_indep", "sd_corr")
    )


def test_repeat_runs_are_bitwise_identical():
    cols = columns()
    raw1, sec1, meta1 = eltq.quantize_arrays(*cols, **PARAMS)
    raw2, sec2, meta2 = eltq.quantize_arrays(*cols, **PARAMS)
    assert np.array_equal(raw1, raw2)
    assert np.array_equal(sec1, sec2)
    assert meta1.total_rate == meta2.total_rate
    assert meta1.quadrature_secondary_mean == meta2.quadrature_secondary_mean


def test_thread_count_sweep_is_bitwise_identical():
    cols = columns()
    # Small chunks force many chunks so the fold order genuinely matters.
    base_raw, base_sec, base_meta = eltq.quantize_arrays(
        *cols, **PARAMS, threads=1, chunk_rows=512
    )
    for threads in (2, 4, None):
        raw, secondary, meta = eltq.quantize_arrays(
            *cols, **PARAMS, threads=threads, chunk_rows=512
        )
        assert np.array_equal(base_raw, raw), f"threads={threads}"
        assert np.array_equal(base_sec, secondary), f"threads={threads}"
        assert base_meta.total_rate == meta.total_rate
        assert base_meta.secondary_mean == meta.secondary_mean


def test_multiunit_views_bitwise_invariant_across_threads_and_paths():
    """Multi-unit route (1.10.0): same input twice, a thread sweep, and both
    chunk sizes give bitwise-identical views on the Rust backend."""
    import eltq
    from conftest import load_multi_fixture

    if eltq.BACKEND != "rust":
        pytest.skip("determinism contract is a property of the Rust engine")
    df, terms = load_multi_fixture()
    cols = [df[c] for c in ("event_id", "unit_id", "rate", "mean_loss", "sd_indep",
                            "sd_corr", "max_loss")]
    base, meta = eltq.multiunit_views(*cols, terms=terms, bucket_size=10.0, chunk_rows=3)
    again, _ = eltq.multiunit_views(*cols, terms=terms, bucket_size=10.0, chunk_rows=3)
    for view in meta.views:
        assert np.array_equal(base[view], again[view])
    for threads in (1, 2, 3):
        other, other_meta = eltq.multiunit_views(*cols, terms=terms, bucket_size=10.0,
                                                 chunk_rows=3, threads=threads)
        for view in meta.views:
            assert np.array_equal(base[view], other[view]), (view, threads)
        assert other_meta.total_rate == meta.total_rate
        assert other_meta.analytic_view_mean == meta.analytic_view_mean
    # Direct and FFT convolution agree to floating-point tolerance.
    direct, dm = eltq.multiunit_views(*cols, terms=terms, bucket_size=10.0, convolution="direct")
    fft, fm = eltq.multiunit_views(*cols, terms=terms, bucket_size=10.0, convolution="fft")
    assert dm.engine["fft_paths"] == 0 and fm.engine["direct_paths"] == 0
    for view in meta.views:
        assert np.allclose(direct[view], fft[view], rtol=0.0, atol=1e-14 * meta.total_rate)
