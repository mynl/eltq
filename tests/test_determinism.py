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
