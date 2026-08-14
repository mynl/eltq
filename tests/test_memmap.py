"""Memory-mapped workflows: quantize_into and quantize_to_npy."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from conftest import FIXTURES_DIR, load_fixture

eltq = pytest.importorskip("eltq")

PARAMS = {"bucket_size": 0.5, "quadrature_points": 8}


def small_columns():
    df = load_fixture("small")
    return tuple(
        df[c].to_numpy(dtype=np.float64)
        for c in ("rate", "max_loss", "mean_loss", "sd_indep", "sd_corr")
    )


def direct_result():
    return eltq.quantize_arrays(*small_columns(), **PARAMS)


def test_quantize_into_plain_arrays():
    raw_direct, sec_direct, meta_direct = direct_result()
    n = meta_direct.grid_length
    raw_out = np.zeros(n)
    sec_out = np.zeros(n)
    meta = eltq.quantize_into(
        small_columns(), raw_out=raw_out, secondary_out=sec_out, **PARAMS
    )
    assert np.array_equal(raw_out, raw_direct)
    assert np.array_equal(sec_out, sec_direct)
    assert meta.grid_length == n
    assert meta.total_rate == meta_direct.total_rate


def test_quantize_into_memmap(tmp_path):
    raw_direct, sec_direct, meta_direct = direct_result()
    n = meta_direct.grid_length
    raw_path = tmp_path / "raw.dat"
    sec_path = tmp_path / "secondary.dat"
    raw_out = np.memmap(raw_path, dtype=np.float64, mode="w+", shape=(n,))
    sec_out = np.memmap(sec_path, dtype=np.float64, mode="w+", shape=(n,))
    meta = eltq.quantize_into(
        small_columns(), raw_out=raw_out, secondary_out=sec_out, **PARAMS
    )
    raw_out.flush()
    sec_out.flush()
    del raw_out, sec_out
    reopened = np.memmap(raw_path, dtype=np.float64, mode="r", shape=(n,))
    assert np.array_equal(np.asarray(reopened), raw_direct)
    assert meta.valid_rows == meta_direct.valid_rows


def test_quantize_into_csv_source():
    raw_direct, sec_direct, meta_direct = direct_result()
    n = meta_direct.grid_length
    raw_out = np.zeros(n)
    sec_out = np.zeros(n)
    eltq.quantize_into(
        FIXTURES_DIR / "small_elt.csv",
        raw_out=raw_out,
        secondary_out=sec_out,
        **PARAMS,
    )
    assert np.array_equal(raw_out, raw_direct)


def test_quantize_into_validation():
    n = 201
    good = np.zeros(n)
    cases = [
        ("dtype float64", np.zeros(n, dtype=np.float32), good),
        ("one-dimensional", np.zeros((n, 1)), good),
        ("C-contiguous", np.zeros(2 * n)[::2], good),
        ("must match", np.zeros(n + 1), good),
    ]
    for match, raw_out, sec_out in cases:
        with pytest.raises(eltq.EltqError, match=match):
            eltq.quantize_into(
                small_columns(), raw_out=raw_out, secondary_out=sec_out, **PARAMS
            )
    frozen = np.zeros(n)
    frozen.setflags(write=False)
    with pytest.raises(eltq.EltqError, match="writable"):
        eltq.quantize_into(
            small_columns(), raw_out=frozen, secondary_out=good, **PARAMS
        )
    with pytest.raises(eltq.EltqError, match="conflicts"):
        eltq.quantize_into(
            small_columns(), raw_out=good, secondary_out=np.zeros(n),
            grid_length=n + 5, **PARAMS,
        )
    # Outputs that are too short for the input are a grid overflow.
    with pytest.raises(eltq.GridOverflowError):
        eltq.quantize_into(
            small_columns(), raw_out=np.zeros(10), secondary_out=np.zeros(10),
            **PARAMS,
        )
    with pytest.raises(eltq.EltqError, match="source must be"):
        eltq.quantize_into(
            {"not": "a source"}, raw_out=good, secondary_out=np.zeros(n), **PARAMS
        )


def test_quantize_to_npy(tmp_path):
    raw_direct, sec_direct, meta_direct = direct_result()
    out = tmp_path / "small_grids.npy"
    meta = eltq.quantize_to_npy(small_columns(), out, **PARAMS)
    loaded = np.load(out, mmap_mode="r")
    assert loaded.shape == (2, meta.grid_length)
    assert loaded.dtype == np.float64
    assert np.array_equal(np.asarray(loaded[0]), raw_direct)
    assert np.array_equal(np.asarray(loaded[1]), sec_direct)
    meta_path = tmp_path / "small_grids.meta.json"
    stored = json.loads(meta_path.read_text(encoding="utf-8"))
    assert stored["grid_length"] == meta.grid_length
    assert math.isclose(stored["total_rate"], meta.total_rate, rel_tol=0.0)
    assert stored["diagnostics"]["degenerate_atom"]["count"] == 5
    # No temp litter left behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "small_grids.meta.json",
        "small_grids.npy",
    ]


def test_quantize_to_npy_overwrites(tmp_path):
    out = tmp_path / "grids.npy"
    eltq.quantize_to_npy(small_columns(), out, **PARAMS)
    before = out.stat().st_size
    meta = eltq.quantize_to_npy(small_columns(), out, **PARAMS)
    assert out.stat().st_size == before
    assert meta.grid_length > 0


def test_quantize_to_npy_csv_source(tmp_path):
    raw_direct, _, _ = direct_result()
    out = tmp_path / "from_csv.npy"
    eltq.quantize_to_npy(FIXTURES_DIR / "small_elt.csv", out, **PARAMS)
    loaded = np.load(out)
    assert np.array_equal(loaded[0], raw_direct)
