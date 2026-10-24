"""eltq package API tests: CSV route, conversions, exceptions, metadata.

The mathematical behavior is covered by the impl-parameterized acceptance and
contract suites; these tests cover the Python surface specific to the Rust
package. Skipped entirely until the extension is installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import FIXTURES_DIR, load_fixture, run

eltq = pytest.importorskip("eltq")


def elt_impl():
    from conftest import IMPLS

    return next(i for i in IMPLS if i.name == "eltq")


@pytest.mark.parametrize("name", ["small", "medium"])
def test_csv_route_matches_array_route(name):
    df = load_fixture(name)
    params = {"bucket_size": 0.5 if name == "small" else 5000.0,
              "method": "quantiles", "quadrature_points": 8 if name == "small" else 32}
    raw_a, sec_a, meta_a = run(elt_impl(), df, **params)
    raw_c, sec_c, meta_c = eltq.quantize_csv(FIXTURES_DIR / f"{name}_elt.csv",
                                             **params)
    assert np.array_equal(raw_a, raw_c)
    assert np.array_equal(sec_a, sec_c)
    assert meta_a.total_rate == meta_c.total_rate
    assert meta_a.valid_rows == meta_c.valid_rows


def test_csv_custom_column_names(tmp_path):
    path = tmp_path / "renamed.csv"
    path.write_text(
        "id,lam,M,mu,s1,s2\nev1,0.01,100.0,40.0,15.0,10.0\n", encoding="utf-8"
    )
    raw, secondary, meta = eltq.quantize_csv(
        path,
        bucket_size=0.5,
        columns={"event_id": "id", "rate": "lam", "max_loss": "M",
                 "mean_loss": "mu", "sd_indep": "s1", "sd_corr": "s2"},
    )
    assert meta.valid_rows == 1
    assert math.isclose(raw.sum(), 0.01, rel_tol=1e-12)


def test_csv_missing_column_raises(tmp_path):
    path = tmp_path / "missing.csv"
    path.write_text("event_id,rate,max_loss\nev1,0.01,100.0\n", encoding="utf-8")
    with pytest.raises(eltq.EltqError, match="mean_loss"):
        eltq.quantize_csv(path, bucket_size=0.5)


def test_csv_bad_cells_reported_with_event_ids(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "event_id,rate,max_loss,mean_loss,sd_indep,sd_corr\n"
        "good,0.01,100.0,40.0,15.0,10.0\n"
        "broken,oops,100.0,40.0,15.0,10.0\n",
        encoding="utf-8",
    )
    raw, secondary, meta = eltq.quantize_csv(path, bucket_size=0.5)
    assert meta.valid_rows == 1
    assert meta.diagnostics["nonfinite"].count == 1
    assert meta.diagnostics["nonfinite"].examples == ["broken"]


def test_csv_unknown_column_key_rejected(tmp_path):
    path = tmp_path / "any.csv"
    path.write_text(
        "event_id,rate,max_loss,mean_loss,sd_indep,sd_corr\n", encoding="utf-8"
    )
    with pytest.raises(eltq.EltqError, match="unknown column key"):
        eltq.quantize_csv(path, bucket_size=0.5, columns={"nope": "x"})


def test_array_inputs_are_converted():
    # Lists and non-float64 dtypes are converted explicitly by the wrapper.
    raw, secondary, meta = eltq.quantize_arrays(
        [0.01, 0.02],
        np.array([100.0, 50.0], dtype=np.float32),
        (40.0, 20.0),
        np.array([15.0, 5.0]),
        [10.0, 0.0],
        bucket_size=0.5,
    )
    assert raw.dtype == np.float64 and raw.flags.c_contiguous
    assert math.isclose(raw.sum(), 0.03, rel_tol=1e-12)
    with pytest.raises(eltq.EltqError, match="one-dimensional"):
        eltq.quantize_arrays(
            np.zeros((2, 2)), [1.0], [0.5], [0.1], [0.0], bucket_size=0.5
        )


def test_event_id_translation_in_examples():
    raw, secondary, meta = eltq.quantize_arrays(
        [0.01, float("nan")], [100.0, 100.0], [40.0, 40.0], [15.0, 15.0],
        [0.0, 0.0], bucket_size=0.5, event_ids=["a", "b"],
    )
    assert meta.diagnostics["nonfinite"].examples == ["b"]


def test_exception_hierarchy():
    assert issubclass(eltq.VarianceError, eltq.EltqError)
    assert issubclass(eltq.GridOverflowError, eltq.EltqError)


def test_versions_in_sync():
    from eltq._eltq import __rust_version__

    assert eltq.__version__ == __rust_version__


def test_quantize_dataframe_convenience():
    df = load_fixture("small")
    raw_d, sec_d, meta_d = eltq.quantize_dataframe(
        df, bucket_size=0.5, quadrature_points=8
    )
    raw_a, sec_a, meta_a = run(elt_impl(), df, bucket_size=0.5,
                               quadrature_points=8)
    assert np.array_equal(raw_d, raw_a)
    assert meta_d.diagnostics.keys() == meta_a.diagnostics.keys()
