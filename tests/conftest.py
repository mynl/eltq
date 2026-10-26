"""Shared pytest fixtures: implementation registry and fixture loading.

Every implementation of the mathematical contract is exposed through one
``Impl`` record with a unified call signature, so the acceptance suite runs
identically against the SciPy reference and (from 1.2.0) the Rust ``eltq``
package.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

TESTS_DIR = Path(__file__).parent
REPO_ROOT = TESTS_DIR.parent
FIXTURES_DIR = REPO_ROOT / "fixtures"
GOLDEN_DIR = FIXTURES_DIR / "golden"

sys.path.insert(0, str(TESTS_DIR))

from eltq import reference  # noqa: E402  (also the eltq fallback backend)


@dataclass
class Impl:
    """One implementation under test plus its exception types."""

    name: str
    quantize: Callable[..., tuple[np.ndarray, np.ndarray, Any]]
    base_error: type[Exception]
    variance_error: type[Exception]
    overflow_error: type[Exception]


def _available_impls() -> list[Impl]:
    impls = [
        Impl(
            name="reference",
            quantize=reference.quantize_reference,
            base_error=reference.EltqError,
            variance_error=reference.VarianceError,
            overflow_error=reference.GridOverflowError,
        )
    ]
    try:
        import eltq
    except ImportError:
        return impls
    if hasattr(eltq, "quantize_arrays"):
        impls.append(
            Impl(
                name="eltq",
                quantize=eltq.quantize_arrays,
                base_error=eltq.EltqError,
                variance_error=eltq.VarianceError,
                overflow_error=eltq.GridOverflowError,
            )
        )
    return impls


IMPLS = _available_impls()


@pytest.fixture(params=IMPLS, ids=[i.name for i in IMPLS])
def impl(request) -> Impl:
    return request.param


def run(impl: Impl, elt, **params):
    """Call an implementation on an ELT given as a DataFrame or dict of columns."""
    if isinstance(elt, pd.DataFrame):
        cols = {c: elt[c].to_numpy(dtype=np.float64)
                for c in ("rate", "max_loss", "mean_loss", "sd_indep", "sd_corr")}
        params.setdefault("event_ids", list(elt["event_id"]))
    else:
        cols = {k: np.asarray(v, dtype=np.float64)
                for k, v in elt.items() if k != "event_id"}
        if "event_id" in elt:
            params.setdefault("event_ids", list(elt["event_id"]))
    return impl.quantize(
        cols["rate"], cols["max_loss"], cols["mean_loss"],
        cols["sd_indep"], cols["sd_corr"], **params,
    )


def elt(rows) -> dict:
    """Build column dict from (rate, max_loss, mean_loss, sd_indep, sd_corr) rows."""
    a = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
    return {
        "rate": a[:, 0], "max_loss": a[:, 1], "mean_loss": a[:, 2],
        "sd_indep": a[:, 3], "sd_corr": a[:, 4],
    }


def load_fixture(name: str) -> pd.DataFrame:
    # round_trip: correctly rounded floats, bit-identical to Rust's parser.
    return pd.read_csv(FIXTURES_DIR / f"{name}_elt.csv",
                       float_precision="round_trip")


def load_golden(name: str):
    """Return (raw, secondary, meta_dict) for a committed golden output."""
    with np.load(GOLDEN_DIR / f"{name}.npz") as z:
        raw, secondary = z["raw"].copy(), z["secondary"].copy()
    meta = json.loads((GOLDEN_DIR / f"{name}.meta.json").read_text(encoding="utf-8"))
    return raw, secondary, meta


def grid_of(meta) -> np.ndarray:
    return meta.bucket_size * np.arange(meta.grid_length, dtype=np.float64)


def diag_count(meta, category: str) -> int:
    issues = meta.diagnostics.get(category)
    if issues is None:
        return 0
    return issues.count if hasattr(issues, "count") else issues["count"]


# ---------------------------------------------------------------------------
# Multi-unit route (1.7.0): the SciPy reference and the package API.

from eltq import reference_multiunit  # noqa: E402

MULTI_COLUMNS = ("event_id", "unit_id", "rate", "mean_loss", "sd_indep",
                 "sd_corr", "max_loss")


@dataclass
class MultiImpl:
    """One implementation of the multi-unit contract plus its error types."""

    name: str
    views: Callable[..., tuple[dict, Any]]
    joint: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, Any]]
    base_error: type[Exception]
    variance_error: type[Exception]
    overflow_error: type[Exception]
    structure_error: type[Exception]


def _reference_joint(unit1, unit2, **params):
    result = reference_multiunit.secondary_two_units([unit1, unit2], **params)
    return result.pmf, result.x1, result.x2, result.meta


def _available_multi_impls() -> list[MultiImpl]:
    impls = [
        MultiImpl(
            name="reference",
            views=reference_multiunit.multiunit_views_reference,
            joint=_reference_joint,
            base_error=reference.EltqError,
            variance_error=reference.VarianceError,
            overflow_error=reference.GridOverflowError,
            structure_error=reference_multiunit.MultiUnitError,
        )
    ]
    import eltq

    def eltq_joint(unit1, unit2, *, n_z=32, **params):
        return eltq.joint_two_units(unit1, unit2, z_nodes=n_z, **params)

    impls.append(
        MultiImpl(
            name="eltq",
            views=eltq.multiunit_views,
            joint=eltq_joint,
            base_error=eltq.EltqError,
            variance_error=eltq.VarianceError,
            overflow_error=eltq.GridOverflowError,
            structure_error=eltq.MultiUnitError,
        )
    )
    return impls


MULTI_IMPLS = _available_multi_impls()


@pytest.fixture(params=MULTI_IMPLS, ids=[i.name for i in MULTI_IMPLS])
def multi(request) -> MultiImpl:
    return request.param


def multi_rows(rows):
    """Column dict from (event_id, unit_id, rate, mu, sd_indep, sd_corr, M) rows."""
    cols = {name: [] for name in MULTI_COLUMNS}
    for row in rows:
        for name, value in zip(MULTI_COLUMNS, row):
            cols[name].append(value)
    return {name: (np.asarray(v, dtype=np.float64) if name not in ("event_id", "unit_id")
                   else list(v)) for name, v in cols.items()}


def run_multi(impl: MultiImpl, elt, **params):
    """Run a multi-unit implementation on a row list or a DataFrame."""
    if isinstance(elt, pd.DataFrame):
        cols = {c: (elt[c].to_numpy(dtype=np.float64) if c not in ("event_id", "unit_id")
                    else list(elt[c].astype(str))) for c in MULTI_COLUMNS}
    else:
        cols = multi_rows(elt) if isinstance(elt, list) else elt
    return impl.views(*(cols[c] for c in MULTI_COLUMNS), **params)


def load_multi_fixture():
    """Return (DataFrame, terms text) for the multi-unit fixture."""
    df = pd.read_csv(FIXTURES_DIR / "multiunit_small_elt.csv",
                     float_precision="round_trip")
    terms = (FIXTURES_DIR / "multiunit_small_terms.txt").read_text(encoding="utf-8")
    return df, terms


def load_multi_golden():
    """Return (views dict, meta dict) for the multi-unit golden output."""
    with np.load(GOLDEN_DIR / "multiunit_small.npz") as z:
        views = {k: z[k].copy() for k in z.files}
    meta = json.loads((GOLDEN_DIR / "multiunit_small.meta.json").read_text(encoding="utf-8"))
    return views, meta


def load_joint_golden():
    """Return (pmf, meta dict) for the two-unit joint golden output."""
    pmf = np.loadtxt(GOLDEN_DIR / "joint_small.pmf.csv", delimiter=",")
    meta = json.loads((GOLDEN_DIR / "joint_small.meta.json").read_text(encoding="utf-8"))
    return pmf, meta
