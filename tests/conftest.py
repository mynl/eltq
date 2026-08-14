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

import reference  # noqa: E402


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
    return pd.read_csv(FIXTURES_DIR / f"{name}_elt.csv")


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
