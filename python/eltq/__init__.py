"""eltq: ferociously fast ELT secondary-uncertainty quantization.

Converts a portfolio event loss table into two evenly spaced, rate-weighted
loss-grid distributions (``raw`` and ``secondary``) via a mass- and
mean-preserving linear deposit rule. The mathematical contract lives in
``secondary-uncertainty-mathematical-specification.md``; computation happens
in the Rust extension ``eltq._eltq``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike

import numpy as np

from eltq._eltq import (
    EltqError,
    GridOverflowError,
    VarianceError,
    __rust_version__,
)
from eltq._eltq import (
    quantize_arrays as _quantize_arrays,
)
from eltq._eltq import (
    quantize_csv as _quantize_csv,
)

__version__ = "1.2.0"

__all__ = [
    "EltqError",
    "GridOverflowError",
    "Meta",
    "RowIssues",
    "VarianceError",
    "quantize_arrays",
    "quantize_csv",
    "quantize_dataframe",
]


@dataclass
class RowIssues:
    """Count plus a bounded set of representative event ids for one category."""

    count: int = 0
    examples: list = field(default_factory=list)


@dataclass
class Meta:
    """Metadata and diagnostics accompanying the two grid distributions.

    Masses and means describe the arrays actually returned (normalized when
    ``normalized``); ``total_rate``, ``analytic_aal``,
    ``quadrature_secondary_mean`` and ``secondary_mean_error`` are always in
    rate-weighted units. Diagnostic examples are event ids when available,
    0-based row indices otherwise.
    """

    bucket_size: float
    grid_length: int
    quadrature_points: int
    normalized: bool
    total_rate: float
    analytic_aal: float
    raw_mass: float
    raw_mean: float
    secondary_mass: float
    secondary_mean: float
    quadrature_secondary_mean: float
    secondary_mean_error: float
    input_rows: int
    valid_rows: int
    diagnostics: dict[str, RowIssues]


def _build_meta(meta_dict: dict, event_ids=None) -> Meta:
    diagnostics = {}
    for category, entry in meta_dict.pop("diagnostics").items():
        examples = entry["examples"]
        if event_ids is not None and examples and isinstance(examples[0], int):
            examples = [event_ids[row] for row in examples]
        diagnostics[category] = RowIssues(count=entry["count"], examples=examples)
    return Meta(diagnostics=diagnostics, **meta_dict)


def _column(name: str, values) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise EltqError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def quantize_arrays(
    rate,
    max_loss,
    mean_loss,
    sd_indep,
    sd_corr,
    *,
    bucket_size,
    grid_length=None,
    quadrature_points=32,
    normalize=False,
    mean_is_loss_rate=False,
    variance_policy="clamp",
    clamp_factor=0.999,
    event_ids=None,
):
    """Quantize ELT columns; return ``(raw, secondary, meta)``.

    Parameters
    ----------
    rate, max_loss, mean_loss, sd_indep, sd_corr : array_like
        One-dimensional numeric columns of equal length; converted to
        contiguous float64 explicitly (no copy when already contiguous).
    bucket_size : float
        Grid spacing h (> 0).
    grid_length : int, optional
        Number of grid nodes n; auto-derived as ceil(max M / h) + 1 when None.
    quadrature_points : int, default 32
        Number K of midpoint beta quantiles per nondegenerate event.
    normalize : bool, default False
        Divide both outputs by the total contributing rate.
    mean_is_loss_rate : bool, default False
        Interpret mean_loss as a rate on [0, 1], scaled by max_loss.
    variance_policy : {"clamp", "error", "drop", "atom"}, default "clamp"
        Treatment of rows with inadmissible beta variance.
    clamp_factor : float, default 0.999
        Clamp target as a fraction of the admissible sigma bound.
    event_ids : sequence, optional
        Per-row identifiers used in diagnostic examples (audit only).

    Returns
    -------
    raw, secondary : numpy.ndarray
        Length-n float64 rate-weight arrays; the loss at element i is i * h.
    meta : Meta
    """
    raw, secondary, meta_dict = _quantize_arrays(
        _column("rate", rate),
        _column("max_loss", max_loss),
        _column("mean_loss", mean_loss),
        _column("sd_indep", sd_indep),
        _column("sd_corr", sd_corr),
        bucket_size=bucket_size,
        grid_length=grid_length,
        quadrature_points=quadrature_points,
        normalize=normalize,
        mean_is_loss_rate=mean_is_loss_rate,
        variance_policy=variance_policy,
        clamp_factor=clamp_factor,
    )
    ids = list(event_ids) if event_ids is not None else None
    return raw, secondary, _build_meta(meta_dict, ids)


def quantize_csv(
    path: str | PathLike,
    *,
    bucket_size,
    grid_length=None,
    quadrature_points=32,
    normalize=False,
    mean_is_loss_rate=False,
    variance_policy="clamp",
    clamp_factor=0.999,
    columns: dict[str, str] | None = None,
):
    """Quantize an ELT CSV file; return ``(raw, secondary, meta)``.

    The file must carry a header row with the canonical columns
    ``event_id, rate, max_loss, mean_loss, sd_indep, sd_corr`` (``event_id``
    optional); ``columns`` maps canonical names to the actual header names,
    e.g. ``{"rate": "annual_rate"}``. Unparseable numeric cells become NaN
    and surface in the ``nonfinite`` diagnostic category. Parsing and
    computation both run in Rust.
    """
    raw, secondary, meta_dict = _quantize_csv(
        path,
        bucket_size=bucket_size,
        grid_length=grid_length,
        quadrature_points=quadrature_points,
        normalize=normalize,
        mean_is_loss_rate=mean_is_loss_rate,
        variance_policy=variance_policy,
        clamp_factor=clamp_factor,
        columns=columns,
    )
    return raw, secondary, _build_meta(meta_dict)


def quantize_dataframe(df, *, event_id_column: str = "event_id", **options):
    """Quantize a pandas DataFrame with canonical columns; see quantize_arrays.

    Convenience wrapper: extracts contiguous float64 columns explicitly and
    passes the id column (when present) for diagnostics. pandas is not a
    dependency of eltq; any object with DataFrame-like column access works.
    """
    event_ids = (
        list(df[event_id_column]) if event_id_column in df.columns else None
    )
    return quantize_arrays(
        df["rate"],
        df["max_loss"],
        df["mean_loss"],
        df["sd_indep"],
        df["sd_corr"],
        event_ids=event_ids,
        **options,
    )
