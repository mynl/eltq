"""eltq: ferociously fast ELT secondary-uncertainty quantization.

Converts a portfolio event loss table into two evenly spaced, rate-weighted
loss-grid distributions (``raw`` and ``secondary``) via a mass- and
mean-preserving linear deposit rule. The mathematical contract lives in
``secondary-uncertainty-mathematical-specification.md``.

Backends: computation normally happens in the Rust extension ``eltq._eltq``
(``BACKEND == "rust"``). When the compiled extension is not importable, the
same API is served by the pure-Python SciPy reference implementation
(``eltq.reference``, ``BACKEND == "reference"``) — identical mathematics,
roughly forty times slower. The reference backend ignores ``threads`` and
``chunk_rows`` (it is single-threaded and exactly summed).
"""

from __future__ import annotations

from os import PathLike

import numpy as np

from eltq._meta import Meta, RowIssues

__version__ = "1.5.0"

try:
    from eltq._eltq import (
        EltqError,
        GridOverflowError,
        VarianceError,
        __rust_version__,
    )
    from eltq._eltq import (
        quantize_arrays as _rust_quantize_arrays,
    )
    from eltq._eltq import (
        quantize_csv as _rust_quantize_csv,
    )

    BACKEND = "rust"
except ImportError:
    try:
        from eltq.reference import EltqError, GridOverflowError, VarianceError
        from eltq import reference as _reference
    except ImportError as scipy_missing:
        raise ImportError(
            "the compiled eltq._eltq extension is unavailable and the SciPy "
            "fallback backend could not load; install scipy to use the "
            "pure-Python reference backend"
        ) from scipy_missing

    BACKEND = "reference"
    __rust_version__ = None

__all__ = [
    "BACKEND",
    "EltqError",
    "GridOverflowError",
    "Meta",
    "RowIssues",
    "VarianceError",
    "quantize_arrays",
    "quantize_csv",
    "quantize_dataframe",
    "quantize_into",
    "quantize_to_npy",
]

_CANONICAL_COLUMNS = ("rate", "max_loss", "mean_loss", "sd_indep", "sd_corr")


def _build_meta(meta_dict: dict, event_ids=None) -> Meta:
    """Convert the Rust extension's metadata dict into the Meta dataclass."""
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


def _read_csv_fallback(path, columns):
    """Stdlib CSV reader mirroring the Rust route's behavior: unparseable or
    empty numeric cells become NaN (reported under ``nonfinite``); missing
    columns and unknown mapping keys are hard errors."""
    import csv

    mapping = {key: key for key in ("event_id", *_CANONICAL_COLUMNS)}
    if columns:
        for key, value in columns.items():
            if key not in mapping:
                raise EltqError(f"unknown column key {key!r} in columns mapping")
            mapping[key] = value
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise EltqError(f"CSV read failed: {path} is empty")
        index = {name: i for i, name in enumerate(header)}
        column_index = {}
        for key in _CANONICAL_COLUMNS:
            name = mapping[key]
            if name not in index:
                raise EltqError(f"CSV is missing required column {name!r}")
            column_index[key] = index[name]
        id_index = index.get(mapping["event_id"])
        data = {key: [] for key in _CANONICAL_COLUMNS}
        ids = [] if id_index is not None else None
        for row in reader:
            if not row:
                continue
            for key, i in column_index.items():
                try:
                    value = float(row[i])
                except (ValueError, IndexError):
                    value = float("nan")
                data[key].append(value)
            if ids is not None:
                ids.append(row[id_index] if id_index < len(row) else "")
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
    return arrays, ids


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
    threads=None,
    chunk_rows=None,
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
    threads : int, optional
        Worker threads; None uses all cores. Results are bitwise identical
        for any thread count. Ignored by the reference backend.
    chunk_rows : int, optional
        Rows per parallel chunk (default 65536); part of the deterministic
        contract. Ignored by the reference backend.
    event_ids : sequence, optional
        Per-row identifiers used in diagnostic examples (audit only).

    Returns
    -------
    raw, secondary : numpy.ndarray
        Length-n float64 rate-weight arrays; the loss at element i is i * h.
    meta : Meta
    """
    cols = tuple(
        _column(name, values)
        for name, values in zip(
            _CANONICAL_COLUMNS, (rate, max_loss, mean_loss, sd_indep, sd_corr)
        )
    )
    ids = list(event_ids) if event_ids is not None else None
    common = {
        "bucket_size": bucket_size,
        "grid_length": grid_length,
        "quadrature_points": quadrature_points,
        "normalize": normalize,
        "mean_is_loss_rate": mean_is_loss_rate,
        "variance_policy": variance_policy,
        "clamp_factor": clamp_factor,
    }
    if BACKEND == "rust":
        raw, secondary, meta_dict = _rust_quantize_arrays(
            *cols, **common, threads=threads, chunk_rows=chunk_rows
        )
        return raw, secondary, _build_meta(meta_dict, ids)
    return _reference.quantize_reference(*cols, **common, event_ids=ids)


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
    threads=None,
    chunk_rows=None,
    columns: dict[str, str] | None = None,
):
    """Quantize an ELT CSV file; return ``(raw, secondary, meta)``.

    The file must carry a header row with the canonical columns
    ``event_id, rate, max_loss, mean_loss, sd_indep, sd_corr`` (``event_id``
    optional); ``columns`` maps canonical names to the actual header names,
    e.g. ``{"rate": "annual_rate"}``. Unparseable numeric cells become NaN
    and surface in the ``nonfinite`` diagnostic category. On the Rust backend
    parsing and computation both run in Rust.
    """
    common = {
        "bucket_size": bucket_size,
        "grid_length": grid_length,
        "quadrature_points": quadrature_points,
        "normalize": normalize,
        "mean_is_loss_rate": mean_is_loss_rate,
        "variance_policy": variance_policy,
        "clamp_factor": clamp_factor,
    }
    if BACKEND == "rust":
        raw, secondary, meta_dict = _rust_quantize_csv(
            path, **common, threads=threads, chunk_rows=chunk_rows,
            columns=columns,
        )
        return raw, secondary, _build_meta(meta_dict)
    data, ids = _read_csv_fallback(path, columns)
    return _reference.quantize_reference(
        *(data[key] for key in _CANONICAL_COLUMNS), **common, event_ids=ids
    )


def _quantize_source(source, options):
    """Dispatch a source (CSV path, DataFrame-like, or 5-tuple of columns)."""
    if isinstance(source, (str, PathLike)):
        return quantize_csv(source, **options)
    if hasattr(source, "columns"):
        return quantize_dataframe(source, **options)
    if isinstance(source, (tuple, list)) and len(source) == 5:
        return quantize_arrays(*source, **options)
    raise EltqError(
        "source must be a CSV path, a DataFrame with canonical columns, or a "
        "(rate, max_loss, mean_loss, sd_indep, sd_corr) tuple"
    )


def _check_output(name: str, array) -> None:
    if not isinstance(array, np.ndarray):
        raise EltqError(f"{name} must be a numpy.ndarray (or memmap), "
                        f"got {type(array).__name__}")
    if array.dtype != np.float64:
        raise EltqError(f"{name} must have dtype float64, got {array.dtype}")
    if array.ndim != 1:
        raise EltqError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not array.flags.c_contiguous:
        raise EltqError(f"{name} must be C-contiguous")
    if not array.flags.writeable:
        raise EltqError(f"{name} must be writable")


def quantize_into(source, *, raw_out, secondary_out, grid_length=None, **options):
    """Quantize into caller-provided writable arrays; return ``Meta``.

    ``raw_out`` and ``secondary_out`` must be writable, C-contiguous float64
    one-dimensional arrays of equal length — including ``numpy.memmap``
    instances — and their length defines the grid (``grid_length``, if also
    given, must match). Computation happens in backend-owned buffers; the
    results are copied into the borrowed arrays in one final assignment.
    Memmap callers flush/close as usual.
    """
    _check_output("raw_out", raw_out)
    _check_output("secondary_out", secondary_out)
    if raw_out.size != secondary_out.size:
        raise EltqError(
            f"raw_out has {raw_out.size} elements but secondary_out has "
            f"{secondary_out.size}; they must match"
        )
    if grid_length is not None and grid_length != raw_out.size:
        raise EltqError(
            f"grid_length {grid_length} conflicts with output length {raw_out.size}"
        )
    raw, secondary, meta = _quantize_source(
        source, {**options, "grid_length": raw_out.size}
    )
    raw_out[:] = raw
    secondary_out[:] = secondary
    return meta


def quantize_to_npy(source, path, **options):
    """Quantize and write a ``(2, n)`` float64 ``.npy`` file; return ``Meta``.

    Row 0 is the raw distribution, row 1 the secondary; reopen with
    ``np.load(path, mmap_mode="r")``. Metadata is written to an adjacent
    ``<stem>.meta.json``. Both files are written transactionally (temp file
    then atomic replace), so an interruption cannot leave a completed-looking
    partial result.
    """
    import dataclasses
    import json
    import os
    from pathlib import Path

    path = Path(path)
    raw, secondary, meta = _quantize_source(source, options)
    stacked = np.stack([raw, secondary])
    tmp_npy = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    meta_path = path.with_name(f"{path.stem}.meta.json")
    tmp_meta = meta_path.with_name(f"{meta_path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp_npy, "wb") as handle:
            np.save(handle, stacked)
        tmp_meta.write_text(
            json.dumps(dataclasses.asdict(meta), indent=2), encoding="utf-8"
        )
        os.replace(tmp_npy, path)
        os.replace(tmp_meta, meta_path)
    finally:
        tmp_npy.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)
    return meta


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
        *(df[name] for name in _CANONICAL_COLUMNS),
        event_ids=event_ids,
        **options,
    )
