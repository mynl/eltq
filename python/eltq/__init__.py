"""eltq: ferociously fast ELT secondary-uncertainty quantization.

Converts a portfolio event loss table into two evenly spaced, rate-weighted
loss-grid distributions: ``raw``, an atom per event at its conditional mean
by the mass- and mean-preserving linear deposit rule, and ``secondary``,
each event's conditional loss as a beta on [0, M] with sd sigma_I + sigma_C,
bucketed from the portfolio exceedance curve by the nearest rule (the
survival method; the K-point quantiles method remains selectable). The
exceedance curve itself is available through ``exceedance_curve``. The
mathematical contract lives in
``secondary-uncertainty-mathematical-specification.md``.

Backends: computation normally happens in the Rust extension ``eltq._eltq``
(``BACKEND == "rust"``). When the compiled extension is not importable, the
same API is served by the pure-Python SciPy reference implementation
(``eltq.reference``, ``BACKEND == "reference"``) — identical mathematics,
roughly forty times slower. The reference backend ignores ``threads`` and
``chunk_rows`` (it is single-threaded and exactly summed).

Multi-unit route (1.7.0): ``multiunit_views`` accumulates the total event
severity by view (ground_up, gross, ceded, net) for event x unit rows with
DecL-style unit terms, and ``joint_two_units`` returns the exact bucketed
bivariate pmf of one event across two units. Both run on the Rust extension
(``joint_two_units`` from 1.8.0, ``multiunit_views`` from 1.10.0) and on the
SciPy reference ``eltq.reference_multiunit`` otherwise.
"""

from __future__ import annotations

from os import PathLike

import numpy as np

from eltq._meta import (
    DEFAULT_NODES_PER_DECADE, DEFAULT_NODES_PER_SIGMA, CurveMeta, FirstMomentMeta, GridMeta, JointMeta, Meta,
    ModelMoments, MultiMeta, PairMeta, RowIssues, TailMoments,
)
from eltq.terms import VIEWS, Clause, TermsError, UnitTerms, parse_terms

__version__ = "2.13.0"

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

    try:
        from eltq._eltq import joint_two_units as _rust_joint_two_units
    except ImportError:  # extension older than 1.8.0
        _rust_joint_two_units = None
    try:
        from eltq._eltq import multiunit_views as _rust_multiunit_views
    except ImportError:  # extension older than 1.10.0
        _rust_multiunit_views = None
    try:
        from eltq._eltq import exceedance_curve as _rust_exceedance_curve
    except ImportError:  # extension older than 1.17.0
        _rust_exceedance_curve = None
    try:
        from eltq._eltq import moments_arrays as _rust_moments_arrays
    except ImportError:  # extension older than 2.11.0
        _rust_moments_arrays = None
    try:
        from eltq._eltq import tail_moments_arrays as _rust_tail_moments_arrays
    except ImportError:  # extension older than 2.12.0
        _rust_tail_moments_arrays = None
    try:
        from eltq._eltq import multiunit_moments as _rust_multiunit_moments
    except ImportError:  # extension older than 2.13.0
        _rust_multiunit_moments = None

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
    _rust_joint_two_units = None
    _rust_multiunit_views = None
    _rust_exceedance_curve = None
    _rust_moments_arrays = None
    _rust_tail_moments_arrays = None
    _rust_multiunit_moments = None

if BACKEND == "rust":
    try:
        from eltq._eltq import MultiUnitError
    except ImportError:  # extension older than 1.10.0

        class MultiUnitError(EltqError):
            """Structural problem in a multi-unit ELT: events not contiguous,
            a rate varying within an event, or a unit repeated within an
            event."""

else:
    from eltq.reference_multiunit import MultiUnitError

__all__ = [
    "BACKEND",
    "MultiUnitError",
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
    "exceedance_curve",
    "ExceedanceCurve",
    "moments_arrays",
    "ModelMoments",
    "tail_moments_arrays",
    "TailMoments",
    "multiunit_moments",
    "CurveMeta",
    "GridMeta",
    "joint_two_units",
    "multiunit_views",
    "multiunit_views_dataframe",
    "multiunit_views_into",
    "occurrence_pair_joint",
    "occurrence_unit_first_moment",
    "annual_conditional_mean",
    "PairMeta",
    "FirstMomentMeta",
    "parse_terms",
    "TermsError",
    "UnitTerms",
    "Clause",
    "VIEWS",
    "JointMeta",
    "MultiMeta",
]

_CANONICAL_COLUMNS = ("rate", "max_loss", "mean_loss", "sd_indep", "sd_corr")
_METHODS = ("quantiles", "survival")
_OVERFLOW_POLICIES = ("top_bucket", "drop", "error")


def _build_meta(meta_dict: dict, event_ids=None) -> Meta:
    """Convert the Rust extension's metadata dict into the Meta dataclass."""
    diagnostics = {}
    for category, entry in meta_dict.pop("diagnostics").items():
        examples = entry["examples"]
        if event_ids is not None and examples and isinstance(examples[0], int):
            examples = [event_ids[row] for row in examples]
        diagnostics[category] = RowIssues(count=entry["count"], examples=examples)
    return Meta(diagnostics=diagnostics, **meta_dict)


def _resolve_mesh(nodes_per_decade, nodes_per_sigma) -> dict:
    """Validate the survival method's mesh keywords; return them resolved
    to the package defaults where None."""
    density = DEFAULT_NODES_PER_DECADE if nodes_per_decade is None else int(nodes_per_decade)
    if density < 1:
        raise EltqError(f"nodes_per_decade must be >= 1, got {nodes_per_decade}")
    sigma = DEFAULT_NODES_PER_SIGMA if nodes_per_sigma is None else int(nodes_per_sigma)
    if sigma < 1:
        raise EltqError(f"nodes_per_sigma must be >= 1, got {nodes_per_sigma}")
    return {"nodes_per_decade": density, "nodes_per_sigma": sigma}


def _check_method_options(method, nodes_per_decade, overflow, nodes_per_sigma=None) -> dict:
    """Validate the method keywords; return the resolved mesh keywords."""
    if method not in _METHODS:
        raise EltqError(f"unknown method {method!r} (expected quantiles or survival)")
    if overflow not in _OVERFLOW_POLICIES:
        raise EltqError(
            f"unknown overflow {overflow!r} (expected top_bucket, drop, or error)")
    return _resolve_mesh(nodes_per_decade, nodes_per_sigma)


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
    method="survival",
    quadrature_points=32,
    nodes_per_decade=None,
    nodes_per_sigma=None,
    overflow="top_bucket",
    normalize=False,
    mean_is_loss_rate=False,
    variance_policy="bernoulli",
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
        Number of grid nodes n; auto-derived as ceil(max M / h) + 1 when
        None, which covers every loss. A shorter explicit length is a
        window, handled by ``overflow``.
    method : {"survival", "quantiles"}, default "survival"
        How the ``secondary`` grid is built. ``survival``: the portfolio
        exceedance rate evaluated exactly at log-spaced nodes, interpolated
        monotonically and bucketed by the nearest rule, each event's
        endpoint handled per event (spec section 5A). ``quantiles``: K
        midpoint beta quantiles per event, deposited by the linear rule
        (spec section 5); it cuts every event's survival function at 1/(2K),
        which loses expected loss at small alpha.
    quadrature_points : int, default 32
        Number K of midpoint beta quantiles per nondegenerate event
        (``quantiles`` method).
    nodes_per_decade : int, optional
        Base node density of the ``survival`` method; the package default
        (100) when None.
    nodes_per_sigma : int, optional
        The ``survival`` method's accuracy knob: nodes per standard deviation
        a bell-shaped event (both beta parameters above 1) asks of the fine
        mesh around its mean. The default, 12, holds the curve within 1e-4
        relative of each event's survival where that is at least 1e-4 and
        within 1e-3 where it is at least 1e-8; cost grows linearly with it,
        and only in such events' ranges. An event narrower than a base
        interval takes the narrow route instead, so the fine density is at
        most this times ``nodes_per_decade``.
    overflow : {"top_bucket", "drop", "error"}, default "top_bucket"
        What happens to mass above the top bucket's upper edge, (n - 1/2) h,
        when ``grid_length`` is a window: added to the top node (rate exact,
        mean understated), discarded (rate short), or a ``GridOverflowError``.
        Either non-error policy reports the overflow's rate and displaced
        moment in ``meta.rate_above_grid`` and ``meta.mean_above_grid``.
    normalize : bool, default False
        Divide both outputs by the total contributing rate.
    mean_is_loss_rate : bool, default False
        Interpret mean_loss as a rate on [0, 1], scaled by max_loss.
    variance_policy : {"bernoulli", "error", "drop", "atom"}, default "bernoulli"
        Treatment of rows with inadmissible beta variance, sigma^2 >= mu (M - mu).
        ``bernoulli`` replaces the row by the variance maximizer on [0, M]:
        two atoms, rate (1 - p) at 0 and rate p at M with p = mu / M, so the
        rate and mean are exact. ``error`` raises; ``drop`` removes the row;
        ``atom`` collapses it to an atom at mu. The ``clamp`` of versions
        before 2.0.0 (sigma clamped to 0.999 of the bound) is refused.
    threads : int, optional
        Worker threads; None uses all cores. Results are bitwise identical
        for any thread count. Ignored by the reference backend.
    chunk_rows : int, optional
        Rows per parallel chunk; part of the deterministic contract. The
        default is a rule of the row count and the grid length under the
        quantiles method, ``clamp(max(ceil(rows / 64), ceil(n / 64)), 4096,
        65536)``, and 256 under the survival method. Ignored by the reference
        backend.
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
    mesh = _check_method_options(method, nodes_per_decade, overflow, nodes_per_sigma)
    common = {
        "bucket_size": bucket_size,
        "grid_length": grid_length,
        "quadrature_points": quadrature_points,
        "normalize": normalize,
        "mean_is_loss_rate": mean_is_loss_rate,
        "variance_policy": variance_policy,
    }
    survival = {"method": method, "overflow": overflow, **mesh}
    if BACKEND == "rust":
        raw, secondary, meta_dict = _rust_quantize_arrays(
            *cols, **common, **survival, threads=threads, chunk_rows=chunk_rows
        )
        return raw, secondary, _build_meta(meta_dict, ids)
    return _reference.quantize_reference(*cols, **common, **survival, event_ids=ids)


def moments_arrays(
    rate,
    max_loss,
    mean_loss,
    sd_indep,
    sd_corr,
    *,
    mean_is_loss_rate=False,
    variance_policy="bernoulli",
    event_ids=None,
) -> ModelMoments:
    """Model moments of a one-row ELT with no loss grid (spec section 4B).

    Classifies the rows exactly as ``quantize_arrays`` does and returns the
    total rate, the rate-weighted raw moments ``C_r = sum lambda P(X^r)``
    for ``r = 1, 2, 3`` (the annual cumulants under compound Poisson
    occurrence), the severity moments ``C_r / total_rate``, the zero rate,
    the support of every row's effective law, the counts by law and the
    diagnostics. Each row is an atom (``P X^r = mu^r``), a beta
    (``mu M^(r-1) prod_{k<r} (alpha + k) / (alpha + beta + k)``) or a
    two-point Bernoulli row (``mu M^(r-1)``), all closed forms, so there is
    no bucket size or grid length and nothing depends on a future grid.

    Parameters
    ----------
    rate, max_loss, mean_loss, sd_indep, sd_corr : array_like
        One-dimensional numeric columns of equal length, as
        ``quantize_arrays`` takes them.
    mean_is_loss_rate : bool, default False
        Interpret mean_loss as a rate on [0, 1], scaled by max_loss.
    variance_policy : {"bernoulli", "error", "drop", "atom"}, default "bernoulli"
        Treatment of rows with inadmissible beta variance, as in
        ``quantize_arrays``.
    event_ids : sequence, optional
        Per-row identifiers used in diagnostic examples (audit only).

    Returns
    -------
    ModelMoments
        ``method`` is ``closed_form``; an order whose moment leaves double
        precision is infinite and flagged in ``moment_overflow``.
    """
    cols = tuple(
        _column(name, values)
        for name, values in zip(
            _CANONICAL_COLUMNS, (rate, max_loss, mean_loss, sd_indep, sd_corr)
        )
    )
    ids = list(event_ids) if event_ids is not None else None
    if _rust_moments_arrays is None:
        from eltq import reference as _ref

        return _ref.moments_reference(*cols, mean_is_loss_rate=mean_is_loss_rate,
                                      variance_policy=variance_policy, event_ids=ids)
    out = _rust_moments_arrays(*cols, mean_is_loss_rate=mean_is_loss_rate, variance_policy=variance_policy)
    diagnostics = {}
    for category, entry in out.pop("diagnostics").items():
        examples = entry["examples"]
        if ids is not None:
            examples = [ids[row] for row in examples]
        diagnostics[category] = RowIssues(count=entry["count"], examples=examples)
    total_rate = out["total_rate"]
    raw = tuple(out.pop("raw_moments"))
    return ModelMoments(
        raw_moments=raw,
        severity_moments=tuple(c / total_rate if total_rate > 0.0 else float("nan") for c in raw),
        analytic_aal=raw[0],
        moment_overflow=tuple(out.pop("moment_overflow")),
        diagnostics=diagnostics,
        **out,
    )


def tail_moments_arrays(
    rate,
    max_loss,
    mean_loss,
    sd_indep,
    sd_corr,
    *,
    edge,
    top_node=None,
    mean_is_loss_rate=False,
    variance_policy="bernoulli",
    threads=None,
    event_ids=None,
) -> TailMoments:
    """Tail measures of a one-row ELT at candidate edges, with no grid (spec
    section 4B).

    For each edge ``t`` with node ``c``: the tail rate ``R(t) = sum lambda
    P(X > t)``, the tail moments ``T_r(t) = sum lambda P(X^r; X > t)`` and
    the excess moments ``D_r(t, c) = sum lambda P(X^r - c^r; X > t)`` for
    ``r = 1, 2, 3``, and apart from them the atoms in the final half bucket
    ``(c, t]``. A consumer sizing a grid of spacing ``h`` and length ``n``
    asks at ``t = (n - 1/2) h`` and ``c = (n - 1) h``: under the
    ``top_bucket`` overflow policy the annual cumulants fall by exactly
    ``D_r``, and ``R`` and ``D_1`` are the grid's ``rate_above_grid`` and
    ``mean_above_grid``. Every row is exact by its law (the partial-moment
    identity ``P(X^r; X > t) = P(X^r) I-bar_{t/M}(alpha + r, beta)`` and a
    centred recurrence that keeps a short tail's digits; the narrow kernel
    and the normal law at large parameters); ``R`` and ``T_r`` are
    nonincreasing in ``t``, so a caller can bisect for an edge.

    Parameters
    ----------
    rate, max_loss, mean_loss, sd_indep, sd_corr : array_like
        One-dimensional numeric columns of equal length, as
        ``quantize_arrays`` takes them.
    edge : array_like
        Candidate edges ``t >= 0``, any order.
    top_node : array_like, optional
        The node ``c`` of each edge, ``0 <= c <= t``; the edge itself when
        None (then ``D_r`` is the excess over the edge).
    mean_is_loss_rate, variance_policy : as ``quantize_arrays``.
    threads : int, optional
        Worker threads; None uses all cores. Results are bitwise identical
        for any thread count. Ignored by the reference backend.
    event_ids : sequence, optional
        Per-row identifiers used in diagnostic examples (audit only).

    Returns
    -------
    TailMoments
        Arrays per edge: ``rate_above`` and ``top_half_rate`` of shape
        ``(E,)``; ``moments_above``, ``excess_above`` and ``top_half_excess``
        of shape ``(E, 3)``, column ``r - 1`` for order ``r``.
    """
    cols = tuple(
        _column(name, values)
        for name, values in zip(
            _CANONICAL_COLUMNS, (rate, max_loss, mean_loss, sd_indep, sd_corr)
        )
    )
    edges = np.ascontiguousarray(np.atleast_1d(np.asarray(edge, dtype=np.float64)))
    nodes = None if top_node is None else np.ascontiguousarray(
        np.atleast_1d(np.asarray(top_node, dtype=np.float64)))
    if edges.ndim != 1:
        raise EltqError(f"edge must be one-dimensional, got shape {edges.shape}")
    ids = list(event_ids) if event_ids is not None else None
    if _rust_tail_moments_arrays is None:
        from eltq import reference as _ref

        return _ref.tail_moments_reference(*cols, edge=edges, top_node=nodes,
                                           mean_is_loss_rate=mean_is_loss_rate,
                                           variance_policy=variance_policy, event_ids=ids)
    out = _rust_tail_moments_arrays(*cols, edge=edges, top_node=nodes, mean_is_loss_rate=mean_is_loss_rate,
                                    variance_policy=variance_policy, threads=threads)
    diagnostics = {}
    for category, entry in out.pop("diagnostics").items():
        examples = entry["examples"]
        if ids is not None:
            examples = [ids[row] for row in examples]
        diagnostics[category] = RowIssues(count=entry["count"], examples=examples)
    for name in ("moments_above", "excess_above", "top_half_excess"):
        out[name] = out[name].reshape(-1, 3)
    return TailMoments(diagnostics=diagnostics, moment_overflow=tuple(out.pop("moment_overflow")), **out)


def quantize_csv(
    path: str | PathLike,
    *,
    bucket_size,
    grid_length=None,
    method="survival",
    quadrature_points=32,
    nodes_per_decade=None,
    nodes_per_sigma=None,
    overflow="top_bucket",
    normalize=False,
    mean_is_loss_rate=False,
    variance_policy="bernoulli",
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
    parsing and computation both run in Rust. Keywords are those of
    ``quantize_arrays``.
    """
    mesh = _check_method_options(method, nodes_per_decade, overflow, nodes_per_sigma)
    common = {
        "bucket_size": bucket_size,
        "grid_length": grid_length,
        "quadrature_points": quadrature_points,
        "normalize": normalize,
        "mean_is_loss_rate": mean_is_loss_rate,
        "variance_policy": variance_policy,
    }
    survival = {"method": method, "overflow": overflow, **mesh}
    if BACKEND == "rust":
        raw, secondary, meta_dict = _rust_quantize_csv(
            path, **common, **survival, threads=threads, chunk_rows=chunk_rows,
            columns=columns,
        )
        return raw, secondary, _build_meta(meta_dict)
    data, ids = _read_csv_fallback(path, columns)
    cols = tuple(data[key] for key in _CANONICAL_COLUMNS)
    return _reference.quantize_reference(*cols, **common, **survival, event_ids=ids)


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


class ExceedanceCurve:
    """The portfolio exceedance curve of an ELT (spec section 5A).

    Built by ``exceedance_curve``. ``loss`` holds the nodes and
    ``exceedance_rate`` the exact exceedance rate of the beta events at
    them; ``atom_loss`` and ``atom_rate`` the atoms (degenerate rows, policy
    atoms, both atoms of a Bernoulli row), which ``exceedance`` adds as
    steps. ``meta`` is a ``CurveMeta``. A curve built once serves any number
    of grids with ``bucket_size >= 2 min_loss``.
    """

    def __init__(self, backend, meta: CurveMeta):
        self._backend = backend
        self.meta = meta

    @property
    def loss(self) -> np.ndarray:
        return np.asarray(self._backend.loss, dtype=np.float64)

    @property
    def exceedance_rate(self) -> np.ndarray:
        return np.asarray(self._backend.exceedance_rate, dtype=np.float64)

    @property
    def atom_loss(self) -> np.ndarray:
        return np.asarray(self._backend.atom_loss, dtype=np.float64)

    @property
    def atom_rate(self) -> np.ndarray:
        return np.asarray(self._backend.atom_rate, dtype=np.float64)

    def exceedance(self, x) -> np.ndarray:
        """``Lambda(x)``, the rate of events with loss above ``x``, atoms
        included; NaN below ``min_loss``, 0 above the largest loss."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        return np.asarray(self._backend.exceedance(x.ravel()), dtype=np.float64).reshape(x.shape)

    def grid(self, bucket_size, grid_length=None, *, normalize=False,
             overflow="top_bucket"):
        """The secondary grid at ``bucket_size`` (at least twice
        ``min_loss``) with ``grid_length`` nodes, derived to cover the
        largest loss when None, under the overflow policy; returns
        ``(array, GridMeta)``."""
        if overflow not in _OVERFLOW_POLICIES:
            raise EltqError(
                f"unknown overflow {overflow!r} (expected top_bucket, drop, or error)")
        secondary, meta = self._backend.grid(
            float(bucket_size), None if grid_length is None else int(grid_length),
            normalize=bool(normalize), overflow=overflow)
        if not isinstance(meta, GridMeta):
            meta = GridMeta(**meta)
        return np.asarray(secondary, dtype=np.float64), meta

    def oep_pml(self, return_periods) -> np.ndarray:
        """The occurrence exceedance loss at each return period ``T``
        under Poisson occurrence: the loss ``x`` with
        ``Lambda(x) = -log(1 - 1 / T)``, found by bisection in log loss
        (an atom's loss when the step at that atom jumps the target). NaN
        where the target rate exceeds ``Lambda(min_loss)``, that is, below
        the curve's range; 0 for ``T <= 1``."""
        t = np.ascontiguousarray(return_periods, dtype=np.float64)
        shape = t.shape
        t = t.ravel()
        out = np.full(t.shape, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            target = np.where(t > 1.0, -np.log1p(-1.0 / t), np.inf)
        lo_x = self.meta.min_loss
        hi_x = max(self.meta.max_loss, lo_x * (1.0 + 1e-12))
        top = float(self.exceedance(np.array([lo_x]))[0])
        solve = (t > 1.0) & (target <= top)
        out[t <= 1.0] = 0.0
        if not solve.any():
            return out.reshape(shape)
        lo = np.full(solve.sum(), np.log(lo_x))
        hi = np.full(solve.sum(), np.log(hi_x))
        goal = target[solve]
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            above = self.exceedance(np.exp(mid)) > goal
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)
            if np.all(hi - lo <= 1e-15 * np.abs(hi)):
                break
        # The bracket is a rounding step wide: an atom inside it is the
        # step that jumps the target, and its loss is the answer.
        x_lo, x_hi = np.exp(lo), np.exp(hi)
        found = x_hi.copy()
        atoms = np.sort(self.atom_loss)
        if atoms.size:
            # The first atom at or above x_lo, if any; an answer above every
            # atom has none (2.0.1: the clipped index used to return the
            # largest atom there).
            j = np.searchsorted(atoms, x_lo, side="left")
            inside = j < atoms.size
            inside[inside] = atoms[j[inside]] <= x_hi[inside]
            found[inside] = atoms[j[inside]]
        out[solve] = found
        return out.reshape(shape)


def exceedance_curve(
    rate,
    max_loss,
    mean_loss,
    sd_indep,
    sd_corr,
    *,
    min_loss,
    nodes_per_decade=None,
    nodes_per_sigma=None,
    mean_is_loss_rate=False,
    variance_policy="bernoulli",
    threads=None,
    chunk_rows=None,
    event_ids=None,
) -> ExceedanceCurve:
    """Build the portfolio exceedance curve of an ELT from ``min_loss`` up.

    Parameters
    ----------
    rate, max_loss, mean_loss, sd_indep, sd_corr : array_like
        The ELT columns, as for ``quantize_arrays``.
    min_loss : float
        The curve's lowest node, the smallest loss it answers for; a grid
        made from it needs ``bucket_size >= 2 * min_loss``. Required: there
        is no natural default.
    nodes_per_decade, nodes_per_sigma : int, optional
        The base density and the accuracy knob, as for ``quantize_arrays``;
        the package defaults when None.
    mean_is_loss_rate, variance_policy, threads, chunk_rows, event_ids
        As for ``quantize_arrays``.

    Returns
    -------
    ExceedanceCurve
    """
    cols = tuple(
        _column(name, values)
        for name, values in zip(
            _CANONICAL_COLUMNS, (rate, max_loss, mean_loss, sd_indep, sd_corr)
        )
    )
    ids = list(event_ids) if event_ids is not None else None
    mesh = _resolve_mesh(nodes_per_decade, nodes_per_sigma)
    if _rust_exceedance_curve is not None:
        backend = _rust_exceedance_curve(
            *cols, min_loss=float(min_loss), **mesh,
            mean_is_loss_rate=mean_is_loss_rate, variance_policy=variance_policy,
            threads=threads, chunk_rows=chunk_rows,
        )
        meta_dict = backend.meta()
        diagnostics = {}
        for category, entry in meta_dict.pop("diagnostics").items():
            examples = entry["examples"]
            if ids is not None and examples:
                examples = [ids[row] for row in examples]
            diagnostics[category] = RowIssues(count=entry["count"], examples=examples)
        return ExceedanceCurve(backend, CurveMeta(diagnostics=diagnostics, **meta_dict))
    from eltq import reference as _ref

    backend, meta = _ref.exceedance_curve_reference(
        *cols, min_loss=float(min_loss), **mesh,
        mean_is_loss_rate=mean_is_loss_rate, variance_policy=variance_policy,
        event_ids=ids,
    )
    return ExceedanceCurve(backend, meta)


_MULTI_COLUMNS = ("event_id", "unit_id", "rate", "mean_loss", "sd_indep",
                  "sd_corr", "max_loss")


def _reference_multiunit():
    try:
        from eltq import reference_multiunit
    except ImportError as scipy_missing:
        raise ImportError(
            "the multi-unit route currently runs on the SciPy reference "
            "(eltq.reference_multiunit); install scipy to use it"
        ) from scipy_missing
    return reference_multiunit


def _via_reference(function, *args, **kwargs):
    """Call a reference-module function, translating its exception classes
    into the package's (they coincide on the reference backend)."""
    module = _reference_multiunit()
    if BACKEND != "rust":
        return function(module)(*args, **kwargs)
    from eltq import reference as _ref

    try:
        return function(module)(*args, **kwargs)
    except _ref.GridOverflowError as error:
        translated = GridOverflowError(str(error))
        translated.required_length = error.required_length
        translated.examples = error.examples
        raise translated from None
    except _ref.VarianceError as error:
        raise VarianceError(str(error)) from None
    except module.MultiUnitError as error:
        raise MultiUnitError(str(error)) from None
    except _ref.EltqError as error:
        raise EltqError(str(error)) from None


def _factorize(values):
    """Return ``(codes, uniques)``: dense int64 codes by first appearance and
    the distinct values in that order. Integer arrays pass through as their
    own codes (uniques still in first-seen order)."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise EltqError(f"identifier columns must be one-dimensional, got shape {array.shape}")
    try:
        import pandas as pd

        codes, uniques = pd.factorize(array, sort=False)
        return np.ascontiguousarray(codes, dtype=np.int64), list(uniques)
    except ImportError:
        pass
    table: dict = {}
    uniques: list = []
    codes = np.empty(array.size, dtype=np.int64)
    for i, value in enumerate(array.tolist()):
        code = table.get(value)
        if code is None:
            code = len(uniques)
            table[value] = code
            uniques.append(value)
        codes[i] = code
    return codes, uniques


def _rust_multiunit(event_id, unit_id, columns, *, terms, bucket_size, grid_length, views,
                    z_nodes, z_rule, normalize, variance_policy, clamp_factor, threads,
                    chunk_rows, roll_up_diagnostic, convolution, coupling, pair=None,
                    first_moments=False):
    """Run the Rust engine on factorized identifiers; build ``MultiMeta``.
    With ``pair`` or ``first_moments`` the occurrence outputs come back as a
    third value, in the reference's form."""
    from eltq.terms import check_grid_alignment

    event_codes, event_uniques = _factorize(event_id)
    unit_codes, unit_uniques = _factorize(unit_id)
    unit_ids = [str(u) for u in unit_uniques]
    try:
        parsed = parse_terms(terms)
        for unit_terms in parsed.values():
            check_grid_alignment(unit_terms, float(bucket_size))
    except TermsError as error:
        raise EltqError(str(error)) from error
    spec = []
    for uid in unit_ids:
        t = parsed.get(uid)
        if t is None:
            spec.append((uid, float("inf"), 0.0, []))
        else:
            spec.append((uid, float(t.limit), float(t.ded),
                         [(c.share, c.limit, c.attach) for c in t.clauses]))
    pair_codes = None
    if pair is not None:
        pair = tuple(str(u) for u in pair)
        if len(pair) != 2:
            raise EltqError(f"pair must name two units, got {pair}")
        pair_codes = tuple(unit_ids.index(u) if u in unit_ids else len(unit_ids) for u in pair)
        for u, code in zip(pair, pair_codes):
            if code == len(unit_ids):
                raise EltqError(f"unit {u!r} does not appear in the ELT")
    views_out, meta_dict, extras = _rust_multiunit_views(
        event_codes, unit_codes, *columns, terms=spec, bucket_size=float(bucket_size),
        grid_length=grid_length, z_nodes=int(z_nodes), z_rule=z_rule,
        views=list(views), normalize=normalize, variance_policy=variance_policy,
        clamp_factor=clamp_factor, threads=threads, chunk_rows=chunk_rows,
        roll_up_diagnostic=roll_up_diagnostic, convolution=convolution, coupling=coupling,
        pair=pair_codes, first_moments=first_moments,
    )
    event_labels = np.asarray(event_id)
    unit_labels = np.asarray(unit_id)
    diagnostics = {}
    for category, entry in meta_dict.pop("diagnostics").items():
        examples = [f"{event_labels[row]}/{unit_labels[row]}" for row in entry["examples"]]
        diagnostics[category] = RowIssues(count=entry["count"], examples=examples)
    roll_up = meta_dict.pop("roll_up")
    if roll_up is not None:
        for entry in roll_up:
            entry["event_id"] = event_labels[entry.pop("event_row")]
    matched_shortfall = meta_dict.pop("matched_shortfall")
    if matched_shortfall is not None:
        for entry in matched_shortfall:
            entry["event_id"] = event_labels[entry.pop("event_row")]
    meta = MultiMeta(
        views=tuple(meta_dict.pop("views")),
        unit_ids=tuple(unit_ids),
        terms_echo={uid: t.text() for uid, t in parsed.items()},
        unused_terms=tuple(uid for uid in parsed if uid not in set(unit_ids)),
        diagnostics=diagnostics,
        roll_up=roll_up,
        matched_shortfall=matched_shortfall,
        **meta_dict,
    )
    del event_uniques
    if pair is None and not first_moments:
        return views_out, meta
    out = {}
    if "pair" in extras:
        pmf, n_u, n_w, rate, events = extras["pair"]
        h = float(bucket_size)
        out["pair"] = (pmf, h * np.arange(n_u), h * np.arange(n_w), rate, events)
    if "first_moments" in extras:
        arrays, identity_error = extras["first_moments"]
        out["first_moments"] = ({uid: arrays[code] for code, uid in enumerate(unit_ids)}, identity_error)
    return views_out, meta, out


def multiunit_views(
    event_id,
    unit_id,
    rate,
    mean_loss,
    sd_indep,
    sd_corr,
    max_loss,
    *,
    terms=None,
    bucket_size,
    grid_length=None,
    views=VIEWS,
    z_nodes=32,
    z_rule="stratified",
    normalize=False,
    variance_policy="bernoulli",
    clamp_factor=0.999,
    threads=None,
    chunk_rows=None,
    roll_up_diagnostic=False,
    convolution="auto",
    coupling="burke",
):
    """Total event severity by view for a multi-unit ELT; return ``(views, meta)``.

    Rows are event x unit (contiguous per event, one row per unit, rate
    constant within an event). Per unit, the conditional loss is a beta on
    [0, M] with total sd ``sd_indep + sd_corr``, or the two-point law on 0
    and M when that sd is inadmissible for a beta (spec section 4A); units
    of one event share a common factor through Burke's one-factor Gaussian
    copula and are independent given it. Per-unit DecL-style ``terms`` (see
    ``eltq.terms``) define the views: ``ground_up`` = X, ``gross`` after
    deductible and limit, ``ceded`` by the occurrence clauses applied to
    gross, ``net`` = gross - ceded. The total across units of each view is
    accumulated across events weighted by rate, exactly as ``quantize_arrays``
    accumulates single-unit rows.

    Parameters
    ----------
    event_id, unit_id : array_like
        Row identifiers (any hashable / string values).
    rate, mean_loss, sd_indep, sd_corr, max_loss : array_like
        Numeric columns; ``rate`` constant within an event.
    terms : str | sequence of str | Mapping | None
        One ``unit <id> [<limit> xs <ded>] [occurrence net of ...]`` line per
        unit, or a mapping of structured terms; units without terms are
        untouched (gross = ground_up, ceded = 0).
    bucket_size : float
        Grid spacing h; every monetary breakpoint in ``terms`` must be a
        multiple of h (no silent rounding).
    grid_length : int, optional
        Number of grid nodes; auto-derived from the largest event span
        (each unit's support rounded up to the grid, summed) when None.
    views : sequence of str, default all four
        Views to compute and return.
    z_nodes : int, default 32
        Shared-factor nodes Q under a node rule; the number of equal body
        strata under the stratified rule, whose upper-tail ladder is fixed
        (``meta.z_strata`` reports the total).
    z_rule : {"stratified", "midpoint", "gauss_hermite"}, default "stratified"
        Shared-factor rule (spec section 4A). The stratified rule averages
        each unit's conditional law over strata of the factor exactly, so
        every marginal is exact; the units are independent given the
        stratum, and the tail ladder holds that dependence error to about
        3e-3 of the event total's exceedance at every level
        (`benchmarks/z-convergence-2.6.0.md`). The node rules cut or miss
        the tails of units with a loading near one.
    normalize : bool, default False
        Divide every view by the total contributing rate.
    variance_policy : {"bernoulli", "clamp", "error", "drop", "atom"}, default "bernoulli"
        Treatment of a unit row with inadmissible beta variance: the
        two-point law on 0 and M with ``P(X = M) = mu / M`` given the shared
        factor (reported under ``variance_bernoulli``), the clamp of versions
        before 2.5.0 (deprecated, with a ``DeprecationWarning``; removed at
        3.0.0), a ``VarianceError``, a dropped row, or an atom at the mean.
    clamp_factor : float, default 0.999
        Clamp target as a fraction of the admissible sigma bound.
    threads : int, optional
        Worker threads; None uses all cores. Results are bitwise identical
        for any thread count. Ignored by the reference backend.
    chunk_rows : int, optional
        Rows per parallel chunk (rounded up to whole events; default 2048);
        part of the deterministic contract. Ignored by the reference backend.
    roll_up_diagnostic : bool, default False
        Per event with two or more units, report the sd and 99th-percentile
        ratios of the exact ground-up total against the rolled single-row
        beta (``meta.roll_up``).
    convolution : {"auto", "direct", "fft"}, default "auto"
        How units are convolved across an event on the Rust backend: by
        estimated cost, or forced. Ignored by the reference backend.
    coupling : {"burke", "comonotone", "independent", "homer", "matched"}, default "burke"
        The coupling rule of spec section 4A, applied to every event.
        ``burke``: the loading ``sd_corr / hypot(sd_indep, sd_corr)`` on the
        common factor (Burke 2016). ``comonotone``: every loading one, one
        uniform per occurrence, computed exactly as the comonotone sum of the
        bucketed view marginals. ``independent``: every loading zero.
        ``homer``: Homer and Li 2017, the event's rows rolled into one and
        the rolled loss allocated pro rata to the means, uncapped as
        published (a unit can be allocated more than its exposed value;
        ``meta.homer_excess_mean`` reports the expected excess), the rolled
        row bucketed directly and pushed through the composed allocation map
        per view. ``matched``: Burke's loadings scaled by a common factor
        per event so that the event total's variance equals the roll-up
        convention's ``(sd_I,R + sd_C,R)^2`` (the one alternative with a
        rationale; no published source), saturating at the largest loading
        one, with the events out of reach in ``meta.matched_shortfall``; it
        costs a bisection of about thirty variance evaluations per event.
        Echoed as ``meta.coupling``.

    Returns
    -------
    views : dict[str, numpy.ndarray]
        View name -> length-n float64 rate-weight array; the loss at element
        i is i * h.
    meta : MultiMeta
    """
    if _rust_multiunit_views is not None:
        _reference_multiunit().warn_if_clamp(variance_policy)
        columns = tuple(
            _column(name, values)
            for name, values in zip(
                _MULTI_COLUMNS[2:], (rate, mean_loss, sd_indep, sd_corr, max_loss)
            )
        )
        return _rust_multiunit(
            event_id, unit_id, columns, terms=terms, bucket_size=bucket_size,
            grid_length=grid_length, views=views, z_nodes=z_nodes, z_rule=z_rule,
            normalize=normalize, variance_policy=variance_policy,
            clamp_factor=clamp_factor, threads=threads, chunk_rows=chunk_rows,
            roll_up_diagnostic=roll_up_diagnostic, convolution=convolution,
            coupling=coupling,
        )
    return _via_reference(
        lambda module: module.multiunit_views_reference,
        event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss,
        terms=terms, bucket_size=bucket_size, grid_length=grid_length,
        views=views, z_nodes=z_nodes, z_rule=z_rule, normalize=normalize,
        variance_policy=variance_policy, clamp_factor=clamp_factor,
        roll_up_diagnostic=roll_up_diagnostic, coupling=coupling,
    )


def multiunit_moments(
    event_id,
    unit_id,
    rate,
    mean_loss,
    sd_indep,
    sd_corr,
    max_loss,
    *,
    coupling="burke",
    variance_policy="bernoulli",
    clamp_factor=0.999,
    rtol=1e-6,
    max_refinements=3,
    threads=None,
    chunk_rows=None,
) -> ModelMoments:
    """Model moments of a multi-unit ELT's event totals, with no grid (spec
    section 4B).

    The rows are validated and classified exactly as ``multiunit_views``
    does. Each event total ``S = sum_u X_u`` contributes its raw moments
    ``P(S^r)``, ``r = 1, 2, 3``, rate-weighted: ``raw_moments`` are the annual
    cumulants of the compound Poisson total. Given the shared factor the
    units are independent, so the moments of ``S`` follow from adding the
    units one at a time; the marginal parts are closed forms and the first
    moment is exact. An event with one contributing unit, and every event
    under ``independent`` or ``homer`` (the rolled row), takes closed forms;
    under ``burke`` and ``comonotone`` the cross parts are integrated over
    the factor (each unit's conditional moments from a table of its share
    against its normal score), at two levels of the rules compared and
    refined until they agree within ``rtol`` in ``C_2`` and ``C_3``. These
    are the moments of the coupling's law, not of the stratified rule's
    approximation of it that the ``multiunit_views`` grids realize.

    Parameters
    ----------
    event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss : array_like
        As ``multiunit_views`` takes them.
    coupling : {"burke", "comonotone", "independent", "homer"}, default "burke"
        The coupling rule of spec section 4A. ``matched`` is refused: its
        scale is defined through a grid (spec section 11).
    variance_policy, clamp_factor : as ``multiunit_views``.
    rtol : float, default 1e-6
        Relative tolerance of ``C_2`` and ``C_3``, per event.
    max_refinements : int, default 3
        Rounds of refinement (each halving the rules' widths and tightening
        the tables sixteenfold) before an event is reported unconverged.
    threads : int, optional
        Worker threads; None uses all cores. Results are bitwise identical
        for any thread count. Ignored by the reference backend.
    chunk_rows : int, optional
        Rows per parallel chunk (rounded up to whole events; default 2048).
        Ignored by the reference backend.

    Returns
    -------
    ModelMoments
        With ``support`` per event (the sum of its units' supports, raised to
        the rolled row's under ``homer``; NaN where no unit contributes),
        ``method`` ``factor_integration`` when some event was integrated,
        ``achieved_error`` the rate-weighted differences of the two levels
        relative to ``C_2`` and ``C_3``, ``marginal_residual``,
        ``evaluations`` and the events that did not converge.
    """
    if _rust_multiunit_moments is None:
        return _via_reference(
            lambda module: module.multiunit_moments_reference,
            event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss,
            coupling=coupling, variance_policy=variance_policy, clamp_factor=clamp_factor,
            rtol=rtol, max_refinements=max_refinements,
        )
    _reference_multiunit().warn_if_clamp(variance_policy)
    if coupling == "matched":
        raise EltqError("the matched coupling's scale is defined through a grid; multiunit_moments refuses it "
                        "until that scale is defined without one (spec section 11)")
    columns = tuple(
        _column(name, values)
        for name, values in zip(_MULTI_COLUMNS[2:], (rate, mean_loss, sd_indep, sd_corr, max_loss))
    )
    event_codes, _ = _factorize(event_id)
    unit_codes, _ = _factorize(unit_id)
    out = _rust_multiunit_moments(
        event_codes, unit_codes, *columns, coupling=coupling, variance_policy=variance_policy,
        clamp_factor=clamp_factor, rtol=float(rtol), max_refinements=int(max_refinements),
        threads=threads, chunk_rows=chunk_rows,
    )
    event_labels = np.asarray(event_id)
    unit_labels = np.asarray(unit_id)
    diagnostics = {}
    for category, entry in out.pop("diagnostics").items():
        examples = [f"{event_labels[row]}/{unit_labels[row]}" for row in entry["examples"]]
        diagnostics[category] = RowIssues(count=entry["count"], examples=examples)
    unconverged = [{"event_id": event_labels[row], "error": tuple(error)} for row, error in out.pop("unconverged")]
    total_rate = out["total_rate"]
    raw = tuple(out.pop("raw_moments"))
    return ModelMoments(
        raw_moments=raw,
        severity_moments=tuple(c / total_rate if total_rate > 0.0 else float("nan") for c in raw),
        analytic_aal=raw[0],
        moment_overflow=tuple(out.pop("moment_overflow")),
        achieved_error=tuple(out.pop("achieved_error")),
        diagnostics=diagnostics,
        unconverged=unconverged,
        rtol=float(rtol),
        converged=out["unconverged_events"] == 0,
        coupling=coupling,
        **out,
    )


def _occurrence_run(event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss, *, view,
                    terms, bucket_size, pair, first_moments, **options):
    """One engine run for the occurrence outputs of ``view``, rate-weighted."""
    if view not in VIEWS:
        raise EltqError(f"unknown view {view!r}; expected one of {VIEWS}")
    if _rust_multiunit_views is not None:
        _reference_multiunit().warn_if_clamp(options.get("variance_policy", "bernoulli"))
        columns = tuple(
            _column(name, values)
            for name, values in zip(_MULTI_COLUMNS[2:], (rate, mean_loss, sd_indep, sd_corr, max_loss))
        )
        rust_options = dict(
            z_nodes=32, z_rule="stratified", variance_policy="bernoulli", clamp_factor=0.999,
            threads=None, chunk_rows=None, convolution="auto", coupling="burke",
        )
        rust_options.update(options)
        return _rust_multiunit(
            event_id, unit_id, columns, terms=terms, bucket_size=bucket_size, grid_length=None,
            views=(view,), normalize=False, roll_up_diagnostic=False, pair=pair,
            first_moments=first_moments, **rust_options,
        )
    options = {k: v for k, v in options.items() if k not in ("threads", "chunk_rows", "convolution")}
    return _via_reference(
        lambda module: module.multiunit_views_reference,
        event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss,
        terms=terms, bucket_size=bucket_size, views=(view,), pair=pair,
        first_moments=first_moments, **options,
    )


def occurrence_pair_joint(
    event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss, *, units, view="ground_up",
    terms=None, bucket_size, **options,
):
    """The rate-weighted joint occurrence severity of two units for a view.

    ``S_uw = sum_j lambda_j J_{j,uw}`` on the common grid (spec section 4A,
    bivariate outputs), with ``J_{j,uw}`` the joint law of the two units'
    view losses given event j: each unit's view map is applied to its
    conditional row before the outer product over the factor rule, so the
    joint is of the losses after each unit's occurrence terms, which is
    what terms on a unit's annual loss and an allocation act on. An event
    exposing one of the units puts the other at zero; an event exposing
    neither contributes nothing, so ``meta.mass`` is the Poisson rate of the
    pair's own occurrences and ``pmf / meta.mass`` the discrete bivariate
    severity to compound (``BivariateAggregate(mode="discrete")`` in the
    aggregate package). Under ``coupling="comonotone"`` the merged
    breakpoints deposit each interval at the pair of indices; under
    ``"homer"`` each rolled bucket goes to the pair of its allocations'
    images.

    Parameters
    ----------
    event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss
        The multi-unit ELT, as for ``multiunit_views``.
    units : tuple of two unit ids
        The pair, in the order of the returned axes.
    view : {"ground_up", "gross", "ceded", "net"}
        The view of each unit's loss.
    terms, bucket_size, **options
        As for ``multiunit_views`` (``z_nodes``, ``z_rule``,
        ``variance_policy``, ``clamp_factor``, ``coupling``, ``threads``,
        ``chunk_rows``, ``convolution``).

    Returns
    -------
    pmf : numpy.ndarray
        Row-major ``(n_u, n_w)`` rate weights.
    x_u, x_w : numpy.ndarray
        The two loss grids, ``i * h``.
    meta : PairMeta
    """
    units = tuple(units)
    if len(units) != 2:
        raise EltqError(f"units must name two units, got {units}")
    _, meta, extras = _occurrence_run(
        event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss, view=view, terms=terms,
        bucket_size=bucket_size, pair=units, first_moments=False, **options,
    )
    pmf, x_u, x_w, mass, events = extras["pair"]
    if mass > 0.0:
        p = pmf / mass
        marg_u, marg_w = p.sum(axis=1), p.sum(axis=0)
        mean_u, mean_w = float(marg_u @ x_u), float(marg_w @ x_w)
        sd_u = float(np.sqrt(max(marg_u @ x_u**2 - mean_u**2, 0.0)))
        sd_w = float(np.sqrt(max(marg_w @ x_w**2 - mean_w**2, 0.0)))
        cross = float(x_u @ p @ x_w)
        corr = (cross - mean_u * mean_w) / (sd_u * sd_w) if sd_u > 0.0 and sd_w > 0.0 else float("nan")
    else:
        mean_u = mean_w = sd_u = sd_w = 0.0
        corr = float("nan")
    pair_meta = PairMeta(
        bucket_size=float(bucket_size), units=tuple(str(u) for u in units), view=view,
        coupling=meta.coupling, z_nodes=meta.z_nodes, z_rule=meta.z_rule, z_strata=meta.z_strata,
        grid_lengths=(x_u.size, x_w.size), mass=float(mass), events=int(events),
        grid_mean=(mean_u, mean_w), grid_sd=(sd_u, sd_w), grid_correlation=float(corr),
    )
    return pmf, x_u, x_w, pair_meta


def occurrence_unit_first_moment(
    event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss, *, units=None,
    view="ground_up", terms=None, bucket_size, **options,
):
    """The rate-weighted first-moment vector of each unit at each total.

    ``m_u(y) = sum_j lambda_j P(X_uj; X_j = y)`` on the grid of the view's
    total ``X_j`` across the event's units (spec section 4A): what the
    annual conditional mean of a unit given the portfolio total needs
    (``annual_conditional_mean``), for any number of units, with memory one
    grid per unit. ``sum_u m_u(y) = y s(y)`` with ``s`` the view's
    rate-weighted severity (the ``multiunit_views`` total); the largest
    violation is ``meta.identity_error``.

    Parameters
    ----------
    units : sequence of unit ids, optional
        The units to return; all by default.
    view, terms, bucket_size, **options
        As for ``occurrence_pair_joint``.

    Returns
    -------
    moments : dict[str, numpy.ndarray]
        Unit id -> length-n array of rate-weighted first moments.
    x : numpy.ndarray
        The grid of the view's total, ``i * h``.
    meta : FirstMomentMeta
    """
    _, meta, extras = _occurrence_run(
        event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss, view=view, terms=terms,
        bucket_size=bucket_size, pair=None, first_moments=True, **options,
    )
    moments, identity_error = extras["first_moments"]
    if units is not None:
        wanted = [str(u) for u in units]
        missing = [u for u in wanted if u not in moments]
        if missing:
            raise EltqError(f"units {missing} do not appear in the ELT")
        moments = {u: moments[u] for u in wanted}
    x = float(bucket_size) * np.arange(meta.grid_length)
    fm_meta = FirstMomentMeta(
        bucket_size=float(bucket_size), grid_length=meta.grid_length, units=tuple(moments),
        view=view, coupling=meta.coupling, z_nodes=meta.z_nodes, z_rule=meta.z_rule,
        z_strata=meta.z_strata, total_rate=meta.total_rate, identity_error=float(identity_error),
        total_mean=float(sum(m.sum() for m in moments.values())),
    )
    return moments, x, fm_meta


def annual_conditional_mean(first_moment, annual_density):
    """The annual conditional mean of a unit given the portfolio total.

    By the Mecke identity for the Poisson process of occurrences,
    ``P(A_u; A = s) = sum_y m_u(y) f_A(s - y)``, so ``kappa_u(s) = (m_u *
    f_A)(s) / f_A(s)``: a one-dimensional convolution of the rate-weighted
    first-moment vector ``m_u`` of ``occurrence_unit_first_moment`` with the
    annual density ``f_A`` of the same view's total on the same bucket size
    and origin, which the caller takes from the aggregate package or from
    any compound Poisson calculation on the view's severity (spec section
    4A, bivariate outputs; the multi-unit article, section on the annual
    bivariate aggregate). The convolution is by real FFT, zero-padded to the
    sum of the two lengths so that nothing wraps, and truncated to the
    length of ``f_A``; only ``f_A`` below that length enters, so the result
    is as exact as ``f_A`` is there. The result is NaN where ``f_A`` is
    zero; filtering by a probability floor is the caller's.

    Parameters
    ----------
    first_moment : numpy.ndarray or dict[str, numpy.ndarray]
        ``m_u`` for one unit, or the dict of several from
        ``occurrence_unit_first_moment``.
    annual_density : numpy.ndarray
        ``f_A`` on the grid ``i * h`` from zero.

    Returns
    -------
    numpy.ndarray or dict[str, numpy.ndarray]
        ``kappa_u`` on the grid of ``annual_density``, in the form of
        ``first_moment``.
    """
    f_a = np.asarray(annual_density, dtype=np.float64)
    if isinstance(first_moment, dict):
        return {u: annual_conditional_mean(m, f_a) for u, m in first_moment.items()}
    m_u = np.asarray(first_moment, dtype=np.float64)
    size = m_u.size + f_a.size - 1
    numerator = np.fft.irfft(np.fft.rfft(m_u, size) * np.fft.rfft(f_a, size), size)[:f_a.size]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f_a > 0.0, numerator / f_a, np.nan)


def multiunit_views_dataframe(df, *, terms=None, **options):
    """``multiunit_views`` for a DataFrame with the canonical multi-unit columns
    ``event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss``."""
    return multiunit_views(*(df[name] for name in _MULTI_COLUMNS), terms=terms,
                           **options)


def multiunit_views_into(
    event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss, *, out,
    grid_length=None, **options,
):
    """``multiunit_views`` into caller-provided writable arrays; return ``MultiMeta``.

    ``out`` maps view names to writable, C-contiguous float64 one-dimensional
    arrays of one common length (``numpy.memmap`` included); the views
    computed are exactly ``out``'s keys and the arrays' length defines the
    grid (``grid_length``, if also given, must match). Computation happens in
    backend-owned buffers; results are copied into the borrowed arrays in one
    final assignment per view.
    """
    if not out:
        raise EltqError("out must map at least one view to an output array")
    names = tuple(out)
    length = None
    for name in names:
        _check_output(f"out[{name!r}]", out[name])
        if length is None:
            length = out[name].size
        elif out[name].size != length:
            raise EltqError("all output arrays must have the same length")
    if grid_length is not None and grid_length != length:
        raise EltqError(f"grid_length {grid_length} conflicts with output length {length}")
    views, meta = multiunit_views(
        event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss,
        views=names, grid_length=length, **options,
    )
    for name in names:
        out[name][:] = views[name]
    return meta


def joint_two_units(
    unit1,
    unit2,
    *,
    bucket_size,
    n1,
    n2,
    z_nodes=32,
    z_rule="stratified",
    variance_policy="bernoulli",
    clamp_factor=0.999,
    coupling="burke",
):
    """Exact bucketed bivariate pmf of one event across two units.

    ``unit1`` and ``unit2`` are ``(mean_loss, sd_indep, sd_corr, max_loss)``
    tuples in loss currency units. Returns ``(pmf, x1, x2, meta)``: ``pmf[j,
    k]`` is the probability that unit 1 loses ``x1[j]`` and unit 2 loses
    ``x2[k]`` on the common grid of spacing ``bucket_size`` with ``n1`` and
    ``n2`` nodes (each must cover its unit's maximum loss); ``meta`` is a
    ``JointMeta``. The shared factor is integrated by the stratified rule
    (``z_nodes`` body strata plus the tail ladder; marginals exact) or by
    ``z_nodes`` midpoint or Gauss-Hermite nodes, as ``multiunit_views``. A unit whose
    variance is inadmissible follows ``variance_policy`` as in
    ``multiunit_views`` (``bernoulli``, the deprecated ``clamp``, ``error``
    or ``atom``), and ``coupling`` selects the rule as there; under
    ``comonotone`` and ``homer`` the joint is exact with no factor rule
    (under ``homer`` a unit's grid must reach its share of the rolled
    exposure, ``s_k M_R``).
    """
    if _rust_joint_two_units is not None:
        _reference_multiunit().warn_if_clamp(variance_policy)
        pmf, x1, x2, meta_dict = _rust_joint_two_units(
            tuple(float(v) for v in unit1), tuple(float(v) for v in unit2),
            bucket_size=float(bucket_size), n1=int(n1), n2=int(n2),
            z_nodes=int(z_nodes), z_rule=z_rule, variance_policy=variance_policy,
            clamp_factor=clamp_factor, coupling=coupling,
        )
        meta_dict["notes"] = tuple(tuple(notes) for notes in meta_dict["notes"])
        return pmf, x1, x2, JointMeta(**meta_dict)
    result = _via_reference(
        lambda module: module.secondary_two_units,
        [unit1, unit2], bucket_size, n1, n2, n_z=z_nodes, z_rule=z_rule,
        variance_policy=variance_policy, clamp_factor=clamp_factor, coupling=coupling,
    )
    return result.pmf, result.x1, result.x2, result.meta
