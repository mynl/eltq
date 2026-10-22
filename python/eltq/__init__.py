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

from eltq._meta import JointMeta, Meta, MultiMeta, RowIssues
from eltq.terms import VIEWS, Clause, TermsError, UnitTerms, parse_terms

__version__ = "1.11.6"

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
    "joint_two_units",
    "multiunit_views",
    "multiunit_views_dataframe",
    "multiunit_views_into",
    "parse_terms",
    "TermsError",
    "UnitTerms",
    "Clause",
    "VIEWS",
    "JointMeta",
    "MultiMeta",
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
                    chunk_rows, roll_up_diagnostic, convolution):
    """Run the Rust engine on factorized identifiers; build ``MultiMeta``."""
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
    views_out, meta_dict = _rust_multiunit_views(
        event_codes, unit_codes, *columns, terms=spec, bucket_size=float(bucket_size),
        grid_length=grid_length, z_nodes=int(z_nodes), z_rule=z_rule,
        views=list(views), normalize=normalize, variance_policy=variance_policy,
        clamp_factor=clamp_factor, threads=threads, chunk_rows=chunk_rows,
        roll_up_diagnostic=roll_up_diagnostic, convolution=convolution,
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
    meta = MultiMeta(
        views=tuple(meta_dict.pop("views")),
        unit_ids=tuple(unit_ids),
        terms_echo={uid: t.text() for uid, t in parsed.items()},
        unused_terms=tuple(uid for uid in parsed if uid not in set(unit_ids)),
        diagnostics=diagnostics,
        roll_up=roll_up,
        **meta_dict,
    )
    del event_uniques
    return views_out, meta


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
    z_rule="midpoint",
    normalize=False,
    variance_policy="clamp",
    clamp_factor=0.999,
    threads=None,
    chunk_rows=None,
    roll_up_diagnostic=False,
    convolution="auto",
):
    """Total event severity by view for a multi-unit ELT; return ``(views, meta)``.

    Rows are event x unit (contiguous per event, one row per unit, rate
    constant within an event). Per unit, the conditional loss is a beta on
    [0, M] with total sd ``sd_indep + sd_corr``; units of one event share a
    common factor through Burke's one-factor Gaussian copula and are
    independent given it. Per-unit DecL-style ``terms`` (see
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
        Shared-factor quadrature nodes Q.
    z_rule : {"midpoint", "gauss_hermite"}, default "midpoint"
        Shared-factor quadrature rule.
    normalize : bool, default False
        Divide every view by the total contributing rate.
    variance_policy : {"clamp", "error", "drop", "atom"}, default "clamp"
        Treatment of a unit row with inadmissible beta variance.
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

    Returns
    -------
    views : dict[str, numpy.ndarray]
        View name -> length-n float64 rate-weight array; the loss at element
        i is i * h.
    meta : MultiMeta
    """
    if _rust_multiunit_views is not None:
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
        )
    return _via_reference(
        lambda module: module.multiunit_views_reference,
        event_id, unit_id, rate, mean_loss, sd_indep, sd_corr, max_loss,
        terms=terms, bucket_size=bucket_size, grid_length=grid_length,
        views=views, z_nodes=z_nodes, z_rule=z_rule, normalize=normalize,
        variance_policy=variance_policy, clamp_factor=clamp_factor,
        roll_up_diagnostic=roll_up_diagnostic,
    )


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
    z_nodes=1001,
    z_rule="midpoint",
    variance_policy="clamp",
    clamp_factor=0.999,
):
    """Exact bucketed bivariate pmf of one event across two units.

    ``unit1`` and ``unit2`` are ``(mean_loss, sd_indep, sd_corr, max_loss)``
    tuples in loss currency units. Returns ``(pmf, x1, x2, meta)``: ``pmf[j,
    k]`` is the probability that unit 1 loses ``x1[j]`` and unit 2 loses
    ``x2[k]`` on the common grid of spacing ``bucket_size`` with ``n1`` and
    ``n2`` nodes (each must cover its unit's maximum loss); ``meta`` is a
    ``JointMeta``. The shared factor is integrated by ``z_nodes`` quadrature
    nodes (``z_rule`` midpoint or gauss_hermite); marginals of the pmf agree
    with directly bucketing each beta up to quadrature error.
    """
    if _rust_joint_two_units is not None:
        pmf, x1, x2, meta_dict = _rust_joint_two_units(
            tuple(float(v) for v in unit1), tuple(float(v) for v in unit2),
            bucket_size=float(bucket_size), n1=int(n1), n2=int(n2),
            z_nodes=int(z_nodes), z_rule=z_rule, variance_policy=variance_policy,
            clamp_factor=clamp_factor,
        )
        meta_dict["notes"] = tuple(tuple(notes) for notes in meta_dict["notes"])
        return pmf, x1, x2, JointMeta(**meta_dict)
    result = _via_reference(
        lambda module: module.secondary_two_units,
        [unit1, unit2], bucket_size, n1, n2, n_z=z_nodes, z_rule=z_rule,
        variance_policy=variance_policy, clamp_factor=clamp_factor,
    )
    return result.pmf, result.x1, result.x2, result.meta
