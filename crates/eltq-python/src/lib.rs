//! PyO3 bindings for eltq-core: NumPy conversion and error translation only.
//!
//! Inputs are borrowed as contiguous `float64` slices without copying (the
//! pure-Python wrapper performs explicit contiguous conversion first);
//! results transfer `Vec<f64>` ownership into NumPy via `IntoPyArray` with no
//! completed-array copy. Compute runs detached from the interpreter.
//! Diagnostics report 0-based row indices; the CSV route translates them to
//! event ids here (the ids never enter the computational path), while the
//! array route leaves translation to the Python wrapper.

use std::collections::HashMap;
use std::path::PathBuf;

use eltq_core::csv_input::{CsvColumns, read_elt_csv};
use eltq_core::{
    Clause, Config, Convolution, EltInput, Error as CoreError, JointConfig, Meta, MultiConfig,
    MultiInput, UnitSpec, UnitTerms, VariancePolicy, View, ZRule,
    joint_two_units as core_joint_two_units, multiunit_views as core_multiunit_views, quantize,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;

create_exception!(_eltq, EltqError, PyException, "Base error for eltq.");
create_exception!(
    _eltq,
    VarianceError,
    EltqError,
    "Inadmissible beta variance under the strict policy."
);
create_exception!(
    _eltq,
    GridOverflowError,
    EltqError,
    "Grid does not cover the maximum contributing loss."
);
create_exception!(
    _eltq,
    MultiUnitError,
    EltqError,
    "Structural problem in a multi-unit ELT: events not contiguous, a rate \
     varying within an event, or a unit repeated within an event."
);

fn to_py_err(py: Python<'_>, error: CoreError) -> PyErr {
    match &error {
        CoreError::InadmissibleVariance { .. } | CoreError::UnitInadmissibleVariance { .. } => {
            VarianceError::new_err(error.to_string())
        }
        CoreError::GridOverflow {
            required,
            example_rows,
            ..
        } => {
            // Mirror the reference exception's attributes.
            let build = || -> PyResult<PyErr> {
                let value = py.get_type::<GridOverflowError>().call1((error.to_string(),))?;
                value.setattr("required_length", *required)?;
                value.setattr("examples", example_rows.clone())?;
                Ok(PyErr::from_value(value))
            };
            build().unwrap_or_else(|_| GridOverflowError::new_err(error.to_string()))
        }
        CoreError::MultiUnitStructure(_) => MultiUnitError::new_err(error.to_string()),
        _ => EltqError::new_err(error.to_string()),
    }
}

fn parse_variance_policy(variance_policy: &str, clamp_factor: f64) -> PyResult<VariancePolicy> {
    match variance_policy {
        "clamp" => Ok(VariancePolicy::ClampSigma {
            factor: clamp_factor,
        }),
        "error" => Ok(VariancePolicy::Error),
        "drop" => Ok(VariancePolicy::Drop),
        "atom" => Ok(VariancePolicy::Atom),
        other => Err(EltqError::new_err(format!(
            "unknown variance_policy {other:?} (expected clamp, error, drop, or atom)"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_config(
    bucket_size: f64,
    grid_length: Option<usize>,
    quadrature_points: usize,
    normalize: bool,
    mean_is_loss_rate: bool,
    variance_policy: &str,
    clamp_factor: f64,
    threads: Option<usize>,
    chunk_rows: Option<usize>,
) -> PyResult<Config> {
    let variance_policy = parse_variance_policy(variance_policy, clamp_factor)?;
    let mut config = Config::new(bucket_size);
    config.grid_length = grid_length;
    config.quadrature_points = quadrature_points;
    config.normalize = normalize;
    config.mean_is_loss_rate = mean_is_loss_rate;
    config.variance_policy = variance_policy;
    config.threads = threads;
    if let Some(chunk_rows) = chunk_rows {
        config.chunk_rows = chunk_rows;
    }
    Ok(config)
}

/// Build the metadata dict; diagnostic examples become event-id strings when
/// `ids` is available, 0-based row indices otherwise.
fn meta_to_dict<'py>(
    py: Python<'py>,
    meta: &Meta,
    ids: Option<&[String]>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("bucket_size", meta.bucket_size)?;
    dict.set_item("grid_length", meta.grid_length)?;
    dict.set_item("quadrature_points", meta.quadrature_points)?;
    dict.set_item("normalized", meta.normalized)?;
    dict.set_item("total_rate", meta.total_rate)?;
    dict.set_item("analytic_aal", meta.analytic_aal)?;
    dict.set_item("raw_mass", meta.raw_mass)?;
    dict.set_item("raw_mean", meta.raw_mean)?;
    dict.set_item("secondary_mass", meta.secondary_mass)?;
    dict.set_item("secondary_mean", meta.secondary_mean)?;
    dict.set_item("quadrature_secondary_mean", meta.quadrature_secondary_mean)?;
    dict.set_item("secondary_mean_error", meta.secondary_mean_error)?;
    dict.set_item("input_rows", meta.input_rows)?;
    dict.set_item("valid_rows", meta.valid_rows)?;
    let diagnostics = PyDict::new(py);
    for (category, issues) in meta.diagnostics.iter_nonzero() {
        let entry = PyDict::new(py);
        entry.set_item("count", issues.count)?;
        match ids {
            Some(ids) => entry.set_item(
                "examples",
                issues
                    .examples
                    .iter()
                    .map(|&row| ids[row].clone())
                    .collect::<Vec<_>>(),
            )?,
            None => entry.set_item("examples", issues.examples.clone())?,
        }
        diagnostics.set_item(category.as_str(), entry)?;
    }
    dict.set_item("diagnostics", diagnostics)?;
    Ok(dict)
}

type QuantizeReturn<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyDict>,
);

/// Borrow a contiguous float64 slice from a NumPy array without copying.
fn contiguous<'a>(name: &str, array: &'a PyReadonlyArray1<'_, f64>) -> PyResult<&'a [f64]> {
    array
        .as_slice()
        .map_err(|_| EltqError::new_err(format!("{name} array must be C-contiguous float64")))
}

#[pyfunction]
#[pyo3(signature = (rate, max_loss, mean_loss, sd_indep, sd_corr, *,
    bucket_size, grid_length=None, quadrature_points=32, normalize=false,
    mean_is_loss_rate=false, variance_policy="clamp",
    clamp_factor=eltq_core::DEFAULT_CLAMP_FACTOR, threads=None,
    chunk_rows=None))]
#[allow(clippy::too_many_arguments)]
fn quantize_arrays<'py>(
    py: Python<'py>,
    rate: PyReadonlyArray1<'py, f64>,
    max_loss: PyReadonlyArray1<'py, f64>,
    mean_loss: PyReadonlyArray1<'py, f64>,
    sd_indep: PyReadonlyArray1<'py, f64>,
    sd_corr: PyReadonlyArray1<'py, f64>,
    bucket_size: f64,
    grid_length: Option<usize>,
    quadrature_points: usize,
    normalize: bool,
    mean_is_loss_rate: bool,
    variance_policy: &str,
    clamp_factor: f64,
    threads: Option<usize>,
    chunk_rows: Option<usize>,
) -> PyResult<QuantizeReturn<'py>> {
    let config = build_config(
        bucket_size,
        grid_length,
        quadrature_points,
        normalize,
        mean_is_loss_rate,
        variance_policy,
        clamp_factor,
        threads,
        chunk_rows,
    )?;
    let input = EltInput {
        rate: contiguous("rate", &rate)?,
        max_loss: contiguous("max_loss", &max_loss)?,
        mean_loss: contiguous("mean_loss", &mean_loss)?,
        sd_indep: contiguous("sd_indep", &sd_indep)?,
        sd_corr: contiguous("sd_corr", &sd_corr)?,
    };
    let result = py
        .detach(|| quantize(&input, &config))
        .map_err(|e| to_py_err(py, e))?;
    let meta = meta_to_dict(py, &result.meta, None)?;
    Ok((
        result.raw.into_pyarray(py),
        result.secondary.into_pyarray(py),
        meta,
    ))
}

#[pyfunction]
#[pyo3(signature = (path, *, bucket_size, grid_length=None,
    quadrature_points=32, normalize=false, mean_is_loss_rate=false,
    variance_policy="clamp", clamp_factor=eltq_core::DEFAULT_CLAMP_FACTOR,
    threads=None, chunk_rows=None, columns=None))]
#[allow(clippy::too_many_arguments)]
fn quantize_csv<'py>(
    py: Python<'py>,
    path: PathBuf,
    bucket_size: f64,
    grid_length: Option<usize>,
    quadrature_points: usize,
    normalize: bool,
    mean_is_loss_rate: bool,
    variance_policy: &str,
    clamp_factor: f64,
    threads: Option<usize>,
    chunk_rows: Option<usize>,
    columns: Option<HashMap<String, String>>,
) -> PyResult<QuantizeReturn<'py>> {
    let config = build_config(
        bucket_size,
        grid_length,
        quadrature_points,
        normalize,
        mean_is_loss_rate,
        variance_policy,
        clamp_factor,
        threads,
        chunk_rows,
    )?;
    let mut csv_columns = CsvColumns::default();
    if let Some(mapping) = columns {
        for (key, value) in mapping {
            let slot = match key.as_str() {
                "event_id" => &mut csv_columns.event_id,
                "rate" => &mut csv_columns.rate,
                "max_loss" => &mut csv_columns.max_loss,
                "mean_loss" => &mut csv_columns.mean_loss,
                "sd_indep" => &mut csv_columns.sd_indep,
                "sd_corr" => &mut csv_columns.sd_corr,
                other => {
                    return Err(EltqError::new_err(format!(
                        "unknown column key {other:?} in columns mapping"
                    )));
                }
            };
            *slot = value;
        }
    }
    let (result, event_ids) = py
        .detach(|| {
            let cols = read_elt_csv(&path, &csv_columns)?;
            let result = quantize(&cols.input(), &config)?;
            Ok::<_, CoreError>((result, cols.event_ids))
        })
        .map_err(|e| to_py_err(py, e))?;
    let meta = meta_to_dict(py, &result.meta, event_ids.as_deref())?;
    Ok((
        result.raw.into_pyarray(py),
        result.secondary.into_pyarray(py),
        meta,
    ))
}

type JointReturn<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyDict>,
);

/// Exact bucketed bivariate pmf of one event across two units (1.8.0).
/// Units are `(mean_loss, sd_indep, sd_corr, max_loss)` tuples; returns
/// `(pmf[n1, n2], x1, x2, meta_dict)` matching `eltq.reference_multiunit`.
#[pyfunction]
#[pyo3(signature = (unit1, unit2, *, bucket_size, n1, n2, z_nodes=1001,
    z_rule="midpoint", variance_policy="clamp",
    clamp_factor=eltq_core::DEFAULT_CLAMP_FACTOR))]
#[allow(clippy::too_many_arguments)]
fn joint_two_units<'py>(
    py: Python<'py>,
    unit1: (f64, f64, f64, f64),
    unit2: (f64, f64, f64, f64),
    bucket_size: f64,
    n1: usize,
    n2: usize,
    z_nodes: usize,
    z_rule: &str,
    variance_policy: &str,
    clamp_factor: f64,
) -> PyResult<JointReturn<'py>> {
    let spec = |u: (f64, f64, f64, f64)| UnitSpec {
        mean_loss: u.0,
        sd_indep: u.1,
        sd_corr: u.2,
        max_loss: u.3,
    };
    let config = JointConfig {
        bucket_size,
        n1,
        n2,
        z_nodes,
        z_rule: ZRule::parse(z_rule).map_err(|e| to_py_err(py, e))?,
        variance_policy: parse_variance_policy(variance_policy, clamp_factor)?,
    };
    let result = py
        .detach(|| core_joint_two_units([spec(unit1), spec(unit2)], &config))
        .map_err(|e| to_py_err(py, e))?;
    let meta = &result.meta;
    let dict = PyDict::new(py);
    dict.set_item("bucket_size", meta.bucket_size)?;
    dict.set_item("z_nodes", meta.z_nodes)?;
    dict.set_item("z_rule", meta.z_rule.as_str())?;
    dict.set_item("alpha", (meta.alpha[0], meta.alpha[1]))?;
    dict.set_item("beta", (meta.beta[0], meta.beta[1]))?;
    dict.set_item("sigma_used", (meta.sigma_used[0], meta.sigma_used[1]))?;
    dict.set_item("copula_normal_correlation", meta.copula_normal_correlation)?;
    dict.set_item("total_mass", meta.total_mass)?;
    dict.set_item("grid_mean", (meta.grid_mean[0], meta.grid_mean[1]))?;
    dict.set_item("analytic_mean", (meta.analytic_mean[0], meta.analytic_mean[1]))?;
    dict.set_item("grid_sd", (meta.grid_sd[0], meta.grid_sd[1]))?;
    dict.set_item("grid_correlation", meta.grid_correlation)?;
    dict.set_item(
        "notes",
        (
            meta.notes[0].iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            meta.notes[1].iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        ),
    )?;
    let grid = |n: usize| (0..n).map(|i| i as f64 * bucket_size).collect::<Vec<f64>>();
    let (n1, n2) = (result.n1, result.n2);
    let pmf = result.pmf.into_pyarray(py).reshape([n1, n2])?;
    Ok((pmf, grid(n1).into_pyarray(py), grid(n2).into_pyarray(py), dict))
}

/// Structured terms as passed from Python: `(unit_id, limit, ded,
/// [(share, limit, attach), ...])` per unit code.
type TermsSpec = Vec<(String, f64, f64, Vec<(f64, f64, f64)>)>;

/// Total event severity by view for a multi-unit ELT (1.10.0). Event and
/// unit codes are integer codings (unit codes index `terms`); returns
/// `(views_dict, meta_dict)` with diagnostics as 0-based row indices and
/// roll-up entries keyed by event start row, translated by the wrapper.
#[pyfunction]
#[pyo3(signature = (event_code, unit_code, rate, mean_loss, sd_indep, sd_corr,
    max_loss, *, terms, bucket_size, grid_length=None, z_nodes=32,
    z_rule="midpoint", views=None, normalize=false, variance_policy="clamp",
    clamp_factor=eltq_core::DEFAULT_CLAMP_FACTOR, threads=None,
    chunk_rows=None, roll_up_diagnostic=false, convolution="auto"))]
#[allow(clippy::too_many_arguments)]
fn multiunit_views<'py>(
    py: Python<'py>,
    event_code: PyReadonlyArray1<'py, i64>,
    unit_code: PyReadonlyArray1<'py, i64>,
    rate: PyReadonlyArray1<'py, f64>,
    mean_loss: PyReadonlyArray1<'py, f64>,
    sd_indep: PyReadonlyArray1<'py, f64>,
    sd_corr: PyReadonlyArray1<'py, f64>,
    max_loss: PyReadonlyArray1<'py, f64>,
    terms: TermsSpec,
    bucket_size: f64,
    grid_length: Option<usize>,
    z_nodes: usize,
    z_rule: &str,
    views: Option<Vec<String>>,
    normalize: bool,
    variance_policy: &str,
    clamp_factor: f64,
    threads: Option<usize>,
    chunk_rows: Option<usize>,
    roll_up_diagnostic: bool,
    convolution: &str,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let err = |e: CoreError| to_py_err(py, e);
    let mut config = MultiConfig::new(bucket_size);
    config.grid_length = grid_length;
    config.z_nodes = z_nodes;
    config.z_rule = ZRule::parse(z_rule).map_err(err)?;
    if let Some(names) = views {
        config.views = names
            .iter()
            .map(|name| View::parse(name).map_err(err))
            .collect::<PyResult<Vec<_>>>()?;
    }
    config.normalize = normalize;
    config.variance_policy = parse_variance_policy(variance_policy, clamp_factor)?;
    config.threads = threads;
    if let Some(chunk_rows) = chunk_rows {
        config.chunk_rows = chunk_rows;
    }
    config.roll_up_diagnostic = roll_up_diagnostic;
    config.convolution = Convolution::parse(convolution).map_err(err)?;
    let terms: Vec<UnitTerms> = terms
        .into_iter()
        .map(|(unit_id, limit, ded, clauses)| {
            let clauses = clauses
                .into_iter()
                .map(|(share, limit, attach)| Clause::new(share, limit, attach))
                .collect::<Result<Vec<_>, _>>()?;
            UnitTerms::new(&unit_id, limit, ded, clauses)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    let event_code = event_code
        .as_slice()
        .map_err(|_| EltqError::new_err("event_code array must be C-contiguous int64"))?;
    let unit_code = unit_code
        .as_slice()
        .map_err(|_| EltqError::new_err("unit_code array must be C-contiguous int64"))?;
    let input = MultiInput {
        event_code,
        unit_code,
        rate: contiguous("rate", &rate)?,
        mean_loss: contiguous("mean_loss", &mean_loss)?,
        sd_indep: contiguous("sd_indep", &sd_indep)?,
        sd_corr: contiguous("sd_corr", &sd_corr)?,
        max_loss: contiguous("max_loss", &max_loss)?,
    };
    let result = py
        .detach(|| core_multiunit_views(&input, &terms, &config))
        .map_err(err)?;
    let meta = &result.meta;
    let names: Vec<&str> = meta.views.iter().map(|v| v.as_str()).collect();
    let views_dict = PyDict::new(py);
    for (name, array) in names.iter().zip(result.views) {
        views_dict.set_item(name, array.into_pyarray(py))?;
    }
    let per_view = |values: &[f64]| -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, v) in names.iter().zip(values) {
            d.set_item(name, *v)?;
        }
        Ok(d)
    };
    let dict = PyDict::new(py);
    dict.set_item("bucket_size", meta.bucket_size)?;
    dict.set_item("grid_length", meta.grid_length)?;
    dict.set_item("z_nodes", meta.z_nodes)?;
    dict.set_item("z_rule", meta.z_rule.as_str())?;
    dict.set_item("views", names.clone())?;
    dict.set_item("normalized", meta.normalized)?;
    dict.set_item("total_rate", meta.total_rate)?;
    dict.set_item("analytic_aal", meta.analytic_aal)?;
    dict.set_item("view_mass", per_view(&meta.view_mass)?)?;
    dict.set_item("view_mean", per_view(&meta.view_mean)?)?;
    dict.set_item("analytic_view_mean", per_view(&meta.analytic_view_mean)?)?;
    dict.set_item("view_mean_error", per_view(&meta.view_mean_error)?)?;
    dict.set_item("input_rows", meta.input_rows)?;
    dict.set_item("valid_rows", meta.valid_rows)?;
    dict.set_item("events", meta.events)?;
    dict.set_item("valid_events", meta.valid_events)?;
    let diagnostics = PyDict::new(py);
    for (category, issues) in meta.diagnostics.iter_nonzero() {
        let entry = PyDict::new(py);
        entry.set_item("count", issues.count)?;
        entry.set_item("examples", issues.examples.clone())?;
        diagnostics.set_item(category.as_str(), entry)?;
    }
    dict.set_item("diagnostics", diagnostics)?;
    match &meta.roll_up {
        None => dict.set_item("roll_up", py.None())?,
        Some(entries) => {
            let list: Vec<Bound<'py, PyDict>> = entries
                .iter()
                .map(|e| {
                    let d = PyDict::new(py);
                    d.set_item("event_row", e.event_row)?;
                    d.set_item("units", e.units)?;
                    d.set_item("sd_exact", e.sd_exact)?;
                    d.set_item("sd_rolled", e.sd_rolled)?;
                    d.set_item("sd_ratio", e.sd_ratio)?;
                    d.set_item("q99_exact", e.q99_exact)?;
                    d.set_item("q99_rolled", e.q99_rolled)?;
                    d.set_item("q99_ratio", e.q99_ratio)?;
                    Ok(d)
                })
                .collect::<PyResult<_>>()?;
            dict.set_item("roll_up", list)?;
        }
    }
    let engine = PyDict::new(py);
    engine.set_item("max_mass_error", meta.max_mass_error)?;
    engine.set_item("direct_paths", meta.direct_paths)?;
    engine.set_item("fft_paths", meta.fft_paths)?;
    engine.set_item("convolution", config.convolution.as_str())?;
    dict.set_item("engine", engine)?;
    Ok((views_dict, dict))
}

#[pymodule]
fn _eltq(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__rust_version__", env!("CARGO_PKG_VERSION"))?;
    m.add("EltqError", py.get_type::<EltqError>())?;
    m.add("VarianceError", py.get_type::<VarianceError>())?;
    m.add("GridOverflowError", py.get_type::<GridOverflowError>())?;
    m.add("MultiUnitError", py.get_type::<MultiUnitError>())?;
    m.add_function(wrap_pyfunction!(quantize_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_csv, m)?)?;
    m.add_function(wrap_pyfunction!(joint_two_units, m)?)?;
    m.add_function(wrap_pyfunction!(multiunit_views, m)?)?;
    Ok(())
}
