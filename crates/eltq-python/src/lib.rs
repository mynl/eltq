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
use eltq_core::{Config, EltInput, Error as CoreError, Meta, VariancePolicy, quantize};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
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

fn to_py_err(error: CoreError) -> PyErr {
    match &error {
        CoreError::InadmissibleVariance { .. } => VarianceError::new_err(error.to_string()),
        CoreError::GridOverflow { .. } => GridOverflowError::new_err(error.to_string()),
        _ => EltqError::new_err(error.to_string()),
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
) -> PyResult<Config> {
    let variance_policy = match variance_policy {
        "clamp" => VariancePolicy::ClampSigma {
            factor: clamp_factor,
        },
        "error" => VariancePolicy::Error,
        "drop" => VariancePolicy::Drop,
        "atom" => VariancePolicy::Atom,
        other => {
            return Err(EltqError::new_err(format!(
                "unknown variance_policy {other:?} (expected clamp, error, drop, or atom)"
            )));
        }
    };
    let mut config = Config::new(bucket_size);
    config.grid_length = grid_length;
    config.quadrature_points = quadrature_points;
    config.normalize = normalize;
    config.mean_is_loss_rate = mean_is_loss_rate;
    config.variance_policy = variance_policy;
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
    clamp_factor=eltq_core::DEFAULT_CLAMP_FACTOR))]
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
) -> PyResult<QuantizeReturn<'py>> {
    let config = build_config(
        bucket_size,
        grid_length,
        quadrature_points,
        normalize,
        mean_is_loss_rate,
        variance_policy,
        clamp_factor,
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
        .map_err(to_py_err)?;
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
    columns=None))]
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
        .map_err(to_py_err)?;
    let meta = meta_to_dict(py, &result.meta, event_ids.as_deref())?;
    Ok((
        result.raw.into_pyarray(py),
        result.secondary.into_pyarray(py),
        meta,
    ))
}

#[pymodule]
fn _eltq(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__rust_version__", env!("CARGO_PKG_VERSION"))?;
    m.add("EltqError", py.get_type::<EltqError>())?;
    m.add("VarianceError", py.get_type::<VarianceError>())?;
    m.add("GridOverflowError", py.get_type::<GridOverflowError>())?;
    m.add_function(wrap_pyfunction!(quantize_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_csv, m)?)?;
    Ok(())
}
