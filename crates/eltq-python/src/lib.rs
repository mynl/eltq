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
use std::sync::Arc;

use eltq_core::{
    ClassifyOptions, Clause, Config, Convolution, Coupling, Curve, EltInput, Error as CoreError, JointConfig, Meta,
    Method, MultiConfig, MultiInput, Overflow, UnitSpec, UnitTerms, VariancePolicy, View, ZRule, build_curve,
    classify_rows, joint_two_units as core_joint_two_units, model_moments, tail_moments,
    multiunit_views as core_multiunit_views, quantize,
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

/// The one-row route's policies: the clamp is refused here (2.0.0); the
/// multi-unit route parses its own with `parse_variance_policy`.
fn parse_one_row_variance_policy(variance_policy: &str) -> PyResult<VariancePolicy> {
    match variance_policy {
        "bernoulli" => Ok(VariancePolicy::Bernoulli),
        "error" => Ok(VariancePolicy::Error),
        "drop" => Ok(VariancePolicy::Drop),
        "atom" => Ok(VariancePolicy::Atom),
        "clamp" => Err(EltqError::new_err(
            "the clamp variance policy was removed from the one-row route at 2.0.0 \
             (use bernoulli, error, drop, or atom)",
        )),
        other => Err(EltqError::new_err(format!(
            "unknown variance_policy {other:?} (expected bernoulli, error, drop, or atom)"
        ))),
    }
}

fn parse_variance_policy(variance_policy: &str, clamp_factor: f64) -> PyResult<VariancePolicy> {
    match variance_policy {
        "bernoulli" => Ok(VariancePolicy::Bernoulli),
        "clamp" => Ok(VariancePolicy::ClampSigma {
            factor: clamp_factor,
        }),
        "error" => Ok(VariancePolicy::Error),
        "drop" => Ok(VariancePolicy::Drop),
        "atom" => Ok(VariancePolicy::Atom),
        other => Err(EltqError::new_err(format!(
            "unknown variance_policy {other:?} (expected bernoulli, clamp, error, drop, or atom)"
        ))),
    }
}

/// The one-row route's keyword arguments, shared by the array and CSV entry
/// points.
struct Options<'a> {
    bucket_size: f64,
    grid_length: Option<usize>,
    method: &'a str,
    quadrature_points: usize,
    nodes_per_decade: usize,
    nodes_per_sigma: usize,
    overflow: &'a str,
    normalize: bool,
    mean_is_loss_rate: bool,
    variance_policy: &'a str,
    threads: Option<usize>,
    chunk_rows: Option<usize>,
}

fn build_config(py: Python<'_>, o: &Options<'_>) -> PyResult<Config> {
    let mut config = Config::new(o.bucket_size);
    config.grid_length = o.grid_length;
    config.method = Method::parse(o.method).map_err(|e| to_py_err(py, e))?;
    config.quadrature_points = o.quadrature_points;
    config.nodes_per_decade = o.nodes_per_decade;
    config.nodes_per_sigma = o.nodes_per_sigma;
    config.overflow = Overflow::parse(o.overflow).map_err(|e| to_py_err(py, e))?;
    config.normalize = o.normalize;
    config.mean_is_loss_rate = o.mean_is_loss_rate;
    config.variance_policy = parse_one_row_variance_policy(o.variance_policy)?;
    config.threads = o.threads;
    config.chunk_rows = o.chunk_rows;
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
    dict.set_item("method", meta.method.as_str())?;
    dict.set_item("quadrature_points", meta.quadrature_points)?;
    dict.set_item("nodes", meta.nodes)?;
    dict.set_item("nodes_per_decade", meta.nodes_per_decade)?;
    dict.set_item("interpolation_error", meta.interpolation_error)?;
    dict.set_item("narrow_events", meta.narrow_events)?;
    dict.set_item("fine_nodes", meta.fine_nodes)?;
    dict.set_item("requesting_events", meta.requesting_events)?;
    dict.set_item("overflow", meta.overflow.as_str())?;
    let per_output = |values: [f64; 2]| -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("raw", values[0])?;
        d.set_item("secondary", values[1])?;
        Ok(d)
    };
    dict.set_item("rate_above_grid", per_output(meta.rate_above_grid)?)?;
    dict.set_item("mean_above_grid", per_output(meta.mean_above_grid)?)?;
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
    dict.set_item("diagnostics", diagnostics_to_dict(py, &meta.diagnostics, ids)?)?;
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
    bucket_size, grid_length=None, method="survival", quadrature_points=32,
    nodes_per_decade=eltq_core::DEFAULT_NODES_PER_DECADE,
    nodes_per_sigma=eltq_core::DEFAULT_NODES_PER_SIGMA, overflow="top_bucket",
    normalize=false, mean_is_loss_rate=false, variance_policy="bernoulli",
    threads=None, chunk_rows=None))]
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
    method: &str,
    quadrature_points: usize,
    nodes_per_decade: usize,
    nodes_per_sigma: usize,
    overflow: &str,
    normalize: bool,
    mean_is_loss_rate: bool,
    variance_policy: &str,
    threads: Option<usize>,
    chunk_rows: Option<usize>,
) -> PyResult<QuantizeReturn<'py>> {
    let config = build_config(py, &Options {
        bucket_size,
        grid_length,
        method,
        quadrature_points,
        nodes_per_decade,
        nodes_per_sigma,
        overflow,
        normalize,
        mean_is_loss_rate,
        variance_policy,
        threads,
        chunk_rows,
    })?;
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
#[pyo3(signature = (path, *, bucket_size, grid_length=None, method="survival",
    quadrature_points=32, nodes_per_decade=eltq_core::DEFAULT_NODES_PER_DECADE,
    nodes_per_sigma=eltq_core::DEFAULT_NODES_PER_SIGMA,
    overflow="top_bucket", normalize=false, mean_is_loss_rate=false,
    variance_policy="bernoulli", threads=None, chunk_rows=None, columns=None))]
#[allow(clippy::too_many_arguments)]
fn quantize_csv<'py>(
    py: Python<'py>,
    path: PathBuf,
    bucket_size: f64,
    grid_length: Option<usize>,
    method: &str,
    quadrature_points: usize,
    nodes_per_decade: usize,
    nodes_per_sigma: usize,
    overflow: &str,
    normalize: bool,
    mean_is_loss_rate: bool,
    variance_policy: &str,
    threads: Option<usize>,
    chunk_rows: Option<usize>,
    columns: Option<HashMap<String, String>>,
) -> PyResult<QuantizeReturn<'py>> {
    let config = build_config(py, &Options {
        bucket_size,
        grid_length,
        method,
        quadrature_points,
        nodes_per_decade,
        nodes_per_sigma,
        overflow,
        normalize,
        mean_is_loss_rate,
        variance_policy,
        threads,
        chunk_rows,
    })?;
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

/// Diagnostics as a dict of `{category: {count, examples}}`, examples as
/// 0-based rows or as event ids when `ids` is given.
fn diagnostics_to_dict<'py>(
    py: Python<'py>,
    diagnostics: &eltq_core::Diagnostics,
    ids: Option<&[String]>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (category, issues) in diagnostics.iter_nonzero() {
        let entry = PyDict::new(py);
        entry.set_item("count", issues.count)?;
        match ids {
            Some(ids) => entry.set_item(
                "examples",
                issues.examples.iter().map(|&row| ids[row].clone()).collect::<Vec<_>>(),
            )?,
            None => entry.set_item("examples", issues.examples.clone())?,
        }
        dict.set_item(category.as_str(), entry)?;
    }
    Ok(dict)
}

/// The exceedance curve of an ELT (spec section 5A): built once, it answers
/// exceedance queries and makes the even grid at any `bucket_size >= 2
/// min_loss`. Arrays are copied out on access; the curve itself is shared.
#[pyclass(name = "ExceedanceCurve", module = "eltq._eltq", frozen)]
struct PyCurve {
    inner: Arc<Curve>,
    config: Config,
}

#[pymethods]
impl PyCurve {
    #[getter]
    fn loss<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.loss.clone().into_pyarray(py)
    }

    #[getter]
    fn exceedance_rate<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.exceedance_rate.clone().into_pyarray(py)
    }

    #[getter]
    fn atom_loss<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.atom_loss.clone().into_pyarray(py)
    }

    #[getter]
    fn atom_rate<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.atom_rate.clone().into_pyarray(py)
    }

    /// The curve's metadata as a dict (diagnostic examples as 0-based rows).
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let c = &self.inner;
        let dict = PyDict::new(py);
        dict.set_item("min_loss", c.min_loss)?;
        dict.set_item("max_loss", c.max_loss)?;
        dict.set_item("nodes", c.nodes())?;
        dict.set_item("nodes_per_decade", c.nodes_per_decade)?;
        dict.set_item("interpolation_error", c.interpolation_error)?;
        dict.set_item("narrow_events", c.narrow_events())?;
        dict.set_item("fine_nodes", c.fine_nodes())?;
        dict.set_item("requesting_events", c.requesting_events())?;
        dict.set_item("total_rate", c.total_rate)?;
        dict.set_item("analytic_aal", c.analytic_aal)?;
        dict.set_item("input_rows", c.input_rows)?;
        dict.set_item("valid_rows", c.valid_rows)?;
        dict.set_item("diagnostics", diagnostics_to_dict(py, &c.diagnostics, None)?)?;
        Ok(dict)
    }

    /// `Lambda(x)` with the atoms included, elementwise; NaN below `min_loss`.
    fn exceedance<'py>(&self, py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let xs = contiguous("x", &x)?;
        let curve = Arc::clone(&self.inner);
        let out: Vec<f64> = py.detach(move || xs.iter().map(|&v| curve.exceedance(v)).collect());
        Ok(out.into_pyarray(py))
    }

    /// The secondary grid at `bucket_size` with `grid_length` nodes (derived
    /// when None) under the overflow policy; returns `(array, meta_dict)`.
    #[pyo3(signature = (bucket_size, grid_length=None, *, normalize=false, overflow="top_bucket"))]
    fn grid<'py>(
        &self,
        py: Python<'py>,
        bucket_size: f64,
        grid_length: Option<usize>,
        normalize: bool,
        overflow: &str,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)> {
        let policy = Overflow::parse(overflow).map_err(|e| to_py_err(py, e))?;
        let curve = Arc::clone(&self.inner);
        let config = self.config.clone();
        let outcome = py
            .detach(move || curve.grid(bucket_size, grid_length, policy, &config))
            .map_err(|e| to_py_err(py, e))?;
        let mut secondary = outcome.secondary;
        // Compensated, as `quantize` sums them, so the two agree bitwise.
        let moment = eltq_core::sum::grid_moment(&secondary, bucket_size);
        let mass = eltq_core::sum::sum(&secondary);
        if normalize {
            if self.inner.total_rate <= 0.0 {
                return Err(to_py_err(py, CoreError::ZeroTotalRate));
            }
            for w in secondary.iter_mut() {
                *w /= self.inner.total_rate;
            }
        }
        let dict = PyDict::new(py);
        dict.set_item("bucket_size", bucket_size)?;
        dict.set_item("grid_length", outcome.grid_length)?;
        dict.set_item("normalized", normalize)?;
        dict.set_item("overflow", policy.as_str())?;
        dict.set_item("secondary_mass", if normalize { mass / self.inner.total_rate } else { mass })?;
        dict.set_item("secondary_mean", if normalize { moment / self.inner.total_rate } else { moment })?;
        dict.set_item("secondary_mean_error", moment - self.inner.analytic_aal)?;
        dict.set_item("rate_above_grid", outcome.overflow.rate.value())?;
        dict.set_item("mean_above_grid", outcome.overflow.moment.value())?;
        Ok((secondary.into_pyarray(py), dict))
    }
}

/// Build the exceedance curve from `min_loss` up (spec section 5A).
#[pyfunction]
#[pyo3(signature = (rate, max_loss, mean_loss, sd_indep, sd_corr, *, min_loss,
    nodes_per_decade=eltq_core::DEFAULT_NODES_PER_DECADE,
    nodes_per_sigma=eltq_core::DEFAULT_NODES_PER_SIGMA, mean_is_loss_rate=false,
    variance_policy="bernoulli", threads=None, chunk_rows=None))]
#[allow(clippy::too_many_arguments)]
fn exceedance_curve<'py>(
    py: Python<'py>,
    rate: PyReadonlyArray1<'py, f64>,
    max_loss: PyReadonlyArray1<'py, f64>,
    mean_loss: PyReadonlyArray1<'py, f64>,
    sd_indep: PyReadonlyArray1<'py, f64>,
    sd_corr: PyReadonlyArray1<'py, f64>,
    min_loss: f64,
    nodes_per_decade: usize,
    nodes_per_sigma: usize,
    mean_is_loss_rate: bool,
    variance_policy: &str,
    threads: Option<usize>,
    chunk_rows: Option<usize>,
) -> PyResult<PyCurve> {
    // The bucket size is irrelevant to the curve; a valid placeholder.
    let config = build_config(py, &Options {
        bucket_size: 2.0 * min_loss,
        grid_length: None,
        method: "survival",
        quadrature_points: 32,
        nodes_per_decade,
        nodes_per_sigma,
        overflow: "top_bucket",
        normalize: false,
        mean_is_loss_rate,
        variance_policy,
        threads,
        chunk_rows,
    })?;
    let input = EltInput {
        rate: contiguous("rate", &rate)?,
        max_loss: contiguous("max_loss", &max_loss)?,
        mean_loss: contiguous("mean_loss", &mean_loss)?,
        sd_indep: contiguous("sd_indep", &sd_indep)?,
        sd_corr: contiguous("sd_corr", &sd_corr)?,
    };
    let curve = py
        .detach(|| {
            let classified = classify_rows(&input, &config)?;
            build_curve(&input, &config, &classified, min_loss)
        })
        .map_err(|e| to_py_err(py, e))?;
    Ok(PyCurve {
        inner: Arc::new(curve),
        config,
    })
}

/// Model moments of a one-row ELT with no grid (spec section 4B): a dict
/// of the total rate, the rate-weighted raw moments, the zero rate, the
/// per-row support, the counts by effective law, the overflow flags and the
/// diagnostics (0-based rows, translated by the Python wrapper).
#[pyfunction]
#[pyo3(signature = (rate, max_loss, mean_loss, sd_indep, sd_corr, *,
    mean_is_loss_rate=false, variance_policy="bernoulli"))]
#[allow(clippy::too_many_arguments)]
fn moments_arrays<'py>(
    py: Python<'py>,
    rate: PyReadonlyArray1<'py, f64>,
    max_loss: PyReadonlyArray1<'py, f64>,
    mean_loss: PyReadonlyArray1<'py, f64>,
    sd_indep: PyReadonlyArray1<'py, f64>,
    sd_corr: PyReadonlyArray1<'py, f64>,
    mean_is_loss_rate: bool,
    variance_policy: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let options = ClassifyOptions {
        mean_is_loss_rate,
        variance_policy: parse_one_row_variance_policy(variance_policy)?,
    };
    let input = EltInput {
        rate: contiguous("rate", &rate)?,
        max_loss: contiguous("max_loss", &max_loss)?,
        mean_loss: contiguous("mean_loss", &mean_loss)?,
        sd_indep: contiguous("sd_indep", &sd_indep)?,
        sd_corr: contiguous("sd_corr", &sd_corr)?,
    };
    let m = py
        .detach(|| model_moments(&input, &options))
        .map_err(|e| to_py_err(py, e))?;
    let dict = PyDict::new(py);
    dict.set_item("total_rate", m.total_rate)?;
    dict.set_item("raw_moments", m.raw_moments.to_vec())?;
    dict.set_item("rate_at_zero", m.rate_at_zero)?;
    dict.set_item("max_support", m.max_support)?;
    dict.set_item("input_rows", m.input_rows)?;
    dict.set_item("valid_rows", m.valid_rows)?;
    dict.set_item("atom_rows", m.atom_rows)?;
    dict.set_item("beta_rows", m.beta_rows)?;
    dict.set_item("two_point_rows", m.two_point_rows)?;
    dict.set_item("dropped_rows", m.dropped_rows)?;
    dict.set_item("moment_overflow", m.moment_overflow.to_vec())?;
    dict.set_item("diagnostics", diagnostics_to_dict(py, &m.diagnostics, None)?)?;
    dict.set_item("support", m.support.into_pyarray(py))?;
    Ok(dict)
}

/// Tail measures of a one-row ELT at candidate edges with no grid (spec
/// section 4B): a dict of flat arrays (the per-order ones edge-major, three
/// per edge, reshaped by the Python wrapper), the regimes, the overflow flags
/// and the diagnostics.
#[pyfunction]
#[pyo3(signature = (rate, max_loss, mean_loss, sd_indep, sd_corr, *, edge,
    top_node=None, mean_is_loss_rate=false, variance_policy="bernoulli", threads=None))]
#[allow(clippy::too_many_arguments)]
fn tail_moments_arrays<'py>(
    py: Python<'py>,
    rate: PyReadonlyArray1<'py, f64>,
    max_loss: PyReadonlyArray1<'py, f64>,
    mean_loss: PyReadonlyArray1<'py, f64>,
    sd_indep: PyReadonlyArray1<'py, f64>,
    sd_corr: PyReadonlyArray1<'py, f64>,
    edge: PyReadonlyArray1<'py, f64>,
    top_node: Option<PyReadonlyArray1<'py, f64>>,
    mean_is_loss_rate: bool,
    variance_policy: &str,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let options = ClassifyOptions {
        mean_is_loss_rate,
        variance_policy: parse_one_row_variance_policy(variance_policy)?,
    };
    let input = EltInput {
        rate: contiguous("rate", &rate)?,
        max_loss: contiguous("max_loss", &max_loss)?,
        mean_loss: contiguous("mean_loss", &mean_loss)?,
        sd_indep: contiguous("sd_indep", &sd_indep)?,
        sd_corr: contiguous("sd_corr", &sd_corr)?,
    };
    let edges = contiguous("edge", &edge)?;
    let nodes = match &top_node {
        Some(c) => Some(contiguous("top_node", c)?),
        None => None,
    };
    let t = py
        .detach(|| tail_moments(&input, &options, edges, nodes, threads))
        .map_err(|e| to_py_err(py, e))?;
    let flat = |v: &[[f64; 3]]| -> Vec<f64> { v.iter().flatten().copied().collect() };
    let dict = PyDict::new(py);
    dict.set_item("moments_above", flat(&t.moments_above).into_pyarray(py))?;
    dict.set_item("excess_above", flat(&t.excess_above).into_pyarray(py))?;
    dict.set_item("top_half_excess", flat(&t.top_half_excess).into_pyarray(py))?;
    dict.set_item("edge", t.edge.into_pyarray(py))?;
    dict.set_item("top_node", t.top_node.into_pyarray(py))?;
    dict.set_item("rate_above", t.rate_above.into_pyarray(py))?;
    dict.set_item("top_half_rate", t.top_half_rate.into_pyarray(py))?;
    dict.set_item("total_rate", t.total_rate)?;
    let regimes = PyDict::new(py);
    regimes.set_item("incomplete_beta", t.regimes.incomplete_beta)?;
    regimes.set_item("narrow_kernel", t.regimes.narrow_kernel)?;
    regimes.set_item("normal_law", t.regimes.normal_law)?;
    regimes.set_item("atoms", t.regimes.atoms)?;
    regimes.set_item("two_point", t.regimes.two_point)?;
    dict.set_item("regimes", regimes)?;
    dict.set_item("moment_overflow", t.moment_overflow.to_vec())?;
    dict.set_item("input_rows", t.input_rows)?;
    dict.set_item("valid_rows", t.valid_rows)?;
    dict.set_item("diagnostics", diagnostics_to_dict(py, &t.diagnostics, None)?)?;
    Ok(dict)
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
#[pyo3(signature = (unit1, unit2, *, bucket_size, n1, n2, z_nodes=32,
    z_rule="stratified", variance_policy="bernoulli",
    clamp_factor=eltq_core::DEFAULT_CLAMP_FACTOR, coupling="burke"))]
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
    coupling: &str,
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
        coupling: Coupling::parse(coupling).map_err(|e| to_py_err(py, e))?,
    };
    let result = py
        .detach(|| core_joint_two_units([spec(unit1), spec(unit2)], &config))
        .map_err(|e| to_py_err(py, e))?;
    let meta = &result.meta;
    let dict = PyDict::new(py);
    dict.set_item("bucket_size", meta.bucket_size)?;
    dict.set_item("coupling", meta.coupling.as_str())?;
    match &meta.matched {
        None => dict.set_item("matched", py.None())?,
        Some(m) => {
            let d = PyDict::new(py);
            d.set_item("scale", m.scale)?;
            d.set_item("sd_target", m.sd_target)?;
            d.set_item("sd_achieved", m.sd_achieved)?;
            d.set_item("saturated", m.saturated)?;
            dict.set_item("matched", d)?;
        }
    }
    dict.set_item("z_nodes", meta.z_nodes)?;
    dict.set_item("z_rule", meta.z_rule.as_str())?;
    dict.set_item("z_strata", meta.z_strata)?;
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
    z_rule="stratified", views=None, normalize=false, variance_policy="bernoulli",
    clamp_factor=eltq_core::DEFAULT_CLAMP_FACTOR, threads=None,
    chunk_rows=None, roll_up_diagnostic=false, convolution="auto", coupling="burke",
    pair=None, first_moments=false))]
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
    coupling: &str,
    pair: Option<(usize, usize)>,
    first_moments: bool,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>, Bound<'py, PyDict>)> {
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
    config.coupling = Coupling::parse(coupling).map_err(err)?;
    config.pair = pair.map(|(u, w)| [u, w]);
    config.first_moments = first_moments;
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
    // Occurrence outputs (2.9.0), when requested.
    let extras = PyDict::new(py);
    if let Some(p) = result.pair {
        let pmf = p.pmf.into_pyarray(py).reshape([p.n_u, p.n_w])?;
        extras.set_item("pair", (pmf, p.n_u, p.n_w, p.rate, p.events))?;
    }
    if let Some(m) = result.first_moments {
        let arrays: Vec<Bound<'py, PyArray1<f64>>> = m.moments.into_iter().map(|v| v.into_pyarray(py)).collect();
        extras.set_item("first_moments", (arrays, m.identity_error))?;
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
    dict.set_item("coupling", meta.coupling.as_str())?;
    dict.set_item("z_nodes", meta.z_nodes)?;
    dict.set_item("z_rule", meta.z_rule.as_str())?;
    dict.set_item("z_strata", meta.z_strata)?;
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
    dict.set_item("one_unit_events", meta.one_unit_events)?;
    dict.set_item("homer_excess_mean", meta.homer_excess_mean)?;
    match &meta.matched_shortfall {
        None => dict.set_item("matched_shortfall", py.None())?,
        Some(entries) => {
            let list: Vec<Bound<'py, PyDict>> = entries
                .iter()
                .map(|e| {
                    let d = PyDict::new(py);
                    d.set_item("event_row", e.event_row)?;
                    d.set_item("scale", e.scale.scale)?;
                    d.set_item("sd_target", e.scale.sd_target)?;
                    d.set_item("sd_achieved", e.scale.sd_achieved)?;
                    Ok(d)
                })
                .collect::<PyResult<_>>()?;
            dict.set_item("matched_shortfall", list)?;
        }
    }
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
    Ok((views_dict, dict, extras))
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
    m.add_class::<PyCurve>()?;
    m.add_function(wrap_pyfunction!(exceedance_curve, m)?)?;
    m.add_function(wrap_pyfunction!(joint_two_units, m)?)?;
    m.add_function(wrap_pyfunction!(multiunit_views, m)?)?;
    m.add_function(wrap_pyfunction!(moments_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(tail_moments_arrays, m)?)?;
    Ok(())
}
