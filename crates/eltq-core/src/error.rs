//! Typed errors. Invalid rows never panic; they are either classified into
//! diagnostics or surface as one of these variants.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("bucket_size must be finite and positive, got {0}")]
    BadBucketSize(f64),
    #[error("quadrature_points must be >= 1, got {0}")]
    BadQuadraturePoints(usize),
    #[error("grid_length must be >= 1, got {0}")]
    BadGridLength(usize),
    #[error("clamp factor must be in (0, 1), got {0}")]
    BadClampFactor(f64),
    #[error("column {name} has {got} rows, expected {expected}")]
    ColumnLengthMismatch {
        name: &'static str,
        got: usize,
        expected: usize,
    },
    #[error(
        "row {row}: sigma^2 = {sigma2} >= mu (M - mu) = {v_max} \
         is inadmissible for a beta"
    )]
    InadmissibleVariance { row: usize, sigma2: f64, v_max: f64 },
    #[error(
        "grid of {grid_length} nodes at spacing {bucket_size} covers only \
         {top}, but the maximum contributing loss is {max_loss}; \
         grid_length >= {required} required"
    )]
    GridOverflow {
        grid_length: usize,
        bucket_size: f64,
        top: f64,
        max_loss: f64,
        required: usize,
        /// Bounded set of representative offending row indices.
        example_rows: Vec<usize>,
    },
    #[error("cannot normalize: total contributing rate is zero")]
    ZeroTotalRate,
    #[error(
        "beta quantile evaluation failed for row {row} \
         (alpha = {alpha}, beta = {beta})"
    )]
    KernelFailure { row: usize, alpha: f64, beta: f64 },
}
