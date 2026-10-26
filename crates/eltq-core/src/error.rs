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
        "grid of {grid_length} nodes at spacing {bucket_size} covers losses up to \
         {top}, but the maximum contributing loss is {max_loss}; \
         grid_length >= {required} required (or select the top_bucket or drop \
         overflow policy)"
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
    #[error("chunk_rows must be >= 1, got {0}")]
    BadChunkRows(usize),
    #[error("nodes_per_decade must be >= 1, got {0}")]
    BadNodesPerDecade(usize),
    #[error("nodes_per_sigma must be >= 1, got {0}")]
    BadNodesPerSigma(usize),
    #[error("min_loss must be finite and positive, got {0}")]
    BadMinLoss(f64),
    #[error(
        "row {row}: the incomplete beta continued fraction did not converge at \
         alpha = {alpha}, beta = {beta}; the event lies outside the regime the \
         shared curve evaluates and was not routed around it"
    )]
    IncompleteBetaDiverged { row: usize, alpha: f64, beta: f64 },
    #[error(
        "the clamp variance policy was removed from the one-row route at 2.0.0 \
         (use bernoulli, error, drop, or atom)"
    )]
    ClampPolicyRemoved,
    #[error(
        "bucket_size {bucket_size} is below twice the curve's lowest node {min_loss}; \
         the curve knows nothing below that loss"
    )]
    BucketBelowCurve { bucket_size: f64, min_loss: f64 },
    #[error("unknown method {0:?} (expected quantiles or survival)")]
    BadMethod(String),
    #[error("unknown overflow {0:?} (expected top_bucket, drop, or error)")]
    BadOverflow(String),
    #[error("threads must be >= 1 when given, got {0}")]
    BadThreads(usize),
    #[error("failed to build worker thread pool: {0}")]
    ThreadPool(String),
    #[error("CSV read failed: {0}")]
    Csv(String),
    #[error("CSV is missing required column {0:?}")]
    MissingColumn(String),
    #[error(
        "beta quantile evaluation failed for row {row} \
         (alpha = {alpha}, beta = {beta})"
    )]
    KernelFailure { row: usize, alpha: f64, beta: f64 },
    #[error("z_nodes must be >= 1, got {0}")]
    BadZNodes(usize),
    #[error("unknown z_rule {0:?} (expected midpoint, gauss_hermite, or stratified)")]
    BadZRule(String),
    #[error("{unit}: {message}")]
    UnitInvalid { unit: String, message: String },
    #[error("{unit}: sigma^2 = {sigma2} >= mu (M - mu) = {v_max} is inadmissible for a beta")]
    UnitInadmissibleVariance { unit: String, sigma2: f64, v_max: f64 },
    #[error("the drop variance policy is not available for the two-unit joint")]
    BadJointPolicy,
    #[error("terms: {0}")]
    Terms(String),
    #[error("multi-unit structure: {0}")]
    MultiUnitStructure(String),
    #[error("unit code {code} at row {row} has no terms entry (terms table has {available})")]
    UnitCodeOutOfRange { row: usize, code: i64, available: usize },
    #[error("at least one view is required")]
    NoViews,
    #[error("unknown convolution mode {0:?} (expected auto, direct, or fft)")]
    BadConvolution(String),
}
