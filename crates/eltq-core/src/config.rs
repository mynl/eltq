//! Quantization configuration and policies.

use crate::error::Error;

pub const DEFAULT_CLAMP_FACTOR: f64 = 0.999;
pub const DEFAULT_QUADRATURE_POINTS: usize = 32;

/// Treatment of rows whose stated variance is inadmissible for a beta,
/// i.e. sigma^2 >= mu (M - mu) with 0 < mu < M.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VariancePolicy {
    /// Clamp sigma to `factor * sqrt(mu (M - mu))`, keeping the row as a
    /// (very U-shaped) beta. Reported under `sigma_clamped`.
    ClampSigma { factor: f64 },
    /// Fail with [`Error::InadmissibleVariance`].
    Error,
    /// Drop the row from both outputs. Reported under `variance_dropped`.
    Drop,
    /// Collapse the row to an atom at mu. Reported under `variance_atom`.
    Atom,
}

impl Default for VariancePolicy {
    fn default() -> Self {
        VariancePolicy::ClampSigma {
            factor: DEFAULT_CLAMP_FACTOR,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Grid spacing h (> 0).
    pub bucket_size: f64,
    /// Number of grid nodes n; None auto-derives ceil(max M / h) + 1.
    pub grid_length: Option<usize>,
    /// Number K of midpoint beta quantiles per nondegenerate event.
    pub quadrature_points: usize,
    /// Divide both outputs by the total contributing rate.
    pub normalize: bool,
    /// Interpret mean_loss as a rate on [0, 1], scaled by max_loss.
    pub mean_is_loss_rate: bool,
    pub variance_policy: VariancePolicy,
}

impl Config {
    pub fn new(bucket_size: f64) -> Self {
        Config {
            bucket_size,
            grid_length: None,
            quadrature_points: DEFAULT_QUADRATURE_POINTS,
            normalize: false,
            mean_is_loss_rate: false,
            variance_policy: VariancePolicy::default(),
        }
    }

    pub fn validate(&self) -> Result<(), Error> {
        if !self.bucket_size.is_finite() || self.bucket_size <= 0.0 {
            return Err(Error::BadBucketSize(self.bucket_size));
        }
        if self.quadrature_points < 1 {
            return Err(Error::BadQuadraturePoints(self.quadrature_points));
        }
        if let Some(n) = self.grid_length
            && n < 1
        {
            return Err(Error::BadGridLength(n));
        }
        if let VariancePolicy::ClampSigma { factor } = self.variance_policy
            && !(factor.is_finite() && factor > 0.0 && factor < 1.0)
        {
            return Err(Error::BadClampFactor(factor));
        }
        Ok(())
    }
}
