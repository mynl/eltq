//! Quantization configuration and policies.

use crate::error::Error;

pub const DEFAULT_CLAMP_FACTOR: f64 = 0.999;
pub const DEFAULT_QUADRATURE_POINTS: usize = 32;
/// Rows per chunk for the quantiles method (each fold adds grid-length
/// vectors) and for the survival method (node-length partials, so smaller
/// chunks keep every thread busy).
pub const DEFAULT_CHUNK_ROWS: usize = 64 * 1024;
pub const DEFAULT_SURVIVAL_CHUNK_ROWS: usize = 256;
/// Node density of the survival method; matches
/// `eltq._meta.DEFAULT_NODES_PER_DECADE` (`benchmarks/report-1.16.0.md`).
pub const DEFAULT_NODES_PER_DECADE: usize = 100;
/// Nodes per standard deviation a bell-shaped event asks of the fine mesh
/// around its mean (spec section 5A); matches
/// `eltq._meta.DEFAULT_NODES_PER_SIGMA`. Twelve holds the shared cubic
/// within 1e-4 relative where an event's survival is at least 1e-4 and
/// within 1e-3 where it is at least 1e-8 (`dev/done/plan-peer-review-response.md`).
pub const DEFAULT_NODES_PER_SIGMA: usize = 12;

/// How the secondary grid is built (spec sections 5 and 5A).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// K midpoint beta quantiles per event, deposited by the linear rule.
    Quantiles,
    /// The exceedance rate on geometric nodes, bucketed by the nearest rule.
    Survival,
}

impl Method {
    pub fn parse(name: &str) -> Result<Self, Error> {
        match name {
            "quantiles" => Ok(Method::Quantiles),
            "survival" => Ok(Method::Survival),
            other => Err(Error::BadMethod(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Method::Quantiles => "quantiles",
            Method::Survival => "survival",
        }
    }
}

/// What happens to mass above the top bucket's upper edge when the grid is
/// a window short of the largest loss (spec section 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overflow {
    /// Added to the top node: rate exact, first moment understated.
    TopBucket,
    /// Discarded: rate short by the overflow.
    Drop,
    /// A typed error whenever a contributing loss exceeds the top edge.
    Error,
}

impl Overflow {
    pub fn parse(name: &str) -> Result<Self, Error> {
        match name {
            "top_bucket" => Ok(Overflow::TopBucket),
            "drop" => Ok(Overflow::Drop),
            "error" => Ok(Overflow::Error),
            other => Err(Error::BadOverflow(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Overflow::TopBucket => "top_bucket",
            Overflow::Drop => "drop",
            Overflow::Error => "error",
        }
    }
}

/// Treatment of rows whose stated variance is inadmissible for a beta,
/// i.e. sigma^2 >= mu (M - mu) with 0 < mu < M.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VariancePolicy {
    /// Replace the row by the variance maximizer on [0, M]: two atoms,
    /// `rate (1 - p)` at 0 and `rate p` at M with `p = mu / M`. Rate and
    /// mean are exact and the variance is the admissible supremum. The
    /// default for the one-row route from 1.13.0; reported under
    /// `variance_bernoulli`. Not available on the multi-unit route, which
    /// keeps the clamp until it adopts this treatment.
    Bernoulli,
    /// Clamp sigma to `factor * sqrt(mu (M - mu))`, keeping the row as a
    /// (very U-shaped) beta. Reported under `sigma_clamped`. The multi-unit
    /// routes' default; the one-row route's default until 1.12.0 and
    /// refused by it from 2.0.0 (`Error::ClampPolicyRemoved`).
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
        VariancePolicy::Bernoulli
    }
}

impl VariancePolicy {
    /// The clamp at the default factor: the multi-unit routes' default.
    pub fn clamp_default() -> Self {
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
    pub method: Method,
    /// Number K of midpoint beta quantiles per nondegenerate event.
    pub quadrature_points: usize,
    /// Node density of the survival method's base mesh.
    pub nodes_per_decade: usize,
    /// Nodes per standard deviation a bell-shaped event asks of the fine
    /// mesh (the accuracy knob of the survival method).
    pub nodes_per_sigma: usize,
    pub overflow: Overflow,
    /// Divide both outputs by the total contributing rate.
    pub normalize: bool,
    /// Interpret mean_loss as a rate on [0, 1], scaled by max_loss.
    pub mean_is_loss_rate: bool,
    pub variance_policy: VariancePolicy,
    /// Worker threads; None uses all cores. Results are bitwise identical
    /// for any thread count (deterministic in-order chunk fold).
    pub threads: Option<usize>,
    /// Rows per parallel chunk; None resolves per method. Part of the
    /// deterministic contract: results are bitwise reproducible for a fixed
    /// (input, config) including this.
    pub chunk_rows: Option<usize>,
}

impl Config {
    pub fn new(bucket_size: f64) -> Self {
        Config {
            bucket_size,
            grid_length: None,
            method: Method::Survival,
            quadrature_points: DEFAULT_QUADRATURE_POINTS,
            nodes_per_decade: DEFAULT_NODES_PER_DECADE,
            nodes_per_sigma: DEFAULT_NODES_PER_SIGMA,
            overflow: Overflow::TopBucket,
            normalize: false,
            mean_is_loss_rate: false,
            variance_policy: VariancePolicy::default(),
            threads: None,
            chunk_rows: None,
        }
    }

    /// The chunk size in force: the explicit one, or the method's default.
    pub fn resolved_chunk_rows(&self) -> usize {
        self.chunk_rows.unwrap_or(match self.method {
            Method::Quantiles => DEFAULT_CHUNK_ROWS,
            Method::Survival => DEFAULT_SURVIVAL_CHUNK_ROWS,
        })
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
        if self.chunk_rows == Some(0) {
            return Err(Error::BadChunkRows(0));
        }
        if self.nodes_per_decade < 1 {
            return Err(Error::BadNodesPerDecade(self.nodes_per_decade));
        }
        if self.nodes_per_sigma < 1 {
            return Err(Error::BadNodesPerSigma(self.nodes_per_sigma));
        }
        if self.threads == Some(0) {
            return Err(Error::BadThreads(0));
        }
        Ok(())
    }
}
