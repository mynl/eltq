//! Quantization configuration and policies.

use crate::error::Error;

pub const DEFAULT_CLAMP_FACTOR: f64 = 0.999;
pub const DEFAULT_QUADRATURE_POINTS: usize = 32;
/// The quantiles method's default chunk is a rule of the row count and the
/// grid length, `clamp(max(ceil(n_rows / 64), ceil(grid_length / 64)),
/// 4096, 65536)`. The row term makes 64 chunks of any input between 262,144
/// and 4,194,304 rows, where the fixed 65,536 gave 262,144 rows four chunks
/// and left the rest of a 32-thread machine idle (peer review, finding P1).
/// The grid term bounds the chunks of a large grid, whose two grid-length
/// partial vectors are allocated, zeroed and folded per chunk: from 2^22
/// buckets the chunk is the old 65,536 and nothing regresses, and at 2^20
/// the 16 chunks still gain (`benchmarks/report-2.4.2.md`).
/// The rule depends on the input and the configuration, never on the thread
/// count, so results stay a function of those two. The survival method's
/// partials are node-length, so its chunk is small and fixed.
pub const DEFAULT_CHUNK_ROWS: usize = 64 * 1024;
pub const MIN_QUANTILES_CHUNK_ROWS: usize = 4096;
pub const QUANTILES_CHUNK_DIVISOR: usize = 64;
pub const QUANTILES_CHUNK_GRID_DIVISOR: usize = 64;
pub const DEFAULT_SURVIVAL_CHUNK_ROWS: usize = 256;
/// Memory the partials of one wave of chunks may hold, 1 GiB: a pass runs
/// its chunks in waves of `min(2 * threads, max(1, budget / partial_bytes))`,
/// bounding the partials in flight by memory as well as by threads (a
/// quantiles partial is two grid-length vectors, 256 MiB at 2^24 buckets).
pub const DEFAULT_WAVE_MEMORY_BYTES: usize = 1 << 30;
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
    /// default for the one-row route from 1.13.0 and for the multi-unit
    /// route from 2.5.0, where the unit is the two-point law of spec
    /// section 4A given the shared factor; reported under
    /// `variance_bernoulli`.
    Bernoulli,
    /// Clamp sigma to `factor * sqrt(mu (M - mu))`, keeping the row as a
    /// (very U-shaped) beta. Reported under `sigma_clamped`. The one-row
    /// route's default until 1.12.0 and refused by it from 2.0.0
    /// (`Error::ClampPolicyRemoved`); the multi-unit routes' default until
    /// 2.5.0, deprecated there and removed at 3.0.0.
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
    /// The clamp at the default factor (the multi-unit routes' default
    /// before 2.5.0).
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
    /// Rows per parallel chunk; None resolves per method (the row-count
    /// rule under the quantiles method, 256 under the survival method). Part
    /// of the deterministic contract: results are bitwise reproducible for a
    /// fixed (input, config) including this.
    pub chunk_rows: Option<usize>,
    /// Bytes the partials of one wave of chunks may hold
    /// (`DEFAULT_WAVE_MEMORY_BYTES`); affects scheduling only, never results.
    pub wave_memory_bytes: usize,
}

/// The options row classification reads (spec section 4): the convention
/// of the mean and the variance policy. No grid option is among them, so a
/// classification made from these alone (the model moments and tail
/// measures of spec section 4B) cannot depend on a bucket size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassifyOptions {
    pub mean_is_loss_rate: bool,
    pub variance_policy: VariancePolicy,
}

impl Default for ClassifyOptions {
    fn default() -> Self {
        ClassifyOptions {
            mean_is_loss_rate: false,
            variance_policy: VariancePolicy::default(),
        }
    }
}

impl ClassifyOptions {
    /// Reject a clamp factor outside (0, 1); the one-row route refuses the
    /// clamp itself row by row (`Error::ClampPolicyRemoved`).
    pub fn validate(&self) -> Result<(), Error> {
        if let VariancePolicy::ClampSigma { factor } = self.variance_policy
            && !(factor.is_finite() && factor > 0.0 && factor < 1.0)
        {
            return Err(Error::BadClampFactor(factor));
        }
        Ok(())
    }
}

impl Config {
    /// The classification options of this configuration.
    pub fn classify_options(&self) -> ClassifyOptions {
        ClassifyOptions {
            mean_is_loss_rate: self.mean_is_loss_rate,
            variance_policy: self.variance_policy,
        }
    }

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
            wave_memory_bytes: DEFAULT_WAVE_MEMORY_BYTES,
        }
    }

    /// The chunk size in force for `n_rows` rows on a grid of `grid_length`
    /// buckets: the explicit one, or the method's default (the rule above
    /// under the quantiles method; the survival method's chunk depends on
    /// neither argument).
    pub fn resolved_chunk_rows(&self, n_rows: usize, grid_length: usize) -> usize {
        self.chunk_rows.unwrap_or(match self.method {
            Method::Quantiles => n_rows
                .div_ceil(QUANTILES_CHUNK_DIVISOR)
                .max(grid_length.div_ceil(QUANTILES_CHUNK_GRID_DIVISOR))
                .clamp(MIN_QUANTILES_CHUNK_ROWS, DEFAULT_CHUNK_ROWS),
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

#[cfg(test)]
mod tests {
    use super::*;

    // [schedule] The quantiles default chunk is a rule of the row count and
    // the grid length: 4,096 up to 262,144 rows on grids to 2^18, then 64
    // chunks, 65,536 from 4,194,304 rows or 2^22 buckets.
    #[test]
    fn quantiles_chunk_rule() {
        let mut c = Config::new(1.0);
        c.method = Method::Quantiles;
        assert_eq!(c.resolved_chunk_rows(0, 1), 4096);
        assert_eq!(c.resolved_chunk_rows(1_000, 1 << 16), 4096);
        assert_eq!(c.resolved_chunk_rows(262_144, 1 << 18), 4096);
        assert_eq!(c.resolved_chunk_rows(262_144, 1 << 20), 16_384);
        assert_eq!(c.resolved_chunk_rows(1_000_000, 1 << 16), 15_625);
        assert_eq!(c.resolved_chunk_rows(1_000_000, 1 << 20), 16_384);
        assert_eq!(c.resolved_chunk_rows(262_144, 1 << 22), 65_536);
        assert_eq!(c.resolved_chunk_rows(262_144, 1 << 24), 65_536);
        assert_eq!(c.resolved_chunk_rows(4_194_304, 1 << 16), 65_536);
        assert_eq!(c.resolved_chunk_rows(100_000_000, 1 << 16), 65_536);
        c.chunk_rows = Some(100);
        assert_eq!(c.resolved_chunk_rows(1_000_000, 1 << 24), 100);
        c.chunk_rows = None;
        c.method = Method::Survival;
        assert_eq!(c.resolved_chunk_rows(1_000_000, 1 << 24), 256);
    }
}
