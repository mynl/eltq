//! eltq-core: pure-Rust core for portfolio ELT secondary-uncertainty
//! quantization.
//!
//! Converts an ELT (structure-of-arrays columns rate / max_loss / mean_loss /
//! sd_indep / sd_corr) into two evenly spaced, rate-weighted loss-grid
//! distributions: `raw` (atom at mu per event) and `secondary` (K midpoint
//! beta quantiles per nondegenerate event), via a mass- and mean-preserving
//! linear deposit rule.
//!
//! Mathematical contract: `secondary-uncertainty-mathematical-specification.md`
//! at the repository root. This crate stays free of PyO3/NumPy/CLI types.

pub mod batched;
pub mod beta;
pub mod config;
pub mod csv_input;
pub mod deposit;
pub mod incbeta;
pub mod diagnostics;
pub mod error;
pub mod kernel;
pub mod quantize;
pub mod sum;
pub mod validate;

pub use config::{
    Config, DEFAULT_CHUNK_ROWS, DEFAULT_CLAMP_FACTOR, DEFAULT_QUADRATURE_POINTS, VariancePolicy,
};
pub use diagnostics::{Category, Diagnostics, MAX_EXAMPLES, Meta, QuantizeResult, RowIssues};
pub use error::Error;
pub use quantize::{EltInput, quantize};
