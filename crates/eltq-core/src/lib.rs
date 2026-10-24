//! eltq-core: pure-Rust core for portfolio ELT secondary-uncertainty
//! quantization.
//!
//! Converts an ELT (structure-of-arrays columns rate / max_loss / mean_loss /
//! sd_indep / sd_corr) into two evenly spaced, rate-weighted loss-grid
//! distributions: `raw` (atom at mu per event) and `secondary` (a beta on
//! [0, M] per nondegenerate event, bucketed by the survival method of
//! `curve` or deposited as K midpoint quantiles by the linear rule).
//!
//! Mathematical contract: `secondary-uncertainty-mathematical-specification.md`
//! at the repository root. This crate stays free of PyO3/NumPy/CLI types.
//!
//! Multi-unit route (1.8.0+): `multiunit` holds the per-unit preparation,
//! conditional pmfs given the shared event factor, and the two-unit joint;
//! `quadrature` the shared-factor rules; `normal` the normal cdf/quantile;
//! `terms` the DecL-style unit terms grammar and compiled view maps.

pub mod batched;
pub mod beta;
pub mod config;
pub mod convolve;
pub mod csv_input;
pub mod curve;
pub mod deposit;
pub mod incbeta;
pub mod diagnostics;
pub mod error;
pub mod kernel;
pub mod multiunit;
pub mod normal;
pub mod quadrature;
pub mod quantize;
pub mod sum;
pub mod terms;
pub mod validate;
pub mod views;

pub use curve::{
    Curve, GridOutcome, NARROW_STEP_PARAMETER_SUM, NARROW_STEPS, NarrowEvent, NarrowKernel, TAIL_INTERVALS,
    TAIL_RATE_FLOOR, build_curve,
};
pub use config::{
    Config, DEFAULT_CHUNK_ROWS, DEFAULT_CLAMP_FACTOR, DEFAULT_NODES_PER_DECADE,
    DEFAULT_QUADRATURE_POINTS, DEFAULT_SURVIVAL_CHUNK_ROWS, Method, Overflow, VariancePolicy,
};
pub use diagnostics::{Category, Diagnostics, MAX_EXAMPLES, Meta, QuantizeResult, RowIssues};
pub use error::Error;
pub use multiunit::{JointConfig, JointMeta, JointResult, UnitKind, UnitPlan, UnitSpec, joint_two_units};
pub use quadrature::{ZRule, z_quadrature};
pub use quantize::{Classified, EltInput, classify_rows, quantize};
pub use terms::{Clause, UnitTerms, View, ViewMap, check_grid_alignment, parse_terms, parse_terms_line};
pub use views::{Convolution, MultiConfig, MultiInput, MultiMeta, MultiResult, RollUpEntry, multiunit_views};
