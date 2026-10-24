//! Per-category row diagnostics and result metadata.

use crate::config::{Method, Overflow};

/// Maximum number of representative row indices retained per category.
pub const MAX_EXAMPLES: usize = 10;

/// Canonical diagnostic categories, in canonical order. Names match the
/// Python reference (`tests/reference.py::CATEGORIES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// Any field NaN/Inf: row dropped.
    Nonfinite,
    /// rate, max_loss, mean_loss, or an sd < 0: row dropped.
    NegativeField,
    /// mu > M (after optional loss-rate scaling): row dropped.
    MeanExceedsMax,
    /// rate == 0: row carries no mass, skipped.
    ZeroRate,
    /// sigma == 0, mu == 0, mu == M, or M == 0: single atom.
    DegenerateAtom,
    /// Inadmissible variance repaired by the clamp policy.
    SigmaClamped,
    /// Inadmissible variance, row dropped by the drop policy.
    VarianceDropped,
    /// Inadmissible variance collapsed to an atom by policy.
    VarianceAtom,
    /// Inadmissible variance replaced by the Bernoulli maximizer (two atoms).
    VarianceBernoulli,
}

impl Category {
    pub const ALL: [Category; 9] = [
        Category::Nonfinite,
        Category::NegativeField,
        Category::MeanExceedsMax,
        Category::ZeroRate,
        Category::DegenerateAtom,
        Category::SigmaClamped,
        Category::VarianceDropped,
        Category::VarianceAtom,
        Category::VarianceBernoulli,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Category::Nonfinite => "nonfinite",
            Category::NegativeField => "negative_field",
            Category::MeanExceedsMax => "mean_exceeds_max",
            Category::ZeroRate => "zero_rate",
            Category::DegenerateAtom => "degenerate_atom",
            Category::SigmaClamped => "sigma_clamped",
            Category::VarianceDropped => "variance_dropped",
            Category::VarianceAtom => "variance_atom",
            Category::VarianceBernoulli => "variance_bernoulli",
        }
    }
}

/// Count plus a bounded set of representative row indices for one category.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct RowIssues {
    pub count: u64,
    pub examples: Vec<usize>,
}

impl RowIssues {
    fn add(&mut self, row: usize) {
        self.count += 1;
        if self.examples.len() < MAX_EXAMPLES {
            self.examples.push(row);
        }
    }
}

/// Diagnostics across all categories. Row indices are 0-based input rows;
/// callers holding event ids translate at the API boundary.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Diagnostics {
    issues: [RowIssues; 9],
}

impl Diagnostics {
    pub fn note(&mut self, category: Category, row: usize) {
        self.issues[category as usize].add(row);
    }

    pub fn get(&self, category: Category) -> &RowIssues {
        &self.issues[category as usize]
    }

    /// Fold `other` into `self` (counts add, examples extend up to the cap);
    /// merging in a fixed order keeps the examples deterministic.
    pub fn merge(&mut self, other: &Diagnostics) {
        for (mine, theirs) in self.issues.iter_mut().zip(&other.issues) {
            mine.count += theirs.count;
            for &row in &theirs.examples {
                if mine.examples.len() >= MAX_EXAMPLES {
                    break;
                }
                mine.examples.push(row);
            }
        }
    }

    /// Categories with at least one affected row, in canonical order.
    pub fn iter_nonzero(&self) -> impl Iterator<Item = (Category, &RowIssues)> {
        Category::ALL
            .iter()
            .map(|&c| (c, self.get(c)))
            .filter(|(_, issues)| issues.count > 0)
    }
}

/// Metadata accompanying the two grid distributions. Masses and means
/// describe the arrays actually returned (normalized when `normalized`);
/// `total_rate`, `analytic_aal`, `quadrature_secondary_mean` and
/// `secondary_mean_error` are always in rate-weighted units, the last being
/// the secondary grid's first moment less the analytic AAL under either
/// method. The quantiles fields are `None` under the survival method and
/// the survival fields `None` under the quantiles method. Overflow figures
/// are `[raw, secondary]`.
#[derive(Debug, Clone)]
pub struct Meta {
    pub bucket_size: f64,
    pub grid_length: usize,
    pub method: Method,
    pub quadrature_points: Option<usize>,
    pub nodes: Option<usize>,
    pub nodes_per_decade: Option<usize>,
    pub interpolation_error: Option<f64>,
    /// Beta events kept out of the curve as narrow (survival method).
    pub narrow_events: Option<usize>,
    pub overflow: Overflow,
    pub rate_above_grid: [f64; 2],
    pub mean_above_grid: [f64; 2],
    pub normalized: bool,
    pub total_rate: f64,
    pub analytic_aal: f64,
    pub raw_mass: f64,
    pub raw_mean: f64,
    pub secondary_mass: f64,
    pub secondary_mean: f64,
    pub quadrature_secondary_mean: Option<f64>,
    pub secondary_mean_error: f64,
    pub input_rows: usize,
    pub valid_rows: usize,
    pub diagnostics: Diagnostics,
}

/// The two grid distributions plus metadata.
#[derive(Debug, Clone)]
pub struct QuantizeResult {
    pub raw: Vec<f64>,
    pub secondary: Vec<f64>,
    pub meta: Meta,
}
