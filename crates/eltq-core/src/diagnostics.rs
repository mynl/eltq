//! Per-category row diagnostics and result metadata.

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
}

impl Category {
    pub const ALL: [Category; 8] = [
        Category::Nonfinite,
        Category::NegativeField,
        Category::MeanExceedsMax,
        Category::ZeroRate,
        Category::DegenerateAtom,
        Category::SigmaClamped,
        Category::VarianceDropped,
        Category::VarianceAtom,
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
    issues: [RowIssues; 8],
}

impl Diagnostics {
    pub fn note(&mut self, category: Category, row: usize) {
        self.issues[category as usize].add(row);
    }

    pub fn get(&self, category: Category) -> &RowIssues {
        &self.issues[category as usize]
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
/// `secondary_mean_error` are always in rate-weighted units.
#[derive(Debug, Clone)]
pub struct Meta {
    pub bucket_size: f64,
    pub grid_length: usize,
    pub quadrature_points: usize,
    pub normalized: bool,
    pub total_rate: f64,
    pub analytic_aal: f64,
    pub raw_mass: f64,
    pub raw_mean: f64,
    pub secondary_mass: f64,
    pub secondary_mean: f64,
    pub quadrature_secondary_mean: f64,
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
