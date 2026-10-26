//! Model moments of an ELT with no loss grid (spec section 4B): the total
//! rate, the rate-weighted raw moments `C_r = sum_j lambda_j P(X_j^r)` for
//! `r = 1, 2, 3`, the zero rate and the support, from the classification of
//! spec section 4 and closed forms of the three effective laws.
//!
//! Nothing here takes a bucket size or a grid length, so no output can
//! depend on a consumer's future grid, and the cost is one classification
//! pass and a few multiplications per row whatever the exposed values
//! (`dev/done/plan-model-moments-and-windowed-views.md`,
//! [Eltq-Model-Moments]).

use crate::beta::beta_params;
use crate::config::ClassifyOptions;
use crate::diagnostics::Diagnostics;
use crate::error::Error;
use crate::quantize::{EltInput, classify_rows_with};
use crate::sum::Neumaier;
use crate::validate::RowKind;

/// The effective law of a contributing row or unit (spec section 4B).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RowLaw {
    /// All the mass at `mu`.
    Atom { mu: f64 },
    /// `M Beta(alpha, beta)` moment-matched to `(mu, sd)` on `[0, max_loss]`.
    Beta { mu: f64, sd: f64, max_loss: f64 },
    /// `0` or `max_loss`, with `P(X = max_loss) = mu / max_loss` (a
    /// Bernoulli row).
    TwoPoint { mu: f64, max_loss: f64 },
}

impl RowLaw {
    /// `P(X^r)` for `r = 1, 2, 3`.
    ///
    /// Notes
    /// -----
    /// The beta's rising-factorial moment `M^r (alpha)_r / (alpha + beta)_r`
    /// is formed as `mu` times the ratios `P(X^(k+1)) / P(X^k) = M (alpha +
    /// k) / (alpha + beta + k)`, each at most `M`, so no intermediate exceeds
    /// the larger of `M` and the result and the parameters enter only
    /// through the ratios that `beta_params` forms (spec section 4); at
    /// `r = 2` the result is `mu^2 + sigma^2`. A moment beyond double
    /// precision comes back infinite, for the caller to report.
    pub fn raw_moments(&self) -> [f64; 3] {
        match *self {
            RowLaw::Atom { mu } => {
                let second = mu * mu;
                [mu, second, second * mu]
            }
            RowLaw::Beta { mu, sd, max_loss } => {
                let (alpha, beta) = beta_params(max_loss, mu, sd);
                let second = mu * (max_loss * ((alpha + 1.0) / (alpha + beta + 1.0)));
                [mu, second, second * (max_loss * ((alpha + 2.0) / (alpha + beta + 2.0)))]
            }
            RowLaw::TwoPoint { mu, max_loss } => {
                let second = mu * max_loss;
                [mu, second, second * max_loss]
            }
        }
    }

    /// The largest loss the law reaches: `mu` for an atom, the exposed
    /// value otherwise.
    pub fn support(&self) -> f64 {
        match *self {
            RowLaw::Atom { mu } => mu,
            RowLaw::Beta { max_loss, .. } | RowLaw::TwoPoint { max_loss, .. } => max_loss,
        }
    }

    /// `P(X = 0)`.
    pub fn zero_probability(&self) -> f64 {
        match *self {
            RowLaw::Atom { mu } => {
                if mu == 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            RowLaw::Beta { .. } => 0.0,
            RowLaw::TwoPoint { mu, max_loss } => (max_loss - mu) / max_loss,
        }
    }
}

/// Model moments of a one-row ELT (spec section 4B). `raw_moments` are
/// rate-weighted; an order whose sum leaves double precision is infinite
/// and flagged in `moment_overflow`.
#[derive(Debug, Clone)]
pub struct ModelMoments {
    pub total_rate: f64,
    pub raw_moments: [f64; 3],
    /// `sum_j lambda_j P(X_j = 0)`.
    pub rate_at_zero: f64,
    /// Per input row, the support of its effective law; NaN where dropped.
    pub support: Vec<f64>,
    pub max_support: f64,
    pub input_rows: usize,
    pub valid_rows: usize,
    pub atom_rows: usize,
    pub beta_rows: usize,
    pub two_point_rows: usize,
    pub dropped_rows: usize,
    pub moment_overflow: [bool; 3],
    pub diagnostics: Diagnostics,
}

/// Rate-weighted sums of the three raw moments, the rate and the zero rate,
/// each compensated, with a moment that leaves double precision flagged
/// rather than summed (an infinite term would turn the compensation into
/// NaN).
#[derive(Debug, Default)]
pub(crate) struct MomentSums {
    rate: Neumaier,
    zero: Neumaier,
    moments: [Neumaier; 3],
    overflow: [bool; 3],
}

impl MomentSums {
    pub(crate) fn add(&mut self, rate: f64, moments: [f64; 3], zero_probability: f64) {
        self.rate.add(rate);
        self.zero.add(rate * zero_probability);
        for r in 0..3 {
            let term = rate * moments[r];
            if term.is_finite() {
                self.moments[r].add(term);
            } else {
                self.overflow[r] = true;
            }
        }
    }

    /// `(total rate, raw moments, zero rate, overflow)`.
    pub(crate) fn finish(&self) -> (f64, [f64; 3], f64, [bool; 3]) {
        let mut overflow = self.overflow;
        let mut moments = [0.0; 3];
        for r in 0..3 {
            let value = self.moments[r].value();
            if overflow[r] || !value.is_finite() {
                overflow[r] = true;
                moments[r] = f64::INFINITY;
            } else {
                moments[r] = value;
            }
        }
        (self.rate.value(), moments, self.zero.value(), overflow)
    }
}

/// The law a classified one-row ELT row contributes, or None for a dropped
/// row.
pub fn row_law(kind: RowKind, mu: f64, sd: f64, max_loss: f64) -> Option<RowLaw> {
    match kind {
        RowKind::Drop => None,
        RowKind::Atom => Some(RowLaw::Atom { mu }),
        RowKind::Beta => Some(RowLaw::Beta { mu, sd, max_loss }),
        RowKind::Bernoulli => Some(RowLaw::TwoPoint { mu, max_loss }),
    }
}

/// Model moments of a one-row ELT (spec section 4B): one classification
/// pass, exactly as the grid route classifies the same input, and closed
/// forms per row, summed in row order with compensation, so the result is
/// a function of the input alone.
pub fn model_moments(input: &EltInput<'_>, options: &ClassifyOptions) -> Result<ModelMoments, Error> {
    let classified = classify_rows_with(input, options)?;
    let n_rows = input.rate.len();
    let mut sums = MomentSums::default();
    let mut support = vec![f64::NAN; n_rows];
    let mut max_support = 0.0_f64;
    let (mut atom_rows, mut beta_rows, mut two_point_rows) = (0, 0, 0);
    for row in 0..n_rows {
        let kind = classified.kinds[row];
        let Some(law) = row_law(kind, classified.mu_eff[row], classified.sd_eff[row], input.max_loss[row]) else {
            continue;
        };
        match law {
            RowLaw::Atom { .. } => atom_rows += 1,
            RowLaw::Beta { .. } => beta_rows += 1,
            RowLaw::TwoPoint { .. } => two_point_rows += 1,
        }
        sums.add(input.rate[row], law.raw_moments(), law.zero_probability());
        support[row] = law.support();
        max_support = max_support.max(support[row]);
    }
    let (total_rate, raw_moments, rate_at_zero, moment_overflow) = sums.finish();
    let valid_rows = atom_rows + beta_rows + two_point_rows;
    Ok(ModelMoments {
        total_rate,
        raw_moments,
        rate_at_zero,
        support,
        max_support,
        input_rows: n_rows,
        valid_rows,
        atom_rows,
        beta_rows,
        two_point_rows,
        dropped_rows: n_rows - valid_rows,
        moment_overflow,
        diagnostics: classified.diagnostics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, VariancePolicy};
    use crate::quantize::quantize;

    fn input<'a>(cols: &'a [Vec<f64>; 5]) -> EltInput<'a> {
        EltInput {
            rate: &cols[0],
            max_loss: &cols[1],
            mean_loss: &cols[2],
            sd_indep: &cols[3],
            sd_corr: &cols[4],
        }
    }

    // [Eltq-Model-Moments] Spec test 35, the aggregate package's oracle rows:
    // the beta (2.625, 2.625) on [0, 100], the same row as an atom and as a
    // two-point row, and a positive-rate row at zero.
    #[test]
    fn oracle_rows() {
        let options = ClassifyOptions::default();
        let cases: [([f64; 5], [f64; 3], f64, f64); 3] = [
            ([0.1, 100.0, 50.0, 10.0, 10.0], [5.0, 290.0, 18_500.0], 0.0, 100.0),
            ([0.1, 100.0, 50.0, 0.0, 0.0], [5.0, 250.0, 12_500.0], 0.0, 50.0),
            ([0.1, 100.0, 50.0, 30.0, 30.0], [5.0, 500.0, 50_000.0], 0.05, 100.0),
        ];
        for (row, want, zero, support) in cases {
            let cols = row.map(|v| vec![v]);
            let m = model_moments(&input(&cols), &options).unwrap();
            for r in 0..3 {
                assert!((m.raw_moments[r] - want[r]).abs() <= 1e-13 * want[r], "{row:?} r={r}: {}", m.raw_moments[r]);
            }
            assert!((m.rate_at_zero - zero).abs() < 1e-17);
            assert_eq!((m.total_rate, m.max_support), (0.1, support));
        }
        let cols = [vec![0.1, 0.2], vec![100.0, 0.0], vec![0.0, 0.0], vec![5.0, 0.0], vec![0.0, 0.0]];
        let m = model_moments(&input(&cols), &options).unwrap();
        assert_eq!((m.total_rate, m.rate_at_zero, m.raw_moments), (0.30000000000000004, 0.30000000000000004, [0.0; 3]));
        assert_eq!((m.atom_rows, m.dropped_rows, m.max_support), (2, 0, 0.0));
    }

    // The classification is the grid route's: the same rate, AAL, counts and
    // diagnostics, a dropped row with NaN support, and no bucket size
    // anywhere; a clamp is still refused by the one-row classification.
    #[test]
    fn classification_matches_the_grid_route() {
        let cols = [
            vec![0.1, 0.2, 0.0, 0.3, f64::NAN, 0.05],
            vec![1e6, 2e6, 1e6, 5e5, 1e6, 1e3],
            vec![2e4, 1e5, 1e4, 6e5, 1e4, 1e3],
            vec![3e4, 1e5, 0.0, 1e4, 0.0, 0.0],
            vec![3e4, 9e5, 0.0, 1e4, 0.0, 0.0],
        ];
        let m = model_moments(&input(&cols), &ClassifyOptions::default()).unwrap();
        let q = quantize(&input(&cols), &Config::new(1000.0)).unwrap();
        assert_eq!(m.total_rate, q.meta.total_rate);
        assert!((m.raw_moments[0] - q.meta.analytic_aal).abs() <= 1e-15 * q.meta.analytic_aal);
        assert_eq!(m.valid_rows, q.meta.valid_rows);
        assert_eq!((m.beta_rows, m.two_point_rows, m.atom_rows, m.dropped_rows), (1, 1, 1, 3));
        assert!(m.support[2].is_nan() && m.support[3].is_nan() && m.support[4].is_nan());
        assert_eq!(m.max_support, 2e6);
        for (category, issues) in q.meta.diagnostics.iter_nonzero() {
            assert_eq!(m.diagnostics.get(category).count, issues.count);
        }
        let clamp = ClassifyOptions { variance_policy: VariancePolicy::clamp_default(), ..Default::default() };
        assert!(matches!(model_moments(&input(&cols), &clamp), Err(Error::ClampPolicyRemoved)));
    }

    // Moments scale as s^r; a scale at which C_3 leaves double precision is
    // flagged and infinite, never a silent value.
    #[test]
    fn scaling_and_overflow() {
        let base = [0.1, 100.0, 40.0, 10.0, 15.0];
        let unit = model_moments(&input(&base.map(|v| vec![v])), &ClassifyOptions::default()).unwrap();
        for scale in [1e100, 1e-100] {
            let row = [base[0], base[1] * scale, base[2] * scale, base[3] * scale, base[4] * scale];
            let m = model_moments(&input(&row.map(|v| vec![v])), &ClassifyOptions::default()).unwrap();
            for r in 0..3 {
                let want = unit.raw_moments[r] * scale.powi(r as i32 + 1);
                assert!((m.raw_moments[r] - want).abs() <= 1e-14 * want, "scale {scale} r={r}");
            }
            assert_eq!(m.moment_overflow, [false; 3]);
        }
        let row = [base[0], base[1] * 1e105, base[2] * 1e105, base[3] * 1e105, base[4] * 1e105];
        let m = model_moments(&input(&row.map(|v| vec![v])), &ClassifyOptions::default()).unwrap();
        assert_eq!(m.moment_overflow, [false, false, true]);
        assert!(m.raw_moments[1].is_finite() && m.raw_moments[2] == f64::INFINITY);
    }
}
