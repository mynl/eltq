//! Row validation and classification.

use crate::config::{Config, VariancePolicy};
use crate::diagnostics::{Category, Diagnostics};
use crate::error::Error;

/// How a row contributes to the outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    /// Contributes to neither output (invalid, zero-rate, or drop policy).
    Drop,
    /// Atom at mu in both raw and secondary.
    Atom,
    /// Atom at mu in raw; K beta midpoint quantiles in secondary.
    Beta,
}

/// Classification outcome: kind plus the effective mean and variance.
#[derive(Debug, Clone, Copy)]
pub struct RowPlan {
    pub kind: RowKind,
    pub mu: f64,
    pub var: f64,
}

const DROP: RowPlan = RowPlan {
    kind: RowKind::Drop,
    mu: 0.0,
    var: 0.0,
};

/// Classify one row, recording any diagnostic. Only the strict variance
/// policy can fail; structurally invalid rows are dropped and reported.
/// The flat scalar signature is deliberate for the hot path.
#[allow(clippy::too_many_arguments)]
pub fn classify_row(
    row: usize,
    rate: f64,
    max_loss: f64,
    mean_loss: f64,
    sd_indep: f64,
    sd_corr: f64,
    config: &Config,
    diagnostics: &mut Diagnostics,
) -> Result<RowPlan, Error> {
    if !(rate.is_finite()
        && max_loss.is_finite()
        && mean_loss.is_finite()
        && sd_indep.is_finite()
        && sd_corr.is_finite())
    {
        diagnostics.note(Category::Nonfinite, row);
        return Ok(DROP);
    }
    if rate < 0.0 || max_loss < 0.0 || mean_loss < 0.0 || sd_indep < 0.0 || sd_corr < 0.0 {
        diagnostics.note(Category::NegativeField, row);
        return Ok(DROP);
    }
    let mu = if config.mean_is_loss_rate {
        mean_loss * max_loss
    } else {
        mean_loss
    };
    if mu > max_loss {
        diagnostics.note(Category::MeanExceedsMax, row);
        return Ok(DROP);
    }
    if rate == 0.0 {
        diagnostics.note(Category::ZeroRate, row);
        return Ok(DROP);
    }
    // Total conditional sd is the arithmetic sum of the two components
    // (Burke, Rau-Chaplin, Varghese 2016, section III.E, step i); the
    // admissibility test then compares sigma^2 against mu (M - mu).
    let sigma = sd_indep + sd_corr;
    let sigma2 = sigma * sigma;
    if max_loss == 0.0 || sigma2 == 0.0 || mu == 0.0 || mu == max_loss {
        // Boundary/degenerate treatment: a single atom at mu. (For a boundary
        // mean with sigma > 0 the variance is formally inadmissible; the
        // specified degenerate treatment is the resolution.)
        diagnostics.note(Category::DegenerateAtom, row);
        return Ok(RowPlan {
            kind: RowKind::Atom,
            mu,
            var: 0.0,
        });
    }
    let v_max = mu * (max_loss - mu);
    if sigma2 >= v_max {
        return match config.variance_policy {
            VariancePolicy::Error => Err(Error::InadmissibleVariance { row, sigma2, v_max }),
            VariancePolicy::Drop => {
                diagnostics.note(Category::VarianceDropped, row);
                Ok(DROP)
            }
            VariancePolicy::Atom => {
                diagnostics.note(Category::VarianceAtom, row);
                Ok(RowPlan {
                    kind: RowKind::Atom,
                    mu,
                    var: 0.0,
                })
            }
            VariancePolicy::ClampSigma { factor } => {
                diagnostics.note(Category::SigmaClamped, row);
                Ok(RowPlan {
                    kind: RowKind::Beta,
                    mu,
                    var: factor * factor * v_max,
                })
            }
        };
    }
    Ok(RowPlan {
        kind: RowKind::Beta,
        mu,
        var: sigma2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(
        rate: f64,
        m: f64,
        mu: f64,
        s: f64,
        config: &Config,
    ) -> (Result<RowPlan, Error>, Diagnostics) {
        let mut diag = Diagnostics::default();
        let plan = classify_row(0, rate, m, mu, s, 0.0, config, &mut diag);
        (plan, diag)
    }

    #[test]
    fn structural_invalids_drop_with_category() {
        let c = Config::new(1.0);
        let cases = [
            (f64::NAN, 1.0, 0.5, 0.1, Category::Nonfinite),
            (0.1, f64::INFINITY, 0.5, 0.1, Category::Nonfinite),
            (-0.1, 1.0, 0.5, 0.1, Category::NegativeField),
            (0.1, 1.0, -0.5, 0.1, Category::NegativeField),
            (0.1, 1.0, 1.5, 0.1, Category::MeanExceedsMax),
            (0.0, 1.0, 0.5, 0.1, Category::ZeroRate),
        ];
        for (rate, m, mu, s, category) in cases {
            let (plan, diag) = classify(rate, m, mu, s, &c);
            assert_eq!(plan.unwrap().kind, RowKind::Drop);
            assert_eq!(diag.get(category).count, 1);
            assert_eq!(diag.get(category).examples, vec![0]);
        }
    }

    #[test]
    fn degenerate_rows_become_atoms() {
        let c = Config::new(1.0);
        for (m, mu, s) in [(1.0, 0.4, 0.0), (1.0, 0.0, 0.2), (1.0, 1.0, 0.2), (0.0, 0.0, 0.0)] {
            let (plan, diag) = classify(0.1, m, mu, s, &c);
            let plan = plan.unwrap();
            assert_eq!(plan.kind, RowKind::Atom);
            assert_eq!(plan.mu, mu);
            assert_eq!(diag.get(Category::DegenerateAtom).count, 1);
        }
    }

    #[test]
    fn variance_policies() {
        // sigma^2 = 0.25 >= mu (M - mu) = 0.24.
        let (m, mu, s) = (1.0, 0.4, 0.5);
        let mut c = Config::new(1.0);

        let (plan, diag) = classify(0.1, m, mu, s, &c);
        let plan = plan.unwrap();
        assert_eq!(plan.kind, RowKind::Beta);
        let expected = 0.999 * 0.999 * 0.24;
        assert!((plan.var - expected).abs() < 1e-15);
        assert_eq!(diag.get(Category::SigmaClamped).count, 1);

        c.variance_policy = VariancePolicy::Error;
        assert!(matches!(
            classify(0.1, m, mu, s, &c).0,
            Err(Error::InadmissibleVariance { row: 0, .. })
        ));

        c.variance_policy = VariancePolicy::Drop;
        let (plan, diag) = classify(0.1, m, mu, s, &c);
        assert_eq!(plan.unwrap().kind, RowKind::Drop);
        assert_eq!(diag.get(Category::VarianceDropped).count, 1);

        c.variance_policy = VariancePolicy::Atom;
        let (plan, diag) = classify(0.1, m, mu, s, &c);
        assert_eq!(plan.unwrap().kind, RowKind::Atom);
        assert_eq!(diag.get(Category::VarianceAtom).count, 1);
    }

    #[test]
    fn total_sd_is_the_arithmetic_sum() {
        // sigma = 3 + 4 = 7, variance 49 (not 3^2 + 4^2 = 25).
        let c = Config::new(1.0);
        let mut diag = Diagnostics::default();
        let plan = classify_row(0, 0.1, 100.0, 40.0, 3.0, 4.0, &c, &mut diag).unwrap();
        assert_eq!(plan.kind, RowKind::Beta);
        assert_eq!(plan.var, 49.0);
        // Each component alone is admissible; their sum is not: 30 + 25 = 55,
        // 55^2 = 3025 >= 40 * 60 = 2400, so the clamp policy engages.
        let (plan, diag) = {
            let mut diag = Diagnostics::default();
            let plan = classify_row(0, 0.1, 100.0, 40.0, 30.0, 25.0, &c, &mut diag).unwrap();
            (plan, diag)
        };
        assert_eq!(plan.kind, RowKind::Beta);
        assert_eq!(diag.get(Category::SigmaClamped).count, 1);
    }

    #[test]
    fn admissible_beta_passes_through() {
        let c = Config::new(1.0);
        let (plan, diag) = classify(0.1, 100.0, 40.0, 15.0, &c);
        let plan = plan.unwrap();
        assert_eq!(plan.kind, RowKind::Beta);
        assert_eq!(plan.var, 225.0);
        assert!(diag.iter_nonzero().next().is_none());
    }

    #[test]
    fn loss_rate_mode_scales_mean_only() {
        let mut c = Config::new(1.0);
        c.mean_is_loss_rate = true;
        let (plan, _) = classify(0.1, 200.0, 0.3, 40.0, &c);
        assert_eq!(plan.unwrap().mu, 60.0);
        let (plan, diag) = classify(0.1, 200.0, 1.5, 40.0, &c);
        assert_eq!(plan.unwrap().kind, RowKind::Drop);
        assert_eq!(diag.get(Category::MeanExceedsMax).count, 1);
    }
}
