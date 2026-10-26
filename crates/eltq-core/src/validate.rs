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
    /// Atom at mu in raw; a beta on [0, M] in secondary.
    Beta,
    /// Inadmissible variance under the Bernoulli policy: atom at mu in raw;
    /// `rate (1 - p)` at 0 and `rate p` at M in secondary, `p = mu / M`.
    Bernoulli,
}

/// Classification outcome: kind plus the effective mean and standard
/// deviation (the sd rather than the variance, so that nothing downstream
/// holds a product of two monetary amounts).
#[derive(Debug, Clone, Copy)]
pub struct RowPlan {
    pub kind: RowKind,
    pub mu: f64,
    pub sd: f64,
}

const DROP: RowPlan = RowPlan {
    kind: RowKind::Drop,
    mu: 0.0,
    sd: 0.0,
};

/// Classify one row, recording any diagnostic. Only the strict variance
/// policy (and the removed clamp) can fail; structurally invalid rows are
/// dropped and reported.
/// The flat scalar signature is deliberate for the hot path.
#[allow(clippy::too_many_arguments)]
/// `2^-48`: a beta row with `sigma < ATOM_RESOLUTION * mu` is an atom at
/// `mu` (sixteen ulps of the loss at the mean is the finest width the
/// standardized coordinate resolves).
pub const ATOM_RESOLUTION: f64 = 3.552713678800501e-15;

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
    if max_loss == 0.0 || sigma == 0.0 || mu == 0.0 || mu == max_loss {
        // Boundary/degenerate treatment: a single atom at mu. (For a boundary
        // mean with sigma > 0 the variance is formally inadmissible; the
        // specified degenerate treatment is the resolution.)
        diagnostics.note(Category::DegenerateAtom, row);
        return Ok(RowPlan {
            kind: RowKind::Atom,
            mu,
            sd: 0.0,
        });
    }
    // A standard deviation below 2^-48 of the mean (sixteen ulps of the loss
    // at the mean): the standardized coordinate no longer resolves the law
    // and the row is an atom at mu, reported apart from a stated zero so
    // that a consumer can tell the two (spec section 4;
    // `dev/done/plan-peer-review-response.md`, [atoms]).
    if sigma < mu * ATOM_RESOLUTION {
        diagnostics.note(Category::AtomByResolution, row);
        return Ok(RowPlan {
            kind: RowKind::Atom,
            mu,
            sd: 0.0,
        });
    }
    // The test sigma^2 >= mu (M - mu) in ratios, (sigma / mu)^2 against
    // M / mu - 1, so that no product of two monetary amounts is formed:
    // sigma^2 overflows from a scale of about 1e154 and underflows below
    // about 1e-154 while every field stays finite, and the row was then a
    // Bernoulli row or an atom that its unit-scale twin is not (peer review,
    // finding I3). `beta_params` forms the shape parameters from the same
    // ratios. The error carries the two sides in monetary units as before.
    let cv = sigma / mu;
    let bound = (max_loss - mu) / mu;
    if cv * cv >= bound {
        return match config.variance_policy {
            VariancePolicy::Bernoulli => {
                diagnostics.note(Category::VarianceBernoulli, row);
                Ok(RowPlan {
                    kind: RowKind::Bernoulli,
                    mu,
                    // sqrt(mu (M - mu)), the maximizer's sd, in ratios.
                    sd: mu * bound.sqrt(),
                })
            }
            VariancePolicy::Error => Err(Error::InadmissibleVariance {
                row,
                sigma2: sigma * sigma,
                v_max: mu * (max_loss - mu),
            }),
            VariancePolicy::Drop => {
                diagnostics.note(Category::VarianceDropped, row);
                Ok(DROP)
            }
            VariancePolicy::Atom => {
                diagnostics.note(Category::VarianceAtom, row);
                Ok(RowPlan {
                    kind: RowKind::Atom,
                    mu,
                    sd: 0.0,
                })
            }
            // The clamp stays in the enum for the multi-unit route; the
            // one-row route refuses it from 2.0.0 (spec section 4).
            VariancePolicy::ClampSigma { .. } => Err(Error::ClampPolicyRemoved),
        };
    }
    Ok(RowPlan {
        kind: RowKind::Beta,
        mu,
        sd: sigma,
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

        // Default: the Bernoulli maximizer, variance exactly mu (M - mu).
        let (plan, diag) = classify(0.1, m, mu, s, &c);
        let plan = plan.unwrap();
        assert_eq!(plan.kind, RowKind::Bernoulli);
        assert_eq!(plan.mu, mu);
        assert!((plan.sd - 0.24_f64.sqrt()).abs() < 1e-15);
        assert_eq!(diag.get(Category::VarianceBernoulli).count, 1);

        c.variance_policy = VariancePolicy::clamp_default();
        assert!(matches!(classify(0.1, m, mu, s, &c).0, Err(Error::ClampPolicyRemoved)));

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
        assert_eq!(plan.sd, 7.0);
        // Each component alone is admissible; their sum is not: 30 + 25 = 55,
        // 55^2 = 3025 >= 40 * 60 = 2400, so the variance policy engages.
        let (plan, diag) = {
            let mut diag = Diagnostics::default();
            let plan = classify_row(0, 0.1, 100.0, 40.0, 30.0, 25.0, &c, &mut diag).unwrap();
            (plan, diag)
        };
        assert_eq!(plan.kind, RowKind::Bernoulli);
        assert_eq!(diag.get(Category::VarianceBernoulli).count, 1);
    }

    #[test]
    fn admissible_beta_passes_through() {
        let c = Config::new(1.0);
        let (plan, diag) = classify(0.1, 100.0, 40.0, 15.0, &c);
        let plan = plan.unwrap();
        assert_eq!(plan.kind, RowKind::Beta);
        assert_eq!(plan.sd, 15.0);
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

    // [atoms] A standard deviation below 2^-48 of the mean is an atom under
    // its own category; one ulp above the line stays a beta.
    #[test]
    fn unresolvable_width_is_an_atom() {
        let c = Config::new(1.0);
        let (plan, diag) = classify(1.0, 1.0, 0.5, 1e-16, &c);
        let plan = plan.unwrap();
        assert_eq!(plan.kind, RowKind::Atom);
        assert_eq!(plan.sd, 0.0);
        assert_eq!(diag.get(Category::AtomByResolution).count, 1);
        assert_eq!(diag.get(Category::DegenerateAtom).count, 0);
        let (plan, diag) = classify(1.0, 1.0, 0.5, 0.5 * ATOM_RESOLUTION * (1.0 + 1e-12), &c);
        assert_eq!(plan.unwrap().kind, RowKind::Beta);
        assert_eq!(diag.get(Category::AtomByResolution).count, 0);
    }

    // [ratios] The classification is the same at a monetary scale of 1e160
    // (sigma^2 and mu (M - mu) overflow) and 1e-200 (both underflow): the
    // admissible row stays a beta, the inadmissible one a Bernoulli row, and
    // the degenerate tests act on the fields themselves (peer review,
    // finding I3).
    #[test]
    fn classification_is_scale_invariant_across_the_double_range() {
        let c = Config::new(1.0);
        for scale in [1.0, 1e160, 1e-200] {
            let (plan, diag) = classify(0.1, scale, 0.4 * scale, 0.1 * scale, &c);
            let plan = plan.unwrap();
            assert_eq!(plan.kind, RowKind::Beta, "{scale}");
            assert_eq!(plan.sd, 0.1 * scale);
            assert!(diag.iter_nonzero().next().is_none());
            let (plan, diag) = classify(0.1, scale, 0.4 * scale, 0.5 * scale, &c);
            let plan = plan.unwrap();
            assert_eq!(plan.kind, RowKind::Bernoulli, "{scale}");
            assert!((plan.sd / (0.24_f64.sqrt() * scale) - 1.0).abs() < 1e-15);
            assert_eq!(diag.get(Category::VarianceBernoulli).count, 1);
            let (plan, diag) = classify(0.1, scale, 0.4 * scale, 0.0, &c);
            assert_eq!(plan.unwrap().kind, RowKind::Atom);
            assert_eq!(diag.get(Category::DegenerateAtom).count, 1);
        }
    }
}
