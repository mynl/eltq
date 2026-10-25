//! Own regularized incomplete-beta CDF and PDF with per-event constants.
//!
//! The continued fraction of Numerical Recipes in C, second edition
//! (Press, Teukolsky, Vetterling, Flannery, 1992; `Press1992`), section 6.4:
//! Abramowitz and Stegun 26.5.8 evaluated by the modified Lentz method with
//! the symmetry switch at x = (alpha + 1) / (alpha + beta + 2). The per-event
//! constant ln B(alpha, beta) is computed once per event, not per
//! probability, the key setup reuse the scalar statrs path cannot offer. For
//! a large parameter it comes from the difference of two Stirling series
//! (section 6.1 of the same source), not from three ln Gamma values, whose
//! cancellation costs seven digits at beta = 1e8
//! (`dev/done/plan-ln-beta-large-parameters.md`).
//!
//! The fraction converges in tens of iterations when one parameter is small
//! and stops converging when both are large (about 1e-10 near the mean at
//! alpha + beta = 1e4, nothing usable above 1e7 at the old cap of 400). A fraction that has not
//! converged after `MAX_CF_ITERS` is reported, never returned as a value:
//! `betacf` yields `None`, the probabilities become NaN, and the callers
//! route around the regime through `is_valid_regime` or raise
//! (`dev/done/plan-peer-review-response.md`, [kernel]).

use statrs::function::gamma::ln_gamma;

use crate::batched::LARGE_PARAMETER_FROM;

/// At the mean the fraction needs about 1.1 sqrt(alpha + beta) iterations
/// when both parameters are large (315 at 2e5 symmetric, 443 at 2e5 and
/// 6e7, 526 at 1e6 symmetric): 1,000 covers min(alpha, beta) to about 2e6
/// at every ratio, ten times the regime the engine evaluates.
const MAX_CF_ITERS: usize = 1000;
const CF_EPS: f64 = 1e-16;
const FPMIN: f64 = 1e-300;

/// The larger parameter from which `ln B` takes the Stirling route. Six
/// terms of the series are exact to double at 100; below it the sum of
/// three `ln_gamma` values cancels by at most 1e-13.
const LN_BETA_STIRLING_FROM: f64 = 100.0;

/// The Stirling correction `sigma(z) = sum B_2k / (2k (2k - 1) z^(2k - 1))`,
/// six terms, for `z >= LN_BETA_STIRLING_FROM`: `ln Gamma(z) = (z - 1/2) ln z
/// - z + ln(2 pi) / 2 + sigma(z)`. The coefficients are the Bernoulli numbers
/// B_2 to B_12 over 2k (2k - 1).
#[inline]
fn stirling_correction(z: f64) -> f64 {
    let w = 1.0 / (z * z);
    (1.0 / 12.0
        + w * (-1.0 / 360.0
            + w * (1.0 / 1260.0
                + w * (-1.0 / 1680.0 + w * (1.0 / 1188.0 + w * (-691.0 / 360360.0))))))
        / z
}

/// `ln B(a, b)` for `a <= b` with `b >= LN_BETA_STIRLING_FROM`, as
/// `ln Gamma(a) + [ln Gamma(b) - ln Gamma(a + b)]` with the bracket from the
/// two Stirling series subtracted symbolically:
/// `a - (b - 1/2) ln(1 + a / b) - a ln(a + b) + sigma(b) - sigma(a + b)`.
/// Every term is O(a) or O(a ln b), so nothing of size b cancels; the naive
/// difference of two values near b ln b loses about log10(b) digits.
fn ln_beta_large(a: f64, b: f64) -> f64 {
    let bracket = a - (b - 0.5) * (a / b).ln_1p() - a * (a + b).ln()
        + stirling_correction(b)
        - stirling_correction(a + b);
    ln_gamma(a) + bracket
}

/// `ln B(alpha, beta)`: the three-term `ln_gamma` sum while both parameters
/// are below `LN_BETA_STIRLING_FROM`, the Stirling route otherwise.
fn ln_beta_fn(alpha: f64, beta: f64) -> f64 {
    let (a, b) = if alpha <= beta { (alpha, beta) } else { (beta, alpha) };
    if b >= LN_BETA_STIRLING_FROM {
        ln_beta_large(a, b)
    } else {
        ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
    }
}

/// Per-event beta function bundle: shapes plus ln B(alpha, beta).
#[derive(Debug, Clone, Copy)]
pub struct BetaFn {
    pub alpha: f64,
    pub beta: f64,
    ln_beta: f64,
}

impl BetaFn {
    pub fn new(alpha: f64, beta: f64) -> Self {
        BetaFn { alpha, beta, ln_beta: ln_beta_fn(alpha, beta) }
    }

    pub fn ln_beta(&self) -> f64 {
        self.ln_beta
    }

    /// Regularized lower incomplete beta I_x(alpha, beta).
    pub fn cdf(&self, x: f64) -> f64 {
        let (a, b) = (self.alpha, self.beta);
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        // ln of x^a (1 - x)^b / B(a, b); symmetric in the reflection below.
        let ln_front = a * x.ln() + b * (-x).ln_1p() - self.ln_beta;
        let front = ln_front.exp();
        if x < (a + 1.0) / (a + b + 2.0) {
            betacf(a, b, x).map_or(f64::NAN, |cf| front * cf / a)
        } else {
            betacf(b, a, 1.0 - x).map_or(f64::NAN, |cf| 1.0 - front * cf / b)
        }
    }

    /// Whether the continued fraction is trusted at these parameters: false
    /// from `min(alpha, beta) = LARGE_PARAMETER_FROM` (2e5), the threshold
    /// from which the quantiles method takes the Cornish-Fisher expansion
    /// instead of inverting the fraction. The survival method keeps such an
    /// event out of the shared curve whatever the node density, so that the
    /// fraction is never asked to work where it fails.
    pub fn is_valid_regime(&self) -> bool {
        self.alpha.min(self.beta) < LARGE_PARAMETER_FROM
    }

    /// Regularized upper incomplete beta, the survival function 1 - I_x,
    /// computed directly in each branch of the continued fraction so that
    /// the upper tail keeps full relative precision (no `1 - cdf`).
    pub fn sf(&self, x: f64) -> f64 {
        let (a, b) = (self.alpha, self.beta);
        if x <= 0.0 {
            return 1.0;
        }
        if x >= 1.0 {
            return 0.0;
        }
        let ln_front = a * x.ln() + b * (-x).ln_1p() - self.ln_beta;
        let front = ln_front.exp();
        if x < (a + 1.0) / (a + b + 2.0) {
            betacf(a, b, x).map_or(f64::NAN, |cf| 1.0 - front * cf / a)
        } else {
            betacf(b, a, 1.0 - x).map_or(f64::NAN, |cf| front * cf / b)
        }
    }

    /// `(sf(y), y f(y))`: the survival function and the density times its
    /// argument, which share the logarithms of `y` and `1 - y`. The second
    /// is `x Lambda'`'s per-event factor and stays finite as y -> 0 for
    /// every alpha > 0.
    pub fn sf_and_density_times_y(&self, y: f64) -> (f64, f64) {
        let (a, b) = (self.alpha, self.beta);
        if y <= 0.0 {
            return (1.0, 0.0);
        }
        if y >= 1.0 {
            return (0.0, 0.0);
        }
        let ln_y = y.ln();
        let ln_1my = (-y).ln_1p();
        let front = (a * ln_y + b * ln_1my - self.ln_beta).exp();
        let sf = if y < (a + 1.0) / (a + b + 2.0) {
            betacf(a, b, y).map_or(f64::NAN, |cf| 1.0 - front * cf / a)
        } else {
            betacf(b, a, 1.0 - y).map_or(f64::NAN, |cf| front * cf / b)
        };
        (sf, (a * ln_y + (b - 1.0) * ln_1my - self.ln_beta).exp())
    }

    /// Density f(x); zero outside the open interval (callers keep brackets
    /// open, so the endpoint singularities for shapes below one never enter).
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return 0.0;
        }
        ((self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (-x).ln_1p() - self.ln_beta).exp()
    }

    /// Logarithmic derivative of the density, f'(x) / f(x).
    #[inline]
    pub fn dlnf(&self, x: f64) -> f64 {
        (self.alpha - 1.0) / x - (self.beta - 1.0) / (1.0 - x)
    }

    /// ln of the lower-tail closed-form quantile estimate
    /// x ~ (u alpha B(alpha, beta))^(1 / alpha), exact as x -> 0.
    pub fn ln_lower_tail_start(&self, u: f64) -> f64 {
        (u.ln() + self.alpha.ln() + self.ln_beta) / self.alpha
    }
}

/// Modified Lentz evaluation of the incomplete-beta continued fraction:
/// `None` when `MAX_CF_ITERS` pass without the step reaching `CF_EPS`.
fn betacf(a: f64, b: f64, x: f64) -> Option<f64> {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_CF_ITERS {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() <= CF_EPS {
            return Some(h);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::distribution::{Beta, ContinuousCDF};

    #[test]
    fn matches_statrs_across_regimes() {
        let shapes = [
            (1.0, 1.0),
            (2.5, 7.5),
            (0.5, 0.5),
            (300.0, 100.0),
            (6e-4, 1.4e-3),
            (1e-3, 1.0),
            (500.0, 500.0),
            (40.0, 2.0),
        ];
        let xs = [1e-12, 1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0 - 1e-9];
        for (a, b) in shapes {
            let ours = BetaFn::new(a, b);
            let oracle = Beta::new(a, b).unwrap();
            for &x in &xs {
                let got = ours.cdf(x);
                let want = oracle.cdf(x);
                // statrs forms ln B from three ln_gamma values, which at
                // (500, 500) costs it 2.6e-13 at x = 0.5 (exact: 0.5); ours
                // is within 4e-14 there (the Stirling route), so the oracle
                // is held to its own precision on that shape.
                let slack = if a + b >= 1000.0 { 5e-13 } else { 1e-13 };
                let tolerance = 1e-15 + slack * want.abs();
                assert!(
                    (got - want).abs() <= tolerance,
                    "cdf mismatch a={a} b={b} x={x}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn sf_matches_statrs_and_keeps_tail_precision() {
        let shapes = [(1.0, 1.0), (2.5, 7.5), (0.5, 0.5), (300.0, 100.0), (6e-4, 1.4e-3), (1e-3, 1.0), (500.0, 500.0), (40.0, 2.0)];
        let xs = [1e-12, 1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0 - 1e-9];
        for (a, b) in shapes {
            let ours = BetaFn::new(a, b);
            let oracle = Beta::new(a, b).unwrap();
            for &x in &xs {
                let got = ours.sf(x);
                // statrs's own sf loses digits as 1 - cdf near 1; hold ours
                // to the better of the two readings.
                let want = if x < 0.5 { 1.0 - oracle.cdf(x) } else { oracle.sf(x) };
                // Large symmetric shapes at the branch switch agree to 1e-12.
                assert!(
                    (got - want).abs() <= 1e-15 + 2e-12 * want.abs(),
                    "sf mismatch a={a} b={b} x={x}: got {got}, want {want}"
                );
                let (sf2, dy) = ours.sf_and_density_times_y(x);
                assert_eq!(sf2, got);
                assert!((dy - x * ours.pdf(x)).abs() <= 1e-13 * dy.abs().max(1e-300));
            }
        }
        // Deep upper tail: beta(3, 300) at y = 0.3 has S ~ 1e-40, which
        // 1 - cdf would return as zero.
        let f = BetaFn::new(3.0, 300.0);
        let s = f.sf(0.3);
        assert!(s > 0.0 && s < 1e-30, "{s}");
        assert!(((1.0 - f.cdf(0.3)) - s).abs() <= 1e-16);
        // Deep into the representable range (about 1e-194 here) the
        // survival stays finite and positive.
        let g = BetaFn::new(2.0, 2000.0);
        let s = g.sf(0.2);
        assert!(s > 0.0 && s.is_finite() && s < 1e-150, "{s}");
    }

    #[test]
    fn ln_beta_matches_mpmath_at_large_parameters() {
        // mpmath at 40 digits, 2026-10-24:
        //   import mpmath as mp; mp.mp.dps = 40
        //   for a, b in pairs: print(mp.nstr(mp.log(mp.beta(a, b)), 20))
        // The first row is the three-term sum (b below the Stirling
        // threshold); the rest take the Stirling route. The naive sum is off
        // by 3.5e-8 at (0.13, 8e7) and 3.8e-8 at (0.5, 1e7).
        let table = [
            (0.13, 50.0, 1.4708432867497706688),
            (0.13, 100.0, 1.380166568528025525),
            (0.13, 8e7, -0.38740733800432604908),
            (0.5, 1e3, -2.8813876965715767707),
            (0.5, 1e7, -7.486682870054459807),
            (1.0, 1e4, -9.2103403719761827361),
            (1.5, 1e4, -13.9363302943495662),
            (2.0, 1e6, -27.631022115928048209),
            (20.0, 1e6, -236.97051697085100008),
        ];
        for (a, b, want) in table {
            let got = BetaFn::new(a, b).ln_beta();
            assert!((got - want).abs() <= 1e-13, "ln B({a}, {b}): got {got}, want {want}");
            // Symmetric by construction: the parameters are ordered first.
            assert_eq!(BetaFn::new(b, a).ln_beta(), got);
        }
    }

    #[test]
    fn uniform_cdf_is_identity() {
        let f = BetaFn::new(1.0, 1.0);
        for &x in &[0.0, 0.125, 0.5, 0.875, 1.0] {
            assert!((f.cdf(x) - x).abs() < 1e-15);
        }
    }

    #[test]
    fn pdf_matches_closed_form() {
        // alpha = 2, beta = 1: f(x) = 2x.
        let f = BetaFn::new(2.0, 1.0);
        for &x in &[0.1, 0.5, 0.9] {
            assert!((f.pdf(x) - 2.0 * x).abs() < 1e-14);
        }
        assert_eq!(f.pdf(0.0), 0.0);
        assert_eq!(f.pdf(1.0), 0.0);
    }

    #[test]
    fn lower_tail_start_tracks_true_quantile() {
        // For small alpha the closed form is essentially exact.
        let f = BetaFn::new(0.05, 2.0);
        let u = 0.3;
        let x = f.ln_lower_tail_start(u).exp();
        assert!(x > 0.0 && x < 1e-6);
        assert!((f.cdf(x) - u).abs() < 1e-3);
        // Shapes so extreme the true quantile underflows: the estimate
        // underflows with it rather than misbehaving.
        let g = BetaFn::new(1e-3, 2.0);
        assert_eq!(g.ln_lower_tail_start(u).exp(), 0.0);
    }

    // [kernel] The fraction reports failure where both parameters are
    // large instead of returning an unconverged value (the review's I2:
    // 1.45 at a = b = 1e9), and the validity flag routes around it.
    #[test]
    fn unconverged_fraction_is_reported_not_returned() {
        assert!(betacf(1e9, 1e9, 0.5).is_none());
        let bf = BetaFn::new(1e9, 1e9);
        assert!(bf.sf(0.5).is_nan());
        assert!(bf.cdf(0.5).is_nan());
        assert!(bf.sf_and_density_times_y(0.5).0.is_nan());
        assert!(!bf.is_valid_regime());
        // One small parameter converges in tens of iterations at any beta.
        let small = BetaFn::new(0.5, 1e9);
        assert!(small.is_valid_regime());
        assert!(small.sf(1e-9).is_finite());
        assert!(BetaFn::new(1.9e5, 1e7).is_valid_regime());
        assert!(!BetaFn::new(2e5, 2e5).is_valid_regime());
    }
}
