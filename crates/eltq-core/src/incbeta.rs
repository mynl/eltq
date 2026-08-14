//! Own regularized incomplete-beta CDF and PDF with per-event constants.
//!
//! Cephes/Numerical-Recipes-style modified Lentz continued fraction with the
//! symmetry switch at x = (alpha + 1) / (alpha + beta + 2). The per-event
//! constant ln B(alpha, beta) is computed once per event, not per
//! probability — the key setup reuse the scalar statrs path cannot offer.

use statrs::function::gamma::ln_gamma;

const MAX_CF_ITERS: usize = 400;
const CF_EPS: f64 = 1e-16;
const FPMIN: f64 = 1e-300;

/// Per-event beta function bundle: shapes plus ln B(alpha, beta).
#[derive(Debug, Clone, Copy)]
pub struct BetaFn {
    pub alpha: f64,
    pub beta: f64,
    ln_beta: f64,
}

impl BetaFn {
    pub fn new(alpha: f64, beta: f64) -> Self {
        BetaFn {
            alpha,
            beta,
            ln_beta: ln_gamma(alpha) + ln_gamma(beta) - ln_gamma(alpha + beta),
        }
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
            front * betacf(a, b, x) / a
        } else {
            1.0 - front * betacf(b, a, 1.0 - x) / b
        }
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

/// Modified Lentz evaluation of the incomplete-beta continued fraction.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
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
            break;
        }
    }
    h
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
                let tolerance = 1e-15 + 1e-13 * want.abs();
                assert!(
                    (got - want).abs() <= tolerance,
                    "cdf mismatch a={a} b={b} x={x}: got {got}, want {want}"
                );
            }
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
}
