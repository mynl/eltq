//! The law of a beta loss share at every parameter size (spec section 5A):
//! the regularized incomplete beta while its continued fraction is trusted,
//! `min(alpha, beta) < LARGE_PARAMETER_FROM`; from there the narrow kernel's
//! quadrature of the density on the bulk, which is itself the standardized
//! normal law in closed form from `NARROW_NORMAL_FROM`.
//!
//! The one-row survival route has carried such rows on its narrow route
//! since 2.3.3; this module gives the same evaluation to every other caller
//! (the multi-unit route's beta units from 2.10.1, `dev/done/plan-model-
//! moments-and-windowed-views.md`, [Eltq-Multiunit-Narrow-Units]), so that
//! none of them meets the fraction's NaN beyond its regime.

use crate::batched::LARGE_PARAMETER_FROM;
use crate::curve::{NarrowKernel, TAIL_RATE_FLOOR, narrow_bulk};
use crate::incbeta::BetaFn;

/// A beta loss share's distribution function, survival function and
/// limited mean, through the incomplete beta or the narrow kernel.
#[derive(Debug, Clone, Copy)]
pub enum BetaLaw {
    /// `min(alpha, beta)` below `LARGE_PARAMETER_FROM`: the continued
    /// fraction, exactly as `BetaFn` evaluates it.
    Regular(BetaFn),
    /// From `LARGE_PARAMETER_FROM`: the narrow kernel on its bulk.
    Narrow(NarrowLaw),
}

/// The narrow kernel normalized on the bulk `(lower, stop)` of the share:
/// the density relative to its value at the mean, integrated by 8-point
/// Gauss and Legendre on half standard deviations (or in closed form under
/// the normal law), outside which the law has nothing below `lower` and
/// everything below `stop`, each within `TAIL_RATE_FLOOR` of the unit mass.
#[derive(Debug, Clone, Copy)]
pub struct NarrowLaw {
    kernel: NarrowKernel,
    lower: f64,
    stop: f64,
    norm: f64,
}

impl BetaLaw {
    /// The law of `Beta(alpha, beta)`; `None` when the narrow bulk cannot be
    /// found within `NARROW_STEPS` standard deviations or its mass is not
    /// positive and finite, which does not happen for parameters this large
    /// (their coefficient of variation is at most 2.2e-3) but is reported
    /// rather than assumed.
    pub fn new(alpha: f64, beta: f64) -> Option<BetaLaw> {
        if alpha.min(beta) < LARGE_PARAMETER_FROM {
            return Some(BetaLaw::Regular(BetaFn::new(alpha, beta)));
        }
        let kernel = NarrowKernel::new(alpha, beta);
        let (lower, stop) = narrow_bulk(&kernel, 1.0, kernel.mean(), kernel.sd(), 1.0, TAIL_RATE_FLOOR)?;
        let norm = kernel.integral(lower, stop);
        (norm > 0.0 && norm.is_finite()).then_some(BetaLaw::Narrow(NarrowLaw { kernel, lower, stop, norm }))
    }

    /// Whether the law is the narrow kernel's.
    pub fn is_narrow(&self) -> bool {
        matches!(self, BetaLaw::Narrow(_))
    }

    /// `P(Y <= y)`.
    pub fn cdf(&self, y: f64) -> f64 {
        match self {
            BetaLaw::Regular(bf) => bf.cdf(y),
            BetaLaw::Narrow(law) => law.cdf(y),
        }
    }

    /// `P(Y > y)`, with full relative precision in the upper tail.
    pub fn sf(&self, y: f64) -> f64 {
        match self {
            BetaLaw::Regular(bf) => bf.sf(y),
            BetaLaw::Narrow(law) => law.sf(y),
        }
    }
}

impl NarrowLaw {
    /// `P(Y <= y)`: the kernel's integral from the nearer end of the bulk,
    /// so that both tails keep their relative precision.
    pub fn cdf(&self, y: f64) -> f64 {
        if y <= self.lower {
            0.0
        } else if y >= self.stop {
            1.0
        } else if y <= self.kernel.mean() {
            (self.kernel.integral(self.lower, y) / self.norm).min(1.0)
        } else {
            (1.0 - self.kernel.integral(y, self.stop) / self.norm).max(0.0)
        }
    }

    /// `P(Y > y)`, the mirror of `cdf`.
    pub fn sf(&self, y: f64) -> f64 {
        if y <= self.lower {
            1.0
        } else if y >= self.stop {
            0.0
        } else if y <= self.kernel.mean() {
            (1.0 - self.kernel.integral(self.lower, y) / self.norm).max(0.0)
        } else {
            (self.kernel.integral(y, self.stop) / self.norm).min(1.0)
        }
    }

    /// `P(min(Y, y))`, the limited mean in the share: the first moment below
    /// `y` by the kernel plus `y` times the survival there.
    pub fn limited_mean(&self, y: f64) -> f64 {
        if y <= self.lower {
            return y.max(0.0);
        }
        let (_, moment) = self.kernel.integrals(self.lower, y.min(self.stop));
        moment / self.norm + y * self.sf(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta::beta_params;

    // Below the threshold the law is the continued fraction bit for bit; at
    // and above it the narrow kernel agrees with the fraction where the
    // fraction still converges (min(alpha, beta) of 2e5 to 1e6), and at 1e9
    // and 1e24 it is a distribution symmetric about the mean.
    #[test]
    fn law_is_the_fraction_below_and_the_kernel_above() {
        let (alpha, beta) = (3.0, 7.5);
        let law = BetaLaw::new(alpha, beta).unwrap();
        let bf = BetaFn::new(alpha, beta);
        assert!(!law.is_narrow());
        for y in [0.01, 0.2, 0.5, 0.9] {
            assert_eq!(law.cdf(y).to_bits(), bf.cdf(y).to_bits());
        }
        for (alpha, beta) in [(2.5e5, 7.5e5), (1e6, 1e6)] {
            let law = BetaLaw::new(alpha, beta).unwrap();
            let bf = BetaFn::new(alpha, beta);
            assert!(law.is_narrow());
            let kernel = NarrowKernel::new(alpha, beta);
            for k in -6..=6 {
                let y = kernel.mean() + k as f64 * kernel.sd();
                let (got, want) = (law.cdf(y), bf.cdf(y));
                assert!((got - want).abs() < 1e-9, "alpha={alpha} k={k}: {got} vs {want}");
                assert!((law.cdf(y) + law.sf(y) - 1.0).abs() < 1e-15);
            }
        }
        for m in [1.25e9, 1e24] {
            let law = BetaLaw::new(m, m).unwrap();
            assert!((law.cdf(0.5) - 0.5).abs() < 1e-12);
            let BetaLaw::Narrow(narrow) = law else { unreachable!() };
            assert!((narrow.limited_mean(1.0) - 0.5).abs() < 1e-12);
            assert_eq!(law.cdf(0.0), 0.0);
            assert_eq!(law.cdf(1.0), 1.0);
        }
        // The unit of the 2026-10-26 probe: mu = 1/2, sigma = 1e-5 on [0, 1].
        let (alpha, beta) = beta_params(1.0, 0.5, 1e-5);
        let law = BetaLaw::new(alpha, beta).unwrap();
        assert!((law.cdf(0.5) - 0.5).abs() < 1e-12);
        assert!(law.cdf(0.3) == 0.0 && law.cdf(0.7) == 1.0);
    }

    // The limited mean against its definition by the survival function,
    // `P(min(Y, y)) = int_0^y S(u) du`, on a fine midpoint sum over the bulk.
    #[test]
    fn narrow_limited_mean_integrates_the_survival() {
        let law = match BetaLaw::new(3e5, 9e5).unwrap() {
            BetaLaw::Narrow(law) => law,
            BetaLaw::Regular(_) => unreachable!(),
        };
        let (lo, mean, sd) = (law.lower, law.kernel.mean(), law.kernel.sd());
        for k in [-3.0, 0.0, 2.5] {
            let y = mean + k * sd;
            let steps = 20_000;
            let width = (y - lo) / steps as f64;
            let integral: f64 = lo + (0..steps).map(|i| law.sf(lo + (i as f64 + 0.5) * width) * width).sum::<f64>();
            assert!((law.limited_mean(y) - integral).abs() < 1e-11, "k={k}");
        }
    }
}
