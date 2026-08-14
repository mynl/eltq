//! Inverse-beta midpoint-quantile kernel: reference implementation.
//!
//! Computes normalized beta quantiles Q(alpha, beta; u_k) at the midpoint
//! probabilities u_k = (k + 1/2) / K. This 1.1.0 kernel builds on statrs's
//! regularized incomplete-beta CDF/PDF and drives them with a safeguarded
//! Newton solve (log-space fallback plus bisection inside a maintained
//! bracket), giving full-precision quantiles across the entire admissible
//! shape domain — including the extreme U-shaped betas produced by the clamp
//! policy. It stays available as the test oracle for the batched 1.4.0
//! kernel.

use statrs::distribution::{Beta, Continuous, ContinuousCDF};

/// Nonfinite CDF or invalid shape parameters; carries no detail because the
/// caller attaches row context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelError;

const MAX_ITERS: usize = 500;

/// Fill `out[k] = Q(alpha, beta; (k + 1/2) / K)` with `K = out.len()`.
///
/// Quantiles in the upper half are solved on the reflected distribution
/// (`Q(a, b; u) = 1 - Q(b, a; 1 - u)`), with the complement probability
/// `1 - u_k = (K - k - 1/2) / K` formed exactly, so both tails enjoy full
/// relative precision.
pub fn midpoint_quantiles_reference(
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) -> Result<(), KernelError> {
    let k_total = out.len() as f64;
    let dist = Beta::new(alpha, beta).map_err(|_| KernelError)?;
    let reflected = Beta::new(beta, alpha).map_err(|_| KernelError)?;
    let cdf_half = dist.cdf(0.5);
    if !cdf_half.is_finite() {
        return Err(KernelError);
    }
    for (k, slot) in out.iter_mut().enumerate() {
        let u = (k as f64 + 0.5) / k_total;
        *slot = if u <= cdf_half {
            solve_lower(&dist, u)?
        } else {
            let u_complement = (k_total - k as f64 - 0.5) / k_total;
            1.0 - solve_lower(&reflected, u_complement)?
        };
    }
    Ok(())
}

/// Solve `cdf(x) = p` for the lower-half root (`p <= cdf(1/2)`, so the
/// answer lies in `[0, 1/2]`).
///
/// Newton steps are safeguarded by a maintained bracket. When a plain step
/// leaves the bracket — typical in power-law regions where the CDF behaves
/// like x^alpha and is linear in ln x, not x — a log-space Newton step is
/// tried before falling back to bisection. Terminates on bracket collapse to
/// machine precision; hard shapes cannot diverge.
fn solve_lower(dist: &Beta, p: f64) -> Result<f64, KernelError> {
    let mut lo = 0.0_f64;
    let mut hi = 0.5_f64;
    let mut x = 0.25_f64;
    let mut best = x;
    let mut best_err = f64::INFINITY;
    for _ in 0..MAX_ITERS {
        let cdf = dist.cdf(x);
        if !cdf.is_finite() {
            return Err(KernelError);
        }
        let e = cdf - p;
        if e.abs() < best_err {
            best_err = e.abs();
            best = x;
        }
        if e == 0.0 {
            return Ok(x);
        }
        if e > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        if hi - lo <= 2.0 * f64::EPSILON * hi {
            return Ok(best);
        }
        if hi <= f64::MIN_POSITIVE {
            // The true quantile underflows f64; the atom at zero is exact.
            return Ok(0.0);
        }
        let f = dist.pdf(x);
        let mut next = f64::NAN;
        if f.is_finite() && f > 0.0 {
            let cand = x - e / f;
            if cand > lo && cand < hi {
                next = cand;
            } else {
                let dt = (e / (f * x)).clamp(-700.0, 700.0);
                let cand = x * (-dt).exp();
                if cand > lo && cand < hi {
                    next = cand;
                }
            }
        }
        if !next.is_finite() || next <= lo || next >= hi {
            next = 0.5 * (lo + hi);
        }
        x = next;
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantiles(alpha: f64, beta: f64, k: usize) -> Vec<f64> {
        let mut out = vec![0.0; k];
        midpoint_quantiles_reference(alpha, beta, &mut out).unwrap();
        out
    }

    fn assert_monotone(xs: &[f64]) {
        for w in xs.windows(2) {
            assert!(w[1] >= w[0] - 1e-14, "non-monotone: {} then {}", w[0], w[1]);
        }
    }

    #[test]
    fn uniform_is_exact() {
        let xs = quantiles(1.0, 1.0, 8);
        for (k, &x) in xs.iter().enumerate() {
            assert!((x - (k as f64 + 0.5) / 8.0).abs() < 1e-14);
        }
    }

    #[test]
    fn closed_form_power_laws() {
        // alpha = 2, beta = 1: cdf x^2, Q(u) = sqrt(u).
        for (k, &x) in quantiles(2.0, 1.0, 16).iter().enumerate() {
            let u = (k as f64 + 0.5) / 16.0;
            assert!((x - u.sqrt()).abs() < 1e-13);
        }
        // alpha = 1, beta = 2: Q(u) = 1 - sqrt(1 - u); exercises reflection.
        for (k, &x) in quantiles(1.0, 2.0, 16).iter().enumerate() {
            let u_complement = (16.0 - k as f64 - 0.5) / 16.0;
            assert!((x - (1.0 - u_complement.sqrt())).abs() < 1e-13);
        }
    }

    #[test]
    fn symmetric_median() {
        let xs = quantiles(5.0, 5.0, 1);
        assert!((xs[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn round_trip_well_conditioned() {
        for (a, b) in [(2.5, 7.5), (300.0, 100.0), (0.5, 0.5), (40.0, 40.0)] {
            let dist = Beta::new(a, b).unwrap();
            let xs = quantiles(a, b, 32);
            assert_monotone(&xs);
            for (k, &x) in xs.iter().enumerate() {
                let u = (k as f64 + 0.5) / 32.0;
                assert!(
                    (dist.cdf(x) - u).abs() < 1e-12,
                    "round trip a={a} b={b} u={u}: x={x} cdf={}",
                    dist.cdf(x)
                );
            }
        }
    }

    #[test]
    fn extreme_clamped_shapes_stay_bracketed() {
        // Shapes like those from the clamp policy (kappa ~ 1/c^2 - 1).
        for (a, b) in [(6e-4, 1.4e-3), (1e-3, 1.0), (1.0, 1e-3), (5e-4, 5e-4)] {
            let xs = quantiles(a, b, 64);
            assert_monotone(&xs);
            for &x in &xs {
                assert!(x.is_finite() && (0.0..=1.0).contains(&x));
            }
        }
    }

    #[test]
    fn concentrated_large_shapes() {
        let xs = quantiles(500.0, 500.0, 8);
        assert_monotone(&xs);
        for &x in &xs {
            assert!((0.4..0.6).contains(&x));
        }
    }
}
