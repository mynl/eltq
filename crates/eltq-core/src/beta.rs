//! Beta parameterization from (M, mu, sigma).

/// Return `(alpha, beta)` for admissible `(M, mu, sigma)`.
///
/// `kappa = mu (M - mu) / sigma^2 - 1`, formed in ratios as
/// `((M - mu) / mu) / (sigma / mu)^2 - 1` so that no product of two monetary
/// amounts appears: `sigma^2` alone overflows from a monetary scale of about
/// 1e154 and underflows below about 1e-154 while every field stays finite
/// (peer review, finding I3). The parameters are then the same function of
/// the row at every scale the double range admits, to rounding. The caller
/// must ensure `0 < mu < M` and `0 < sigma^2 < mu (M - mu)`.
#[inline]
pub fn beta_params(max_loss: f64, mean_loss: f64, sd: f64) -> (f64, f64) {
    let p = mean_loss / max_loss;
    let cv = sd / mean_loss;
    let kappa = ((max_loss - mean_loss) / mean_loss) / (cv * cv) - 1.0;
    (p * kappa, (1.0 - p) * kappa)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relative_error(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
    }

    // Twin of Python acceptance test 6: reconstructed parameters reproduce
    // the stated mean and variance.
    #[test]
    fn moments_round_trip() {
        let cases = [
            (100.0, 40.0, 0.3),
            (100.0, 2.0, 0.7),
            (100.0, 99.0, 0.5),
            (1e6, 3e5, 0.05),
            (0.001, 0.0004, 0.9),
            (50.0, 25.0, 0.999),
        ];
        for (m, mu, frac) in cases {
            let var = frac * mu * (m - mu);
            let (a, b) = beta_params(m, mu, var.sqrt());
            assert!(a > 0.0 && b > 0.0, "shapes must be positive");
            let mean_back = m * a / (a + b);
            let var_back = m * m * a * b / ((a + b) * (a + b) * (a + b + 1.0));
            assert!(relative_error(mean_back, mu) < 1e-12);
            assert!(relative_error(var_back, var) < 1e-11);
        }
    }

    // [ratios] The parameters are the same function of the row at a monetary
    // scale of 1e160, where sigma^2 and mu (M - mu) overflow, and at 1e-200,
    // where they underflow (peer review, finding I3).
    #[test]
    fn parameters_are_scale_invariant_across_the_double_range() {
        let (a0, b0) = beta_params(1.0, 0.4, 0.1);
        for scale in [1e160, 1e-200, 64.0, 3.0] {
            let (a, b) = beta_params(scale, 0.4 * scale, 0.1 * scale);
            assert!(relative_error(a, a0) < 1e-14 && relative_error(b, b0) < 1e-14, "{scale}: {a} {b}");
        }
    }
}
