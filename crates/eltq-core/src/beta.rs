//! Beta parameterization from (M, mu, sigma^2).

/// Return `(alpha, beta)` for admissible `(M, mu, sigma^2)`.
///
/// Uses `kappa = mu (M - mu) / sigma^2 - 1`, which avoids forming `M^2` and
/// so is stable across the full monetary scale range. The caller must ensure
/// `0 < mu < M` and `0 < sigma^2 < mu (M - mu)`.
#[inline]
pub fn beta_params(max_loss: f64, mean_loss: f64, variance: f64) -> (f64, f64) {
    let p = mean_loss / max_loss;
    let kappa = mean_loss * (max_loss - mean_loss) / variance - 1.0;
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
            let (a, b) = beta_params(m, mu, var);
            assert!(a > 0.0 && b > 0.0, "shapes must be positive");
            let mean_back = m * a / (a + b);
            let var_back = m * m * a * b / ((a + b) * (a + b) * (a + b + 1.0));
            assert!(relative_error(mean_back, mu) < 1e-12);
            assert!(relative_error(var_back, var) < 1e-11);
        }
    }
}
