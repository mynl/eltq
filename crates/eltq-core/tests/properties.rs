//! Property-based tests: mass conservation, deposit mean preservation,
//! dyadic scale invariance, and beta-parameterization round trips.

use eltq_core::{Config, EltInput, deposit::deposit, quantize};
use proptest::prelude::*;

/// One synthetic row: rate, max loss, mean position p in [0, 1], and
/// variance fraction f of the admissible maximum (f >= 1 is inadmissible and
/// exercises the Bernoulli policy).
fn row_strategy() -> impl Strategy<Value = (f64, f64, f64, f64)> {
    (
        1e-6_f64..1e3,
        prop_oneof![Just(0.0_f64), 1e-3_f64..1e9],
        0.0_f64..=1.0,
        prop_oneof![Just(0.0_f64), 1e-6_f64..1.5],
    )
}

fn columns(rows: &[(f64, f64, f64, f64)]) -> [Vec<f64>; 5] {
    let mut cols: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::new());
    for &(rate, m, p, f) in rows {
        let mu = p * m;
        let sigma = (f * mu * (m - mu)).max(0.0).sqrt();
        cols[0].push(rate);
        cols[1].push(m);
        cols[2].push(mu);
        cols[3].push(sigma);
        cols[4].push(0.0);
    }
    cols
}

fn input<'a>(cols: &'a [Vec<f64>; 5]) -> EltInput<'a> {
    EltInput {
        rate: &cols[0],
        max_loss: &cols[1],
        mean_loss: &cols[2],
        sd_indep: &cols[3],
        sd_corr: &cols[4],
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn deposit_preserves_mass_and_moment(
        frac in 0.0_f64..=1.0,
        w in 1e-9_f64..1e6,
        h in prop_oneof![Just(1.0_f64), 1e-6_f64..1e6],
        n in 2_usize..500,
    ) {
        let x = frac * (n - 1) as f64 * h;
        let mut hist = vec![0.0_f64; n];
        deposit(&mut hist, x, w, 1.0 / h);
        let mass: f64 = hist.iter().sum();
        let moment: f64 = hist.iter().enumerate()
            .map(|(i, &v)| v * i as f64 * h)
            .sum();
        prop_assert!((mass - w).abs() <= 1e-15 * w);
        prop_assert!((moment - w * x).abs() <= 1e-12 * (w * x).abs().max(w * h));
    }

    #[test]
    fn quantize_conserves_mass_and_mean(
        rows in prop::collection::vec(row_strategy(), 1..30),
    ) {
        let cols = columns(&rows);
        let mut config = Config::new(hint_bucket(&cols[1]));
        config.method = eltq_core::Method::Quantiles;
        config.quadrature_points = 4;
        let result = quantize(&input(&cols), &config).unwrap();
        let meta = &result.meta;
        let lambda: f64 = meta.total_rate;
        prop_assert!((meta.raw_mass - lambda).abs() <= 1e-12 * lambda);
        prop_assert!((meta.secondary_mass - lambda).abs() <= 1e-12 * lambda);
        let aal_scale = meta.analytic_aal.abs().max(lambda * 1e-9);
        prop_assert!((meta.raw_mean - meta.analytic_aal).abs() <= 1e-10 * aal_scale);
        prop_assert!(
            (meta.secondary_mean - meta.quadrature_secondary_mean.unwrap()).abs()
                <= 1e-10 * aal_scale
        );
        prop_assert!(result.raw.iter().all(|&v| v.is_finite() && v >= 0.0));
        prop_assert!(result.secondary.iter().all(|&v| v.is_finite() && v >= 0.0));
    }

    #[test]
    fn dyadic_scaling_is_bitwise_invariant(
        rows in prop::collection::vec(row_strategy(), 1..12),
        exponent in -8_i32..=8,
    ) {
        let cols = columns(&rows);
        let mut config = Config::new(hint_bucket(&cols[1]));
        config.method = eltq_core::Method::Quantiles;
        config.quadrature_points = 4;
        let base = quantize(&input(&cols), &config).unwrap();

        let scale = (2.0_f64).powi(exponent);
        let mut scaled_cols = cols.clone();
        for c in [1, 2, 3, 4] {
            for v in scaled_cols[c].iter_mut() {
                *v *= scale;
            }
        }
        config.bucket_size *= scale;
        let scaled = quantize(&input(&scaled_cols), &config).unwrap();
        prop_assert_eq!(&base.raw, &scaled.raw);
        prop_assert_eq!(&base.secondary, &scaled.secondary);
    }

    #[test]
    fn batched_kernel_agrees_with_reference(
        log_alpha in -3.0_f64..3.0,
        log_beta in -3.0_f64..3.0,
        k in prop_oneof![Just(1_usize), Just(7), Just(32)],
    ) {
        let alpha = 10.0_f64.powf(log_alpha);
        let beta = 10.0_f64.powf(log_beta);
        let mut got = vec![0.0; k];
        let mut want = vec![0.0; k];
        eltq_core::batched::beta_midpoint_quantiles(alpha, beta, &mut got).unwrap();
        eltq_core::kernel::midpoint_quantiles_reference(alpha, beta, &mut want).unwrap();
        let moderate = (0.1..=100.0).contains(&alpha) && (0.1..=100.0).contains(&beta);
        let tolerance = if moderate { 1e-13 } else { 1e-10 };
        for (g, w) in got.iter().zip(&want) {
            prop_assert!(
                (g - w).abs() <= tolerance,
                "alpha={alpha} beta={beta} k={k}: {g} vs {w}"
            );
        }
    }

    #[test]
    fn beta_params_round_trip(
        m in 1e-9_f64..1e12,
        p in 1e-6_f64..=0.999999,
        f in 1e-6_f64..=0.999999,
    ) {
        let mu = p * m;
        prop_assume!(mu > 0.0 && mu < m);
        let var = f * mu * (m - mu);
        prop_assume!(var > 0.0);
        let (a, b) = eltq_core::beta::beta_params(m, mu, var.sqrt());
        prop_assert!(a > 0.0 && b > 0.0);
        let mean_back = m * a / (a + b);
        let var_back = m * m * a * b / ((a + b) * (a + b) * (a + b + 1.0));
        prop_assert!((mean_back - mu).abs() <= 1e-9 * mu);
        prop_assert!((var_back - var).abs() <= 1e-7 * var);
    }
}

/// Pick a bucket size that keeps auto grids to a testable length.
fn hint_bucket(max_losses: &[f64]) -> f64 {
    let max = max_losses.iter().cloned().fold(0.0_f64, f64::max);
    if max > 0.0 { max / 256.0 } else { 1.0 }
}
