//! Criterion benchmarks for the numerical kernels.
//!
//! Run with `cargo bench -p eltq-core`. Correctness gates performance: no
//! benchmark is meaningful from a build failing the acceptance tests.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use eltq_core::batched::beta_midpoint_quantiles;
use eltq_core::beta::beta_params;
use eltq_core::deposit::deposit;
use eltq_core::incbeta::BetaFn;
use eltq_core::kernel::midpoint_quantiles_reference;

/// (label, alpha, beta) per parameter regime of specification section 10.
const REGIMES: [(&str, f64, f64); 5] = [
    ("well_conditioned", 2.5, 7.5),
    ("skewed_small_alpha", 0.5, 9.5),
    ("clamped_ushape", 6e-4, 1.4e-3),
    ("concentrated_large", 500.0, 500.0),
    ("asymmetric_large", 300.0, 100.0),
];

fn bench_cdf(c: &mut Criterion) {
    let mut group = c.benchmark_group("incbeta_cdf");
    for (label, a, b) in REGIMES {
        let bf = BetaFn::new(a, b);
        group.bench_function(label, |bencher| {
            bencher.iter(|| {
                let mut acc = 0.0;
                for i in 1..64 {
                    acc += bf.cdf(black_box(i as f64 / 64.0));
                }
                acc
            })
        });
    }
    group.finish();
}

fn bench_quantile_kernels(c: &mut Criterion) {
    let mut out = vec![0.0_f64; 32];
    let mut group = c.benchmark_group("midpoint_quantiles_k32");
    for (label, a, b) in REGIMES {
        group.bench_function(format!("batched/{label}"), |bencher| {
            bencher.iter(|| beta_midpoint_quantiles(black_box(a), black_box(b), &mut out))
        });
        group.bench_function(format!("reference/{label}"), |bencher| {
            bencher.iter(|| midpoint_quantiles_reference(black_box(a), black_box(b), &mut out))
        });
    }
    group.finish();
}

fn bench_deposit(c: &mut Criterion) {
    let mut hist = vec![0.0_f64; 100_000];
    let xs: Vec<f64> = (0..1024).map(|i| (i as f64 * 97.03) % 99_000.0).collect();
    c.bench_function("deposit_1024", |bencher| {
        bencher.iter(|| {
            for &x in &xs {
                deposit(&mut hist, black_box(x), 1e-3, 1.0);
            }
        })
    });
}

fn bench_beta_params(c: &mut Criterion) {
    c.bench_function("beta_params_1024", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0;
            for i in 1..1024 {
                let mu = i as f64 * 0.09;
                let (a, b) = beta_params(black_box(100.0), black_box(mu), black_box((4.0 + mu).sqrt()));
                acc += a + b;
            }
            acc
        })
    });
}

criterion_group!(
    benches,
    bench_cdf,
    bench_quantile_kernels,
    bench_deposit,
    bench_beta_params
);
criterion_main!(benches);
