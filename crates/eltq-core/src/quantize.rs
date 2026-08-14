//! Streaming quantization driver: parallel fixed chunks with a
//! bitwise-deterministic in-order fold.
//!
//! Rows are processed in fixed chunks of `config.chunk_rows`. Each chunk
//! produces private raw and secondary partial histograms plus compensated
//! partial totals (no locks, no atomics in the hot loop); partials are folded
//! sequentially in chunk-index order, so the result is bitwise identical for
//! any thread count and any work-stealing schedule. Chunks run in bounded
//! waves (2x the worker count), keeping peak memory at
//! `waves * 2 * grid_length * 8` bytes regardless of input size.

use rayon::prelude::*;

use crate::batched::beta_midpoint_quantiles;
use crate::beta::beta_params;
use crate::config::Config;
use crate::deposit::deposit;
use crate::diagnostics::{Diagnostics, Meta, QuantizeResult};
use crate::error::Error;
use crate::sum::{Neumaier, grid_moment, sum};
use crate::validate::{RowKind, classify_row};

/// Structure-of-arrays ELT input. Event ids stay outside the computational
/// path; diagnostics report 0-based row indices which callers translate.
#[derive(Debug, Clone, Copy)]
pub struct EltInput<'a> {
    pub rate: &'a [f64],
    pub max_loss: &'a [f64],
    pub mean_loss: &'a [f64],
    pub sd_indep: &'a [f64],
    pub sd_corr: &'a [f64],
}

impl EltInput<'_> {
    fn validated_len(&self) -> Result<usize, Error> {
        let expected = self.rate.len();
        for (name, len) in [
            ("max_loss", self.max_loss.len()),
            ("mean_loss", self.mean_loss.len()),
            ("sd_indep", self.sd_indep.len()),
            ("sd_corr", self.sd_corr.len()),
        ] {
            if len != expected {
                return Err(Error::ColumnLengthMismatch {
                    name,
                    got: len,
                    expected,
                });
            }
        }
        Ok(expected)
    }
}

/// Private per-chunk partial result.
struct ChunkPartial {
    raw: Vec<f64>,
    secondary: Vec<f64>,
    rate: f64,
    aal: f64,
    quadrature: f64,
    valid_rows: usize,
}

/// Quantize an ELT onto the even loss grid; both histograms in one pass.
pub fn quantize(input: &EltInput<'_>, config: &Config) -> Result<QuantizeResult, Error> {
    config.validate()?;
    let n_rows = input.validated_len()?;
    let h = config.bucket_size;
    let k = config.quadrature_points;

    // Pass 1: classify rows. Dropped rows must touch neither output nor the
    // grid-coverage requirement, so classification precedes everything.
    let mut diagnostics = Diagnostics::default();
    let mut kinds = vec![RowKind::Drop; n_rows];
    let mut mu_eff = vec![0.0_f64; n_rows];
    let mut var_eff = vec![0.0_f64; n_rows];
    let mut max_contrib_loss = 0.0_f64;
    for row in 0..n_rows {
        let plan = classify_row(
            row,
            input.rate[row],
            input.max_loss[row],
            input.mean_loss[row],
            input.sd_indep[row],
            input.sd_corr[row],
            config,
            &mut diagnostics,
        )?;
        kinds[row] = plan.kind;
        mu_eff[row] = plan.mu;
        var_eff[row] = plan.var;
        if plan.kind != RowKind::Drop {
            max_contrib_loss = max_contrib_loss.max(input.max_loss[row]);
        }
    }

    // Grid setup with upfront coverage check: (n - 1) h >= max contributing M.
    let auto_length = if max_contrib_loss > 0.0 {
        (max_contrib_loss / h).ceil() as usize + 1
    } else {
        1
    };
    let n = match config.grid_length {
        None => auto_length,
        Some(n) => {
            let top = (n - 1) as f64 * h;
            if max_contrib_loss > top {
                let example_rows = (0..n_rows)
                    .filter(|&row| kinds[row] != RowKind::Drop && input.max_loss[row] > top)
                    .take(crate::diagnostics::MAX_EXAMPLES)
                    .collect();
                return Err(Error::GridOverflow {
                    grid_length: n,
                    bucket_size: h,
                    top,
                    max_loss: max_contrib_loss,
                    required: auto_length,
                    example_rows,
                });
            }
            n
        }
    };

    // Pass 2: chunked deposits. Each chunk owns private histograms and a
    // reused K-quantile buffer; no N x K array ever exists.
    let inv_h = 1.0 / h;
    let process_chunk = |chunk_index: usize| -> Result<ChunkPartial, Error> {
        let start = chunk_index * config.chunk_rows;
        let end = (start + config.chunk_rows).min(n_rows);
        let mut partial = ChunkPartial {
            raw: vec![0.0; n],
            secondary: vec![0.0; n],
            rate: 0.0,
            aal: 0.0,
            quadrature: 0.0,
            valid_rows: 0,
        };
        let mut quantiles = vec![0.0_f64; k];
        let mut rate_acc = Neumaier::default();
        let mut aal_acc = Neumaier::default();
        let mut quad_acc = Neumaier::default();
        for row in start..end {
            if kinds[row] == RowKind::Drop {
                continue;
            }
            partial.valid_rows += 1;
            let lam = input.rate[row];
            let m = input.max_loss[row];
            let mu = mu_eff[row];
            rate_acc.add(lam);
            aal_acc.add(lam * mu);
            deposit(&mut partial.raw, mu, lam, inv_h);
            if kinds[row] == RowKind::Atom {
                deposit(&mut partial.secondary, mu, lam, inv_h);
                quad_acc.add(lam * mu);
            } else {
                let (alpha, beta) = beta_params(m, mu, var_eff[row]);
                beta_midpoint_quantiles(alpha, beta, &mut quantiles)
                    .map_err(|_| Error::KernelFailure { row, alpha, beta })?;
                let w = lam / k as f64;
                let mut row_sum = Neumaier::default();
                for &q in &quantiles {
                    let x = m * q;
                    deposit(&mut partial.secondary, x, w, inv_h);
                    row_sum.add(x);
                }
                quad_acc.add(w * row_sum.value());
            }
        }
        partial.rate = rate_acc.value();
        partial.aal = aal_acc.value();
        partial.quadrature = quad_acc.value();
        Ok(partial)
    };

    let n_chunks = n_rows.div_ceil(config.chunk_rows).max(1);
    let mut raw = vec![0.0_f64; n];
    let mut secondary = vec![0.0_f64; n];
    let mut total_rate = Neumaier::default();
    let mut analytic_aal = Neumaier::default();
    let mut quadrature_mean = Neumaier::default();
    let mut valid_rows = 0_usize;
    let mut fold = |partials: Vec<Result<ChunkPartial, Error>>| -> Result<(), Error> {
        // Sequential fold in chunk-index order: bitwise-deterministic
        // regardless of which thread produced which partial.
        for partial in partials {
            let partial = partial?;
            for (dst, src) in raw.iter_mut().zip(&partial.raw) {
                *dst += src;
            }
            for (dst, src) in secondary.iter_mut().zip(&partial.secondary) {
                *dst += src;
            }
            total_rate.add(partial.rate);
            analytic_aal.add(partial.aal);
            quadrature_mean.add(partial.quadrature);
            valid_rows += partial.valid_rows;
        }
        Ok(())
    };
    let run_waves = |fold: &mut dyn FnMut(Vec<Result<ChunkPartial, Error>>) -> Result<(), Error>|
     -> Result<(), Error> {
        let wave = rayon::current_num_threads().max(1) * 2;
        let mut next = 0_usize;
        while next < n_chunks {
            let end = (next + wave).min(n_chunks);
            let partials: Vec<Result<ChunkPartial, Error>> =
                (next..end).into_par_iter().map(process_chunk).collect();
            fold(partials)?;
            next = end;
        }
        Ok(())
    };
    match config.threads {
        Some(threads) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| Error::ThreadPool(e.to_string()))?;
            pool.install(|| run_waves(&mut fold))?;
        }
        None => run_waves(&mut fold)?,
    }

    let total_rate = total_rate.value();
    let analytic_aal = analytic_aal.value();
    let quadrature_secondary_mean = quadrature_mean.value();
    if config.normalize {
        if total_rate <= 0.0 {
            return Err(Error::ZeroTotalRate);
        }
        for w in raw.iter_mut() {
            *w /= total_rate;
        }
        for w in secondary.iter_mut() {
            *w /= total_rate;
        }
    }

    let meta = Meta {
        bucket_size: h,
        grid_length: n,
        quadrature_points: k,
        normalized: config.normalize,
        total_rate,
        analytic_aal,
        raw_mass: sum(&raw),
        raw_mean: grid_moment(&raw, h),
        secondary_mass: sum(&secondary),
        secondary_mean: grid_moment(&secondary, h),
        quadrature_secondary_mean,
        secondary_mean_error: quadrature_secondary_mean - analytic_aal,
        input_rows: n_rows,
        valid_rows,
        diagnostics,
    };
    Ok(QuantizeResult {
        raw,
        secondary,
        meta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VariancePolicy;

    fn input<'a>(cols: &'a [Vec<f64>; 5]) -> EltInput<'a> {
        EltInput {
            rate: &cols[0],
            max_loss: &cols[1],
            mean_loss: &cols[2],
            sd_indep: &cols[3],
            sd_corr: &cols[4],
        }
    }

    fn columns(rows: &[[f64; 5]]) -> [Vec<f64>; 5] {
        std::array::from_fn(|c| rows.iter().map(|r| r[c]).collect())
    }

    #[test]
    fn zero_uncertainty_matches_raw() {
        let cols = columns(&[
            [0.02, 100.0, 40.0, 0.0, 0.0],
            [0.01, 80.0, 17.3, 0.0, 0.0],
        ]);
        let result = quantize(&input(&cols), &Config::new(0.5)).unwrap();
        assert_eq!(result.raw, result.secondary);
    }

    #[test]
    fn mass_and_mean_conserved() {
        let cols = columns(&[
            [0.02, 100.0, 40.0, 15.0, 10.0],
            [0.01, 80.0, 20.0, 8.0, 6.0],
            [0.03, 100.0, 2.0, 6.0, 4.0],
        ]);
        let result = quantize(&input(&cols), &Config::new(0.25)).unwrap();
        let lambda = 0.06;
        assert!((result.meta.raw_mass - lambda).abs() < 1e-14);
        assert!((result.meta.secondary_mass - lambda).abs() < 1e-14);
        let aal = 0.02 * 40.0 + 0.01 * 20.0 + 0.03 * 2.0;
        assert!((result.meta.raw_mean - aal).abs() / aal < 1e-12);
        assert!(
            (result.meta.secondary_mean - result.meta.quadrature_secondary_mean).abs()
                / aal
                < 1e-12
        );
    }

    #[test]
    fn overflow_is_a_typed_error() {
        let cols = columns(&[[0.01, 100.0, 40.0, 15.0, 0.0]]);
        let mut config = Config::new(1.0);
        config.grid_length = Some(50);
        match quantize(&input(&cols), &config) {
            Err(Error::GridOverflow {
                required,
                example_rows,
                ..
            }) => {
                assert_eq!(required, 101);
                assert_eq!(example_rows, vec![0]);
            }
            other => panic!("expected GridOverflow, got {other:?}"),
        }
    }

    #[test]
    fn auto_grid_length_covers_max_loss() {
        let cols = columns(&[[0.01, 100.0, 40.0, 15.0, 0.0]]);
        let result = quantize(&input(&cols), &Config::new(1.0)).unwrap();
        assert_eq!(result.meta.grid_length, 101);
    }

    #[test]
    fn normalize_divides_by_total_rate() {
        let cols = columns(&[
            [0.02, 100.0, 40.0, 15.0, 0.0],
            [0.02, 50.0, 20.0, 5.0, 0.0],
        ]);
        let mut config = Config::new(0.5);
        config.normalize = true;
        let result = quantize(&input(&cols), &config).unwrap();
        assert!((result.meta.raw_mass - 1.0).abs() < 1e-14);
        assert!((result.meta.total_rate - 0.04).abs() < 1e-17);

        let empty = columns(&[[0.0, 50.0, 20.0, 5.0, 0.0]]);
        assert!(matches!(
            quantize(&input(&empty), &config),
            Err(Error::ZeroTotalRate)
        ));
    }

    #[test]
    fn strict_policy_surfaces_row() {
        let cols = columns(&[[0.01, 100.0, 30.0, 50.0, 0.0]]);
        let mut config = Config::new(0.5);
        config.variance_policy = VariancePolicy::Error;
        assert!(matches!(
            quantize(&input(&cols), &config),
            Err(Error::InadmissibleVariance { row: 0, .. })
        ));
    }

    #[test]
    fn column_length_mismatch_is_rejected() {
        let cols = columns(&[[0.01, 100.0, 30.0, 5.0, 0.0]]);
        let mut bad = input(&cols);
        let short = vec![1.0_f64; 0];
        bad.sd_corr = &short;
        assert!(matches!(
            quantize(&bad, &Config::new(0.5)),
            Err(Error::ColumnLengthMismatch {
                name: "sd_corr",
                ..
            })
        ));
    }

    #[test]
    fn empty_input_yields_single_node_grid() {
        let cols = columns(&[]);
        let result = quantize(&input(&cols), &Config::new(0.5)).unwrap();
        assert_eq!(result.meta.grid_length, 1);
        assert_eq!(result.meta.valid_rows, 0);
        assert_eq!(result.raw, vec![0.0]);
    }

    #[test]
    fn thread_count_and_chunking_are_bitwise_invariant() {
        // 300 mixed rows across several chunks; sweep threads and chunk size.
        let rows: Vec<[f64; 5]> = (0..300)
            .map(|i| {
                let m = 100.0 + i as f64;
                let mu = 0.3 * m;
                [1e-3 + i as f64 * 1e-5, m, mu, 0.4 * mu, 0.1 * mu]
            })
            .collect();
        let cols = columns(&rows);
        let mut config = Config::new(2.0);
        config.chunk_rows = 64;
        let base = quantize(&input(&cols), &config).unwrap();
        for threads in [1, 2, 3, 7] {
            config.threads = Some(threads);
            let other = quantize(&input(&cols), &config).unwrap();
            assert_eq!(base.raw, other.raw, "threads={threads}");
            assert_eq!(base.secondary, other.secondary, "threads={threads}");
            assert_eq!(base.meta.total_rate, other.meta.total_rate);
            assert_eq!(base.meta.quadrature_secondary_mean,
                other.meta.quadrature_secondary_mean);
        }
    }
}
