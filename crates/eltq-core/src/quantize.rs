//! Streaming quantization driver: parallel fixed chunks with a
//! bitwise-deterministic in-order fold.
//!
//! Rows are classified once, then processed in fixed chunks of
//! `config.resolved_chunk_rows(n_rows, n)` (a rule of the row count and the
//! grid length under the quantiles method, 256 rows under the survival
//! method). Under the
//! quantiles method each chunk produces private raw and secondary partial
//! histograms plus compensated partial totals (no locks, no atomics in the
//! hot loop); under the survival method the chunk pass is `curve::build_curve`
//! and adds the node-length curve partials. Partials are folded sequentially
//! in chunk-index order, so the result is bitwise identical for any thread
//! count and any work-stealing schedule. Chunks run in bounded waves of
//! `min(2 * threads, wave_memory_bytes / partial_bytes)` (`curve::run_chunks`),
//! so the partials in flight hold at most `config.wave_memory_bytes`, 1 GiB
//! by default, whatever the grid length; one thread pool serves a call.

use rayon::prelude::*;

use crate::batched::{MidpointScores, beta_midpoint_quantiles_scored};
use crate::beta::beta_params;
use crate::config::{Config, Method, Overflow};
use crate::curve::{Workers, build_curve_with, run_waves};
use crate::deposit::{OverflowAccumulator, place};
use crate::diagnostics::{Diagnostics, Meta, QuantizeResult};
use crate::error::Error;
use crate::sum::{Neumaier, grid_moment, mass_and_moment, sum};
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

/// Private per-chunk partial result of the quantiles method.
struct ChunkPartial {
    raw: Vec<f64>,
    secondary: Vec<f64>,
    rate: f64,
    aal: f64,
    quadrature: f64,
    valid_rows: usize,
    raw_over: OverflowAccumulator,
    sec_over: OverflowAccumulator,
}

/// What either method hands back before normalization.
struct Outcome {
    raw: Vec<f64>,
    secondary: Vec<f64>,
    total_rate: f64,
    analytic_aal: f64,
    quadrature_secondary_mean: Option<f64>,
    valid_rows: usize,
    nodes: Option<usize>,
    interpolation_error: Option<f64>,
    narrow_events: Option<usize>,
    fine_nodes: Option<usize>,
    requesting_events: Option<usize>,
    raw_over: OverflowAccumulator,
    sec_over: OverflowAccumulator,
}

/// The classification pass: every row's kind, effective mean and standard
/// deviation, the largest contributing loss, and the diagnostics.
#[derive(Debug, Clone)]
pub struct Classified {
    pub kinds: Vec<RowKind>,
    pub mu_eff: Vec<f64>,
    pub sd_eff: Vec<f64>,
    pub max_contrib_loss: f64,
    pub diagnostics: Diagnostics,
}

/// Classify every row (validation, policies, diagnostics). Dropped rows
/// touch neither output nor the grid's natural length, so this precedes
/// everything.
pub fn classify_rows(input: &EltInput<'_>, config: &Config) -> Result<Classified, Error> {
    config.validate()?;
    let n_rows = input.validated_len()?;
    let mut diagnostics = Diagnostics::default();
    let mut kinds = vec![RowKind::Drop; n_rows];
    let mut mu_eff = vec![0.0_f64; n_rows];
    let mut sd_eff = vec![0.0_f64; n_rows];
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
        sd_eff[row] = plan.sd;
        if plan.kind != RowKind::Drop {
            max_contrib_loss = max_contrib_loss.max(input.max_loss[row]);
        }
    }
    Ok(Classified {
        kinds,
        mu_eff,
        sd_eff,
        max_contrib_loss,
        diagnostics,
    })
}

/// Quantize an ELT onto the even loss grid; both histograms in one pass.
pub fn quantize(input: &EltInput<'_>, config: &Config) -> Result<QuantizeResult, Error> {
    let h = config.bucket_size;
    let classified = classify_rows(input, config)?;
    let Classified {
        ref kinds,
        ref mu_eff,
        ref sd_eff,
        max_contrib_loss,
        ..
    } = classified;
    let n_rows = kinds.len();

    // Grid setup. The derived length covers max M, so the overflow is
    // exactly zero; an explicit shorter grid is a window handled by the
    // overflow policy, and under `error` any contributing loss above the
    // top edge (n - 1/2) h is refused up front.
    let auto_length = if max_contrib_loss > 0.0 {
        (max_contrib_loss / h).ceil() as usize + 1
    } else {
        1
    };
    let n = match config.grid_length {
        None => auto_length,
        Some(n) => {
            let top_edge = (n as f64 - 0.5) * h;
            if config.overflow == Overflow::Error && max_contrib_loss > top_edge {
                let example_rows = (0..n_rows)
                    .filter(|&row| kinds[row] != RowKind::Drop && input.max_loss[row] > top_edge)
                    .take(crate::diagnostics::MAX_EXAMPLES)
                    .collect();
                return Err(Error::GridOverflow {
                    grid_length: n,
                    bucket_size: h,
                    top: top_edge,
                    max_loss: max_contrib_loss,
                    required: auto_length,
                    example_rows,
                });
            }
            n
        }
    };

    let workers = Workers::new(config)?;
    let chunk_rows = config.resolved_chunk_rows(n_rows, n);
    let outcome = match config.method {
        Method::Quantiles => quantiles_pass(&workers, input, config, kinds, mu_eff, sd_eff, n, h, chunk_rows)?,
        Method::Survival => {
            // The curve from e_0 = h / 2, its grid, and the raw grid.
            let curve = build_curve_with(&workers, input, config, &classified, 0.5 * h)?;
            let grid = curve.grid_with(&workers, h, Some(n), config.overflow, config)?;
            let mut raw = vec![0.0_f64; n];
            let mut raw_over = OverflowAccumulator::default();
            let top_node = (n - 1) as f64 * h;
            let top_edge = (n as f64 - 0.5) * h;
            for row in 0..n_rows {
                if kinds[row] != RowKind::Drop {
                    place(&mut raw, mu_eff[row], input.rate[row], 1.0 / h, top_edge, top_node, config.overflow, &mut raw_over);
                }
            }
            Outcome {
                raw,
                secondary: grid.secondary,
                total_rate: curve.total_rate,
                analytic_aal: curve.analytic_aal,
                quadrature_secondary_mean: None,
                valid_rows: curve.valid_rows,
                nodes: Some(curve.nodes()),
                interpolation_error: Some(curve.interpolation_error),
                narrow_events: Some(curve.narrow_events()),
                fine_nodes: Some(curve.fine_nodes()),
                requesting_events: Some(curve.requesting_events()),
                raw_over,
                sec_over: grid.overflow,
            }
        }
    };
    let Outcome {
        mut raw,
        mut secondary,
        total_rate,
        analytic_aal,
        quadrature_secondary_mean,
        valid_rows,
        nodes,
        interpolation_error,
        narrow_events,
        fine_nodes,
        requesting_events,
        raw_over,
        sec_over,
    } = outcome;

    // The secondary mean error is the grid's first moment in rate units
    // less the analytic AAL, under either method.
    let secondary_moment = grid_moment(&secondary, h);
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

    // The mass and moment of each output in one pass each; without
    // normalization the secondary moment is the one already taken (both
    // bitwise what the separate scans gave: peer review, finding P2).
    let (raw_mass, raw_mean) = mass_and_moment(&raw, h);
    let (secondary_mass, secondary_mean) = if config.normalize {
        mass_and_moment(&secondary, h)
    } else {
        (sum(&secondary), secondary_moment)
    };
    let meta = Meta {
        bucket_size: h,
        grid_length: n,
        method: config.method,
        quadrature_points: (config.method == Method::Quantiles).then_some(config.quadrature_points),
        nodes,
        nodes_per_decade: (config.method == Method::Survival).then_some(config.nodes_per_decade),
        interpolation_error,
        narrow_events,
        fine_nodes,
        requesting_events,
        overflow: config.overflow,
        rate_above_grid: [raw_over.rate.value(), sec_over.rate.value()],
        mean_above_grid: [raw_over.moment.value(), sec_over.moment.value()],
        normalized: config.normalize,
        total_rate,
        analytic_aal,
        raw_mass,
        raw_mean,
        secondary_mass,
        secondary_mean,
        quadrature_secondary_mean,
        secondary_mean_error: secondary_moment - analytic_aal,
        input_rows: n_rows,
        valid_rows,
        diagnostics: classified.diagnostics,
    };
    Ok(QuantizeResult {
        raw,
        secondary,
        meta,
    })
}

/// The quantiles method over the classified rows: chunked deposits of K
/// midpoint quantiles per beta event, each chunk owning private
/// histograms and a reused K-quantile buffer (no N x K array ever exists).
#[allow(clippy::too_many_arguments)]
fn quantiles_pass(
    workers: &Workers,
    input: &EltInput<'_>,
    config: &Config,
    kinds: &[RowKind],
    mu_eff: &[f64],
    sd_eff: &[f64],
    n: usize,
    h: f64,
    chunk_rows: usize,
) -> Result<Outcome, Error> {
    let n_rows = kinds.len();
    let k = config.quadrature_points;
    let inv_h = 1.0 / h;
    let top_node = (n - 1) as f64 * h;
    let top_edge = (n as f64 - 0.5) * h;
    let policy = config.overflow;
    // The midpoint rule's normal scores, once for every event on the
    // Cornish-Fisher route.
    let scores = MidpointScores::new(k);
    let process_chunk = |chunk_index: usize| -> Result<ChunkPartial, Error> {
        let start = chunk_index * chunk_rows;
        let end = (start + chunk_rows).min(n_rows);
        let mut partial = ChunkPartial {
            raw: vec![0.0; n],
            secondary: vec![0.0; n],
            rate: 0.0,
            aal: 0.0,
            quadrature: 0.0,
            valid_rows: 0,
            raw_over: OverflowAccumulator::default(),
            sec_over: OverflowAccumulator::default(),
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
            place(&mut partial.raw, mu, lam, inv_h, top_edge, top_node, policy, &mut partial.raw_over);
            if kinds[row] == RowKind::Atom {
                place(&mut partial.secondary, mu, lam, inv_h, top_edge, top_node, policy, &mut partial.sec_over);
                quad_acc.add(lam * mu);
            } else if kinds[row] == RowKind::Bernoulli {
                // The variance maximizer on [0, M]: rate p at M and the rest
                // at 0, so the row's mean lam * mu is exact.
                let p = mu / m;
                place(&mut partial.secondary, 0.0, lam * (1.0 - p), inv_h, top_edge, top_node, policy, &mut partial.sec_over);
                place(&mut partial.secondary, m, lam * p, inv_h, top_edge, top_node, policy, &mut partial.sec_over);
                quad_acc.add(lam * mu);
            } else {
                let (alpha, beta) = beta_params(m, mu, sd_eff[row]);
                beta_midpoint_quantiles_scored(alpha, beta, Some(&scores), &mut quantiles)
                    .map_err(|_| Error::KernelFailure { row, alpha, beta })?;
                let w = lam / k as f64;
                let mut row_sum = Neumaier::default();
                for &q in &quantiles {
                    let x = m * q;
                    place(&mut partial.secondary, x, w, inv_h, top_edge, top_node, policy, &mut partial.sec_over);
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

    let n_chunks = n_rows.div_ceil(chunk_rows);
    let mut raw = vec![0.0_f64; n];
    let mut secondary = vec![0.0_f64; n];
    let mut total_rate = Neumaier::default();
    let mut analytic_aal = Neumaier::default();
    let mut quadrature_mean = Neumaier::default();
    let mut valid_rows = 0_usize;
    let mut raw_over = OverflowAccumulator::default();
    let mut sec_over = OverflowAccumulator::default();
    // The fold, in chunk-index order per element so that it is bitwise
    // deterministic regardless of which thread produced which partial: the
    // two grid-length histograms of a wave's partials are added in parallel
    // over disjoint blocks of the grid (the serial fold of 61 chunks at 2^22
    // buckets cost more than the kernel on 32 threads), the scalar totals
    // serially. A partial is the two grid-length histograms.
    const FOLD_BLOCK: usize = 1 << 15;
    run_waves(workers, config, n_chunks, 16 * n + 128, process_chunk, |partials: Vec<ChunkPartial>| {
        raw.par_chunks_mut(FOLD_BLOCK)
            .zip(secondary.par_chunks_mut(FOLD_BLOCK))
            .enumerate()
            .for_each(|(block, (raw_block, sec_block))| {
                let lo = block * FOLD_BLOCK;
                let hi = lo + raw_block.len();
                for partial in &partials {
                    for (dst, src) in raw_block.iter_mut().zip(&partial.raw[lo..hi]) {
                        *dst += src;
                    }
                    for (dst, src) in sec_block.iter_mut().zip(&partial.secondary[lo..hi]) {
                        *dst += src;
                    }
                }
            });
        for partial in &partials {
            total_rate.add(partial.rate);
            analytic_aal.add(partial.aal);
            quadrature_mean.add(partial.quadrature);
            valid_rows += partial.valid_rows;
            raw_over.merge(&partial.raw_over);
            sec_over.merge(&partial.sec_over);
        }
        Ok(())
    })?;
    Ok(Outcome {
        raw,
        secondary,
        total_rate: total_rate.value(),
        analytic_aal: analytic_aal.value(),
        quadrature_secondary_mean: Some(quadrature_mean.value()),
        valid_rows,
        nodes: None,
        interpolation_error: None,
        narrow_events: None,
        fine_nodes: None,
        requesting_events: None,
        raw_over,
        sec_over,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VariancePolicy;
    use crate::diagnostics::Category;

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
        for method in [Method::Quantiles, Method::Survival] {
            let mut config = Config::new(0.5);
            config.method = method;
            let result = quantize(&input(&cols), &config).unwrap();
            assert_eq!(result.raw, result.secondary);
        }
    }

    #[test]
    fn mass_and_mean_conserved() {
        let cols = columns(&[
            [0.02, 100.0, 40.0, 15.0, 10.0],
            [0.01, 80.0, 20.0, 8.0, 6.0],
            [0.03, 100.0, 2.0, 6.0, 4.0],
        ]);
        let mut config = Config::new(0.25);
        config.method = Method::Quantiles;
        let result = quantize(&input(&cols), &config).unwrap();
        let lambda = 0.06;
        assert!((result.meta.raw_mass - lambda).abs() < 1e-14);
        assert!((result.meta.secondary_mass - lambda).abs() < 1e-14);
        let aal = 0.02 * 40.0 + 0.01 * 20.0 + 0.03 * 2.0;
        assert!((result.meta.raw_mean - aal).abs() / aal < 1e-12);
        assert!(
            (result.meta.secondary_mean - result.meta.quadrature_secondary_mean.unwrap()).abs()
                / aal
                < 1e-12
        );
    }

    #[test]
    fn survival_mass_exact_and_mean_close() {
        let cols = columns(&[
            [0.02, 100.0, 40.0, 15.0, 10.0],
            [0.01, 80.0, 20.0, 8.0, 6.0],
            [0.03, 100.0, 2.0, 6.0, 4.0],
            [0.005, 100.0, 30.0, 50.0, 0.0], // Bernoulli
            [0.004, 50.0, 25.0, 0.0, 0.0],   // atom
        ]);
        let mut config = Config::new(0.05);
        config.method = Method::Survival;
        let result = quantize(&input(&cols), &config).unwrap();
        let lambda = 0.069;
        assert!((result.meta.secondary_mass - lambda).abs() < 1e-14 * lambda);
        assert!(result.secondary.iter().all(|&w| w >= 0.0 && w.is_finite()));
        let aal = result.meta.analytic_aal;
        assert!((result.meta.secondary_mean_error / aal).abs() < 1e-4);
        assert_eq!(result.meta.method, Method::Survival);
        assert!(result.meta.nodes.unwrap() > 100);
        assert!(result.meta.quadrature_points.is_none());
        assert!(result.meta.interpolation_error.unwrap() < 1e-4);
        assert_eq!(result.meta.rate_above_grid, [0.0, 0.0]);
    }

    #[test]
    fn overflow_policies_on_a_window() {
        let cols = columns(&[[0.01, 100.0, 40.0, 15.0, 0.0], [0.02, 100.0, 90.0, 0.0, 0.0]]);
        for method in [Method::Quantiles, Method::Survival] {
            let mut config = Config::new(1.0);
            config.grid_length = Some(50);
            config.method = method;
            config.overflow = Overflow::Error;
            match quantize(&input(&cols), &config) {
                Err(Error::GridOverflow { required, example_rows, .. }) => {
                    assert_eq!(required, 101);
                    assert_eq!(example_rows, vec![0, 1]);
                }
                other => panic!("expected GridOverflow, got {other:?}"),
            }
            config.overflow = Overflow::TopBucket;
            let top = quantize(&input(&cols), &config).unwrap();
            assert!((top.meta.raw_mass - 0.03).abs() < 1e-15);
            assert!((top.meta.secondary_mass - 0.03).abs() < 1e-14);
            assert_eq!(top.meta.rate_above_grid[0], 0.02);
            assert_eq!(top.meta.mean_above_grid[0], 0.02 * (90.0 - 49.0));
            assert!(top.meta.rate_above_grid[1] > 0.02);
            config.overflow = Overflow::Drop;
            let drop = quantize(&input(&cols), &config).unwrap();
            assert!((drop.meta.raw_mass - 0.01).abs() < 1e-15);
            assert!((drop.meta.secondary_mass - (0.03 - drop.meta.rate_above_grid[1])).abs() < 1e-14);
            assert_eq!(drop.meta.rate_above_grid, top.meta.rate_above_grid);
            assert_eq!(&drop.raw[..49], &top.raw[..49]);
            assert_eq!(&drop.secondary[..49], &top.secondary[..49]);
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
    fn bernoulli_rows_are_two_atoms() {
        // sigma^2 = 2500 >= mu (M - mu) = 2100: p = 0.3, M on node 200.
        let cols = columns(&[[0.01, 100.0, 30.0, 50.0, 0.0]]);
        for method in [Method::Quantiles, Method::Survival] {
            let mut config = Config::new(0.5);
            config.method = method;
            let result = quantize(&input(&cols), &config).unwrap();
            assert_eq!(result.secondary[0], 0.01 * 0.7);
            assert_eq!(result.secondary[200], 0.01 * 0.3);
            assert_eq!(result.secondary.iter().filter(|&&w| w != 0.0).count(), 2);
            assert_eq!(result.meta.diagnostics.get(Category::VarianceBernoulli).count, 1);
            assert!((result.meta.secondary_mean - 0.3).abs() < 1e-15);
            assert_eq!(result.meta.secondary_mean_error, 0.0);
        }
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
        for method in [Method::Quantiles, Method::Survival] {
            let mut config = Config::new(0.5);
            config.method = method;
            let result = quantize(&input(&cols), &config).unwrap();
            assert_eq!(result.meta.grid_length, 1);
            assert_eq!(result.meta.valid_rows, 0);
            assert_eq!(result.raw, vec![0.0]);
            assert_eq!(result.secondary, vec![0.0]);
        }
    }

    // [schedule] At the default chunk, which is a rule of the row count,
    // the quantiles grid is bitwise invariant across thread counts and
    // within summation rounding of the fixed 65,536-row chunk's.
    #[test]
    fn default_chunk_is_a_rule_of_the_input() {
        let rows: Vec<[f64; 5]> = (0..20_000)
            .map(|i| {
                let m = 1e4 + (i % 977) as f64 * 1e3;
                let mu = (0.05 + 0.9 * ((i * 7919) % 1000) as f64 / 1000.0) * m;
                [1e-4, m, mu, 0.3 * mu.min(m - mu), 0.0]
            })
            .collect();
        let cols = columns(&rows);
        let mut config = Config::new(5000.0);
        config.method = Method::Quantiles;
        config.quadrature_points = 8;
        assert_eq!(config.resolved_chunk_rows(rows.len(), 1 << 16), 4096);
        config.threads = Some(1);
        let base = quantize(&input(&cols), &config).unwrap();
        for threads in [3, 8] {
            config.threads = Some(threads);
            let other = quantize(&input(&cols), &config).unwrap();
            assert_eq!(base.secondary, other.secondary, "threads={threads}");
        }
        config.chunk_rows = Some(65_536);
        let fixed = quantize(&input(&cols), &config).unwrap();
        let worst = base
            .secondary
            .iter()
            .zip(&fixed.secondary)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(worst <= 1e-14 * base.meta.total_rate, "{worst}");
    }

    #[test]
    fn thread_count_and_chunking_are_bitwise_invariant() {
        // 300 mixed rows across several chunks; sweep threads under both
        // methods at a fixed chunk size.
        let rows: Vec<[f64; 5]> = (0..300)
            .map(|i| {
                let m = 100.0 + i as f64;
                let mu = 0.3 * m;
                [1e-3 + i as f64 * 1e-5, m, mu, 0.4 * mu, 0.1 * mu]
            })
            .collect();
        let cols = columns(&rows);
        for method in [Method::Quantiles, Method::Survival] {
            let mut config = Config::new(2.0);
            config.method = method;
            config.chunk_rows = Some(64);
            let base = quantize(&input(&cols), &config).unwrap();
            for threads in [1, 2, 3, 7] {
                config.threads = Some(threads);
                let other = quantize(&input(&cols), &config).unwrap();
                assert_eq!(base.raw, other.raw, "threads={threads}");
                assert_eq!(base.secondary, other.secondary, "threads={threads}");
                assert_eq!(base.meta.total_rate, other.meta.total_rate);
                assert_eq!(base.meta.secondary_mean_error, other.meta.secondary_mean_error);
                assert_eq!(base.meta.interpolation_error, other.meta.interpolation_error);
            }
        }
    }
}
