//! The survival method: the portfolio exceedance rate on geometric nodes,
//! truncated per event before its endpoint, interpolated by a monotone cubic
//! and bucketed by the nearest rule (specification section 5A).
//!
//! Mirror of `python/eltq/reference.py` (`build_survival_curve`,
//! `SurvivalCurve`, `_tail_survival`): the same nodes, the same truncation,
//! the same one-sided slopes, the same cubic Hermite with Fritsch and Carlson
//! limiting, and the same per-event tail interpolant in the log distance to
//! the endpoint. Both backends compute the same function; the acceptance
//! suite holds them to each other.
//!
//! Each beta event `j` is split at its truncation node `x_k`, the node
//! `TAIL_INTERVALS` intervals below the first node at or above its endpoint
//! `M_j`. The **curve** carries `lambda_j (S_j(x) - S_j(x_k))` for `x <= x_k`,
//! continuous and vanishing at a node; the **tail**, the mass
//! `lambda_j S_j(x_k)` on `(x_k, M_j]`, is kept per event and bucketed by a
//! monotone cubic Hermite of `log S_j` against `log(M_j - x)`. The summed
//! curve has kinks only at nodes, which carry one-sided slopes.
//!
//! [`Curve`] is the primary object (spec section 5A): built once by
//! [`build_curve`] in a chunk pass folded in order, it answers exceedance
//! queries and makes the even grid at any `h >= 2 min_loss` and any `n`.

use rayon::prelude::*;

use crate::beta::beta_params;
use crate::config::{Config, Overflow};
use crate::deposit::{OverflowAccumulator, place};
use crate::diagnostics::Diagnostics;
use crate::error::Error;
use crate::incbeta::BetaFn;
use crate::quantize::{Classified, EltInput};
use crate::sum::Neumaier;
use crate::validate::RowKind;

/// Node intervals before an event's endpoint left to its exact tail.
pub const TAIL_INTERVALS: usize = 4;
/// Natural log of the lower-skip probability 2^-60: below the closed-form
/// lower-tail quantile at this probability the survival function is
/// exactly 1 in double precision.
pub const LN_LOWER_SKIP_PROBABILITY: f64 = -60.0 * std::f64::consts::LN_2;
/// Interval log-midpoints sampled for the interpolation-error diagnostic.
pub const SAMPLE_POINTS: usize = 64;
/// The exceedance level, as a fraction of the beta events' total rate,
/// below which the interpolation error is not measured.
pub const ERROR_FLOOR: f64 = 1e-12;

/// Geometric nodes from `min_loss` to `max_loss` at `nodes_per_decade`,
/// the endpoints exact. The same arithmetic as NumPy's `linspace` of the
/// logarithms, so both backends share their nodes to the rounding of `exp`.
pub fn geometric_nodes(min_loss: f64, max_loss: f64, nodes_per_decade: usize) -> Vec<f64> {
    if max_loss <= min_loss {
        return vec![min_loss];
    }
    let intervals = ((nodes_per_decade as f64) * (max_loss / min_loss).log10()).ceil() as usize;
    let intervals = intervals.max(1);
    let (ln_min, ln_max) = (min_loss.ln(), max_loss.ln());
    let step = (ln_max - ln_min) / intervals as f64;
    let mut loss: Vec<f64> = (0..=intervals)
        .map(|i| (i as f64 * step + ln_min).exp())
        .collect();
    loss[0] = min_loss;
    loss[intervals] = max_loss;
    loss
}

/// Fritsch and Carlson limiting of one interval's end slopes against its
/// secant `d`: zero on a flat interval, wrong-signed slopes zeroed, and
/// both scaled down together when `a^2 + b^2 > 9` in units of the secant.
#[inline]
pub fn limit_slopes(d: f64, m0: f64, m1: f64) -> (f64, f64) {
    if d == 0.0 {
        return (0.0, 0.0);
    }
    let mut a = (m0 / d).max(0.0);
    let mut b = (m1 / d).max(0.0);
    let s = a * a + b * b;
    if s > 9.0 {
        let tau = 3.0 / s.sqrt();
        a *= tau;
        b *= tau;
    }
    (a * d, b * d)
}

/// Cubic Hermite on one interval at local coordinate `t`.
#[inline]
pub fn hermite(t: f64, y0: f64, y1: f64, dx: f64, m0: f64, m1: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + (t3 - 2.0 * t2 + t) * dx * m0
        + (-2.0 * t3 + 3.0 * t2) * y1
        + (t3 - t2) * dx * m1
}

/// Index of the last node `<= x` (`searchsorted(..., side="right") - 1`).
#[inline]
fn interval_of(loss: &[f64], x: f64) -> usize {
    loss.partition_point(|&v| v <= x) - 1
}

/// `lam P(X; X > e)` for `X = M Beta(a, b)`: the partial expectation is
/// the mean times the survival of the beta with `a + 1`.
fn tail_moment(lam: f64, m: f64, a: f64, b: f64, e: f64) -> f64 {
    if e >= m {
        return 0.0;
    }
    lam * m * (a / (a + b)) * BetaFn::new(a + 1.0, b).sf(e / m)
}

/// `S(x)` of one beta event on its tail interval `(x_k, M]` at the given
/// losses, by the endpoint interpolant: `log S` against `u = log(M - x)`,
/// a monotone cubic Hermite with the exact slopes `(1 - y) f(y) / S(y)` on
/// nodes shrinking the distance to the endpoint by sqrt(2) per step from
/// `M - x_k` down to `min_distance`, and the exact power law `S ~ d^beta`
/// below the last node. `s` is the exact `S(x_k)`, used as the first
/// node's value.
pub fn tail_survival(bf: &BetaFn, m: f64, xk: f64, s: f64, x: &[f64], min_distance: f64) -> Vec<f64> {
    let (a, b) = (bf.alpha, bf.beta);
    let d0 = m - xk;
    let count = 2 * ((d0 / min_distance).log2().ceil().max(1.0) as usize) + 1;
    // Nodes with a positive survival, descending in distance.
    let mut dist = Vec::with_capacity(count);
    let mut log_surv = Vec::with_capacity(count);
    let mut slope = Vec::with_capacity(count);
    for i in 0..count {
        let d = d0 * 0.5_f64.powf(i as f64 / 2.0);
        let y = 1.0 - d / m;
        let surv = if i == 0 { s } else { bf.sf(y) };
        if !(surv > 0.0) {
            break;
        }
        let ls = surv.ln();
        dist.push(d);
        log_surv.push(ls);
        slope.push(((a - 1.0) * y.ln() + b * (d / m).ln() - bf.ln_beta() - ls).exp());
    }
    let mut out = vec![0.0; x.len()];
    if dist.is_empty() {
        return out;
    }
    // Ascending in u = log d.
    let u: Vec<f64> = dist.iter().rev().map(|d| d.ln()).collect();
    let g: Vec<f64> = log_surv.iter().rev().copied().collect();
    let sl: Vec<f64> = slope.iter().rev().copied().collect();
    let last = u.len() - 1;
    for (o, &xq) in out.iter_mut().zip(x) {
        let d = m - xq;
        if d <= 0.0 {
            continue;
        }
        let uq = d.ln();
        let value = if uq < u[0] || last == 0 {
            g[0] + b * (uq - u[0])
        } else {
            let k = (u.partition_point(|&v| v <= uq) - 1).min(last - 1);
            let du = u[k + 1] - u[k];
            let (m0, m1) = limit_slopes((g[k + 1] - g[k]) / du, sl[k], sl[k + 1]);
            hermite((uq - u[k]) / du, g[k], g[k + 1], du, m0, m1)
        };
        *o = value.exp().min(s);
    }
    out
}

/// One beta event's tail: what the curve omits above its truncation node.
#[derive(Debug, Clone, Copy)]
pub struct TailEvent {
    pub rate: f64,
    pub max_loss: f64,
    pub alpha: f64,
    pub beta: f64,
    /// Truncation node index; the tail lives on `(loss[node], max_loss]`.
    pub node: usize,
    /// `S(loss[node])`, exact.
    pub survival: f64,
}

/// The exceedance curve: nodes, the truncated curve with its one-sided
/// slopes, the tails per event, and the atoms (spec section 5A).
#[derive(Debug, Clone)]
pub struct Curve {
    /// The nodes, `loss[0] = min_loss`.
    pub loss: Vec<f64>,
    /// The truncated curve at the nodes.
    pub curve: Vec<f64>,
    /// The exact exceedance rate of the beta events at the nodes.
    pub exceedance_rate: Vec<f64>,
    /// Limited end slopes per interval, `dy/dx`.
    m0: Vec<f64>,
    m1: Vec<f64>,
    /// `tail_above[i]`: total tail mass of the events truncated at node `i`
    /// or later (length `nodes + 1`).
    tail_above: Vec<f64>,
    /// The tails, stably sorted by truncation node; `group_start[k]` is the
    /// first index of node `k`'s group (length `nodes + 1`).
    tails: Vec<TailEvent>,
    group_start: Vec<usize>,
    /// Atoms sorted by loss, with `atom_above[i]` the total rate of atoms
    /// from index `i` on (length `atoms + 1`).
    pub atom_loss: Vec<f64>,
    pub atom_rate: Vec<f64>,
    atom_above: Vec<f64>,
    /// Total rate of the beta events (the curve's value below every event).
    pub beta_rate: f64,
    /// Totals over every contributing row.
    pub total_rate: f64,
    pub analytic_aal: f64,
    pub valid_rows: usize,
    pub input_rows: usize,
    /// Largest contributing loss (beta endpoint or atom).
    pub max_loss: f64,
    pub min_loss: f64,
    pub nodes_per_decade: usize,
    pub interpolation_error: f64,
    pub diagnostics: Diagnostics,
}

/// The even grid made from a curve, before normalization.
#[derive(Debug, Clone)]
pub struct GridOutcome {
    pub secondary: Vec<f64>,
    pub grid_length: usize,
    pub overflow: OverflowAccumulator,
}

/// One chunk's contribution to the curve build.
struct BuildPartial {
    curve: Vec<f64>,
    slope_both: Vec<f64>,
    slope_end: Vec<f64>,
    tail_at: Vec<f64>,
    node_tail: Vec<f64>,
    sample_curve: Vec<f64>,
    sample_total: Vec<f64>,
    tails: Vec<TailEvent>,
    atoms: Vec<(f64, f64)>,
    rate: f64,
    aal: f64,
    beta_rate: f64,
    valid_rows: usize,
}

/// Sample interval indices for the interpolation-error diagnostic:
/// `unique(round_half_even(linspace(0, intervals - 1, SAMPLE_POINTS)))`.
fn sample_intervals(intervals: usize) -> Vec<usize> {
    if intervals == 0 {
        return Vec::new();
    }
    if intervals <= SAMPLE_POINTS {
        return (0..intervals).collect();
    }
    let step = (intervals - 1) as f64 / (SAMPLE_POINTS - 1) as f64;
    let mut picks: Vec<usize> = (0..SAMPLE_POINTS)
        .map(|i| (i as f64 * step).round_ties_even() as usize)
        .collect();
    picks.dedup();
    picks
}

/// Run a chunked pass in bounded waves, folding partials in chunk order.
pub(crate) fn run_chunks<P: Send>(
    config: &Config,
    n_chunks: usize,
    process: impl Fn(usize) -> Result<P, Error> + Sync,
    mut fold: impl FnMut(P) -> Result<(), Error> + Send,
) -> Result<(), Error> {
    let run_waves = |fold: &mut dyn FnMut(P) -> Result<(), Error>| -> Result<(), Error> {
        let wave = rayon::current_num_threads().max(1) * 2;
        let mut next = 0_usize;
        while next < n_chunks {
            let end = (next + wave).min(n_chunks);
            let partials: Vec<Result<P, Error>> = (next..end).into_par_iter().map(&process).collect();
            for partial in partials {
                fold(partial?)?;
            }
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
            pool.install(|| run_waves(&mut fold))
        }
        None => run_waves(&mut fold),
    }
}

/// Build the exceedance curve of the classified rows from `min_loss` up.
///
/// One chunk pass over the rows, folded in chunk order (bitwise identical
/// for any thread count at a fixed `chunk_rows`): each beta event's
/// truncated survival and slope at the nodes, its tail, its exact survival
/// at the nodes its tail covers, and its share of the diagnostic samples;
/// each atom row's atoms. The tails are then stably sorted by truncation
/// node and the atoms by loss.
pub fn build_curve(
    input: &EltInput<'_>,
    config: &Config,
    classified: &Classified,
    min_loss: f64,
) -> Result<Curve, Error> {
    if !(min_loss.is_finite() && min_loss > 0.0) {
        return Err(Error::BadMinLoss(min_loss));
    }
    let kinds = &classified.kinds;
    let n_rows = kinds.len();
    let chunk_rows = config.resolved_chunk_rows();
    let max_beta_m = (0..n_rows)
        .filter(|&row| kinds[row] == RowKind::Beta)
        .map(|row| input.max_loss[row])
        .fold(0.0_f64, f64::max);
    let loss = geometric_nodes(min_loss, max_beta_m, config.nodes_per_decade);
    let m1 = loss.len();
    let intervals = m1 - 1;
    let sample_idx = sample_intervals(intervals);
    let sample_x: Vec<f64> = sample_idx
        .iter()
        .map(|&i| (0.5 * (loss[i].ln() + loss[i + 1].ln())).exp())
        .collect();

    let process_chunk = |chunk_index: usize| -> Result<BuildPartial, Error> {
        let start = chunk_index * chunk_rows;
        let end = (start + chunk_rows).min(n_rows);
        let mut p = BuildPartial {
            curve: vec![0.0; m1],
            slope_both: vec![0.0; m1],
            slope_end: vec![0.0; m1],
            tail_at: vec![0.0; m1],
            node_tail: vec![0.0; m1],
            sample_curve: vec![0.0; sample_x.len()],
            sample_total: vec![0.0; sample_x.len()],
            tails: Vec::new(),
            atoms: Vec::new(),
            rate: 0.0,
            aal: 0.0,
            beta_rate: 0.0,
            valid_rows: 0,
        };
        let mut rate_acc = Neumaier::default();
        let mut aal_acc = Neumaier::default();
        let mut beta_acc = Neumaier::default();
        let mut surv = Vec::with_capacity(m1);
        let mut g = Vec::with_capacity(m1);
        for row in start..end {
            let kind = kinds[row];
            if kind == RowKind::Drop {
                continue;
            }
            p.valid_rows += 1;
            let lam = input.rate[row];
            let m = input.max_loss[row];
            let mu = classified.mu_eff[row];
            rate_acc.add(lam);
            aal_acc.add(lam * mu);
            match kind {
                RowKind::Atom => p.atoms.push((mu, lam)),
                RowKind::Bernoulli => {
                    let prob = mu / m;
                    p.atoms.push((0.0, lam * (1.0 - prob)));
                    p.atoms.push((m, lam * prob));
                }
                RowKind::Beta => {
                    beta_acc.add(lam);
                    if m <= min_loss {
                        // Entirely below the lowest node: counted in the
                        // beta rate, nothing to interpolate.
                        continue;
                    }
                    let (alpha, beta) = beta_params(m, mu, classified.var_eff[row]);
                    let bf = BetaFn::new(alpha, beta);
                    // loss[k] < m <= loss[k + 1], then TAIL_INTERVALS - 1 lower.
                    let above = loss.partition_point(|&v| v < m);
                    let k_signed = above as isize - 1 - (TAIL_INTERVALS as isize - 1);
                    let (k, xk, s) = if k_signed < 0 {
                        (0_usize, loss[0], bf.sf(loss[0] / m))
                    } else {
                        let k = k_signed as usize;
                        surv.clear();
                        g.clear();
                        // Lower skip: below the closed-form lower-tail
                        // quantile at 2^-60 the survival is exactly 1.
                        let ln_plateau = (LN_LOWER_SKIP_PROBABILITY + alpha.ln() + bf.ln_beta()) / alpha;
                        let plateau = if ln_plateau > -745.0 { ln_plateau.exp() } else { 0.0 };
                        for &x in &loss[..=k] {
                            let y = x / m;
                            if y < plateau {
                                surv.push(1.0);
                                g.push(0.0);
                            } else {
                                let (sf, density_y) = bf.sf_and_density_times_y(y);
                                surv.push(sf);
                                g.push(lam * density_y);
                            }
                        }
                        let s = surv[k];
                        for i in 0..=k {
                            p.curve[i] += lam * (surv[i] - s);
                        }
                        for i in 0..k {
                            p.slope_both[i] += g[i];
                        }
                        p.slope_end[k] += g[k];
                        (k, loss[k], s)
                    };
                    p.tail_at[k] += lam * s;
                    // Its exact survival at the nodes its tail covers.
                    for i in k + 1..m1 {
                        if loss[i] >= m {
                            break;
                        }
                        p.node_tail[i] += lam * bf.sf(loss[i] / m);
                    }
                    // The diagnostic samples: exact curve and exact total.
                    for (i, &x) in sample_x.iter().enumerate() {
                        let sf = bf.sf((x / m).min(1.0));
                        p.sample_total[i] += lam * sf;
                        if x < xk {
                            p.sample_curve[i] += lam * (sf - s);
                        }
                    }
                    if s > 0.0 {
                        p.tails.push(TailEvent {
                            rate: lam,
                            max_loss: m,
                            alpha,
                            beta,
                            node: k,
                            survival: s,
                        });
                    }
                }
                RowKind::Drop => unreachable!(),
            }
        }
        p.rate = rate_acc.value();
        p.aal = aal_acc.value();
        p.beta_rate = beta_acc.value();
        Ok(p)
    };

    let n_chunks = n_rows.div_ceil(chunk_rows).max(1);
    let mut curve = vec![0.0_f64; m1];
    let mut slope_both = vec![0.0_f64; m1];
    let mut slope_end = vec![0.0_f64; m1];
    let mut tail_at = vec![0.0_f64; m1];
    let mut node_tail = vec![0.0_f64; m1];
    let mut sample_curve = vec![0.0_f64; sample_x.len()];
    let mut sample_total = vec![0.0_f64; sample_x.len()];
    let mut tails: Vec<TailEvent> = Vec::new();
    let mut atoms: Vec<(f64, f64)> = Vec::new();
    let mut total_rate = Neumaier::default();
    let mut analytic_aal = Neumaier::default();
    let mut beta_rate = Neumaier::default();
    let mut valid_rows = 0_usize;
    run_chunks(config, n_chunks, process_chunk, |p: BuildPartial| {
        for (dst, src) in curve.iter_mut().zip(&p.curve) {
            *dst += src;
        }
        for (dst, src) in slope_both.iter_mut().zip(&p.slope_both) {
            *dst += src;
        }
        for (dst, src) in slope_end.iter_mut().zip(&p.slope_end) {
            *dst += src;
        }
        for (dst, src) in tail_at.iter_mut().zip(&p.tail_at) {
            *dst += src;
        }
        for (dst, src) in node_tail.iter_mut().zip(&p.node_tail) {
            *dst += src;
        }
        for (dst, src) in sample_curve.iter_mut().zip(&p.sample_curve) {
            *dst += src;
        }
        for (dst, src) in sample_total.iter_mut().zip(&p.sample_total) {
            *dst += src;
        }
        tails.extend(p.tails);
        atoms.extend(p.atoms);
        total_rate.add(p.rate);
        analytic_aal.add(p.aal);
        beta_rate.add(p.beta_rate);
        valid_rows += p.valid_rows;
        Ok(())
    })?;
    let beta_rate = beta_rate.value();
    for v in curve.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }

    // The interpolant: limited one-sided slopes per interval.
    let mut m0 = vec![0.0_f64; intervals];
    let mut m1v = vec![0.0_f64; intervals];
    for i in 0..intervals {
        if curve[i] <= 0.0 {
            continue;
        }
        let d = (curve[i + 1] - curve[i]) / (loss[i + 1] - loss[i]);
        let (a, b) = limit_slopes(
            d,
            -slope_both[i] / loss[i],
            -(slope_both[i + 1] + slope_end[i + 1]) / loss[i + 1],
        );
        m0[i] = a;
        m1v[i] = b;
    }
    let mut tail_above = vec![0.0_f64; m1 + 1];
    for i in (0..m1).rev() {
        tail_above[i] = tail_above[i + 1] + tail_at[i];
    }
    let exceedance_rate: Vec<f64> = (0..m1)
        .map(|i| curve[i] + tail_above[i] + node_tail[i])
        .collect();

    // Group the tails by node and sort the atoms by loss, both stably.
    tails.sort_by_key(|t| t.node);
    let mut group_start = vec![0_usize; m1 + 1];
    for t in &tails {
        group_start[t.node + 1] += 1;
    }
    for k in 0..m1 {
        group_start[k + 1] += group_start[k];
    }
    atoms.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let atom_loss: Vec<f64> = atoms.iter().map(|a| a.0).collect();
    let atom_rate: Vec<f64> = atoms.iter().map(|a| a.1).collect();
    let mut atom_above = vec![0.0_f64; atoms.len() + 1];
    for i in (0..atoms.len()).rev() {
        atom_above[i] = atom_above[i + 1] + atom_rate[i];
    }
    let max_loss = max_beta_m.max(atom_loss.last().copied().unwrap_or(0.0));

    let mut built = Curve {
        loss,
        curve,
        exceedance_rate,
        m0,
        m1: m1v,
        tail_above,
        tails,
        group_start,
        atom_loss,
        atom_rate,
        atom_above,
        beta_rate,
        total_rate: total_rate.value(),
        analytic_aal: analytic_aal.value(),
        valid_rows,
        input_rows: n_rows,
        max_loss,
        min_loss,
        nodes_per_decade: config.nodes_per_decade,
        interpolation_error: 0.0,
        diagnostics: classified.diagnostics.clone(),
    };
    // The sampled interpolation error, relative to the exact total.
    let floor = ERROR_FLOOR * beta_rate;
    let mut worst = 0.0_f64;
    for (i, &x) in sample_x.iter().enumerate() {
        if sample_total[i] > floor {
            let err = (built.interpolate(x) - sample_curve[i]).abs() / sample_total[i];
            worst = worst.max(err);
        }
    }
    built.interpolation_error = worst;
    Ok(built)
}

impl Curve {
    pub fn nodes(&self) -> usize {
        self.loss.len()
    }

    /// The truncated curve by the interpolant; 0 above the highest node.
    /// Below the lowest node the curve is undefined (NaN).
    pub fn interpolate(&self, x: f64) -> f64 {
        let loss = &self.loss;
        let m1 = loss.len();
        if x < loss[0] {
            return f64::NAN;
        }
        if x >= loss[m1 - 1] {
            return 0.0;
        }
        let k = interval_of(loss, x);
        if loss[k] == x {
            return self.curve[k];
        }
        let dx = loss[k + 1] - loss[k];
        hermite((x - loss[k]) / dx, self.curve[k], self.curve[k + 1], dx, self.m0[k], self.m1[k]).max(0.0)
    }

    /// `Lambda(x)` of the beta events: the interpolated curve, the tails of
    /// the events truncated above `x`, and the exact survival of the events
    /// whose tail interval contains `x`.
    pub fn exceedance_betas(&self, x: f64) -> f64 {
        let loss = &self.loss;
        let m1 = loss.len();
        if x < loss[0] {
            return f64::NAN;
        }
        if x >= loss[m1 - 1] {
            return 0.0;
        }
        let k = interval_of(loss, x);
        let mut total = self.interpolate(x) + self.tail_above[k + 1];
        let first_group = k.saturating_sub(TAIL_INTERVALS - 1);
        for t in &self.tails[self.group_start[first_group]..self.group_start[k + 1]] {
            if x < t.max_loss {
                total += t.rate * BetaFn::new(t.alpha, t.beta).sf(x / t.max_loss);
            }
        }
        total
    }

    /// `Lambda(x)` with the atoms included; NaN below the lowest node.
    pub fn exceedance(&self, x: f64) -> f64 {
        let betas = self.exceedance_betas(x);
        let i = self.atom_loss.partition_point(|&v| v <= x);
        betas + self.atom_above[i]
    }

    /// The even grid at spacing `h` with `grid_length` nodes (derived to
    /// cover `max_loss` when `None`) under the overflow policy: the curve
    /// differenced at the half-bucket edges, the tails bucketed exactly per
    /// event in a chunk pass folded in order, and the atoms by the linear
    /// rule. Requires `h >= 2 min_loss`.
    pub fn grid(&self, h: f64, grid_length: Option<usize>, policy: Overflow, config: &Config) -> Result<GridOutcome, Error> {
        if !(h.is_finite() && h > 0.0) {
            return Err(Error::BadBucketSize(h));
        }
        if h < 2.0 * self.min_loss * (1.0 - 1e-12) {
            return Err(Error::BucketBelowCurve { bucket_size: h, min_loss: self.min_loss });
        }
        let auto_length = if self.max_loss > 0.0 { (self.max_loss / h).ceil() as usize + 1 } else { 1 };
        let n = match grid_length {
            None => auto_length,
            Some(0) => return Err(Error::BadGridLength(0)),
            Some(n) => n,
        };
        let top_node = (n - 1) as f64 * h;
        let top_edge = (n as f64 - 0.5) * h;
        if policy == Overflow::Error && self.max_loss > top_edge {
            return Err(Error::GridOverflow {
                grid_length: n,
                bucket_size: h,
                top: top_edge,
                max_loss: self.max_loss,
                required: auto_length,
                example_rows: Vec::new(),
            });
        }
        let inv_h = 1.0 / h;
        let mut secondary = vec![0.0_f64; n];
        let mut over = OverflowAccumulator::default();

        // The curve, differenced at the edges and made monotone against
        // rounding so the telescoping sum stays exact.
        if self.loss.len() > 1 {
            let mut prev = self.curve[0];
            let mut values = Vec::with_capacity(n);
            for i in 0..n {
                let e = (i as f64 + 0.5) * h;
                let v = self.interpolate(e).min(prev);
                values.push(v);
                prev = v;
            }
            secondary[0] += self.beta_rate - values[0] - self.tail_above[0];
            for i in 1..n {
                secondary[i] += values[i - 1] - values[i];
            }
            let rest = values[n - 1];
            if rest > 0.0 {
                over.rate.add(rest);
                if policy == Overflow::TopBucket {
                    secondary[n - 1] += rest;
                }
                // Its moment: the events truncated above the top edge.
                for t in &self.tails {
                    let xk = self.loss[t.node];
                    if xk <= top_edge {
                        continue;
                    }
                    let bf = BetaFn::new(t.alpha, t.beta);
                    let mass = t.rate * (bf.sf(top_edge / t.max_loss) - t.survival);
                    let moment = tail_moment(t.rate, t.max_loss, t.alpha, t.beta, top_edge)
                        - tail_moment(t.rate, t.max_loss, t.alpha, t.beta, xk);
                    over.moment.add(moment - top_node * mass);
                }
            }
        } else {
            secondary[0] += self.beta_rate - self.tail_above[0];
        }

        // The tails, per event, in chunks folded in order.
        let chunk = config.resolved_chunk_rows();
        let n_chunks = self.tails.len().div_ceil(chunk);
        let min_distance = 0.5 * self.loss[0];
        let process = |chunk_index: usize| -> Result<(Vec<f64>, OverflowAccumulator), Error> {
            let start = chunk_index * chunk;
            let end = (start + chunk).min(self.tails.len());
            let mut hist = vec![0.0_f64; n];
            let mut over = OverflowAccumulator::default();
            let mut edges = Vec::new();
            for t in &self.tails[start..end] {
                let (lam, m, s) = (t.rate, t.max_loss, t.survival);
                let xk = self.loss[t.node];
                let bf = BetaFn::new(t.alpha, t.beta);
                let first = ((xk / h - 0.5).floor() as isize + 1) as usize;
                let last_signed = (m / h - 0.5).ceil() as isize - 1;
                if last_signed < first as isize {
                    put(&mut hist, first, lam * s, policy, &mut over);
                } else {
                    let last = last_signed as usize;
                    edges.clear();
                    edges.extend((first..=last).map(|i| (i as f64 + 0.5) * h));
                    let tail = tail_survival(&bf, m, xk, s, &edges, min_distance);
                    put(&mut hist, first, lam * (s - tail[0]), policy, &mut over);
                    for (i, w) in (first + 1..=last).zip(tail.windows(2).map(|w| lam * (w[0] - w[1]))) {
                        put(&mut hist, i, w, policy, &mut over);
                    }
                    put(&mut hist, last + 1, lam * tail[last - first], policy, &mut over);
                }
                if m > top_edge {
                    let begin = xk.max(top_edge);
                    let mass = if begin > xk { lam * bf.sf(begin / m) } else { lam * s };
                    over.moment.add(tail_moment(lam, m, t.alpha, t.beta, begin) - top_node * mass);
                }
            }
            Ok((hist, over))
        };
        run_chunks(config, n_chunks, process, |(hist, part): (Vec<f64>, OverflowAccumulator)| {
            for (dst, src) in secondary.iter_mut().zip(&hist) {
                *dst += src;
            }
            over.merge(&part);
            Ok(())
        })?;

        // The atoms, by the linear rule.
        for (&x, &w) in self.atom_loss.iter().zip(&self.atom_rate) {
            place(&mut secondary, x, w, inv_h, top_edge, top_node, policy, &mut over);
        }
        Ok(GridOutcome {
            secondary,
            grid_length: n,
            overflow: over,
        })
    }
}

/// Deposit `w` into bucket `i`, or beyond the grid: the top node under the
/// `top_bucket` policy, nowhere under `drop`; the overflow rate either way.
#[inline]
fn put(hist: &mut [f64], i: usize, w: f64, policy: Overflow, over: &mut OverflowAccumulator) {
    let n = hist.len();
    if i < n {
        hist[i] += w;
    } else {
        over.rate.add(w);
        if policy == Overflow::TopBucket {
            hist[n - 1] += w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_nodes_hit_both_ends() {
        let nodes = geometric_nodes(0.5, 1e6, 150);
        assert_eq!(nodes[0], 0.5);
        assert_eq!(*nodes.last().unwrap(), 1e6);
        assert_eq!(nodes.len(), (150.0_f64 * (2e6_f64).log10()).ceil() as usize + 1);
        assert!(nodes.windows(2).all(|w| w[1] > w[0]));
        assert_eq!(geometric_nodes(2.0, 1.0, 150), vec![2.0]);
    }

    #[test]
    fn limited_hermite_is_monotone() {
        // Random decreasing data with wild slopes: the limited cubic never
        // rises inside any interval.
        let xs: Vec<f64> = (0..50).map(|i| 1.0 + i as f64).collect();
        let mut state = 12345_u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut ys = vec![10.0_f64];
        for _ in 1..50 {
            let last = *ys.last().unwrap();
            ys.push(last * (0.3 + 0.7 * next()));
        }
        for k in 0..49 {
            let d = (ys[k + 1] - ys[k]) / (xs[k + 1] - xs[k]);
            let (m0, m1) = limit_slopes(d, -20.0 * next(), -20.0 * next());
            let mut prev = ys[k];
            for j in 1..=100 {
                let t = j as f64 / 100.0;
                let v = hermite(t, ys[k], ys[k + 1], 1.0, m0, m1);
                assert!(v <= prev + 1e-12 * prev.abs(), "interval {k} rises at t={t}");
                prev = v;
            }
        }
    }

    #[test]
    fn tail_survival_matches_exact_near_the_endpoint() {
        // A J-shaped and a bell-shaped event: the endpoint interpolant
        // follows the exact survival to a few units in 1e-6 across its
        // tail interval (the regular factor R(y) varies most on the first
        // steps, farthest from the endpoint).
        for (a, b) in [(0.4_f64, 0.6_f64), (3.0, 5.0), (0.01, 30.0)] {
            let bf = BetaFn::new(a, b);
            let m = 1e6;
            let xk = 0.94 * m;
            let s = bf.sf(xk / m);
            let xs: Vec<f64> = (1..=600).map(|i| xk + (m - xk) * i as f64 / 600.0).collect();
            let got = tail_survival(&bf, m, xk, s, &xs, 0.25);
            for (x, g) in xs.iter().zip(&got) {
                let want = bf.sf(x / m);
                if want > 1e-300 {
                    assert!(
                        (g - want).abs() <= 5e-6 * want + 1e-300,
                        "a={a} b={b} x={x}: {g} vs {want}"
                    );
                }
            }
            assert!(got.windows(2).all(|w| w[1] <= w[0]));
        }
    }

    #[test]
    fn sample_intervals_follow_numpy_rounding() {
        assert_eq!(sample_intervals(10), (0..10).collect::<Vec<_>>());
        let picks = sample_intervals(541);
        assert_eq!(picks.len(), 64);
        assert_eq!(picks[0], 0);
        assert_eq!(*picks.last().unwrap(), 540);
        assert!(picks.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn curve_exceedance_matches_exact_and_grids_agree() {
        use crate::quantize::classify_rows;
        // Mixed rows: betas of several shapes, an atom, a Bernoulli row.
        let rows: Vec<[f64; 5]> = vec![
            [0.02, 100.0, 40.0, 15.0, 10.0],
            [0.01, 80.0, 20.0, 8.0, 6.0],
            [0.03, 100.0, 2.0, 6.0, 4.0],
            [0.005, 100.0, 30.0, 50.0, 0.0],
            [0.004, 50.0, 25.0, 0.0, 0.0],
            [0.002, 1000.0, 1.0, 20.0, 0.0],
        ];
        let cols: [Vec<f64>; 5] = std::array::from_fn(|c| rows.iter().map(|r| r[c]).collect());
        let input = EltInput {
            rate: &cols[0],
            max_loss: &cols[1],
            mean_loss: &cols[2],
            sd_indep: &cols[3],
            sd_corr: &cols[4],
        };
        let config = Config::new(0.5);
        let classified = classify_rows(&input, &config).unwrap();
        let curve = build_curve(&input, &config, &classified, 0.25).unwrap();
        assert_eq!(curve.nodes(), curve.loss.len());
        assert!((curve.total_rate - 0.071).abs() < 1e-15);
        // Exact Lambda of the betas at a sweep of losses.
        let betas: Vec<(f64, f64, f64, f64)> = [0, 1, 2, 5]
            .iter()
            .map(|&r| {
                let (a, b) = beta_params(rows[r][1], rows[r][2], (rows[r][3] + rows[r][4]).powi(2));
                (rows[r][0], rows[r][1], a, b)
            })
            .collect();
        for i in 0..400 {
            let x = 0.25 * (1.0_f64 + i as f64 / 40.0).powi(8);
            if x >= 1000.0 {
                break;
            }
            let exact: f64 = betas.iter().map(|&(lam, m, a, b)| lam * BetaFn::new(a, b).sf((x / m).min(1.0))).sum();
            let got = curve.exceedance_betas(x);
            assert!((got - exact).abs() <= 1e-4 * exact + 1e-16, "x={x}: {got} vs {exact}");
        }
        // At the nodes the stored rate is exact.
        for (i, &x) in curve.loss.iter().enumerate() {
            let exact: f64 = betas.iter().map(|&(lam, m, a, b)| lam * BetaFn::new(a, b).sf((x / m).min(1.0))).sum();
            assert!((curve.exceedance_rate[i] - exact).abs() <= 1e-12 * exact + 1e-16);
        }
        // Atoms: the Bernoulli pair and the degenerate row step in.
        let with_atoms = curve.exceedance(60.0);
        assert!((with_atoms - curve.exceedance_betas(60.0) - 0.005 * 0.3).abs() < 1e-15);
        // Grids at the build spacing and at twice it both carry the mass.
        let g1 = curve.grid(0.5, None, Overflow::TopBucket, &config).unwrap();
        let g2 = curve.grid(1.0, None, Overflow::TopBucket, &config).unwrap();
        let mass = |v: &[f64]| v.iter().sum::<f64>();
        assert!((mass(&g1.secondary) - 0.071).abs() < 1e-14);
        assert!((mass(&g2.secondary) - 0.071).abs() < 1e-14);
        assert!(matches!(curve.grid(0.25, None, Overflow::TopBucket, &config), Err(Error::BucketBelowCurve { .. })));
    }
}
