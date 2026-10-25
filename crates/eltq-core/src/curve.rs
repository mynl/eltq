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
//! `lambda_j S_j(x_k)` on `(x_k, M_j]`, is kept per event and walked along
//! the bucket edges by a monotone cubic Hermite of `log S_j` against
//! `log(M_j - x)` until its rate reaches `TAIL_RATE_FLOOR` of the total.
//! A **narrow** event, one standard deviation below a node interval at its
//! mean, is kept out of the curve, whose cubic cannot follow a step inside
//! one interval, and walked exactly on its bulk. The summed
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
/// The rate cut of a tail walk, as a fraction of the beta events' total
/// rate: the walk along the bucket edges stops at the first edge where the
/// event's exceedance rate is at or below it and the remainder is lumped
/// into the next bucket (spec section 5A). Shared with `_meta.py`.
pub const TAIL_RATE_FLOOR: f64 = 1e-20;
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

/// One beta event's survival on its tail interval `(x_k, M]`, the
/// endpoint interpolant: `log S` against `u = log(M - x)`, a monotone
/// cubic Hermite with the exact slopes `(1 - y) f(y) / S(y)` on nodes
/// shrinking the distance to the endpoint by sqrt(2) per step from
/// `M - x_k` down to `min_distance`, the limited end slopes fixed per
/// interval at construction, and the exact power law `S ~ d^beta` below
/// the last node. The first node's value is the exact `S(x_k)`, so the
/// tail joins the curve bitwise.
pub struct TailInterpolant {
    max_loss: f64,
    beta: f64,
    survival_at_node: f64,
    /// Nodes ascending in `u`, their `log S`, and the limited end slopes
    /// of each interval `[u[k], u[k + 1]]`.
    u: Vec<f64>,
    g: Vec<f64>,
    m0: Vec<f64>,
    m1: Vec<f64>,
}

impl TailInterpolant {
    pub fn new(bf: &BetaFn, m: f64, xk: f64, s: f64, min_distance: f64) -> Self {
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
        // Ascending in u = log d.
        let u: Vec<f64> = dist.iter().rev().map(|d| d.ln()).collect();
        let g: Vec<f64> = log_surv.iter().rev().copied().collect();
        let sl: Vec<f64> = slope.iter().rev().copied().collect();
        let intervals = u.len().saturating_sub(1);
        let mut m0 = Vec::with_capacity(intervals);
        let mut m1 = Vec::with_capacity(intervals);
        for k in 0..intervals {
            let du = u[k + 1] - u[k];
            let (lo, hi) = limit_slopes((g[k + 1] - g[k]) / du, sl[k], sl[k + 1]);
            m0.push(lo);
            m1.push(hi);
        }
        TailInterpolant { max_loss: m, beta: b, survival_at_node: s, u, g, m0, m1 }
    }

    /// The cursor `at_ascending` starts from: the topmost interval.
    pub fn start(&self) -> usize {
        self.u.len().saturating_sub(2)
    }

    /// `S` at distance `d = M - x > 0` on interval `k`, or by the power
    /// law below the first node.
    fn eval(&self, d: f64, k: usize) -> f64 {
        let (u, g) = (&self.u, &self.g);
        let uq = d.ln();
        let value = if uq < u[0] || u.len() == 1 {
            g[0] + self.beta * (uq - u[0])
        } else {
            let du = u[k + 1] - u[k];
            hermite((uq - u[k]) / du, g[k], g[k + 1], du, self.m0[k], self.m1[k])
        };
        value.exp().min(self.survival_at_node)
    }

    /// `S(x)` with the interval found by search; 0 at and beyond `M`, and
    /// everywhere when no node has a positive survival.
    pub fn at(&self, x: f64) -> f64 {
        let d = self.max_loss - x;
        if d <= 0.0 || self.u.is_empty() {
            return 0.0;
        }
        let k = self.u.partition_point(|&v| v <= d.ln()).saturating_sub(1).min(self.start());
        self.eval(d, k)
    }

    /// `S(x)` for queries ascending in `x`: the interval cursor moves down
    /// as the distance shrinks instead of being searched. Start it at
    /// `start()`.
    pub fn at_ascending(&self, x: f64, cursor: &mut usize) -> f64 {
        let d = self.max_loss - x;
        if d <= 0.0 || self.u.is_empty() {
            return 0.0;
        }
        let uq = d.ln();
        while *cursor > 0 && uq < self.u[*cursor] {
            *cursor -= 1;
        }
        self.eval(d, *cursor)
    }
}

/// `S(x)` of one beta event on its tail interval at the given losses, by
/// `TailInterpolant`.
pub fn tail_survival(bf: &BetaFn, m: f64, xk: f64, s: f64, x: &[f64], min_distance: f64) -> Vec<f64> {
    let interp = TailInterpolant::new(bf, m, xk, s, min_distance);
    x.iter().map(|&xq| interp.at(xq)).collect()
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

/// Steps of one standard deviation each side of the mean within which a
/// narrow event's bulk must reach the rate floor; an event wider than that
/// (its mass piled at an endpoint) takes the regular route.
pub const NARROW_STEPS: usize = 40;
/// Above this `alpha + beta` the density of a narrow event cannot be
/// resolved in double precision (one ulp of the loss share moves it by
/// more than a tenth), and the event is a step at its mean.
pub const NARROW_STEP_PARAMETER_SUM: f64 = 1e15;

/// 8-point Gauss-Legendre nodes (the positive half) and weights on [-1, 1].
const GL_NODES: [f64; 4] = [0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363];
const GL_WEIGHTS: [f64; 4] = [0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763];

/// The density of a narrow event relative to its value at the mean,
/// `f(y) / f(ybar)` in the loss share `y`, from the deviations by `log1p`
/// so that huge `alpha` and `beta` lose no precision, and its integrals by
/// 8-point Gauss-Legendre on pieces of at most half a standard deviation.
/// Within about 1e-12 of SciPy up to `alpha + beta` of 1e8 and within the
/// rounding of `y` itself beyond, where the incomplete beta's continued
/// fraction (400 iterations) returns garbage near the mean.
#[derive(Debug, Clone, Copy)]
pub struct NarrowKernel {
    alpha: f64,
    beta: f64,
    ybar: f64,
    piece: f64,
}

impl NarrowKernel {
    pub fn new(alpha: f64, beta: f64) -> Self {
        let ybar = alpha / (alpha + beta);
        let sd = (ybar * (1.0 - ybar) / (alpha + beta + 1.0)).sqrt();
        NarrowKernel { alpha, beta, ybar, piece: 0.5 * sd }
    }

    fn ratio(&self, y: f64) -> f64 {
        let d = y - self.ybar;
        ((self.alpha - 1.0) * (d / self.ybar).ln_1p() + (self.beta - 1.0) * (-d / (1.0 - self.ybar)).ln_1p()).exp()
    }

    /// `int_lo^hi f(y) / f(ybar) dy` and `int_lo^hi y f(y) / f(ybar) dy`.
    pub fn integrals(&self, lo: f64, hi: f64) -> (f64, f64) {
        if !(hi > lo) {
            return (0.0, 0.0);
        }
        let pieces = ((hi - lo) / self.piece).ceil().max(1.0) as usize;
        let width = (hi - lo) / pieces as f64;
        let (mut mass, mut moment) = (0.0, 0.0);
        for k in 0..pieces {
            let mid = lo + (k as f64 + 0.5) * width;
            let half = 0.5 * width;
            for (&node, &weight) in GL_NODES.iter().zip(&GL_WEIGHTS) {
                for y in [mid - half * node, mid + half * node] {
                    let r = half * weight * self.ratio(y);
                    mass += r;
                    moment += r * y;
                }
            }
        }
        (mass, moment)
    }

    pub fn integral(&self, lo: f64, hi: f64) -> f64 {
        self.integrals(lo, hi).0
    }
}

/// A beta event narrower than a node interval at its mean (spec section
/// 5A): kept out of the shared curve and walked on its bulk by the
/// quadrature of `NarrowKernel`, outside which it counts in full below and
/// nothing above, each within the rate floor.
#[derive(Debug, Clone, Copy)]
pub struct NarrowEvent {
    pub rate: f64,
    pub max_loss: f64,
    pub alpha: f64,
    pub beta: f64,
    /// The lower edge of the bulk: the first step down from the mean where
    /// the mass of a step falls to the floor, or 0.
    pub lower: f64,
    /// `max(lower, min_loss)`: where the walk and the curve queries begin.
    pub start: f64,
    /// The upper edge of the bulk: the first step up where the mass of a
    /// step falls to the floor, or the endpoint.
    pub stop: f64,
    /// The bulk's mass in the kernel's units, `int_lower^stop f / f(ybar)`.
    pub norm: f64,
    /// `S(start)`: 1 unless the bulk reaches below the lowest node.
    pub survival: f64,
}

impl NarrowEvent {
    pub fn kernel(&self) -> NarrowKernel {
        NarrowKernel::new(self.alpha, self.beta)
    }

    /// `S(x)`: in full below the bulk, nothing at or above it, and by
    /// quadrature from the nearer edge inside.
    pub fn survival_at(&self, kernel: &NarrowKernel, x: f64) -> f64 {
        // A step at the mean (lower == stop) counts nothing at its loss,
        // as an atom does.
        if x >= self.stop {
            return 0.0;
        }
        if x <= self.lower {
            return 1.0;
        }
        let m = self.max_loss;
        if x <= m * kernel.ybar {
            (1.0 - kernel.integral(self.lower / m, x / m) / self.norm).max(0.0)
        } else {
            kernel.integral(x / m, self.stop / m) / self.norm
        }
    }

    /// `lam P(X; X > x)` for the walk's overflow moment: the partial
    /// expectation above `x` by the same quadrature.
    pub fn partial_expectation(&self, kernel: &NarrowKernel, x: f64) -> f64 {
        if x >= self.stop {
            return 0.0;
        }
        let m = self.max_loss;
        let (_, moment) = kernel.integrals(x.max(self.lower) / m, self.stop / m);
        self.rate * m * moment / self.norm
    }
}

/// The bulk of a candidate narrow event by steps of `sigma` out from the
/// mean, each side ending at the first step whose mass is at or below a
/// tenth of the floor's share of the running mass (so the mass dropped
/// beyond it is below the floor), or at 0 and `M`: `(lower, stop)`, or
/// `None` when either side needs more than `NARROW_STEPS` steps.
fn narrow_bulk(kernel: &NarrowKernel, m: f64, mu: f64, sigma: f64, lam: f64, floor: f64) -> Option<(f64, f64)> {
    let cut = 0.1 * floor / lam;
    let side = |direction: f64, limit: f64| -> Option<f64> {
        let mut acc = 0.0;
        let mut prev = mu;
        for i in 1..=NARROW_STEPS {
            let x = mu + direction * i as f64 * sigma;
            let clipped = if direction < 0.0 { x.max(limit) } else { x.min(limit) };
            let (lo, hi) = if direction < 0.0 { (clipped, prev) } else { (prev, clipped) };
            let inc = kernel.integral(lo / m, hi / m);
            acc += inc;
            prev = clipped;
            if clipped == limit || inc <= cut * acc {
                return Some(clipped);
            }
        }
        None
    };
    let lower = side(-1.0, 0.0)?;
    let stop = side(1.0, m)?;
    Some((lower, stop))
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
    /// The narrow events sorted by `start`, with `narrow_above[i]` the
    /// total rate from index `i` on, the widest bulk, and the mass their
    /// walks deposit (`sum lam S(start)`).
    narrow: Vec<NarrowEvent>,
    narrow_above: Vec<f64>,
    narrow_max_width: f64,
    narrow_mass: f64,
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
    narrow: Vec<NarrowEvent>,
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

    // The narrow criterion, one node interval at the mean, and the rate
    // floor of the bulk search, on the beta rate summed here serially (the
    // fold below gives the reported value).
    let narrow_ratio = 10.0_f64.powf(1.0 / config.nodes_per_decade as f64) - 1.0;
    let mut serial_beta_rate = Neumaier::default();
    for row in 0..n_rows {
        if kinds[row] == RowKind::Beta {
            serial_beta_rate.add(input.rate[row]);
        }
    }
    let narrow_floor = TAIL_RATE_FLOOR * serial_beta_rate.value();

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
            narrow: Vec::new(),
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
                    // A NaN from the kernel is an unconverged continued
                    // fraction: a typed error, never a value (spec section 8).
                    let checked = |v: f64| -> Result<f64, Error> {
                        if v.is_nan() {
                            Err(Error::IncompleteBetaDiverged { row, alpha, beta })
                        } else {
                            Ok(v)
                        }
                    };
                    // A narrow event: its bulk, a few sigma wide, found by
                    // stepping out from the mean; kept out of the curve.
                    // Only bounded, log-concave densities (both parameters
                    // above 1) qualify. An event narrower than a node
                    // interval goes here, and so does one whose parameters
                    // are both large enough that the continued fraction is
                    // untrustworthy (`is_valid_regime`), whatever the node
                    // density; beyond NARROW_STEP_PARAMETER_SUM the event is
                    // a step at its mean.
                    let sigma = classified.var_eff[row].sqrt();
                    if (sigma < mu * narrow_ratio || !bf.is_valid_regime()) && alpha > 1.0 && beta > 1.0 {
                        let kernel = NarrowKernel::new(alpha, beta);
                        let bulk = if alpha + beta > NARROW_STEP_PARAMETER_SUM {
                            Some((mu, mu, 1.0))
                        } else {
                            narrow_bulk(&kernel, m, mu, sigma, lam, narrow_floor)
                                .map(|(lower, stop)| (lower, stop, kernel.integral(lower / m, stop / m)))
                        };
                        if let Some((lower, stop, norm)) = bulk.filter(|b| b.2 > 0.0 && b.2.is_finite()) {
                            let start = lower.max(min_loss);
                            let event = NarrowEvent {
                                rate: lam,
                                max_loss: m,
                                alpha,
                                beta,
                                lower,
                                start,
                                stop,
                                norm,
                                survival: if start > lower { kernel.integral(start / m, stop / m) / norm } else { 1.0 },
                            };
                            for (i, &x) in sample_x.iter().enumerate() {
                                p.sample_total[i] += lam * event.survival_at(&kernel, x);
                            }
                            if stop > min_loss {
                                // Its rate at the nodes: in full below the
                                // bulk, nothing above it.
                                for (i, &x) in loss.iter().enumerate() {
                                    if x >= stop {
                                        break;
                                    }
                                    p.node_tail[i] += lam * event.survival_at(&kernel, x);
                                }
                                p.narrow.push(event);
                            }
                            continue;
                        }
                    }
                    // loss[k] < m <= loss[k + 1], then TAIL_INTERVALS - 1 lower.
                    let above = loss.partition_point(|&v| v < m);
                    let k_signed = above as isize - 1 - (TAIL_INTERVALS as isize - 1);
                    let (k, xk, s) = if k_signed < 0 {
                        (0_usize, loss[0], checked(bf.sf(loss[0] / m))?)
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
                                surv.push(checked(sf)?);
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
                        p.node_tail[i] += lam * checked(bf.sf(loss[i] / m))?;
                    }
                    // The diagnostic samples: exact curve and exact total.
                    for (i, &x) in sample_x.iter().enumerate() {
                        let sf = checked(bf.sf((x / m).min(1.0)))?;
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
    let mut narrow: Vec<NarrowEvent> = Vec::new();
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
        narrow.extend(p.narrow);
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
    // The narrow events by start (stable), their suffix rates, the widest
    // bulk, and the mass their walks deposit.
    narrow.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
    let mut narrow_above = vec![0.0_f64; narrow.len() + 1];
    for i in (0..narrow.len()).rev() {
        narrow_above[i] = narrow_above[i + 1] + narrow[i].rate;
    }
    let narrow_max_width = narrow.iter().map(|t| t.stop - t.start).fold(0.0_f64, f64::max);
    let mut narrow_mass = Neumaier::default();
    for t in &narrow {
        narrow_mass.add(t.rate * t.survival);
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
        narrow,
        narrow_above,
        narrow_max_width,
        narrow_mass: narrow_mass.value(),
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

    /// The beta events kept out of the curve as narrow.
    pub fn narrow_events(&self) -> usize {
        self.narrow.len()
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
    /// the events truncated above `x`, the exact survival of the events
    /// whose tail interval contains `x`, and the narrow events (in full
    /// above `x`, exact where `x` is in the bulk).
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
        let i = self.narrow.partition_point(|t| t.start <= x);
        total += self.narrow_above[i];
        let lowest = x - self.narrow_max_width;
        for t in self.narrow[..i].iter().rev() {
            if t.start < lowest {
                break;
            }
            if x < t.stop {
                total += t.rate * t.survival_at(&t.kernel(), x);
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
    /// differenced at the half-bucket edges, the tails and the narrow
    /// events walked per event to the rate floor in chunk passes folded in
    /// order, and the atoms by the linear rule. Requires `h >= 2 min_loss`.
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
            secondary[0] += self.beta_rate - values[0] - self.tail_above[0] - self.narrow_mass;
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
            secondary[0] += self.beta_rate - self.tail_above[0] - self.narrow_mass;
        }

        // The tails and the narrow events, per event, in chunk passes
        // folded in order: each chunk's histogram covers only the bucket
        // span its events can touch (the top node stands in for anything
        // beyond the grid), so a chunk costs its span, not the grid length.
        let chunk = config.resolved_chunk_rows();
        let min_distance = 0.5 * self.loss[0];
        let threshold = TAIL_RATE_FLOOR * self.beta_rate;
        let top = n - 1;
        // The edges e_first ..= e_stop of a walk from `start` to `end` lie
        // in (start, end) and on the grid.
        let edge_range = |start: f64, end: f64| -> (usize, isize) {
            let first = ((start / h - 0.5).floor() as isize + 1) as usize;
            let stop = ((end / h - 0.5).ceil() as isize - 1).min(n as isize - 1);
            (first, stop)
        };
        let span = |ranges: &mut dyn Iterator<Item = (usize, isize)>| -> (usize, Vec<f64>) {
            let (mut lo, mut hi) = (top, 0_usize);
            for (first, stop) in ranges {
                lo = lo.min(first.min(top));
                hi = hi.max(((stop + 1).max(first as isize) as usize).min(top));
            }
            (lo, vec![0.0_f64; hi + 1 - lo])
        };
        // A remainder beyond the grid (so the endpoint exceeds the top
        // edge): its moment by partial expectations.
        let overflow_moment = |lam: f64, m: f64, bf: &BetaFn, start: f64, s: f64, over: &mut OverflowAccumulator| {
            let begin = start.max(top_edge);
            let mass = if begin > start { lam * bf.sf(begin / m) } else { lam * s };
            over.moment.add(tail_moment(lam, m, bf.alpha, bf.beta, begin) - top_node * mass);
        };
        let fold = |secondary: &mut [f64], over: &mut OverflowAccumulator, (lo, hist, part): (usize, Vec<f64>, OverflowAccumulator)| {
            for (dst, src) in secondary[lo..].iter_mut().zip(&hist) {
                *dst += src;
            }
            over.merge(&part);
        };

        let n_chunks = self.tails.len().div_ceil(chunk);
        let process = |chunk_index: usize| -> Result<(usize, Vec<f64>, OverflowAccumulator), Error> {
            let start = chunk_index * chunk;
            let end = (start + chunk).min(self.tails.len());
            let tails = &self.tails[start..end];
            let (lo, mut hist) = span(&mut tails.iter().map(|t| edge_range(self.loss[t.node], t.max_loss)));
            let mut over = OverflowAccumulator::default();
            for t in tails {
                let (lam, m, s) = (t.rate, t.max_loss, t.survival);
                let xk = self.loss[t.node];
                let bf = BetaFn::new(t.alpha, t.beta);
                let (first, stop) = edge_range(xk, m);
                let rest = if stop < first as isize || lam * s <= threshold {
                    // No edge to walk, or the whole tail at or below the
                    // floor: one deposit in the bucket containing x_k.
                    put(&mut hist, lo, n, first, lam * s, policy, &mut over);
                    first
                } else {
                    let interp = TailInterpolant::new(&bf, m, xk, s, min_distance);
                    let mut cursor = interp.start();
                    walk_edges(&mut hist, lo, n, first, stop, lam, s, h, threshold, policy, &mut over, |x| {
                        interp.at_ascending(x, &mut cursor)
                    })
                };
                if rest >= n {
                    overflow_moment(lam, m, &bf, xk, s, &mut over);
                }
            }
            Ok((lo, hist, over))
        };
        run_chunks(config, n_chunks, process, |part| {
            fold(&mut secondary, &mut over, part);
            Ok(())
        })?;

        // The narrow events, exactly on their bulk.
        let n_chunks = self.narrow.len().div_ceil(chunk);
        let process = |chunk_index: usize| -> Result<(usize, Vec<f64>, OverflowAccumulator), Error> {
            let start = chunk_index * chunk;
            let end = (start + chunk).min(self.narrow.len());
            let events = &self.narrow[start..end];
            let (lo, mut hist) = span(&mut events.iter().map(|t| edge_range(t.start, t.stop)));
            let mut over = OverflowAccumulator::default();
            for t in events {
                let (lam, m, s) = (t.rate, t.max_loss, t.survival);
                let kernel = t.kernel();
                let (first, stop) = edge_range(t.start, t.stop);
                let rest = if stop < first as isize || lam * s <= threshold {
                    put(&mut hist, lo, n, first, lam * s, policy, &mut over);
                    first
                } else {
                    // Incremental: each edge's survival is the previous
                    // less the bucket's mass, one short quadrature.
                    let (mut prev_x, mut prev_s) = (t.start, s);
                    walk_edges(&mut hist, lo, n, first, stop, lam, s, h, threshold, policy, &mut over, |x| {
                        prev_s = (prev_s - kernel.integral(prev_x / m, x.min(t.stop) / m) / t.norm).max(0.0);
                        prev_x = x;
                        if x >= t.stop { 0.0 } else { prev_s }
                    })
                };
                if rest >= n {
                    // The remainder beyond the grid: its moment by the
                    // kernel's partial expectation.
                    let begin = t.start.max(top_edge);
                    let mass = lam * t.survival_at(&kernel, begin);
                    over.moment.add(t.partial_expectation(&kernel, begin) - top_node * mass);
                }
            }
            Ok((lo, hist, over))
        };
        run_chunks(config, n_chunks, process, |part| {
            fold(&mut secondary, &mut over, part);
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

/// Walk one event's survival along the edges `first ..= stop` (given
/// `first <= stop` and `lam s` above the floor): deposit `lam` times the
/// drop across each edge, break at the last edge or at the first edge
/// where `lam S` reaches the floor, and lump the remainder in the next
/// bucket; returns that bucket.
#[allow(clippy::too_many_arguments)]
fn walk_edges(
    hist: &mut [f64],
    lo: usize,
    n: usize,
    first: usize,
    stop: isize,
    lam: f64,
    s: f64,
    h: f64,
    threshold: f64,
    policy: Overflow,
    over: &mut OverflowAccumulator,
    mut survival_at: impl FnMut(f64) -> f64,
) -> usize {
    let mut prev = s;
    let mut i = first;
    loop {
        let surv = survival_at((i as f64 + 0.5) * h);
        put(hist, lo, n, i, lam * (prev - surv), policy, over);
        prev = surv;
        if i as isize == stop || lam * surv <= threshold {
            break;
        }
        i += 1;
    }
    put(hist, lo, n, i + 1, lam * prev, policy, over);
    i + 1
}

/// Deposit `w` into bucket `i` of a grid of `n`, through a histogram that
/// starts at bucket `lo` and reaches the top node whenever `i` can exceed
/// the grid; beyond the grid, the top node under the `top_bucket` policy,
/// nowhere under `drop`, and the overflow rate either way.
#[inline]
fn put(hist: &mut [f64], lo: usize, n: usize, i: usize, w: f64, policy: Overflow, over: &mut OverflowAccumulator) {
    if i < n {
        hist[i - lo] += w;
    } else {
        over.rate.add(w);
        if policy == Overflow::TopBucket {
            hist[n - 1 - lo] += w;
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
    fn narrow_kernel_matches_incomplete_beta_where_that_converges() {
        // CV 1e-2 and 3e-3 at mean 0.25: the continued fraction converges
        // (to about 1e-10 of its own at alpha of 2e4) and the quadrature
        // agrees with it to 1e-9; at CV 1e-5 the fraction fails and the
        // quadrature survival is still monotone in [0, 1] with the half at
        // the mean.
        let m = 1e6;
        let mu = 2.5e5;
        for (cv, check) in [(1e-2, true), (3e-3, true), (1e-5, false)] {
            let sd = cv * mu;
            let (alpha, beta) = beta_params(m, mu, sd * sd);
            let kernel = NarrowKernel::new(alpha, beta);
            let (lower, stop) = narrow_bulk(&kernel, m, mu, sd, 1.0, 1e-20, ).unwrap();
            let norm = kernel.integral(lower / m, stop / m);
            let event = NarrowEvent { rate: 1.0, max_loss: m, alpha, beta, lower, start: lower, stop, norm, survival: 1.0 };
            let mut prev = 1.0;
            for k in -10..=10 {
                let x = mu + k as f64 * sd;
                let s = event.survival_at(&kernel, x);
                assert!((0.0..=1.0).contains(&s) && s <= prev, "cv={cv} k={k}: {s}");
                prev = s;
                if check {
                    let want = BetaFn::new(alpha, beta).sf(x / m);
                    assert!((s - want).abs() <= 1e-9 * want.max(1e-300) + 1e-14, "cv={cv} k={k}: {s} vs {want}");
                }
            }
            let half = event.survival_at(&kernel, mu);
            assert!((half - 0.5).abs() < 2e-3 * (1.0 + cv * 100.0), "cv={cv}: {half}");
        }
    }

    #[test]
    fn tail_walk_with_cursor_matches_search() {
        // Ascending queries through the cursor give the searched values
        // bitwise, across every interval and into the power law.
        let bf = BetaFn::new(0.7, 40.0);
        let (m, xk) = (1e5, 0.92e5);
        let interp = TailInterpolant::new(&bf, m, xk, bf.sf(xk / m), 0.5);
        let mut cursor = interp.start();
        let mut prev = f64::INFINITY;
        for i in 0..5000 {
            let x = xk + (m - xk) * (i as f64 + 0.5) / 5000.0;
            let walked = interp.at_ascending(x, &mut cursor);
            assert_eq!(walked, interp.at(x), "x={x}");
            assert!(walked <= prev);
            prev = walked;
        }
        assert_eq!(interp.at(m), 0.0);
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
