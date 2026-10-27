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
use crate::normal;
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
/// `2^-60`: the upper skip of a bell-shaped event on its mesh. From the
/// first node where its survival is below this, `lam (S - s)^+` is below
/// `2^-60 lam` (the truncation value `s` is smaller still) and nothing is
/// added, so a requesting event costs its window, not the mesh.
pub const UPPER_SKIP_SURVIVAL: f64 = 8.673617379884035e-19;
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

/// Where the endpoint expansion's first correction `c t`, with
/// `c = (alpha - 1) beta / (beta + 1)` and `t = (M - x) / M`, lies in this
/// band and `c > 1`, the tail ladder steps by 2^(1/4) instead of sqrt(2):
/// the cubic in `u` of `log(1 - c t)` was off by 3e-4 at beta = 5 with
/// sqrt(2) steps (6e-6 at beta = 0.3), 16 times less with the finer step.
pub const TAIL_FINE_ZONE: (f64, f64) = (1e-2, 1e2);
/// The power law below the ladder's last node applies only where `c t` at
/// that node is at most this: the relative error of extrapolating by the
/// pure power law is `c t_L` (0.17 on Beta(1e6, 1/2) at t_L = 5e-7, which
/// moved 0.63 of the event's rate across the wrong edge; peer review,
/// finding A2). Otherwise the queried point is evaluated exactly.
pub const TAIL_POWER_LAW_LIMIT: f64 = 1e-6;

/// One beta event's survival on its tail interval `(x_k, M]`, the
/// endpoint interpolant: `log S` against `u = log(M - x)`, a monotone
/// cubic Hermite with the exact slopes `(1 - y) f(y) / S(y)` on nodes
/// shrinking the distance to the endpoint by sqrt(2) per step from
/// `M - x_k` down to `min_distance` (by 2^(1/4) across `TAIL_FINE_ZONE`
/// when `c > 1`), the limited end slopes fixed per interval at
/// construction, and below the last node the exact power law `S ~ d^beta`
/// where `c t_L <= TAIL_POWER_LAW_LIMIT`, else the exact survival. The
/// first node's value is the exact `S(x_k)`, so the tail joins the curve
/// bitwise. The nodes are `d_0 2^(-k / 4)` for a shared integer sequence
/// `k`, so both backends place them bitwise alike.
pub struct TailInterpolant {
    bf: BetaFn,
    max_loss: f64,
    beta: f64,
    survival_at_node: f64,
    /// Whether the power law holds below the last node.
    asymptotic: bool,
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
        let c = (a - 1.0) * b / (b + 1.0);
        // Nodes with a positive survival, descending in distance: the k-th
        // at d0 2^(-k / 4), k advancing by 2 (sqrt(2)) or by 1 across the
        // fine zone, down to min_distance and at least three nodes.
        let mut dist = Vec::new();
        let mut log_surv = Vec::new();
        let mut slope = Vec::new();
        let mut k = 0_u32;
        let mut d_last = d0;
        loop {
            let d = d0 * 0.5_f64.powf(k as f64 / 4.0);
            let y = 1.0 - d / m;
            let surv = if k == 0 { s } else { bf.sf(y) };
            if !(surv > 0.0) {
                break;
            }
            let ls = surv.ln();
            dist.push(d);
            log_surv.push(ls);
            slope.push(((a - 1.0) * y.ln() + b * (d / m).ln() - bf.ln_beta() - ls).exp());
            d_last = d;
            if d <= min_distance && dist.len() >= 3 {
                break;
            }
            let ct = c * d / m;
            k += if c > 1.0 && ct >= TAIL_FINE_ZONE.0 && ct <= TAIL_FINE_ZONE.1 { 1 } else { 2 };
        }
        let asymptotic = c * d_last / m <= TAIL_POWER_LAW_LIMIT;
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
        TailInterpolant { bf: *bf, max_loss: m, beta: b, survival_at_node: s, asymptotic, u, g, m0, m1 }
    }

    /// The ladder's node count (for the tests).
    pub fn nodes(&self) -> usize {
        self.u.len()
    }

    /// The cursor `at_ascending` starts from: the topmost interval.
    pub fn start(&self) -> usize {
        self.u.len().saturating_sub(2)
    }

    /// `S` at distance `d = M - x > 0`, `uq = ln d`, on interval `k`; below
    /// the first node by the power law where it holds, else exactly.
    fn eval(&self, d: f64, uq: f64, k: usize) -> f64 {
        let (u, g) = (&self.u, &self.g);
        if uq < u[0] || u.len() == 1 {
            if !self.asymptotic {
                return self.bf.sf(1.0 - d / self.max_loss).min(self.survival_at_node);
            }
            return (g[0] + self.beta * (uq - u[0])).exp().min(self.survival_at_node);
        }
        let du = u[k + 1] - u[k];
        hermite((uq - u[k]) / du, g[k], g[k + 1], du, self.m0[k], self.m1[k]).exp().min(self.survival_at_node)
    }

    /// `S(x)` with the interval found by search; 0 at and beyond `M`, and
    /// everywhere when no node has a positive survival.
    pub fn at(&self, x: f64) -> f64 {
        let d = self.max_loss - x;
        if d <= 0.0 || self.u.is_empty() {
            return 0.0;
        }
        let uq = d.ln();
        let k = self.u.partition_point(|&v| v <= uq).saturating_sub(1).min(self.start());
        self.eval(d, uq, k)
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
        self.eval(d, uq, *cursor)
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
/// The narrow route's quadrature needs a density whose derivative is finite
/// at the endpoints, so both parameters at least 2: between 1 and 2 the
/// 8-point rule was off by up to 5e-4 relative (measured at alpha = 100,
/// beta from 1.02 to 1.6), where the shared curve and the tail ladder are
/// within 1e-5. Below this the event stays on a curve.
pub const NARROW_MIN_PARAMETER: f64 = 2.0;
/// From this smaller parameter a narrow event's kernel is the standardized
/// normal law in closed form instead of the quadrature of its density. The
/// density's two logarithms are each of size about `40 sqrt(alpha beta /
/// (alpha + beta))` at the edge of the bulk and cancel, so the quadrature
/// kernel's error in `log f` is about `4e-15 sqrt(min(alpha, beta))`; the
/// normal's relative error is its skewness, `2 / sqrt(min(alpha, beta))`,
/// times a polynomial in `z`. The two cross near 1e16 at about 1e-6 each
/// (`dev/done/plan-peer-review-response.md`, [atoms]). A row too narrow for
/// either, `sigma < 2^-48 mu`, is an atom (`validate::ATOM_RESOLUTION`).
pub const NARROW_NORMAL_FROM: f64 = 1e16;

/// 8-point Gauss-Legendre nodes (the positive half) and weights on [-1, 1].
const GL_NODES: [f64; 4] = [0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363];
const GL_WEIGHTS: [f64; 4] = [0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763];

/// The density of a narrow event relative to its value at the mean,
/// `f(y) / f(ybar)` in the loss share `y`, from the deviations by `log1p`
/// so that huge `alpha` and `beta` lose no precision, and its integrals by
/// 8-point Gauss-Legendre on pieces of at most half a standard deviation;
/// from `NARROW_NORMAL_FROM` the standardized normal law, whose integrals
/// are closed forms in `erf` and `exp`.
/// Within about 1e-12 of SciPy up to `alpha + beta` of 1e8 and within the
/// rounding of `y` itself beyond, where the incomplete beta's continued
/// fraction (400 iterations) returns garbage near the mean.
#[derive(Debug, Clone, Copy)]
pub struct NarrowKernel {
    alpha: f64,
    beta: f64,
    ybar: f64,
    sd: f64,
    piece: f64,
    /// The normal law from `NARROW_NORMAL_FROM`.
    normal: bool,
}

impl NarrowKernel {
    pub fn new(alpha: f64, beta: f64) -> Self {
        let ybar = alpha / (alpha + beta);
        let sd = (ybar * (1.0 - ybar) / (alpha + beta + 1.0)).sqrt();
        let normal = alpha.min(beta) >= NARROW_NORMAL_FROM;
        NarrowKernel { alpha, beta, ybar, sd, piece: 0.5 * sd, normal }
    }

    /// Whether the kernel is the normal law (for the tests and the meta).
    pub fn is_normal(&self) -> bool {
        self.normal
    }

    /// The mean of the loss share, `alpha / (alpha + beta)`.
    pub fn mean(&self) -> f64 {
        self.ybar
    }

    /// The standard deviation of the loss share.
    pub fn sd(&self) -> f64 {
        self.sd
    }

    /// The normal law's `int_lo^hi exp(-z^2 / 2) dy` and the same with `y`,
    /// `z = (y - ybar) / sd`, through `erfc` on the side where the tail is
    /// thin so that a far upper-tail difference keeps its relative precision.
    fn normal_integrals(&self, lo: f64, hi: f64) -> (f64, f64) {
        let (zl, zh) = ((lo - self.ybar) / self.sd, (hi - self.ybar) / self.sd);
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let phi_diff = if zl >= 0.0 {
            normal::erfc(zl * r) - normal::erfc(zh * r)
        } else if zh <= 0.0 {
            normal::erfc(-zh * r) - normal::erfc(-zl * r)
        } else {
            normal::erf(zh * r) - normal::erf(zl * r)
        };
        let mass = self.sd * (0.5 * std::f64::consts::PI).sqrt() * phi_diff;
        let moment = self.ybar * mass + self.sd * self.sd * ((-0.5 * zl * zl).exp() - (-0.5 * zh * zh).exp());
        (mass, moment)
    }

    fn ratio(&self, y: f64) -> f64 {
        let d = y - self.ybar;
        ((self.alpha - 1.0) * (d / self.ybar).ln_1p() + (self.beta - 1.0) * (-d / (1.0 - self.ybar)).ln_1p()).exp()
    }

    /// The logarithm of the kernel's density relative to its value at the
    /// mean (the normal law's `-z^2 / 2` from `NARROW_NORMAL_FROM`); divided
    /// by the kernel's integral it is the density.
    pub fn ln_relative_density(&self, y: f64) -> f64 {
        if self.normal {
            let z = (y - self.ybar) / self.sd;
            return -0.5 * z * z;
        }
        let d = y - self.ybar;
        (self.alpha - 1.0) * (d / self.ybar).ln_1p() + (self.beta - 1.0) * (-d / (1.0 - self.ybar)).ln_1p()
    }

    /// `int_lo^hi f(y) / f(ybar) dy` and `int_lo^hi y f(y) / f(ybar) dy`.
    pub fn integrals(&self, lo: f64, hi: f64) -> (f64, f64) {
        if !(hi > lo) {
            return (0.0, 0.0);
        }
        if self.normal {
            return self.normal_integrals(lo, hi);
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

    /// `int_lo^hi (y - c)^k f(y) / f(ybar) dy` for `k = 0, 1, 2, 3`, the
    /// centred moments the tail measures of spec section 4B need, by the same
    /// quadrature as `integrals` (every integrand nonnegative for `c <= lo`,
    /// so nothing cancels), or under the normal law in closed form.
    ///
    /// Notes
    /// -----
    /// Under the normal law, with `z = (y - ybar) / sd` and `e = (c - ybar) /
    /// sd`, the integral is `sd^(k+1) sqrt(2 pi) J_k` with `J_k = int_zl^zh (z
    /// - e)^k phi(z) dz`, and `phi' = -z phi` integrated by parts gives
    /// `J_(k+1) = (zl - e)^k phi(zl) - (zh - e)^k phi(zh) + k J_(k-1) - e
    /// J_k` from `J_0 = Phi(zh) - Phi(zl)`.
    pub fn centred_integrals(&self, lo: f64, hi: f64, c: f64) -> [f64; 4] {
        if !(hi > lo) {
            return [0.0; 4];
        }
        if self.normal {
            let (zl, zh, e) = ((lo - self.ybar) / self.sd, (hi - self.ybar) / self.sd, (c - self.ybar) / self.sd);
            let r = std::f64::consts::FRAC_1_SQRT_2;
            let j0 = 0.5
                * if zl >= 0.0 {
                    normal::erfc(zl * r) - normal::erfc(zh * r)
                } else if zh <= 0.0 {
                    normal::erfc(-zh * r) - normal::erfc(-zl * r)
                } else {
                    normal::erf(zh * r) - normal::erf(zl * r)
                };
            let phi = |z: f64| (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
            let (pl, ph) = (phi(zl), phi(zh));
            let (dl, dh) = (zl - e, zh - e);
            let j1 = pl - ph - e * j0;
            let j2 = dl * pl - dh * ph + j0 - e * j1;
            let j3 = dl * dl * pl - dh * dh * ph + 2.0 * j1 - e * j2;
            let scale = self.sd * (2.0 * std::f64::consts::PI).sqrt();
            return [
                scale * j0.max(0.0),
                scale * self.sd * j1.max(0.0),
                scale * self.sd * self.sd * j2.max(0.0),
                scale * self.sd * self.sd * self.sd * j3.max(0.0),
            ];
        }
        let pieces = ((hi - lo) / self.piece).ceil().max(1.0) as usize;
        let width = (hi - lo) / pieces as f64;
        let mut out = [0.0_f64; 4];
        for k in 0..pieces {
            let mid = lo + (k as f64 + 0.5) * width;
            let half = 0.5 * width;
            for (&node, &weight) in GL_NODES.iter().zip(&GL_WEIGHTS) {
                for y in [mid - half * node, mid + half * node] {
                    let w = half * weight * self.ratio(y);
                    let v = y - c;
                    out[0] += w;
                    out[1] += w * v;
                    out[2] += w * v * v;
                    out[3] += w * v * v * v;
                }
            }
        }
        out
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

    /// `S(x)`: in full below the bulk, nothing at or above it, and by the
    /// kernel's integral from the nearer edge inside.
    pub fn survival_at(&self, kernel: &NarrowKernel, x: f64) -> f64 {
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
pub(crate) fn narrow_bulk(kernel: &NarrowKernel, m: f64, mu: f64, sigma: f64, lam: f64, floor: f64) -> Option<(f64, f64)> {
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

/// Where a beta event is carried (spec section 5A): the base curve, the
/// fine curve (a bell-shaped event narrower than `nodes_per_sigma` base
/// intervals, which asked for density around its mean), or the narrow route
/// (one narrower than a base interval, or beyond the continued fraction's
/// regime).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Regular,
    Requesting,
    Narrow,
}

/// The base mesh refined: base interval `i` split into `splits[i]` equal
/// pieces in `ln x`, the base nodes kept as they are (so `fine[fine_of_base[i]]
/// == base[i]` bitwise). Both backends place the interior nodes with the
/// same arithmetic, `exp(u0 + k (u1 - u0) / s)`.
pub fn refine_mesh(base: &[f64], splits: &[usize]) -> (Vec<f64>, Vec<usize>) {
    let mut fine = Vec::with_capacity(splits.iter().sum::<usize>() + 1);
    let mut fine_of_base = Vec::with_capacity(base.len());
    for i in 0..base.len() - 1 {
        fine_of_base.push(fine.len());
        fine.push(base[i]);
        let s = splits[i];
        if s > 1 {
            let (u0, u1) = (base[i].ln(), base[i + 1].ln());
            for k in 1..s {
                fine.push((u0 + k as f64 * (u1 - u0) / s as f64).exp());
            }
        }
    }
    fine_of_base.push(fine.len());
    fine.push(base[base.len() - 1]);
    (fine, fine_of_base)
}

/// One beta event's truncated survival `lam (S - s)^+` and slope `-x
/// Lambda'` onto a mesh up to its truncation node `k`, where `S = s`: the
/// plateau below the lower skip as one difference (`+` at node 0, `-` at the
/// first evaluated node, prefix-summed at the fold), the nodes from there to
/// `k` evaluated, the slope at `k` on the incoming side only, and for a
/// bell-shaped event nothing from the upper skip on.
#[allow(clippy::too_many_arguments)]
fn deposit_event<F: Fn(f64) -> Result<f64, Error>>(
    bf: &BetaFn,
    m: f64,
    lam: f64,
    k: usize,
    s: f64,
    mesh: &[f64],
    curve: &mut [f64],
    slope_both: &mut [f64],
    slope_end: &mut [f64],
    plateau: &mut [f64],
    bell: bool,
    checked: &F,
) -> Result<(), Error> {
    let alpha = bf.alpha;
    // Lower skip: below the closed-form lower-tail quantile at 2^-60 the
    // survival is exactly 1.
    let ln_plateau = (LN_LOWER_SKIP_PROBABILITY + alpha.ln() + bf.ln_beta()) / alpha;
    let plateau_y = if ln_plateau > -745.0 { ln_plateau.exp() } else { 0.0 };
    let first = mesh[..=k].partition_point(|&x| x / m < plateau_y);
    if first > 0 {
        plateau[0] += lam * (1.0 - s);
        plateau[first] -= lam * (1.0 - s);
    }
    for i in first..=k {
        let (sf, density_y) = bf.sf_and_density_times_y(mesh[i] / m);
        let sf = checked(sf)?;
        if bell && sf < UPPER_SKIP_SURVIVAL {
            break;
        }
        curve[i] += lam * (sf - s);
        if i < k {
            slope_both[i] += lam * density_y;
        } else {
            slope_end[k] += lam * density_y;
        }
    }
    Ok(())
}

/// A monotone cubic Hermite with the limited slopes `m0`, `m1` per interval
/// of `loss`, at `x`: NaN below the first node, 0 at and beyond the last,
/// exact at a node.
fn cubic_at(loss: &[f64], curve: &[f64], m0: &[f64], m1: &[f64], x: f64) -> f64 {
    let n = loss.len();
    if x < loss[0] {
        return f64::NAN;
    }
    if x >= loss[n - 1] {
        return 0.0;
    }
    let k = interval_of(loss, x);
    if loss[k] == x {
        return curve[k];
    }
    let dx = loss[k + 1] - loss[k];
    hermite((x - loss[k]) / dx, curve[k], curve[k + 1], dx, m0[k], m1[k]).max(0.0)
}

/// The interval cursor of a cubic mesh for queries ascending in `x`: it
/// moves up through the intervals instead of searching for each query, and
/// returns exactly what `cubic_at` returns.
#[derive(Default)]
struct CubicCursor {
    k: usize,
}

impl CubicCursor {
    fn at_ascending(&mut self, loss: &[f64], curve: &[f64], m0: &[f64], m1: &[f64], x: f64) -> f64 {
        let n = loss.len();
        if x < loss[0] {
            return f64::NAN;
        }
        if x >= loss[n - 1] {
            return 0.0;
        }
        while loss[self.k + 1] <= x {
            self.k += 1;
        }
        let k = self.k;
        if loss[k] == x {
            return curve[k];
        }
        let dx = loss[k + 1] - loss[k];
        hermite((x - loss[k]) / dx, curve[k], curve[k + 1], dx, m0[k], m1[k]).max(0.0)
    }
}

/// `int_e^{x_m}` of that cubic: the partial interval holding `e` by the
/// antiderivatives of the four basis functions, then `dx (y0 + y1) / 2 +
/// dx^2 (m0 - m1) / 12` on each whole interval above it.
fn cubic_integral_above(loss: &[f64], curve: &[f64], m0: &[f64], m1: &[f64], e: f64) -> f64 {
    let n = loss.len();
    if n < 2 || e >= loss[n - 1] {
        return 0.0;
    }
    let e = e.max(loss[0]);
    let k = interval_of(loss, e);
    let mut total = Neumaier::default();
    let dx = loss[k + 1] - loss[k];
    let t0 = (e - loss[k]) / dx;
    let (t2, t3, t4) = (t0 * t0, t0 * t0 * t0, t0 * t0 * t0 * t0);
    let h00 = 0.5 - (0.5 * t4 - t3 + t0);
    let h10 = 1.0 / 12.0 - (0.25 * t4 - 2.0 * t3 / 3.0 + 0.5 * t2);
    let h01 = 0.5 - (-0.5 * t4 + t3);
    let h11 = -1.0 / 12.0 - (0.25 * t4 - t3 / 3.0);
    total.add(dx * (h00 * curve[k] + h01 * curve[k + 1] + dx * (h10 * m0[k] + h11 * m1[k])));
    for i in k + 1..n - 1 {
        let dx = loss[i + 1] - loss[i];
        total.add(dx * (0.5 * (curve[i] + curve[i + 1]) + dx * (m0[i] - m1[i]) / 12.0));
    }
    total.value().max(0.0)
}

/// Limited one-sided slopes per interval of a mesh from the accumulated
/// `-x Lambda'` (both sides, and the incoming side alone at truncation
/// nodes).
fn limited_slopes(loss: &[f64], curve: &[f64], slope_both: &[f64], slope_end: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let intervals = loss.len().saturating_sub(1);
    let mut m0 = vec![0.0_f64; intervals];
    let mut m1 = vec![0.0_f64; intervals];
    for i in 0..intervals {
        if curve[i] <= 0.0 {
            continue;
        }
        let d = (curve[i + 1] - curve[i]) / (loss[i + 1] - loss[i]);
        let (a, b) = limit_slopes(d, -slope_both[i] / loss[i], -(slope_both[i + 1] + slope_end[i + 1]) / loss[i + 1]);
        m0[i] = a;
        m1[i] = b;
    }
    (m0, m1)
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
    /// The fine mesh (the base nodes refined where bell-shaped events asked
    /// for density) and the requesting events' truncated curve on it with
    /// its limited slopes; all empty when no event requested.
    fine_loss: Vec<f64>,
    fine_curve: Vec<f64>,
    fine_m0: Vec<f64>,
    fine_m1: Vec<f64>,
    requesting_events: usize,
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
    /// The requesting events' curve and slopes on the fine mesh.
    fine_curve: Vec<f64>,
    fine_slope_both: Vec<f64>,
    fine_slope_end: Vec<f64>,
    /// The plateaus below the events' lower skips as differences, one
    /// element longer than each mesh.
    plateau: Vec<f64>,
    fine_plateau: Vec<f64>,
    requesting: usize,
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

/// The workers of one top-level call: a pool built once when `threads` is
/// given and shared by every chunked pass of the call (the curve build and
/// the grid's two passes, or the quantiles pass), or rayon's global pool.
pub(crate) struct Workers {
    pool: Option<rayon::ThreadPool>,
}

impl Workers {
    pub(crate) fn new(config: &Config) -> Result<Self, Error> {
        let pool = match config.threads {
            Some(threads) => Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|e| Error::ThreadPool(e.to_string()))?,
            ),
            None => None,
        };
        Ok(Workers { pool })
    }

    fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match &self.pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }
}

/// Chunks per wave: twice the workers, and no more than hold `budget` bytes
/// of partials at `partial_bytes` each (peer review, finding P1: at 2^24
/// buckets a quantiles partial is 256 MiB, and 64 in flight were 16 GiB).
pub(crate) fn wave_size(threads: usize, budget: usize, partial_bytes: usize) -> usize {
    (2 * threads.max(1)).min((budget / partial_bytes.max(1)).max(1))
}

/// Run a chunked pass in bounded waves on the call's workers, handing each
/// wave's partials to `fold_wave` in chunk order (so a fold that keeps that
/// order per element is bitwise identical for any thread count at a fixed
/// chunk size). `partial_bytes` is the caller's bound on one chunk's
/// partial, which sizes the wave against `config.wave_memory_bytes`. The
/// fold runs on the workers too, so it may use rayon. An empty pass returns
/// before any work is scheduled.
pub(crate) fn run_waves<P: Send>(
    workers: &Workers,
    config: &Config,
    n_chunks: usize,
    partial_bytes: usize,
    process: impl Fn(usize) -> Result<P, Error> + Sync,
    mut fold_wave: impl FnMut(Vec<P>) -> Result<(), Error> + Send,
) -> Result<(), Error> {
    if n_chunks == 0 {
        return Ok(());
    }
    workers.install(|| {
        let wave = wave_size(rayon::current_num_threads(), config.wave_memory_bytes, partial_bytes);
        let mut next = 0_usize;
        while next < n_chunks {
            let end = (next + wave).min(n_chunks);
            let partials: Vec<P> = (next..end)
                .into_par_iter()
                .map(&process)
                .collect::<Result<Vec<P>, Error>>()?;
            fold_wave(partials)?;
            next = end;
        }
        Ok(())
    })
}

/// `run_waves` with the partials folded one at a time, in chunk order.
pub(crate) fn run_chunks<P: Send>(
    workers: &Workers,
    config: &Config,
    n_chunks: usize,
    partial_bytes: usize,
    process: impl Fn(usize) -> Result<P, Error> + Sync,
    mut fold: impl FnMut(P) -> Result<(), Error> + Send,
) -> Result<(), Error> {
    run_waves(workers, config, n_chunks, partial_bytes, process, |partials: Vec<P>| {
        for partial in partials {
            fold(partial)?;
        }
        Ok(())
    })
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
    let workers = Workers::new(config)?;
    build_curve_with(&workers, input, config, classified, min_loss)
}

/// `build_curve` on the workers of an enclosing call (`quantize` builds
/// the pool once for the curve and its grid).
pub(crate) fn build_curve_with(
    workers: &Workers,
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
    // The survival chunk depends on neither the row count nor the grid.
    let chunk_rows = config.resolved_chunk_rows(n_rows, 0);
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

    // The rate floor of the bulk search, on the beta rate summed here
    // serially (the fold below gives the reported value).
    let mut serial_beta_rate = Neumaier::default();
    for row in 0..n_rows {
        if kinds[row] == RowKind::Beta {
            serial_beta_rate.add(input.rate[row]);
        }
    }
    let narrow_floor = TAIL_RATE_FLOOR * serial_beta_rate.value();

    // The routes (spec section 5A), decided once so that the chunks agree
    // and the fine mesh is known before the pass: a bell-shaped event (both
    // parameters above 1) narrower than a base interval at its mean, or
    // beyond the continued fraction's regime, is narrow; one asking more
    // than the base density, `d_j = T ln(10) mu / sigma` nodes per decade
    // for `nodes_per_sigma` per standard deviation on [mu - 3 sigma,
    // mu + 7 sigma], lives on the fine curve; the rest are regular on the
    // base curve. The fine mesh splits each base interval into the largest
    // number of equal pieces asked of it, at most `nodes_per_sigma` since
    // the narrow route takes over at one interval.
    let t_sigma = config.nodes_per_sigma as f64;
    let narrow_ratio = 10.0_f64.powf(1.0 / config.nodes_per_decade as f64) - 1.0;
    let base_density = config.nodes_per_decade as f64;
    let mut routes = vec![Route::Regular; n_rows];
    let mut splits = vec![1_usize; intervals];
    for row in 0..n_rows {
        if kinds[row] != RowKind::Beta || input.max_loss[row] <= min_loss {
            continue;
        }
        let (m, mu, sigma) = (input.max_loss[row], classified.mu_eff[row], classified.sd_eff[row]);
        let (alpha, beta) = beta_params(m, mu, sigma);
        if !(alpha > 1.0 && beta > 1.0) {
            continue;
        }
        let density = t_sigma * std::f64::consts::LN_10 * mu / sigma;
        let quadrature_ok = alpha.min(beta) >= NARROW_MIN_PARAMETER;
        if quadrature_ok && (alpha.min(beta) >= crate::batched::LARGE_PARAMETER_FROM || sigma < mu * narrow_ratio) {
            routes[row] = Route::Narrow;
        } else if density > base_density {
            // At most `nodes_per_sigma` pieces per base interval: an event
            // narrower than an interval that the quadrature cannot take (a
            // parameter between 1 and 2, its mass against an endpoint) is
            // carried by its tail ladder, not by the fine mesh.
            routes[row] = Route::Requesting;
            let lo = (mu - 3.0 * sigma).max(min_loss);
            let hi = (mu + 7.0 * sigma).min(m);
            if hi > lo {
                let i_lo = interval_of(&loss, lo);
                let i_hi = interval_of(&loss, hi).min(intervals - 1);
                let pieces = ((density / base_density).ceil() as usize).min(config.nodes_per_sigma);
                for sp in &mut splits[i_lo..=i_hi] {
                    *sp = (*sp).max(pieces);
                }
            }
        }
    }
    let has_fine = routes.iter().any(|&r| r == Route::Requesting);
    let (fine_loss, fine_of_base) = if has_fine { refine_mesh(&loss, &splits) } else { (Vec::new(), Vec::new()) };
    let fine_m1 = fine_loss.len();

    let process_chunk = |chunk_index: usize| -> Result<BuildPartial, Error> {
        let start = chunk_index * chunk_rows;
        let end = (start + chunk_rows).min(n_rows);
        let mut p = BuildPartial {
            curve: vec![0.0; m1],
            slope_both: vec![0.0; m1],
            slope_end: vec![0.0; m1],
            tail_at: vec![0.0; m1],
            node_tail: vec![0.0; m1],
            fine_curve: vec![0.0; fine_m1],
            fine_slope_both: vec![0.0; fine_m1],
            fine_slope_end: vec![0.0; fine_m1],
            plateau: vec![0.0; m1 + 1],
            fine_plateau: vec![0.0; fine_m1 + 1],
            requesting: 0,
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
                    let (alpha, beta) = beta_params(m, mu, classified.sd_eff[row]);
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
                    // above 1) qualify: one narrower than a base interval at
                    // its mean, or whose parameters are both large enough
                    // that the continued fraction is untrustworthy
                    // (`is_valid_regime`), whatever the node density; from
                    // NARROW_NORMAL_FROM the kernel is the normal law (a row
                    // narrower than the resolution of its mean never gets
                    // here: `validate` made it an atom).
                    let sigma = classified.sd_eff[row];
                    if routes[row] == Route::Narrow {
                        let kernel = NarrowKernel::new(alpha, beta);
                        let bulk = narrow_bulk(&kernel, m, mu, sigma, lam, narrow_floor)
                            .map(|(lower, stop)| (lower, stop, kernel.integral(lower / m, stop / m)));
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
                        let xk = loss[k];
                        let s = checked(bf.sf(xk / m))?;
                        let bell = alpha > 1.0 && beta > 1.0;
                        if routes[row] == Route::Requesting {
                            // On the fine mesh, whose node for x_k is the
                            // base node itself.
                            p.requesting += 1;
                            deposit_event(
                                &bf, m, lam, fine_of_base[k], s, &fine_loss,
                                &mut p.fine_curve, &mut p.fine_slope_both, &mut p.fine_slope_end, &mut p.fine_plateau,
                                bell, &checked,
                            )?;
                        } else {
                            deposit_event(
                                &bf, m, lam, k, s, &loss,
                                &mut p.curve, &mut p.slope_both, &mut p.slope_end, &mut p.plateau,
                                bell, &checked,
                            )?;
                        }
                        (k, xk, s)
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

    let n_chunks = n_rows.div_ceil(chunk_rows);
    // A partial's bound: its node-length vectors on both meshes, the
    // samples, and the chunk's per-event records.
    let partial_bytes = 8 * (6 * m1 + 4 * fine_m1 + 2 * sample_x.len() + 2) + 64 * chunk_rows;
    let mut curve = vec![0.0_f64; m1];
    let mut slope_both = vec![0.0_f64; m1];
    let mut slope_end = vec![0.0_f64; m1];
    let mut tail_at = vec![0.0_f64; m1];
    let mut node_tail = vec![0.0_f64; m1];
    let mut fine_curve = vec![0.0_f64; fine_m1];
    let mut fine_slope_both = vec![0.0_f64; fine_m1];
    let mut fine_slope_end = vec![0.0_f64; fine_m1];
    let mut plateau = vec![0.0_f64; m1 + 1];
    let mut fine_plateau = vec![0.0_f64; fine_m1 + 1];
    let mut requesting_events = 0_usize;
    let mut sample_curve = vec![0.0_f64; sample_x.len()];
    let mut sample_total = vec![0.0_f64; sample_x.len()];
    let mut tails: Vec<TailEvent> = Vec::new();
    let mut narrow: Vec<NarrowEvent> = Vec::new();
    let mut atoms: Vec<(f64, f64)> = Vec::new();
    let mut total_rate = Neumaier::default();
    let mut analytic_aal = Neumaier::default();
    let mut beta_rate = Neumaier::default();
    let mut valid_rows = 0_usize;
    run_chunks(workers, config, n_chunks, partial_bytes, process_chunk, |p: BuildPartial| {
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
        for (dst, src) in fine_curve.iter_mut().zip(&p.fine_curve) {
            *dst += src;
        }
        for (dst, src) in fine_slope_both.iter_mut().zip(&p.fine_slope_both) {
            *dst += src;
        }
        for (dst, src) in fine_slope_end.iter_mut().zip(&p.fine_slope_end) {
            *dst += src;
        }
        for (dst, src) in plateau.iter_mut().zip(&p.plateau) {
            *dst += src;
        }
        for (dst, src) in fine_plateau.iter_mut().zip(&p.fine_plateau) {
            *dst += src;
        }
        requesting_events += p.requesting;
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
    // The plateaus, prefix-summed into the curves (the events' rate less
    // their truncation value below their lower skips), then no negatives.
    let settle = |curve: &mut [f64], plateau: &[f64]| {
        let mut running = 0.0_f64;
        for (v, d) in curve.iter_mut().zip(plateau) {
            running += d;
            *v += running;
            if *v < 0.0 {
                *v = 0.0;
            }
        }
    };
    settle(&mut curve, &plateau);
    settle(&mut fine_curve, &fine_plateau);

    // The interpolants: limited one-sided slopes per interval of each mesh.
    let (m0, m1v) = limited_slopes(&loss, &curve, &slope_both, &slope_end);
    let (fine_m0, fine_m1v) = limited_slopes(&fine_loss, &fine_curve, &fine_slope_both, &fine_slope_end);
    let mut tail_above = vec![0.0_f64; m1 + 1];
    for i in (0..m1).rev() {
        tail_above[i] = tail_above[i + 1] + tail_at[i];
    }
    let exceedance_rate: Vec<f64> = (0..m1)
        .map(|i| {
            let fine = if has_fine { fine_curve[fine_of_base[i]] } else { 0.0 };
            curve[i] + fine + tail_above[i] + node_tail[i]
        })
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
        fine_loss,
        fine_curve,
        fine_m0,
        fine_m1: fine_m1v,
        requesting_events,
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
        let base = cubic_at(&self.loss, &self.curve, &self.m0, &self.m1, x);
        if self.fine_loss.is_empty() {
            base
        } else {
            base + cubic_at(&self.fine_loss, &self.fine_curve, &self.fine_m0, &self.fine_m1, x)
        }
    }

    /// The fine mesh's node count (0 when no event requested density).
    pub fn fine_nodes(&self) -> usize {
        self.fine_loss.len()
    }

    /// The bell-shaped events carried on the fine curve.
    pub fn requesting_events(&self) -> usize {
        self.requesting_events
    }

    /// `int_e^{x_m} Lambda~(x) dx`, the integral of the interpolated curve
    /// above `e`: the cubic's antiderivative on the partial interval holding
    /// `e`, then the closed form `dx (y0 + y1) / 2 + dx^2 (m0 - m1) / 12` on
    /// each whole interval above it. Zero at and beyond the last node.
    pub fn integral_above(&self, e: f64) -> f64 {
        let base = cubic_integral_above(&self.loss, &self.curve, &self.m0, &self.m1, e);
        if self.fine_loss.is_empty() {
            base
        } else {
            base + cubic_integral_above(&self.fine_loss, &self.fine_curve, &self.fine_m0, &self.fine_m1, e)
        }
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
        let workers = Workers::new(config)?;
        self.grid_with(&workers, h, grid_length, policy, config)
    }

    /// `grid` on the workers of an enclosing call.
    pub(crate) fn grid_with(
        &self,
        workers: &Workers,
        h: f64,
        grid_length: Option<usize>,
        policy: Overflow,
        config: &Config,
    ) -> Result<GridOutcome, Error> {
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
            // The curve's value at its lowest node, base and fine together
            // (clipping from the base alone dropped a requesting event's
            // plateau into bucket 0); then the edges in order, each mesh's
            // interval found by a cursor that only moves up, the differences
            // written as they come (a binary search per edge and a stored
            // value vector before: peer review, finding P2).
            let mut prev = self.interpolate(self.loss[0]);
            let mut base = CubicCursor::default();
            let mut fine = CubicCursor::default();
            for i in 0..n {
                let e = (i as f64 + 0.5) * h;
                let mut v = base.at_ascending(&self.loss, &self.curve, &self.m0, &self.m1, e);
                if !self.fine_loss.is_empty() {
                    v += fine.at_ascending(&self.fine_loss, &self.fine_curve, &self.fine_m0, &self.fine_m1, e);
                }
                let v = v.min(prev);
                if i == 0 {
                    secondary[0] += self.beta_rate - v - self.tail_above[0] - self.narrow_mass;
                } else {
                    secondary[i] += prev - v;
                }
                prev = v;
            }
            let rest = prev;
            if rest > 0.0 {
                over.rate.add(rest);
                if policy == Overflow::TopBucket {
                    secondary[n - 1] += rest;
                }
                // Its moment about the top node: integrating each event's
                // partial expectation by parts and summing gives
                // (e - c) Lambda~(e) + int_e^{x_m} Lambda~(x) dx, the
                // integral of the interpolant itself, closed-form per
                // interval. This covers the events whose endpoint survival
                // underflowed, which leave no tail record (the per-event
                // loop over the tails they were missing from lost their
                // moment: peer review, finding I1).
                over.moment.add((top_edge - top_node) * rest + self.integral_above(top_edge));
            }
        } else {
            secondary[0] += self.beta_rate - self.tail_above[0] - self.narrow_mass;
        }

        // The tails and the narrow events, per event, in chunk passes
        // folded in order: each chunk's histogram covers only the bucket
        // span its events can touch (the top node stands in for anything
        // beyond the grid), so a chunk costs its span, not the grid length.
        let chunk = config.resolved_chunk_rows(self.tails.len(), n);
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
        // The span bound is the grid itself: a chunk's tails can reach any
        // bucket from the lowest truncation node to the top.
        run_chunks(workers, config, n_chunks, 8 * n, process, |part| {
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
        // The span bound is the grid itself: a chunk's tails can reach any
        // bucket from the lowest truncation node to the top.
        run_chunks(workers, config, n_chunks, 8 * n, process, |part| {
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
            let (alpha, beta) = beta_params(m, mu, sd);
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

    // [overflow] The curve's overflow moment on a windowed grid comes from
    // the integral of the interpolant, so an event whose endpoint survival
    // underflows (no tail record) keeps its moment: Beta(2, 1000) on [0, 1]
    // windowed at 0.00195 has the exact excess moment 5.796480662291e-4
    // about the top node (SciPy); the tails loop it replaced reported 0
    // (peer review, finding I1). The integral itself is checked against a
    // fine Riemann sum of the interpolant.
    #[test]
    fn windowed_overflow_moment_survives_tail_underflow() {
        use crate::quantize::classify_rows;
        let (a, b) = (2.0_f64, 1000.0_f64);
        let p = a / (a + b);
        let sd = (p * (1.0 - p) / (a + b + 1.0)).sqrt();
        let rate = [1.0];
        let max_loss = [1.0];
        let mean = [p];
        let sd_i = [sd];
        let sd_c = [0.0];
        let input = EltInput { rate: &rate, max_loss: &max_loss, mean_loss: &mean, sd_indep: &sd_i, sd_corr: &sd_c };
        let (h, n) = (1e-4, 20);
        let config = Config::new(h);
        let classified = classify_rows(&input, &config).unwrap();
        let curve = build_curve(&input, &config, &classified, 0.5 * h).unwrap();
        assert!(curve.tails.is_empty(), "the endpoint survival underflows: no tail record");
        let out = curve.grid(h, Some(n), Overflow::TopBucket, &config).unwrap();
        let exact_rate = 0.41891026155683486;
        let exact_moment = 5.796480662291177e-4;
        let (over_rate, over_moment) = (out.overflow.rate.value(), out.overflow.moment.value());
        assert!((over_rate - exact_rate).abs() <= 1e-7 * exact_rate, "rate {over_rate}");
        assert!((over_moment - exact_moment).abs() <= 1e-4 * exact_moment, "moment {over_moment}");
        // The integral against a Riemann sum of the interpolant.
        let e = (n as f64 - 0.5) * h;
        let top = *curve.loss.last().unwrap();
        let steps = 200_000;
        let dx = (top - e) / steps as f64;
        let riemann: f64 = (0..steps).map(|i| curve.interpolate(e + (i as f64 + 0.5) * dx)).sum::<f64>() * dx;
        let integral = curve.integral_above(e);
        assert!((integral - riemann).abs() <= 1e-6 * riemann, "{integral} vs {riemann}");
    }

    // [atoms] At the handover to the normal law the two kernels agree on
    // mass and moment to a few units in 1e-6 (each is about that far from
    // the truth there, in opposite senses), the survival at the mean of a
    // symmetric event is a half on either side (the review's A1: a step at
    // the mean gave 0 at 1e-8 relative width), and the law stays monotone
    // in [0, 1] to min(alpha, beta) = 1e24.
    #[test]
    fn normal_kernel_takes_over_from_the_quadrature() {
        let m = 1.0;
        for (alpha, beta) in [(1e16, 1e16), (1e16, 4e17), (3e16, 1e16)] {
            let below = NarrowKernel::new(alpha * (1.0 - 1e-9), beta * (1.0 - 1e-9));
            let above = NarrowKernel::new(alpha, beta);
            assert!(!below.is_normal() && above.is_normal());
            let ybar = above.ybar;
            let sd = above.sd;
            for (lo, hi) in [(ybar - 5.0 * sd, ybar + 5.0 * sd), (ybar, ybar + 2.0 * sd), (ybar + 3.0 * sd, ybar + 9.0 * sd)] {
                let (m0, m1) = below.integrals(lo, hi);
                let (n0, n1) = above.integrals(lo, hi);
                assert!((m0 - n0).abs() <= 5e-6 * n0, "{alpha} {beta}: mass {m0} vs {n0}");
                assert!((m1 - n1).abs() <= 5e-6 * n1, "{alpha} {beta}: moment {m1} vs {n1}");
            }
        }
        for sigma in [1e-8, 2e-8, 1e-10, 1e-12] {
            let mu = 0.5;
            let (alpha, beta) = beta_params(m, mu, sigma);
            let kernel = NarrowKernel::new(alpha, beta);
            let (lower, stop) = narrow_bulk(&kernel, m, mu, sigma, 1.0, 1e-20).unwrap();
            let norm = kernel.integral(lower / m, stop / m);
            let event = NarrowEvent { rate: 1.0, max_loss: m, alpha, beta, lower, start: lower, stop, norm, survival: 1.0 };
            let half = event.survival_at(&kernel, mu);
            assert!((half - 0.5).abs() <= 1e-9, "sigma={sigma}: {half}");
            let mut prev = 1.0;
            for k in -12..=12 {
                let s = event.survival_at(&kernel, mu + k as f64 * sigma);
                assert!((0.0..=1.0).contains(&s) && s <= prev, "sigma={sigma} k={k}: {s}");
                prev = s;
            }
            // The partial expectation above the mean: mu / 2 + sigma / sqrt(2 pi).
            let want = 0.5 * mu + sigma / (2.0 * std::f64::consts::PI).sqrt();
            let got = event.partial_expectation(&kernel, mu);
            assert!((got - want).abs() <= 1e-9 * want + 1e-6 * sigma, "sigma={sigma}: {got} vs {want}");
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

    // [tails] The ladder refines across the fine zone when c > 1 and the
    // power law gives way to the exact survival below the last node when
    // c t_L is large: the review's A2 case, Beta(1e6, 1/2), at an edge
    // 5e-7 below M where the pure power law gave 0.054 for 0.6827.
    #[test]
    fn tail_ladder_refines_and_evaluates_below_the_last_node() {
        let m = 1.0;
        let (a, b) = (1e6, 0.5);
        let bf = BetaFn::new(a, b);
        let h = (1.0 - b / a) / 999.5;
        let xk = 1.0 - 10.0 * b / a; // ten mean distances below M
        let s = bf.sf(xk / m);
        let interp = TailInterpolant::new(&bf, m, xk, s, 0.25 * h);
        assert!(!interp.asymptotic);
        let edge = 999.5 * h;
        let got = interp.at(edge);
        let want = bf.sf(edge / m);
        assert!((got - want).abs() <= 1e-9 * want, "{got} vs {want}");
        assert!((want - 0.6826894921171948).abs() < 1e-9);
        // Small alpha: no refinement, the power law holds, the node count is
        // the old 2 ceil(log2(d0 / min)) + 1.
        let small = BetaFn::new(0.7, 40.0);
        let (xk, min_distance) = (0.92e5, 0.5);
        let interp = TailInterpolant::new(&small, 1e5, xk, small.sf(xk / 1e5), min_distance);
        assert!(interp.asymptotic);
        let expected = 2 * (((1e5 - xk) / min_distance).log2().ceil() as usize) + 1;
        assert!(interp.nodes() <= expected && interp.nodes() >= expected - 2, "{} vs {expected}", interp.nodes());
        // Large alpha, moderate beta: the ladder across the fine zone is
        // within 5e-5 of the exact survival at every quarter-octave
        // midpoint (3e-4 before the refinement at beta = 5).
        for (a, b) in [(1e2, 5.0), (1e4, 2.0), (1e6, 5.0)] {
            let bf = BetaFn::new(a, b);
            let mu = a / (a + b);
            let xk = (1.0 - 10.0 * (1.0 - mu)).max(0.5);
            let s = bf.sf(xk);
            let interp = TailInterpolant::new(&bf, 1.0, xk, s, 1e-8);
            let d0 = 1.0 - xk;
            let mut k = 1;
            loop {
                let d = d0 * 0.5_f64.powf(k as f64 / 8.0);
                if d < 1e-8 {
                    break;
                }
                let want = bf.sf(1.0 - d);
                if want < 1e-280 {
                    break;
                }
                let got = interp.at(1.0 - d);
                assert!((got - want).abs() <= 5e-5 * want, "a={a} b={b} d={d}: {got} vs {want}");
                k += 2;
            }
        }
    }

    // [mesh] The refined mesh keeps every base node bitwise and splits the
    // asked intervals evenly in ln x.
    #[test]
    fn refined_mesh_keeps_the_base_nodes() {
        let base = geometric_nodes(1.0, 1e3, 10);
        let mut splits = vec![1_usize; base.len() - 1];
        splits[3] = 4;
        splits[7] = 2;
        let (fine, of_base) = refine_mesh(&base, &splits);
        assert_eq!(fine.len(), base.len() + 3 + 1);
        for (i, &b) in base.iter().enumerate() {
            assert_eq!(fine[of_base[i]], b);
        }
        assert!(fine.windows(2).all(|w| w[1] > w[0]));
        let (u0, u1) = (base[3].ln(), base[4].ln());
        assert_eq!(fine[of_base[3] + 1], (u0 + (u1 - u0) / 4.0).exp());
    }

    // [mesh] The review's A3 case, Beta(1800, 1e6): a bell-shaped event
    // with a coefficient of variation of 2.4%, one base interval wide, was
    // 22.8% off in its upper tail on the shared cubic. On the fine curve it
    // meets the contract: within 1e-4 where S >= 1e-4 and within 1e-3 where
    // S >= 1e-8, at every fine-interval midpoint, against the engine's own
    // incomplete beta (good to 1e-9 here).
    #[test]
    fn requesting_event_meets_the_contract() {
        use crate::quantize::classify_rows;
        let (a, b) = (1800.0_f64, 1e6_f64);
        let p = a / (a + b);
        let sd = (p * (1.0 - p) / (a + b + 1.0)).sqrt();
        let rate = [1.0];
        let max_loss = [1.0];
        let mean = [p];
        let sd_i = [sd];
        let sd_c = [0.0];
        let input = EltInput { rate: &rate, max_loss: &max_loss, mean_loss: &mean, sd_indep: &sd_i, sd_corr: &sd_c };
        let config = Config::new(2.0 * p * 1e-4);
        let classified = classify_rows(&input, &config).unwrap();
        let curve = build_curve(&input, &config, &classified, p * 1e-4).unwrap();
        assert_eq!(curve.requesting_events(), 1);
        assert!(curve.fine_nodes() > curve.nodes(), "{} vs {}", curve.fine_nodes(), curve.nodes());
        let bf = BetaFn::new(a, b);
        let x = 0.001944568270403778;
        let exact = bf.sf(x);
        assert!((exact - 3.252159380340672e-4).abs() < 1e-12);
        let got = curve.exceedance(x);
        assert!((got - exact).abs() <= 1e-4 * exact, "{got} vs {exact}");
        for w in curve.fine_loss.windows(2) {
            let mid = (0.5 * (w[0].ln() + w[1].ln())).exp();
            let want = bf.sf(mid);
            if want < 1e-8 {
                continue;
            }
            let got = curve.exceedance(mid);
            let tol = if want >= 1e-4 { 1e-4 } else { 1e-3 };
            assert!((got - want).abs() <= tol * want, "x={mid}: {got} vs {want}");
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
                let (a, b) = beta_params(rows[r][1], rows[r][2], rows[r][3] + rows[r][4]);
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

    // [schedule] The wave is twice the workers unless the memory budget
    // holds fewer partials; never below one chunk.
    #[test]
    fn wave_is_bounded_by_threads_and_memory() {
        let gib = 1_usize << 30;
        assert_eq!(wave_size(32, gib, 8 * (1 << 16)), 64);
        assert_eq!(wave_size(32, gib, 16 * (1 << 20)), 64);
        assert_eq!(wave_size(32, gib, 16 * (1 << 24)), 4);
        assert_eq!(wave_size(1, gib, 16 * (1 << 24)), 2);
        assert_eq!(wave_size(32, gib, 16 * (1 << 28)), 1);
        assert_eq!(wave_size(0, 0, 0), 1);
    }

    // [tidy] The ascending cursor returns exactly what the searched
    // evaluation returns, at nodes, between them and beyond the mesh.
    #[test]
    fn cubic_cursor_matches_the_search() {
        let loss: Vec<f64> = (0..50).map(|i| 1.0 * 1.1_f64.powi(i)).collect();
        let curve: Vec<f64> = loss.iter().map(|x| 1.0 / (1.0 + x)).collect();
        let slope_both: Vec<f64> = loss.iter().map(|x| x / (1.0 + x).powi(2)).collect();
        let slope_end = vec![0.0; loss.len()];
        let (m0, m1) = limited_slopes(&loss, &curve, &slope_both, &slope_end);
        let mut cursor = CubicCursor::default();
        let mut queries: Vec<f64> = (0..600).map(|i| loss[0] * 1.0137_f64.powi(i)).collect();
        queries.extend(loss.iter().copied());
        queries.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for &x in &queries {
            let walked = cursor.at_ascending(&loss, &curve, &m0, &m1, x);
            let searched = cubic_at(&loss, &curve, &m0, &m1, x);
            assert!(walked.to_bits() == searched.to_bits(), "{x}: {walked} vs {searched}");
        }
    }
}
