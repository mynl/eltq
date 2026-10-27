//! Model moments of shared events with no grid (spec section 4B, test 36):
//! the raw moments of each event total under the declared coupling,
//! rate-weighted over the events.
//!
//! Given the shared factor `Z_c = z` the units are independent, so the
//! conditional moments of the total follow from adding the units one at a
//! time; the marginal parts `sum_u P(X_u^r)` are closed forms, the first
//! moment is exact, and only the cross parts `c_2(z) = 2 sum_{u<v} a_u1
//! a_v1` and `c_3(z)` need the conditional first and second moments
//! `a_ur(z) = P(X_u^r | z)` and a rule over the factor. Per beta unit with a
//! positive loading the share is tabled against its normal score (`ln y`
//! against `v = Phi^{-1}(F(y))`, forward evaluations only, cubic Hermite with
//! exact slopes), and `a_ur(z)` is the integral of `y(w z + s t)^r` against
//! the standard normal in `t`, Gauss-Legendre on pieces split at the
//! table's breakpoints. The factor is integrated by Gauss-Legendre on pieces
//! split at the steps of two-point units at loading one and, for units
//! steep in the factor, at their tables' breakpoints. Two levels of the
//! tables and the rules are compared; an event whose levels disagree by
//! more than `rtol` in `C_2` or `C_3` is refined, and after
//! `MAX_REFINEMENTS` rounds reported unconverged. The algorithm, its
//! constants and its test cases are the reference's
//! (`eltq.reference_multiunit.multiunit_moments_reference`;
//! `dev/done/plan-model-moments-and-windowed-views.md`, [Eltq-Model-Moments]).

use rayon::prelude::*;

use crate::config::VariancePolicy;
use crate::diagnostics::{Diagnostics, MAX_EXAMPLES};
use crate::error::Error;
use crate::law::BetaLaw;
use crate::moments::RowLaw;
use crate::multiunit::{Coupling, UnitKind, UnitPlan, UnitSpec, prepare_unit, rolled_spec, two_point_conditional};
use crate::normal;
use crate::sum::Neumaier;
use crate::views::{DEFAULT_MULTI_CHUNK_ROWS, MultiInput, event_pass};

/// Default relative tolerance in `C_2` and `C_3` (spec section 11, item 10).
pub const MOMENT_RTOL: f64 = 1e-6;
/// The tables' tolerance on `ln y` as a fraction of `rtol`.
const TABLE_TOL_FRACTION: f64 = 1.0 / 30.0;
/// Widest score cell of a table, and largest rise of the share's logit
/// `ln(y / (1 - y))` across one.
const TABLE_MAX_DV: f64 = 1.0;
const TABLE_MAX_DLOGIT: f64 = 1.0;
/// Most nodes a table may have before it is reported unconverged, and the
/// least tolerance a table is held to (the incomplete beta's own accuracy).
const TABLE_MAX_NODES: usize = 16384;
const TABLE_TOL_FLOOR: f64 = 1e-12;
/// The fine level's inner width (in the kernel's standardized variable),
/// inner breakpoint cap (in the logit) and factor width.
const INNER_WIDTH: f64 = 2.5;
const INNER_CAP: f64 = 3.0;
const OUTER_WIDTH: f64 = 2.5;
/// How far the inner rule reaches in standard normal units (`Phi(-8)` is
/// 6e-16), the factor's lower end, and the least upper end of the factor.
const REACH: f64 = 8.0;
const Z_LOW: f64 = -9.0;
const Z_HIGH_MIN: f64 = 9.0;
/// A unit's own factor nodes: pieces of at most this many times its
/// kernel's width in the factor when smooth, at least `STEEP_NATIVE` wide
/// between its table's breakpoints when steep.
const SMOOTH_NATIVE: f64 = 2.5;
const STEEP_NATIVE: f64 = 1.0;
/// Refinement rounds after the first comparison.
pub const MAX_REFINEMENTS: usize = 3;
/// The coarse level's widths and caps relative to the fine level's.
const COARSE_SCALE: f64 = 1.5;
/// A unit whose kernel in the factor, `s / w`, is narrower than this puts
/// its table's breakpoints into the factor rule.
const STEEP_KERNEL: f64 = 0.5;

const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// 8-point Gauss-Legendre on [-1, 1].
const GL8_X: [f64; 8] = [
    -0.960_289_856_497_536_3,
    -0.796_666_477_413_626_7,
    -0.525_532_409_916_329,
    -0.183_434_642_495_649_8,
    0.183_434_642_495_649_8,
    0.525_532_409_916_329,
    0.796_666_477_413_626_7,
    0.960_289_856_497_536_3,
];
const GL8_W: [f64; 8] = [
    0.101_228_536_290_376_3,
    0.222_381_034_453_374_5,
    0.313_706_645_877_887_3,
    0.362_683_783_378_362,
    0.362_683_783_378_362,
    0.313_706_645_877_887_3,
    0.222_381_034_453_374_5,
    0.101_228_536_290_376_3,
];

#[inline]
fn phi(x: f64) -> f64 {
    FRAC_1_SQRT_2PI * (-0.5 * x * x).exp()
}

/// One table node: the score, `ln y`, the slope `d ln y / dv` and the logit.
#[derive(Debug, Clone, Copy)]
struct Node {
    v: f64,
    y: f64,
    lny: f64,
    slope: f64,
    logit: f64,
}

/// The loss share of a beta unit against its normal score (module notes).
#[derive(Debug, Clone)]
pub struct ScoreTable {
    v: Vec<f64>,
    lny: Vec<f64>,
    slope: Vec<f64>,
    logit: Vec<f64>,
    /// False when the node budget ran out before every cell met the tolerance.
    pub converged: bool,
}

impl ScoreTable {
    /// Build the table of `Beta(alpha, beta)`: seeds per decade of `y` from
    /// `1e-4 rtol` of the mean, near one, about the mean (and densely across
    /// the bulk past `LARGE_PARAMETER_FROM`), then cells split until the cubic
    /// Hermite misses no midpoint by more than `tol` in `ln y`, no cell is
    /// wider than `TABLE_MAX_DV` and the logit rises by at most
    /// `TABLE_MAX_DLOGIT` across one.
    pub fn new(alpha: f64, beta: f64, rtol: f64, tol: f64) -> Result<ScoreTable, Error> {
        let law = BetaLaw::new(alpha, beta).ok_or(Error::KernelFailure { row: 0, alpha, beta })?;
        let mean = alpha / (alpha + beta);
        let sd = (alpha * beta / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1.0))).sqrt();
        let lo = (1e-4 * rtol * mean).max(1e-300);
        let mut seeds: Vec<f64> = Vec::new();
        let mut decade = lo.log10().floor() as i32;
        while 10f64.powi(decade) < 0.5 {
            seeds.push(10f64.powi(decade).max(lo));
            decade += 1;
        }
        for j in 0..17 {
            seeds.push(1.0 - 0.5 * 10f64.powi(-j));
        }
        for k in -8..=8 {
            seeds.push(mean + k as f64 * sd);
        }
        if law.is_narrow() {
            for k in 0..33 {
                seeds.push(mean + sd * (-12.0 + 24.0 * k as f64 / 32.0));
            }
        }
        seeds.retain(|&s| lo <= s && s < 1.0);
        seeds.sort_by(f64::total_cmp);
        seeds.dedup();
        let mut nodes: Vec<Node> = Vec::with_capacity(seeds.len());
        for &y in &seeds {
            if let Some(n) = node(&law, y)
                && nodes.last().is_none_or(|last: &Node| n.v > last.v)
            {
                nodes.push(n);
            }
        }
        if nodes.len() < 2 {
            return Err(Error::KernelFailure { row: 0, alpha, beta });
        }
        // Breadth-first passes: every open cell is checked against its
        // midpoint and split or closed, so that a table stopped by the node
        // budget is refined evenly rather than on one side.
        let mut out = nodes;
        let mut open = vec![true; out.len() - 1];
        let mut converged = true;
        while open.iter().any(|&o| o) {
            let mut next = Vec::with_capacity(2 * out.len());
            let mut next_open = Vec::with_capacity(2 * out.len());
            next.push(out[0]);
            for (k, &check) in open.iter().enumerate() {
                let (left, right) = (out[k], out[k + 1]);
                let mut split = false;
                if check
                    && let Some(mid) = node(&law, midpoint(&left, &right))
                    && left.v < mid.v
                    && mid.v < right.v
                    && must_split(&left, &right, &mid, tol)
                {
                    next.push(mid);
                    next.push(right);
                    next_open.push(true);
                    next_open.push(true);
                    split = true;
                }
                if !split {
                    next.push(right);
                    next_open.push(false);
                }
            }
            if next.len() > TABLE_MAX_NODES {
                converged = false;
                break;
            }
            out = next;
            open = next_open;
        }
        Ok(ScoreTable {
            v: out.iter().map(|n| n.v).collect(),
            lny: out.iter().map(|n| n.lny).collect(),
            slope: out.iter().map(|n| n.slope).collect(),
            logit: out.iter().map(|n| n.logit).collect(),
            converged,
        })
    }

    /// Nodes in the table.
    pub fn len(&self) -> usize {
        self.v.len()
    }

    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }

    /// The table on every second node (the last kept): the coarse level.
    pub fn subsampled(&self) -> ScoreTable {
        let n = self.v.len();
        let mut keep: Vec<usize> = (0..n).step_by(2).collect();
        if *keep.last().unwrap() != n - 1 {
            keep.push(n - 1);
        }
        ScoreTable {
            v: keep.iter().map(|&i| self.v[i]).collect(),
            lny: keep.iter().map(|&i| self.lny[i]).collect(),
            slope: keep.iter().map(|&i| self.slope[i]).collect(),
            logit: keep.iter().map(|&i| self.logit[i]).collect(),
            converged: self.converged,
        }
    }

    /// `ln y` at score `v`, held at the end values outside the table.
    pub fn ln_share(&self, v: f64) -> f64 {
        let n = self.v.len();
        let i = self.v.partition_point(|&x| x <= v).saturating_sub(1).min(n - 2);
        self.hermite(i, v)
    }

    /// `ln_share` for scores in ascending order, the cell found by walking
    /// forward from `cursor`.
    #[inline]
    fn ln_share_ascending(&self, v: f64, cursor: &mut usize) -> f64 {
        let last = self.v.len() - 2;
        while *cursor < last && self.v[*cursor + 1] <= v {
            *cursor += 1;
        }
        self.hermite(*cursor, v)
    }

    #[inline]
    fn hermite(&self, i: usize, v: f64) -> f64 {
        let h = self.v[i + 1] - self.v[i];
        let t = ((v - self.v[i]) / h).clamp(0.0, 1.0);
        let (t2, t3) = (t * t, t * t * t);
        (2.0 * t3 - 3.0 * t2 + 1.0) * self.lny[i]
            + (t3 - 2.0 * t2 + t) * h * self.slope[i]
            + (-2.0 * t3 + 3.0 * t2) * self.lny[i + 1]
            + (t3 - t2) * h * self.slope[i + 1]
    }

    /// Scores at which the logit has risen by `cap` since the last, with the
    /// table's ends.
    pub fn breakpoints(&self, cap: f64) -> Vec<f64> {
        let mut out = vec![self.v[0]];
        let mut last = self.logit[0];
        for (&v, &lg) in self.v.iter().zip(&self.logit).skip(1) {
            if lg - last >= cap {
                out.push(v);
                last = lg;
            }
        }
        let end = *self.v.last().unwrap();
        if *out.last().unwrap() != end {
            out.push(end);
        }
        out
    }
}

/// The node at share `y`, or None where the score is not finite.
fn node(law: &BetaLaw, y: f64) -> Option<Node> {
    if !(y > 0.0 && y < 1.0) {
        return None;
    }
    let (cdf, sf) = law.tails(y);
    let v = if sf < 0.5 { -normal::quantile(sf) } else { normal::quantile(cdf) };
    if !v.is_finite() {
        return None;
    }
    let lny = y.ln();
    let slope = (-0.5 * v * v - 0.5 * (2.0 * std::f64::consts::PI).ln() - lny - law.ln_density(y)).exp();
    if !slope.is_finite() {
        return None;
    }
    Some(Node { v, y, lny, slope, logit: lny - (-y).ln_1p() })
}

fn midpoint(left: &Node, right: &Node) -> f64 {
    let (yl, yr) = (left.y, right.y);
    if yr > 2.0 * yl {
        (0.5 * (left.lny + right.lny)).exp()
    } else if 1.0 - yl > 2.0 * (1.0 - yr) {
        1.0 - ((1.0 - yl) * (1.0 - yr)).sqrt()
    } else {
        0.5 * (yl + yr)
    }
}

fn hermite_nodes(left: &Node, right: &Node, v: f64) -> f64 {
    let h = right.v - left.v;
    let t = (v - left.v) / h;
    let (t2, t3) = (t * t, t * t * t);
    (2.0 * t3 - 3.0 * t2 + 1.0) * left.lny
        + (t3 - 2.0 * t2 + t) * h * left.slope
        + (-2.0 * t3 + 3.0 * t2) * right.lny
        + (t3 - t2) * h * right.slope
}

/// Whether the cell `(left, right)` with its midpoint node `mid` must be split.
fn must_split(left: &Node, right: &Node, mid: &Node, tol: f64) -> bool {
    right.v - left.v > TABLE_MAX_DV
        || right.logit - left.logit > TABLE_MAX_DLOGIT
        || (hermite_nodes(left, right, mid.v) - mid.lny).abs() > tol
}

/// Gauss-Legendre points and weights on `[lo, hi]` split at `breaks` and
/// into pieces at most `width` long, appended to `points` and `weights`.
fn pieces<const N: usize>(
    lo: f64,
    hi: f64,
    breaks: impl Iterator<Item = f64>,
    width: f64,
    rule: (&[f64; N], &[f64; N]),
    points: &mut Vec<f64>,
    weights: &mut Vec<f64>,
) {
    points.clear();
    weights.clear();
    let mut edges: Vec<f64> = Vec::with_capacity(16);
    edges.push(lo);
    for b in breaks {
        if lo < b && b < hi && b > *edges.last().unwrap() {
            edges.push(b);
        }
    }
    edges.push(hi);
    for pair in edges.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let k = ((b - a) / width).ceil().max(1.0) as usize;
        let step = (b - a) / k as f64;
        for j in 0..k {
            let left = a + j as f64 * step;
            let right = if j + 1 == k { b } else { a + (j + 1) as f64 * step };
            let (half, mid) = (0.5 * (right - left), 0.5 * (left + right));
            for (&x, &w) in rule.0.iter().zip(rule.1) {
                points.push(mid + half * x);
                weights.push(half * w);
            }
        }
    }
}

/// The widths and caps of one level, and whether the tables are subsampled.
#[derive(Debug, Clone, Copy)]
struct Level {
    inner_width: f64,
    inner_cap: f64,
    outer_width: f64,
    coarse: bool,
}

impl Level {
    fn new(scale: f64, coarse: bool) -> Level {
        let factor = scale * if coarse { COARSE_SCALE } else { 1.0 };
        Level {
            inner_width: INNER_WIDTH * factor,
            inner_cap: INNER_CAP * factor,
            outer_width: OUTER_WIDTH * factor,
            coarse,
        }
    }
}

/// The law of a prepared unit for the closed forms.
fn unit_row_law(plan: &UnitPlan) -> RowLaw {
    match plan.kind {
        UnitKind::Atom | UnitKind::Drop => RowLaw::Atom { mu: plan.mean_loss },
        UnitKind::TwoPoint => RowLaw::TwoPoint { mu: plan.mean_loss, max_loss: plan.max_loss },
        UnitKind::Beta | UnitKind::Comonotone => RowLaw::Beta { mu: plan.mean_loss, sd: plan.sigma_used, max_loss: plan.max_loss },
    }
}

fn zero_probability(plan: &UnitPlan) -> f64 {
    match plan.kind {
        UnitKind::Atom => {
            if plan.mean_loss == 0.0 {
                1.0
            } else {
                0.0
            }
        }
        UnitKind::TwoPoint => (plan.max_loss - plan.mean_loss) / plan.max_loss,
        _ => 0.0,
    }
}

fn is_flat(plan: &UnitPlan) -> bool {
    matches!(plan.kind, UnitKind::Atom | UnitKind::Drop) || plan.w_corr == 0.0
}

/// One unit's conditional first and second moments given the factor.
struct UnitTerm {
    plan: UnitPlan,
    moments: [f64; 3],
    w: f64,
    table: Option<ScoreTable>,
    coarse_table: Option<ScoreTable>,
}

impl UnitTerm {
    fn new(plan: UnitPlan, rtol: f64, tol: f64) -> Result<UnitTerm, Error> {
        let moments = unit_row_law(&plan).raw_moments();
        let w = if matches!(plan.kind, UnitKind::Beta | UnitKind::Comonotone | UnitKind::TwoPoint) { plan.w_corr } else { 0.0 };
        let table = if matches!(plan.kind, UnitKind::Beta | UnitKind::Comonotone) && w > 0.0 {
            Some(ScoreTable::new(plan.alpha, plan.beta, rtol, tol)?)
        } else {
            None
        };
        let coarse_table = table.as_ref().map(ScoreTable::subsampled);
        Ok(UnitTerm { plan, moments, w, table, coarse_table })
    }

    fn flat(&self) -> bool {
        self.w == 0.0
    }

    fn table(&self, level: &Level) -> &ScoreTable {
        if level.coarse { self.coarse_table.as_ref().unwrap() } else { self.table.as_ref().unwrap() }
    }

    /// Factor values at which the conditional moments change character.
    fn outer_breaks(&self, level: &Level, out: &mut Vec<f64>) {
        if self.flat() {
            return;
        }
        if self.plan.kind == UnitKind::TwoPoint {
            if self.plan.w_indep == 0.0 {
                out.push(normal::quantile(1.0 - self.plan.two_point_p()));
            }
            return;
        }
        let s = ((1.0 - self.w) * (1.0 + self.w)).sqrt();
        if s / self.w >= STEEP_KERNEL {
            return;
        }
        out.extend(self.table(level).breakpoints(level.inner_cap).into_iter().map(|v| v / self.w));
    }

    /// `(P(X | z), P(X^2 | z))` at every `z`, added to `a1` and `a2`
    /// (cleared first); returns the integrand evaluations.
    fn conditional(&self, z: &[f64], level: &Level, a1: &mut Vec<f64>, a2: &mut Vec<f64>, scratch: &mut Scratch) -> usize {
        a1.clear();
        a2.clear();
        if self.flat() {
            a1.resize(z.len(), self.moments[0]);
            a2.resize(z.len(), self.moments[1]);
            return 0;
        }
        let m_loss = self.plan.max_loss;
        if self.plan.kind == UnitKind::TwoPoint {
            let p = self.plan.two_point_p();
            for &zz in z {
                let at_max = two_point_conditional(p, self.w, self.plan.w_indep, zz);
                a1.push(m_loss * at_max);
                a2.push(m_loss * m_loss * at_max);
            }
            return z.len();
        }
        let table = self.table(level);
        let breaks = table.breakpoints(level.inner_cap);
        let s = ((1.0 - self.w) * (1.0 + self.w)).sqrt();
        let (y0, yn) = (table.lny[0].exp(), table.lny[table.len() - 1].exp());
        let (v0, vn) = (table.v[0], table.v[table.len() - 1]);
        let mut count = 0;
        for &zz in z {
            let m = self.w * zz;
            if s == 0.0 {
                let y = table.ln_share(m).exp();
                a1.push(m_loss * y);
                a2.push(m_loss * y * m_loss * y);
                count += 1;
                continue;
            }
            let (t0, tn) = ((v0 - m) / s, (vn - m) / s);
            let (lo, hi) = (t0.max(-REACH), tn.min(REACH));
            let (mut first, mut second) = (0.0, 0.0);
            if t0 > -REACH {
                let p = normal::cdf(t0.min(REACH));
                first += y0 * p;
                second += y0 * y0 * p;
            }
            if tn < REACH {
                let p = normal::cdf(-tn.max(-REACH));
                first += yn * p;
                second += yn * yn * p;
            }
            if hi > lo {
                pieces(lo, hi, breaks.iter().map(|&b| (b - m) / s), level.inner_width, (&GL8_X, &GL8_W), &mut scratch.points, &mut scratch.weights);
                let mut cursor = 0;
                for (&t, &w) in scratch.points.iter().zip(&scratch.weights) {
                    let y = table.ln_share_ascending(m + s * t, &mut cursor).exp();
                    let wt = w * phi(t);
                    first += y * wt;
                    second += y * y * wt;
                }
                count += scratch.points.len();
            }
            a1.push(m_loss * first);
            a2.push(m_loss * m_loss * second);
        }
        count
    }
}

/// Reusable buffers of one worker.
#[derive(Default)]
struct Scratch {
    points: Vec<f64>,
    weights: Vec<f64>,
}

/// Barycentric weights of the 8-point Gauss-Legendre nodes.
fn barycentric() -> [f64; 8] {
    std::array::from_fn(|j| 1.0 / (0..8).filter(|&k| k != j).map(|k| GL8_X[j] - GL8_X[k]).product::<f64>())
}

impl UnitTerm {
    /// The piece edges of this beta unit's own factor nodes (a smooth unit
    /// varies on its kernel's width in z, a steep one at its table's
    /// breakpoints mapped to z).
    fn native_edges(&self, level: &Level, z_high: f64) -> Vec<f64> {
        let kernel = ((1.0 - self.w) * (1.0 + self.w)).sqrt() / self.w;
        let (mut breaks, width) = if kernel >= STEEP_KERNEL {
            (Vec::new(), level.outer_width.min(SMOOTH_NATIVE * kernel))
        } else {
            let b: Vec<f64> = self.table(level).breakpoints(level.inner_cap).into_iter().map(|v| v / self.w).collect();
            (b, level.outer_width.min(STEEP_NATIVE.max(1.5 * kernel)))
        };
        breaks.retain(|&b| Z_LOW < b && b < z_high);
        breaks.sort_by(f64::total_cmp);
        breaks.dedup();
        let mut edges = vec![Z_LOW];
        let cut = |a: f64, b: f64, edges: &mut Vec<f64>| {
            let k = ((b - a) / width).ceil().max(1.0) as usize;
            for j in 1..=k {
                edges.push(if j == k { b } else { a + (b - a) * j as f64 / k as f64 });
            }
        };
        let mut previous = Z_LOW;
        for &b in &breaks {
            cut(previous, b, &mut edges);
            previous = b;
        }
        cut(previous, z_high, &mut edges);
        edges
    }
}

/// Values at `z` from values at each piece's 8 Gauss-Legendre nodes, by
/// barycentric Lagrange interpolation of their logarithm within the piece
/// holding `z` (a conditional moment is positive and its logarithm smoother
/// than itself where it climbs by orders of magnitude); `z` ascending.
fn interpolate_log(edges: &[f64], values: &mut [f64], z: &[f64], weights: &[f64; 8], out: &mut Vec<f64>) {
    out.clear();
    for v in values.iter_mut() {
        *v = v.max(1e-300).ln();
    }
    let mut piece = 0;
    let last = edges.len() - 2;
    for &zz in z {
        while piece < last && edges[piece + 1] <= zz {
            piece += 1;
        }
        let (a, b) = (edges[piece], edges[piece + 1]);
        let x = (2.0 * zz - a - b) / (b - a);
        let logs = &values[piece * 8..piece * 8 + 8];
        let (mut num, mut den, mut exact) = (0.0, 0.0, None);
        for j in 0..8 {
            let d = x - GL8_X[j];
            if d == 0.0 {
                exact = Some(logs[j]);
                break;
            }
            let t = weights[j] / d;
            num += t * logs[j];
            den += t;
        }
        out.push(exact.unwrap_or(num / den).exp());
    }
}

/// The cross parts at one level: `(int c_2 phi, int c_3 phi, the factor
/// integrals of each unit's (a1, a2), evaluations)`. Flat units are
/// constants, two-point units and units at loading one are evaluated at the
/// event's factor nodes, the other beta units at their own nodes and
/// interpolated onto the event's.
fn cross_parts(units: &[UnitTerm], level: &Level, z_high: f64, scratch: &mut Scratch) -> (f64, f64, Vec<(f64, f64)>, usize) {
    let mut breaks: Vec<f64> = Vec::new();
    for u in units {
        u.outer_breaks(level, &mut breaks);
    }
    breaks.sort_by(f64::total_cmp);
    breaks.dedup();
    let (mut z, mut wz) = (Vec::new(), Vec::new());
    pieces(Z_LOW, z_high, breaks.into_iter(), level.outer_width, (&GL8_X, &GL8_W), &mut z, &mut wz);
    for (w, &zz) in wz.iter_mut().zip(&z) {
        *w *= phi(zz);
    }
    let n = z.len();
    let (mut t1, mut t2, mut c2, mut c3) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let (mut a1, mut a2) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut b1, mut b2) = (Vec::new(), Vec::new());
    let (mut native, mut native_w) = (Vec::new(), Vec::new());
    let bary = barycentric();
    let mut marginals = Vec::with_capacity(units.len());
    let mut evaluations = 0;
    for u in units {
        if u.table.is_some() && u.w < 1.0 {
            let edges = u.native_edges(level, z_high);
            pieces(edges[0], edges[edges.len() - 1], edges[1..edges.len() - 1].iter().copied(), f64::INFINITY, (&GL8_X, &GL8_W), &mut native, &mut native_w);
            evaluations += u.conditional(&native, level, &mut b1, &mut b2, scratch);
            interpolate_log(&edges, &mut b1, &z, &bary, &mut a1);
            interpolate_log(&edges, &mut b2, &z, &bary, &mut a2);
        } else {
            evaluations += u.conditional(&z, level, &mut a1, &mut a2, scratch);
        }
        let (mut m1, mut m2) = (Neumaier::default(), Neumaier::default());
        for k in 0..n {
            c3[k] += 3.0 * t2[k] * a1[k] + 3.0 * t1[k] * a2[k];
            c2[k] += 2.0 * t1[k] * a1[k];
            t2[k] += 2.0 * t1[k] * a1[k] + a2[k];
            t1[k] += a1[k];
            m1.add(a1[k] * wz[k]);
            m2.add(a2[k] * wz[k]);
        }
        marginals.push((m1.value(), m2.value()));
    }
    let (mut s2, mut s3) = (Neumaier::default(), Neumaier::default());
    for k in 0..n {
        s2.add(c2[k] * wz[k]);
        s3.add(c3[k] * wz[k]);
    }
    (s2.value(), s3.value(), marginals, evaluations)
}

/// How one event's moments were obtained.
#[derive(Debug, Clone, Copy)]
struct Record {
    integrated: bool,
    /// Relative differences of the two levels in `C_2` and `C_3`.
    error: [f64; 2],
    residual: f64,
    evaluations: usize,
    converged: bool,
}

const CLOSED: Record = Record {
    integrated: false,
    error: [0.0, 0.0],
    residual: 0.0,
    evaluations: 0,
    converged: true,
};

/// `int prod_u P(X_u = 0 | z) phi(z) dz`: zero unless every unit is a
/// two-point unit or an atom at zero.
fn integrated_zero(plans: &[UnitPlan], z_high: f64) -> f64 {
    if plans
        .iter()
        .any(|p| matches!(p.kind, UnitKind::Beta | UnitKind::Comonotone) || (p.kind == UnitKind::Atom && p.mean_loss > 0.0))
    {
        return 0.0;
    }
    let mut steps: Vec<f64> = plans
        .iter()
        .filter(|p| p.kind == UnitKind::TwoPoint && p.w_indep == 0.0)
        .map(|p| normal::quantile(1.0 - p.two_point_p()))
        .collect();
    steps.sort_by(f64::total_cmp);
    steps.dedup();
    let (mut z, mut w) = (Vec::new(), Vec::new());
    pieces(Z_LOW, z_high, steps.into_iter(), 0.5, (&GL8_X, &GL8_W), &mut z, &mut w);
    let product = |zz: f64| -> f64 {
        plans
            .iter()
            .filter(|p| p.kind == UnitKind::TwoPoint)
            .map(|p| 1.0 - two_point_conditional(p.two_point_p(), p.w_corr, p.w_indep, zz))
            .product()
    };
    let mut acc = Neumaier::default();
    for (&zz, &ww) in z.iter().zip(&w) {
        acc.add(product(zz) * ww * phi(zz));
    }
    acc.value() + normal::cdf(Z_LOW) * product(z[0])
}

/// Raw moments, zero probability and record of one event's total, its
/// units coupled, with two or more units under `burke` or `comonotone`.
fn integrate_event(plans: &[UnitPlan], rtol: f64, max_refinements: usize, scratch: &mut Scratch) -> Result<([f64; 3], f64, Record), Error> {
    let marginal: Vec<[f64; 3]> = plans.iter().map(|p| unit_row_law(p).raw_moments()).collect();
    let marginal_sum: [f64; 3] = std::array::from_fn(|r| marginal.iter().map(|m| m[r]).sum());
    let support: f64 = plans.iter().map(UnitPlan::support).sum();
    let ratio = (marginal_sum[1] / (support * support)).min(marginal_sum[2] / (support * support * support));
    let z_high = (-normal::quantile(0.01 * rtol * ratio)).clamp(Z_HIGH_MIN, 37.0);
    let mut scale: f64 = 1.0;
    let mut previous: Option<(f64, f64, Vec<(f64, f64)>, usize)> = None;
    let mut record = Record { integrated: true, ..CLOSED };
    let mut moments = marginal_sum;
    for _ in 0..=max_refinements {
        let tol = (rtol * TABLE_TOL_FRACTION * scale.powi(4)).max(TABLE_TOL_FLOOR);
        let units: Vec<UnitTerm> = plans.iter().map(|p| UnitTerm::new(p.clone(), rtol, tol)).collect::<Result<_, _>>()?;
        let fine = cross_parts(&units, &Level::new(scale, false), z_high, scratch);
        record.evaluations += fine.3;
        let coarse = match previous.take() {
            Some(p) => p,
            None => {
                let c = cross_parts(&units, &Level::new(scale, true), z_high, scratch);
                record.evaluations += c.3;
                c
            }
        };
        moments = [marginal_sum[0], marginal_sum[1] + fine.0, marginal_sum[2] + fine.1];
        let error = [(fine.0 - coarse.0).abs(), (fine.1 - coarse.1).abs()];
        record.error = [error[0] / moments[1], error[1] / moments[2]];
        record.residual = fine
            .2
            .iter()
            .zip(&marginal)
            .map(|(&(f1, f2), m)| ((f1 - m[0]).abs() / m[0].max(1e-300)).max((f2 - m[1]).abs() / m[1].max(1e-300)))
            .fold(0.0, f64::max);
        let tables_ok = units.iter().all(|u| u.table.as_ref().is_none_or(|t| t.converged));
        record.converged = tables_ok && error[0] <= rtol * moments[1] && error[1] <= rtol * moments[2];
        if record.converged {
            break;
        }
        previous = Some(fine);
        scale *= 0.5;
    }
    Ok((moments, integrated_zero(plans, z_high), record))
}

/// Configuration of the shared-event moments.
#[derive(Debug, Clone)]
pub struct MomentConfig {
    pub coupling: Coupling,
    pub variance_policy: VariancePolicy,
    pub rtol: f64,
    pub max_refinements: usize,
    pub threads: Option<usize>,
    /// Rows per parallel chunk, rounded up to whole events.
    pub chunk_rows: usize,
}

impl MomentConfig {
    pub fn new() -> MomentConfig {
        MomentConfig {
            coupling: Coupling::Burke,
            variance_policy: VariancePolicy::default(),
            rtol: MOMENT_RTOL,
            max_refinements: MAX_REFINEMENTS,
            threads: None,
            chunk_rows: DEFAULT_MULTI_CHUNK_ROWS,
        }
    }
}

impl Default for MomentConfig {
    fn default() -> Self {
        MomentConfig::new()
    }
}

/// One event that did not converge.
#[derive(Debug, Clone, PartialEq)]
pub struct UnconvergedEvent {
    /// First row of the event (callers translate to the event id).
    pub event_row: usize,
    pub error: [f64; 2],
}

/// Model moments of a multi-unit ELT's event totals (spec section 4B).
#[derive(Debug, Clone)]
pub struct SharedMoments {
    pub total_rate: f64,
    pub raw_moments: [f64; 3],
    pub rate_at_zero: f64,
    /// Per event, the sum of its units' supports; NaN where none contributes.
    pub support: Vec<f64>,
    pub max_support: f64,
    pub input_rows: usize,
    pub valid_rows: usize,
    pub atom_rows: usize,
    pub beta_rows: usize,
    pub two_point_rows: usize,
    pub events: usize,
    pub valid_events: usize,
    pub one_unit_events: usize,
    pub integrated: bool,
    pub achieved_error: [f64; 2],
    pub marginal_residual: f64,
    pub evaluations: usize,
    pub unconverged_events: usize,
    pub unconverged: Vec<UnconvergedEvent>,
    pub moment_overflow: [bool; 3],
    pub diagnostics: Diagnostics,
}

/// Per-chunk partial result.
#[derive(Default)]
struct Partial {
    rate: Neumaier,
    zero: Neumaier,
    moments: [Neumaier; 3],
    overflow: [bool; 3],
    errors: [Neumaier; 2],
    support: Vec<(usize, f64)>,
    rows: [usize; 3],
    valid_events: usize,
    one_unit_events: usize,
    integrated: bool,
    residual: f64,
    evaluations: usize,
    unconverged_events: usize,
    unconverged: Vec<UnconvergedEvent>,
}

/// Model moments of a multi-unit ELT's event totals under the coupling of
/// `config` (spec section 4B): events validated and classified as
/// `multiunit_views` does; per event, one contributing unit and the
/// `independent` and `homer` couplings by closed forms, `burke` and
/// `comonotone` by the integration of the module notes; `matched` refused.
/// Events are processed in fixed chunks folded in chunk order, so the
/// result is bitwise identical for any thread count.
pub fn multiunit_moments(input: &MultiInput<'_>, config: &MomentConfig) -> Result<SharedMoments, Error> {
    if config.coupling == Coupling::Matched {
        return Err(Error::MatchedRefused);
    }
    if !(config.rtol > 0.0 && config.rtol < 1.0) {
        return Err(Error::BadMomentTolerance(config.rtol));
    }
    if let VariancePolicy::ClampSigma { factor } = config.variance_policy
        && !(factor.is_finite() && factor > 0.0 && factor < 1.0)
    {
        return Err(Error::BadClampFactor(factor));
    }
    if config.chunk_rows < 1 {
        return Err(Error::BadChunkRows(config.chunk_rows));
    }
    if config.threads == Some(0) {
        return Err(Error::BadThreads(0));
    }
    let pass = event_pass(input, config.variance_policy, None)?;
    let n_events = pass.events.len();
    let mut chunks: Vec<(usize, usize)> = Vec::new();
    let (mut first, mut rows) = (0, 0);
    for (e, &(s, t)) in pass.events.iter().enumerate() {
        rows += t - s;
        if rows >= config.chunk_rows {
            chunks.push((first, e + 1));
            first = e + 1;
            rows = 0;
        }
    }
    if first < n_events {
        chunks.push((first, n_events));
    }
    let process = |&(lo, hi): &(usize, usize)| -> Result<Partial, Error> {
        let mut partial = Partial::default();
        let mut scratch = Scratch::default();
        for e in lo..hi {
            if !pass.rated[e] {
                continue;
            }
            let (start, end) = pass.events[e];
            let live: Vec<usize> = (start..end).filter(|&row| pass.classes[row].kind != UnitKind::Drop).collect();
            if live.is_empty() {
                continue;
            }
            let lam = input.rate[start];
            partial.valid_events += 1;
            let plans: Vec<UnitPlan> = live
                .iter()
                .map(|&row| UnitPlan::from_row(&pass.classes[row], &input.spec(row)).couple(config.coupling))
                .collect();
            for p in &plans {
                let slot = match p.kind {
                    UnitKind::Atom => 0,
                    UnitKind::TwoPoint => 2,
                    _ => 1,
                };
                partial.rows[slot] += 1;
            }
            let mut support: f64 = plans.iter().map(UnitPlan::support).sum();
            let (moments, zero, record) = if plans.len() == 1 {
                partial.one_unit_events += 1;
                (unit_row_law(&plans[0]).raw_moments(), zero_probability(&plans[0]), CLOSED)
            } else if config.coupling == Coupling::Homer {
                let specs: Vec<UnitSpec> = live.iter().map(|&row| input.spec(row)).collect();
                let rolled = prepare_unit(&rolled_spec(&specs), "rolled", config.variance_policy)?;
                support = support.max(rolled.support());
                if rolled.kind == UnitKind::Drop {
                    ([0.0; 3], 0.0, CLOSED)
                } else {
                    (unit_row_law(&rolled).raw_moments(), zero_probability(&rolled), CLOSED)
                }
            } else if config.coupling == Coupling::Independent || plans.iter().all(is_flat) {
                let mut t = [0.0_f64; 3];
                for p in &plans {
                    let m = unit_row_law(p).raw_moments();
                    t = [t[0] + m[0], t[1] + 2.0 * t[0] * m[0] + m[1], t[2] + 3.0 * t[1] * m[0] + 3.0 * t[0] * m[1] + m[2]];
                }
                (t, plans.iter().map(zero_probability).product(), CLOSED)
            } else {
                integrate_event(&plans, config.rtol, config.max_refinements, &mut scratch).map_err(|error| match error {
                    Error::KernelFailure { alpha, beta, .. } => Error::KernelFailure { row: start, alpha, beta },
                    other => other,
                })?
            };
            partial.support.push((e, support));
            partial.rate.add(lam);
            partial.zero.add(lam * zero);
            for r in 0..3 {
                let term = lam * moments[r];
                if term.is_finite() {
                    partial.moments[r].add(term);
                } else {
                    partial.overflow[r] = true;
                }
            }
            if record.integrated {
                partial.integrated = true;
                partial.errors[0].add(lam * moments[1] * record.error[0]);
                partial.errors[1].add(lam * moments[2] * record.error[1]);
                partial.residual = partial.residual.max(record.residual);
                partial.evaluations += record.evaluations;
                if !record.converged {
                    partial.unconverged_events += 1;
                    if partial.unconverged.len() < MAX_EXAMPLES {
                        partial.unconverged.push(UnconvergedEvent { event_row: start, error: record.error });
                    }
                }
            }
        }
        Ok(partial)
    };
    let run = || -> Result<Vec<Partial>, Error> { chunks.par_iter().map(process).collect() };
    let partials = match config.threads {
        Some(threads) => rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| Error::ThreadPool(e.to_string()))?
            .install(run)?,
        None => run()?,
    };
    let mut out = SharedMoments {
        total_rate: 0.0,
        raw_moments: [0.0; 3],
        rate_at_zero: 0.0,
        support: vec![f64::NAN; n_events],
        max_support: 0.0,
        input_rows: input.rate.len(),
        valid_rows: 0,
        atom_rows: 0,
        beta_rows: 0,
        two_point_rows: 0,
        events: n_events,
        valid_events: 0,
        one_unit_events: 0,
        integrated: false,
        achieved_error: [0.0; 2],
        marginal_residual: 0.0,
        evaluations: 0,
        unconverged_events: 0,
        unconverged: Vec::new(),
        moment_overflow: [false; 3],
        diagnostics: pass.diagnostics.clone(),
    };
    let (mut rate, mut zero) = (Neumaier::default(), Neumaier::default());
    let mut moments = [Neumaier::default(); 3];
    let mut errors = [Neumaier::default(); 2];
    for p in partials {
        rate.add(p.rate.value());
        zero.add(p.zero.value());
        for r in 0..3 {
            moments[r].add(p.moments[r].value());
            out.moment_overflow[r] |= p.overflow[r];
        }
        errors[0].add(p.errors[0].value());
        errors[1].add(p.errors[1].value());
        for (e, s) in p.support {
            out.support[e] = s;
            out.max_support = out.max_support.max(s);
        }
        out.atom_rows += p.rows[0];
        out.beta_rows += p.rows[1];
        out.two_point_rows += p.rows[2];
        out.valid_events += p.valid_events;
        out.one_unit_events += p.one_unit_events;
        out.integrated |= p.integrated;
        out.marginal_residual = out.marginal_residual.max(p.residual);
        out.evaluations += p.evaluations;
        out.unconverged_events += p.unconverged_events;
        for u in p.unconverged {
            if out.unconverged.len() < MAX_EXAMPLES {
                out.unconverged.push(u);
            }
        }
    }
    out.total_rate = rate.value();
    out.rate_at_zero = zero.value();
    for r in 0..3 {
        let value = moments[r].value();
        if out.moment_overflow[r] || !value.is_finite() {
            out.moment_overflow[r] = true;
            out.raw_moments[r] = f64::INFINITY;
        } else {
            out.raw_moments[r] = value;
        }
    }
    for r in 0..2 {
        let c = out.raw_moments[r + 1];
        out.achieved_error[r] = if c > 0.0 && c.is_finite() { errors[r].value() / c } else { 0.0 };
    }
    out.valid_rows = out.atom_rows + out.beta_rows + out.two_point_rows;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(rows: &'a [(i64, i64, f64, f64, f64, f64, f64)], cols: &'a mut Vec<Vec<f64>>, codes: &'a mut Vec<Vec<i64>>) -> MultiInput<'a> {
        *codes = vec![rows.iter().map(|r| r.0).collect(), rows.iter().map(|r| r.1).collect()];
        *cols = (0..5)
            .map(|k| rows.iter().map(|r| [r.2, r.3, r.4, r.5, r.6][k]).collect())
            .collect();
        MultiInput {
            event_code: &codes[0],
            unit_code: &codes[1],
            rate: &cols[0],
            mean_loss: &cols[1],
            sd_indep: &cols[2],
            sd_corr: &cols[3],
            max_loss: &cols[4],
        }
    }

    // [Eltq-Model-Moments] Spec test 36: two two-point units are exact (the
    // cross moment is M_u M_w P(V_u > y_u, V_w > y_w)); one-unit events and
    // the independent coupling are closed forms; results are bitwise across
    // thread counts; matched is refused.
    #[test]
    fn two_point_pair_and_closed_forms() {
        let rows = [(0, 0, 1.0, 30.0, 50.0, 30.0, 100.0), (0, 1, 1.0, 20.0, 40.0, 60.0, 200.0)];
        let (mut cols, mut codes) = (Vec::new(), Vec::new());
        let m = multiunit_moments(&input(&rows, &mut cols, &mut codes), &MomentConfig::new()).unwrap();
        assert!((m.raw_moments[1] - 9387.097506566148).abs() < 1e-9 * 9387.0);
        assert!((m.raw_moments[2] - 2174193.877954766).abs() < 1e-9 * 2.17e6);
        assert!(m.integrated && m.unconverged_events == 0);
        let one = [(0, 0, 0.1, 50.0, 10.0, 10.0, 100.0)];
        let one_m = multiunit_moments(&input(&one, &mut cols, &mut codes), &MomentConfig::new()).unwrap();
        assert!((one_m.raw_moments[2] - 18_500.0).abs() < 1e-9 && !one_m.integrated);
        let mut config = MomentConfig::new();
        config.coupling = Coupling::Independent;
        let ind = multiunit_moments(&input(&rows, &mut cols, &mut codes), &config).unwrap();
        assert!((ind.raw_moments[1] - (30.0 * 100.0 + 20.0 * 200.0 + 2.0 * 30.0 * 20.0)).abs() < 1e-9);
        config.coupling = Coupling::Matched;
        assert!(matches!(multiunit_moments(&input(&rows, &mut cols, &mut codes), &config), Err(Error::MatchedRefused)));
        let betas = [(0, 0, 1.0, 200.0, 100.0, 150.0, 1000.0), (0, 1, 1.0, 500.0, 300.0, 200.0, 2000.0), (1, 0, 0.5, 300.0, 50.0, 400.0, 3000.0), (1, 1, 0.5, 40.0, 30.0, 30.0, 400.0)];
        let mut first = MomentConfig::new();
        first.threads = Some(1);
        first.chunk_rows = 2;
        let a = multiunit_moments(&input(&betas, &mut cols, &mut codes), &first).unwrap();
        first.threads = Some(3);
        let b = multiunit_moments(&input(&betas, &mut cols, &mut codes), &first).unwrap();
        assert_eq!(a.raw_moments.map(f64::to_bits), b.raw_moments.map(f64::to_bits));
        assert!((a.total_rate - 1.5).abs() < 1e-15 && a.valid_events == 2);
    }
}
