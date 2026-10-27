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
    fn native_edges(&self, level: &Level, z_high: f64, extra: &[f64]) -> Vec<f64> {
        let kernel = ((1.0 - self.w) * (1.0 + self.w)).sqrt() / self.w;
        let (mut breaks, width) = if kernel >= STEEP_KERNEL {
            (Vec::new(), level.outer_width.min(SMOOTH_NATIVE * kernel))
        } else {
            let b: Vec<f64> = self.table(level).breakpoints(level.inner_cap).into_iter().map(|v| v / self.w).collect();
            (b, level.outer_width.min(STEEP_NATIVE.max(1.5 * kernel)))
        };
        breaks.extend_from_slice(extra);
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

/// `interpolate_log` from logarithms already formed (exact where the values
/// underflow); a piece whose logarithms are all `-inf` gives zero, one with
/// some is interpolated in the values.
fn interpolate_logs(edges: &[f64], logs: &[f64], z: &[f64], weights: &[f64; 8], out: &mut Vec<f64>) {
    out.clear();
    let mut piece = 0;
    let last = edges.len() - 2;
    for &zz in z {
        while piece < last && edges[piece + 1] <= zz {
            piece += 1;
        }
        let (a, b) = (edges[piece], edges[piece + 1]);
        let x = (2.0 * zz - a - b) / (b - a);
        let local = &logs[piece * 8..piece * 8 + 8];
        let finite = local.iter().all(|v| v.is_finite());
        let (mut num, mut den, mut exact) = (0.0, 0.0, None);
        for j in 0..8 {
            let value = if finite { local[j] } else { local[j].exp() };
            let d = x - GL8_X[j];
            if d == 0.0 {
                exact = Some(value);
                break;
            }
            let t = weights[j] / d;
            num += t * value;
            den += t;
        }
        let result = exact.unwrap_or(num / den);
        out.push(if finite { result.exp() } else { result.max(0.0) });
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
            let edges = u.native_edges(level, z_high, &[]);
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

// ---------------------------------------------------------------------------
// [Eltq-Tail-Queries] tail measures of shared events with no grid (spec
// section 4B, test 38): exact for one-unit events, the homer coupling, events
// of atoms and events whose total is a function of the factor alone;
// otherwise the union bound, its conditional bounds integrated over the
// factor by the moment pass's rule and widened by `TAIL_MARGIN`.

/// Relative margin by which the shared tail query widens its bounds, a
/// hundred times the measured accuracy of the factor integration (about
/// 1e-8, spec test 36): the bounds are valid to that accuracy.
pub const TAIL_MARGIN: f64 = 1e-6;
/// A threshold within this many standard scores of the inner rule's reach
/// has its tail integrated past the reach: beyond it lies more than 5e-13
/// of its tail.
const DEEP_FROM: f64 = 5.0;
/// The span past the reach, in units of `1 / a` at its start `a`, after
/// which the normal weight has fallen by `e^-40`.
const DEEP_SPAN: f64 = 40.0;
/// Width of its Gauss-Legendre pieces in the same units (`e^-5` each).
const DEEP_WIDTH: f64 = 5.0;
/// A unit's tail beyond `x` given the factor switches on as
/// `Phi((w z - v_x) / s)`, over a factor width `k = s / w` about
/// `z = v_x / w`: sharper than the moment pass resolves when `k` is below
/// this. Such a transition gets breakpoints `TRANSITION_STEP k` apart over
/// `TRANSITION_SPAN` in units of `k` (below its start the sigmoid is under
/// `Phi(-10)`).
const TRANSITION_KERNEL: f64 = 1.25;
const TRANSITION_STEP: f64 = 2.0;
const TRANSITION_SPAN: (f64, f64) = (-10.0, 6.0);

const BINOMIAL: [[f64; 4]; 4] = [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 2.0, 1.0, 0.0], [1.0, 3.0, 3.0, 1.0]];

impl UnitTerm {
    /// The score of share `y` of this beta unit, exact (`-inf` at or below
    /// zero, `+inf` at or above one).
    fn score_of(&self, y: f64) -> Result<f64, Error> {
        if y <= 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if y >= 1.0 {
            return Ok(f64::INFINITY);
        }
        let law = BetaLaw::new(self.plan.alpha, self.plan.beta).ok_or(Error::KernelFailure {
            row: 0,
            alpha: self.plan.alpha,
            beta: self.plan.beta,
        })?;
        let (cdf, sf) = law.tails(y);
        Ok(if sf < 0.5 { -normal::quantile(sf) } else { normal::quantile(cdf) })
    }

    /// Breakpoints in the factor where this unit's tails beyond
    /// `thresholds` (with their `scores`) switch on, added to `out`: a step
    /// for a beta unit at loading one, else `TRANSITION_STEP k` apart over
    /// `TRANSITION_SPAN` about each transition sharper than
    /// `TRANSITION_KERNEL`.
    fn transitions(&self, thresholds: &[f64], scores: &[f64], z_high: f64, out: &mut Vec<f64>) {
        let plan = &self.plan;
        if self.w == 0.0 || matches!(plan.kind, UnitKind::Atom | UnitKind::Drop) || self.flat() {
            return;
        }
        let mut centers = Vec::new();
        let k = if plan.kind == UnitKind::TwoPoint {
            // The step at loading one is among the moment pass's breakpoints.
            if plan.w_indep == 0.0 || !thresholds.iter().any(|&x| x < plan.max_loss) {
                return;
            }
            centers.push(normal::quantile(1.0 - plan.two_point_p()) / self.w);
            plan.w_indep / self.w
        } else {
            let s = ((1.0 - self.w) * (1.0 + self.w)).sqrt();
            for (&x, &v) in thresholds.iter().zip(scores) {
                if 0.0 < x && x < plan.max_loss {
                    centers.push(if s == 0.0 { v } else { v / self.w });
                }
            }
            if s == 0.0 {
                out.extend(centers.into_iter().filter(|&v| Z_LOW < v && v < z_high));
                return;
            }
            s / self.w
        };
        if k >= TRANSITION_KERNEL {
            return;
        }
        let steps = ((TRANSITION_SPAN.1 - TRANSITION_SPAN.0) / TRANSITION_STEP).round() as usize;
        for c in centers {
            for j in 0..=steps {
                let b = c + k * (TRANSITION_SPAN.0 + j as f64 * TRANSITION_STEP);
                if Z_LOW < b && b < z_high {
                    out.push(b);
                }
            }
        }
    }

    /// `P(X^k; X > x | z)` for k = 0..3 at every `z` and every threshold
    /// (a negative threshold gives the full moments), written to `out` as
    /// `out[(i * thresholds.len() + j) * 4 + k]`; returns the evaluations.
    fn tails(&self, z: &[f64], thresholds: &[f64], scores: &[f64], out: &mut Vec<f64>, mut logs: Option<&mut Vec<f64>>, scratch: &mut Scratch) -> Result<usize, Error> {
        let n_x = thresholds.len();
        out.clear();
        out.resize(z.len() * n_x * 4, 0.0);
        let plan = &self.plan;
        let powers = |x: f64| [1.0, x, x * x, x * x * x];
        match plan.kind {
            UnitKind::Atom | UnitKind::Drop => {
                let p = powers(plan.mean_loss);
                for i in 0..z.len() {
                    for (j, &x) in thresholds.iter().enumerate() {
                        if plan.mean_loss > x {
                            out[(i * n_x + j) * 4..(i * n_x + j) * 4 + 4].copy_from_slice(&p);
                        }
                    }
                }
                return Ok(0);
            }
            UnitKind::TwoPoint => {
                let m = powers(plan.max_loss);
                for (i, &zz) in z.iter().enumerate() {
                    let p = if self.w > 0.0 { two_point_conditional(plan.two_point_p(), self.w, plan.w_indep, zz) } else { plan.two_point_p() };
                    for (j, &x) in thresholds.iter().enumerate() {
                        let slot = &mut out[(i * n_x + j) * 4..(i * n_x + j) * 4 + 4];
                        if plan.max_loss > x {
                            for k in 0..4 {
                                slot[k] = p * m[k];
                            }
                        }
                        if x < 0.0 {
                            slot[0] = 1.0;
                        }
                    }
                }
                return Ok(z.len());
            }
            _ => {}
        }
        let m_loss = plan.max_loss;
        let scale = powers(m_loss);
        if self.flat() {
            let law = BetaLaw::new(plan.alpha, plan.beta).ok_or(Error::KernelFailure { row: 0, alpha: plan.alpha, beta: plan.beta })?;
            let mut values = vec![[0.0; 4]; n_x];
            for (j, &x) in thresholds.iter().enumerate() {
                values[j] = if x < 0.0 {
                    [1.0, self.moments[0], self.moments[1], self.moments[2]]
                } else if x >= m_loss {
                    [0.0; 4]
                } else {
                    let y = x / m_loss;
                    let k = match law {
                        BetaLaw::Regular(bf) => crate::tails::beta_tail(&bf, y, 0.0).0,
                        BetaLaw::Narrow(narrow) => narrow.centred_tail(y, 0.0),
                    };
                    std::array::from_fn(|r| k[r].max(0.0) * scale[r])
                };
            }
            for i in 0..z.len() {
                for j in 0..n_x {
                    out[(i * n_x + j) * 4..(i * n_x + j) * 4 + 4].copy_from_slice(&values[j]);
                }
            }
            return Ok(0);
        }
        let table = self.table.as_ref().unwrap();
        let s = ((1.0 - self.w) * (1.0 + self.w)).sqrt();
        if s == 0.0 {
            for (i, &zz) in z.iter().enumerate() {
                let x = m_loss * table.ln_share(zz).exp();
                let p = powers(x);
                for (j, &thr) in thresholds.iter().enumerate() {
                    if x > thr {
                        out[(i * n_x + j) * 4..(i * n_x + j) * 4 + 4].copy_from_slice(&p);
                    }
                }
            }
            return Ok(z.len());
        }
        let breaks = table.breakpoints(INNER_CAP);
        if let Some(l) = logs.as_deref_mut() {
            l.clear();
            l.resize(z.len() * n_x * 4, f64::NEG_INFINITY);
        }
        let n = table.len();
        let (v0, vn) = (table.v[0], table.v[n - 1]);
        let (y0, yn) = (table.lny[0].exp(), table.lny[n - 1].exp());
        let mut count = 0;
        let mut edges: Vec<f64> = Vec::new();
        let mut seg_sums: Vec<[f64; 3]> = Vec::new();
        for (i, &zz) in z.iter().enumerate() {
            let m = self.w * zz;
            let (t0, tn) = ((v0 - m) / s, (vn - m) / s);
            let (lo, hi) = (t0.max(-REACH), tn.min(REACH));
            // Segment edges: the range's ends, the table's breakpoints and the
            // thresholds inside it, ascending.
            edges.clear();
            if hi > lo {
                edges.push(lo);
                edges.extend(breaks.iter().map(|&b| (b - m) / s).filter(|&t| lo < t && t < hi));
                edges.extend(scores.iter().map(|&v| (v - m) / s).filter(|&t| lo < t && t < hi));
                edges.push(hi);
                edges.sort_by(f64::total_cmp);
                edges.dedup();
            }
            seg_sums.clear();
            for pair in edges.windows(2) {
                pieces(pair[0], pair[1], std::iter::empty(), INNER_WIDTH, (&GL8_X, &GL8_W), &mut scratch.points, &mut scratch.weights);
                let mut cursor = 0;
                let mut sum = [0.0; 3];
                for (&t, &w) in scratch.points.iter().zip(&scratch.weights) {
                    let y = table.ln_share_ascending(m + s * t, &mut cursor).exp();
                    let wt = w * phi(t);
                    sum[0] += y * wt;
                    sum[1] += y * y * wt;
                    sum[2] += y * y * y * wt;
                }
                count += scratch.points.len();
                seg_sums.push(sum);
            }
            // Cumulative sums from the top: above[k] is the integral above edges[k].
            let mut above = vec![[0.0; 3]; edges.len()];
            for k in (0..seg_sums.len()).rev() {
                for r in 0..3 {
                    above[k][r] = above[k + 1][r] + seg_sums[k][r];
                }
            }
            for (j, &v) in scores.iter().enumerate() {
                let base = (i * n_x + j) * 4;
                let slot = &mut out[base..base + 4];
                let tx = (v - m) / s;
                let ln_q0 = normal::ln_upper_tail(tx);
                slot[0] = ln_q0.exp();
                let mut sum = [0.0; 3];
                if !edges.is_empty() {
                    let start = edges.partition_point(|&e| e < tx);
                    if start < edges.len() {
                        sum = above[start];
                    }
                }
                // The reach suits the moments, not a tail that starts near
                // it: such a threshold's tail is integrated on past it.
                let mut top = tn.min(REACH);
                if tx > REACH - DEEP_FROM && tn > REACH {
                    let a = tx.max(REACH);
                    let b = tn.min(a + DEEP_SPAN / a);
                    top = b;
                    if b > a {
                        let inner = breaks.iter().map(|&bk| (bk - m) / s).filter(|&t| a < t && t < b);
                        pieces(a, b, inner, DEEP_WIDTH / a, (&GL8_X, &GL8_W), &mut scratch.points, &mut scratch.weights);
                        let mut cursor = 0;
                        for (&t, &w) in scratch.points.iter().zip(&scratch.weights) {
                            let y = table.ln_share_ascending(m + s * t, &mut cursor).exp();
                            let wt = w * phi(t);
                            sum[0] += y * wt;
                            sum[1] += y * y * wt;
                            sum[2] += y * y * y * wt;
                        }
                        count += scratch.points.len();
                    }
                }
                // Above the table's last node the share is held at its last value.
                if tn <= top {
                    let p = normal::ln_upper_tail(tn.max(tx).max(-REACH)).exp();
                    sum[0] += yn * p;
                    sum[1] += yn * yn * p;
                    sum[2] += yn * yn * yn * p;
                }
                // Below its first node at its first value, where the threshold lies lower.
                if t0 > tx.max(-REACH) {
                    let p = normal::cdf(t0.min(REACH)) - normal::cdf(tx.max(-REACH));
                    sum[0] += y0 * p;
                    sum[1] += y0 * y0 * p;
                    sum[2] += y0 * y0 * y0 * p;
                }
                for r in 0..3 {
                    slot[r + 1] = sum[r] * scale[r + 1];
                }
                if let Some(l) = logs.as_deref_mut() {
                    l[base] = ln_q0;
                    for k in 1..4 {
                        l[base + k] = slot[k].ln();
                    }
                }
            }
        }
        Ok(count)
    }
}

/// Raw moments of order 0..3 of the sum of two independent parts.
fn combine(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0],
        a[1] * b[0] + a[0] * b[1],
        a[2] * b[0] + 2.0 * a[1] * b[1] + a[0] * b[2],
        a[3] * b[0] + 3.0 * a[2] * b[1] + 3.0 * a[1] * b[2] + a[0] * b[3],
    ]
}

/// Tail measures `(R, T_1..3, D_1..3)` of one event at one edge.
type Measures = [f64; 7];

/// The union bound of an event at every edge (spec section 4B): from above,
/// the units' tails beyond their shares `t_u = t b_u / B` of each edge with
/// the moments of the others; from below, the larger of the probabilities
/// that some unit exceeds the edge and that every unit exceeds its share,
/// and for the moments each unit's own tail beyond the edge. The factor
/// rule is the moment pass's with breakpoints where a unit's tail switches
/// on (`UnitTerm::transitions`); each beta unit below loading one is
/// evaluated at its own nodes and interpolated onto the event's in the
/// logarithm. Each candidate is a smooth function of the factor and is
/// integrated alone; clipping and the larger of two candidates are taken
/// after integration, where a kink at an unknown factor value would cost the
/// rule its accuracy (the integral of a max is at least the max of the
/// integrals, so the bounds stay valid). Returns `(lower, upper)` per edge
/// and the evaluations.
fn union_bound_event(units: &[UnitTerm], edges: &[f64], nodes: &[f64], z_high: f64, scratch: &mut Scratch) -> Result<(Vec<Measures>, Vec<Measures>, usize), Error> {
    let level = Level::new(1.0, false);
    let n_e = edges.len();
    let k_units = units.len();
    let supports: Vec<f64> = units.iter().map(|u| u.plan.support()).collect();
    let total: f64 = supports.iter().sum();
    // Per unit: the thresholds (-1 for the moments, the shares of each edge,
    // each edge), their scores and where its tails switch on.
    let mut thresholds_of: Vec<Vec<f64>> = Vec::with_capacity(k_units);
    let mut scores_of: Vec<Vec<f64>> = Vec::with_capacity(k_units);
    let mut transitions_of: Vec<Vec<f64>> = Vec::with_capacity(k_units);
    let mut breaks: Vec<f64> = Vec::new();
    for (u, &b) in units.iter().zip(&supports) {
        let mut thresholds = vec![-1.0];
        thresholds.extend(edges.iter().map(|&t| t * b / total));
        thresholds.extend(edges.iter().copied());
        let scores: Vec<f64> = if u.table.is_some() {
            thresholds.iter().map(|&x| u.score_of(x / u.plan.max_loss)).collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };
        let mut own = Vec::new();
        u.transitions(&thresholds, &scores, z_high, &mut own);
        u.outer_breaks(&level, &mut breaks);
        breaks.extend_from_slice(&own);
        thresholds_of.push(thresholds);
        scores_of.push(scores);
        transitions_of.push(own);
    }
    breaks.sort_by(f64::total_cmp);
    breaks.dedup();
    let (mut z, mut wz) = (Vec::new(), Vec::new());
    pieces(Z_LOW, z_high, breaks.into_iter(), level.outer_width, (&GL8_X, &GL8_W), &mut z, &mut wz);
    for (w, &zz) in wz.iter_mut().zip(&z) {
        *w *= phi(zz);
    }
    let n = z.len();
    let n_x = 1 + 2 * n_e;
    let bary = barycentric();
    // Per unit, at the event's nodes, `P(X^k; X > x | z)` for k = 0..3 at
    // each threshold.
    let mut profiles: Vec<Vec<f64>> = Vec::with_capacity(k_units);
    let mut evaluations = 0;
    let (mut native, mut native_w) = (Vec::new(), Vec::new());
    let (mut raw, mut raw_logs, mut column, mut interp) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (k, u) in units.iter().enumerate() {
        let (thresholds, scores) = (&thresholds_of[k], &scores_of[k]);
        if u.table.is_some() && u.w < 1.0 {
            let native_edges = u.native_edges(&level, z_high, &transitions_of[k]);
            pieces(native_edges[0], native_edges[native_edges.len() - 1], native_edges[1..native_edges.len() - 1].iter().copied(), f64::INFINITY, (&GL8_X, &GL8_W), &mut native, &mut native_w);
            evaluations += u.tails(&native, thresholds, scores, &mut raw, Some(&mut raw_logs), scratch)?;
            let mut profile = vec![0.0; n * n_x * 4];
            for q in 0..n_x * 4 {
                column.clear();
                column.extend((0..native.len()).map(|i| raw_logs[i * n_x * 4 + q]));
                interpolate_logs(&native_edges, &column, &z, &bary, &mut interp);
                for i in 0..n {
                    profile[i * n_x * 4 + q] = interp[i];
                }
            }
            profiles.push(profile);
        } else {
            let mut profile = Vec::new();
            evaluations += u.tails(&z, thresholds, scores, &mut profile, None, scratch)?;
            profiles.push(profile);
        }
    }
    let at = |p: &Vec<f64>, i: usize, j: usize| -> [f64; 4] {
        let base = (i * n_x + j) * 4;
        [p[base], p[base + 1], p[base + 2], p[base + 3]]
    };
    // Integrals per edge: the upper rate and moments with their values at
    // the first node, the two lower rates, and each unit's own tail moments.
    let mut upper_rate = vec![0.0; n_e];
    let mut upper_moments = vec![[0.0; 3]; n_e];
    let mut first_rate = vec![0.0; n_e];
    let mut first_moments = vec![[0.0; 3]; n_e];
    let mut any_beyond = vec![0.0; n_e];
    let mut all_beyond = vec![0.0; n_e];
    let mut own = vec![[0.0; 3]; n_e * k_units];
    let mut prefix = vec![[0.0; 4]; k_units + 1];
    let mut suffix = vec![[0.0; 4]; k_units + 1];
    for i in 0..n {
        prefix[0] = [1.0, 0.0, 0.0, 0.0];
        for k in 0..k_units {
            let mut m = at(&profiles[k], i, 0);
            m[0] = 1.0;
            prefix[k + 1] = combine(&prefix[k], &m);
        }
        suffix[k_units] = [1.0, 0.0, 0.0, 0.0];
        for k in (0..k_units).rev() {
            let mut m = at(&profiles[k], i, 0);
            m[0] = 1.0;
            suffix[k] = combine(&m, &suffix[k + 1]);
        }
        let w = wz[i];
        for e in 0..n_e {
            let mut up_r = 0.0;
            let mut up_t = [0.0; 3];
            let mut none_beyond = 1.0;
            let mut every_beyond = 1.0;
            for k in 0..k_units {
                let others = combine(&prefix[k], &suffix[k + 1]);
                let q = at(&profiles[k], i, 1 + e);
                let edge_q = at(&profiles[k], i, 1 + n_e + e);
                up_r += q[0];
                for r in 1..=3 {
                    up_t[r - 1] += (0..=r).map(|j| BINOMIAL[r][j] * q[j] * others[r - j]).sum::<f64>();
                    own[e * k_units + k][r - 1] += w * edge_q[r];
                }
                none_beyond *= 1.0 - edge_q[0];
                every_beyond *= q[0];
            }
            upper_rate[e] += w * up_r;
            for r in 0..3 {
                upper_moments[e][r] += w * up_t[r];
            }
            any_beyond[e] += w * (1.0 - none_beyond);
            all_beyond[e] += w * every_beyond;
            if i == 0 {
                first_rate[e] = up_r;
                first_moments[e] = up_t;
            }
        }
    }
    // The omitted factor ranges: below z = -9 the upper integrands, each
    // nondecreasing in the factor, are at most their values at the first
    // node; above z_high at most the trivial bound.
    let (tail_low, tail_high) = (normal::cdf(Z_LOW), normal::cdf(-z_high));
    let mut lower = vec![[0.0; 7]; n_e];
    let mut upper = vec![[0.0; 7]; n_e];
    for e in 0..n_e {
        let (t, c) = (edges[e], nodes[e]);
        let rate_up = (upper_rate[e] + tail_low * first_rate[e] + tail_high).min(1.0);
        let rate_low = any_beyond[e].max(all_beyond[e]);
        lower[e][0] = rate_low;
        upper[e][0] = rate_up;
        for r in 0..3 {
            let power = (r + 1) as i32;
            let moment_up = upper_moments[e][r] + tail_low * first_moments[e][r] + tail_high * total.powi(power);
            let own_largest = (0..k_units).map(|k| own[e * k_units + k][r]).fold(0.0, f64::max);
            let excess_low = (t.powi(power) - c.powi(power)) * rate_low;
            lower[e][1 + r] = (t.powi(power) * rate_low).max(own_largest);
            lower[e][4 + r] = excess_low;
            upper[e][1 + r] = moment_up;
            upper[e][4 + r] = (moment_up - c.powi(power) * rate_low).max(excess_low);
        }
    }
    Ok((lower, upper, evaluations))
}

/// `(R, T_1..3, D_1..3)` at every edge of an event whose total is a
/// nondecreasing function of the factor alone (units at loading one and
/// atoms): `P(S > t) = Phi(-z*)` at the first `z` where the total exceeds
/// `t`, found by bisection, and the moments by the factor rule above it.
fn comonotone_event(units: &[UnitTerm], edges: &[f64], nodes: &[f64], z_high: f64) -> Vec<Measures> {
    let total_at = |zz: f64| -> f64 {
        units
            .iter()
            .map(|u| match u.plan.kind {
                UnitKind::Atom | UnitKind::Drop => u.plan.mean_loss,
                UnitKind::TwoPoint => {
                    if zz > normal::quantile(1.0 - u.plan.two_point_p()) {
                        u.plan.max_loss
                    } else {
                        0.0
                    }
                }
                _ => u.plan.max_loss * u.table.as_ref().unwrap().ln_share(zz).exp(),
            })
            .sum()
    };
    let level = Level::new(1.0, false);
    let mut breaks: Vec<f64> = Vec::new();
    for u in units {
        u.outer_breaks(&level, &mut breaks);
    }
    breaks.sort_by(f64::total_cmp);
    breaks.dedup();
    let mut out = vec![[0.0; 7]; edges.len()];
    let (mut z, mut wz) = (Vec::new(), Vec::new());
    for (e, (&t, &c)) in edges.iter().zip(nodes).enumerate() {
        if total_at(z_high) <= t {
            continue;
        }
        let z_star = if total_at(Z_LOW) > t {
            f64::NEG_INFINITY
        } else {
            let (mut lo, mut hi) = (Z_LOW, z_high);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if total_at(mid) > t {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            hi
        };
        let start = z_star.max(Z_LOW);
        pieces(start, z_high, breaks.iter().copied(), level.outer_width, (&GL8_X, &GL8_W), &mut z, &mut wz);
        let mut sums = [Neumaier::default(); 6];
        for (&zz, &w) in z.iter().zip(&wz) {
            let s = total_at(zz);
            let weight = w * phi(zz);
            let (s2, s3) = (s * s, s * s * s);
            let cpow = [c, c * c, c * c * c];
            for (r, v) in [s, s2, s3].into_iter().enumerate() {
                sums[r].add(weight * v);
                sums[3 + r].add(weight * (v - cpow[r]));
            }
        }
        let mut measures = [0.0; 7];
        measures[0] = if z_star.is_finite() { normal::cdf(-z_star) } else { 1.0 };
        for q in 0..6 {
            measures[1 + q] = sums[q].value();
        }
        if !z_star.is_finite() {
            let (p, base) = (normal::cdf(Z_LOW), total_at(Z_LOW));
            let powers = [base, base * base, base * base * base];
            let cpow = [c, c * c, c * c * c];
            for r in 0..3 {
                measures[1 + r] += powers[r] * p;
                measures[4 + r] += (powers[r] - cpow[r]) * p;
            }
        }
        out[e] = measures;
    }
    out
}

/// Configuration of the shared-event tail query.
#[derive(Debug, Clone)]
pub struct TailConfig {
    pub coupling: Coupling,
    pub variance_policy: VariancePolicy,
    /// Relative width under which a bracket counts as converged; None
    /// requests no target, and `converged` is then None.
    pub rtol: Option<f64>,
    pub threads: Option<usize>,
    pub chunk_rows: usize,
}

impl TailConfig {
    pub fn new() -> TailConfig {
        TailConfig {
            coupling: Coupling::Burke,
            variance_policy: VariancePolicy::default(),
            rtol: None,
            threads: None,
            chunk_rows: DEFAULT_MULTI_CHUNK_ROWS,
        }
    }
}

impl Default for TailConfig {
    fn default() -> Self {
        TailConfig::new()
    }
}

/// Tail measures of a multi-unit ELT's event totals at each edge (spec
/// section 4B): the point values (the upper bounds where an event is
/// bounded), the lower and upper bounds, the half bucket's atoms, and the
/// counts. `[f64; 7]` entries are `(R, T_1..3, D_1..3)`.
#[derive(Debug, Clone)]
pub struct SharedTails {
    pub edge: Vec<f64>,
    pub top_node: Vec<f64>,
    pub point: Vec<[f64; 7]>,
    pub lower: Vec<[f64; 7]>,
    pub upper: Vec<[f64; 7]>,
    /// The rate of the atoms in `(c, t]` and `sum lambda (x^r - c^r)` there.
    pub top_half: Vec<[f64; 4]>,
    /// Per edge, the largest `(upper - lower) / upper` over the measures.
    pub width: Vec<f64>,
    /// Whether every width is within the requested `rtol`; None when none
    /// was requested.
    pub converged: Option<bool>,
    pub total_rate: f64,
    pub input_rows: usize,
    pub valid_rows: usize,
    pub events: usize,
    pub valid_events: usize,
    pub one_unit_events: usize,
    pub exact_events: usize,
    pub bounded_events: usize,
    pub evaluations: usize,
    pub diagnostics: Diagnostics,
}

/// Per-chunk partial of the tail query.
struct TailPartial {
    point: Vec<[Neumaier; 7]>,
    lower: Vec<[Neumaier; 7]>,
    upper: Vec<[Neumaier; 7]>,
    top_half: Vec<[Neumaier; 4]>,
    rate: Neumaier,
    rows: usize,
    valid_events: usize,
    one_unit_events: usize,
    exact_events: usize,
    bounded_events: usize,
    evaluations: usize,
}

/// Tail measures of a multi-unit ELT's event totals at `edge` with nodes
/// `top_node` (the edges themselves when None), with no grid (spec section
/// 4B): exact for one-unit events, under `homer` (the rolled row), for events
/// of atoms and for events whose total is a function of the factor alone;
/// otherwise the union bound (`union_bound_event`). Events are processed in
/// fixed chunks folded in chunk order, so the result is bitwise identical
/// for any thread count.
pub fn multiunit_tail_moments(input: &MultiInput<'_>, edge: &[f64], top_node: Option<&[f64]>, config: &TailConfig) -> Result<SharedTails, Error> {
    if config.coupling == Coupling::Matched {
        return Err(Error::MatchedRefused);
    }
    if let Some(rtol) = config.rtol {
        if !(rtol > 0.0 && rtol < 1.0) {
            return Err(Error::BadMomentTolerance(rtol));
        }
    }
    if config.chunk_rows < 1 {
        return Err(Error::BadChunkRows(config.chunk_rows));
    }
    if config.threads == Some(0) {
        return Err(Error::BadThreads(0));
    }
    let nodes: Vec<f64> = match top_node {
        Some(c) => {
            if c.len() != edge.len() {
                return Err(Error::ColumnLengthMismatch { name: "top_node", got: c.len(), expected: edge.len() });
            }
            c.to_vec()
        }
        None => edge.to_vec(),
    };
    for (&t, &c) in edge.iter().zip(&nodes) {
        if !(t.is_finite() && c.is_finite() && 0.0 <= c && c <= t) {
            return Err(Error::BadTailEdge { edge: t, top_node: c });
        }
    }
    let pass = event_pass(input, config.variance_policy, None)?;
    let n_events = pass.events.len();
    let n_e = edge.len();
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
    let process = |&(lo, hi): &(usize, usize)| -> Result<TailPartial, Error> {
        let mut partial = TailPartial {
            point: vec![[Neumaier::default(); 7]; n_e],
            lower: vec![[Neumaier::default(); 7]; n_e],
            upper: vec![[Neumaier::default(); 7]; n_e],
            top_half: vec![[Neumaier::default(); 4]; n_e],
            rate: Neumaier::default(),
            rows: 0,
            valid_events: 0,
            one_unit_events: 0,
            exact_events: 0,
            bounded_events: 0,
            evaluations: 0,
        };
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
            partial.rows += live.len();
            partial.rate.add(lam);
            let plans: Vec<UnitPlan> = live
                .iter()
                .map(|&row| UnitPlan::from_row(&pass.classes[row], &input.spec(row)).couple(config.coupling))
                .collect();
            let support: f64 = plans.iter().map(UnitPlan::support).sum();
            let failure = |error: Error| match error {
                Error::KernelFailure { alpha, beta, .. } => Error::KernelFailure { row: start, alpha, beta },
                other => other,
            };
            // Exact laws: one unit, the rolled row, or a sum of atoms.
            let exact_law: Option<RowLaw> = if plans.len() == 1 {
                partial.one_unit_events += 1;
                Some(unit_row_law(&plans[0]))
            } else if config.coupling == Coupling::Homer {
                let specs: Vec<UnitSpec> = live.iter().map(|&row| input.spec(row)).collect();
                let rolled = prepare_unit(&rolled_spec(&specs), "rolled", config.variance_policy)?;
                if rolled.kind == UnitKind::Drop {
                    partial.exact_events += 1;
                    continue;
                }
                Some(unit_row_law(&rolled))
            } else if plans.iter().all(|p| p.kind == UnitKind::Atom) {
                Some(RowLaw::Atom { mu: plans.iter().map(|p| p.mean_loss).sum() })
            } else {
                None
            };
            if let Some(law) = exact_law {
                partial.exact_events += 1;
                for k in 0..n_e {
                    let values = crate::tails::law_tail(law, edge[k], nodes[k]).map_err(failure)?;
                    for q in 0..7 {
                        partial.point[k][q].add(lam * values[q]);
                        partial.lower[k][q].add(lam * values[q]);
                        partial.upper[k][q].add(lam * values[q]);
                    }
                    for q in 0..4 {
                        partial.top_half[k][q].add(lam * values[7 + q]);
                    }
                }
                continue;
            }
            // The total is a function of the factor alone when every unit
            // is an atom or loads one on it.
            let comonotone = plans.iter().all(|p| {
                p.kind == UnitKind::Atom
                    || (matches!(p.kind, UnitKind::Beta | UnitKind::Comonotone | UnitKind::TwoPoint) && p.w_corr == 1.0)
            });
            if comonotone {
                partial.exact_events += 1;
            } else {
                partial.bounded_events += 1;
            }
            // An event whose total cannot pass an edge contributes nothing
            // there, and one that passes none costs nothing.
            let active: Vec<usize> = (0..n_e).filter(|&k| support > edge[k]).collect();
            if active.is_empty() {
                continue;
            }
            let units: Vec<UnitTerm> = plans
                .iter()
                .map(|p| UnitTerm::new(p.clone(), MOMENT_RTOL, MOMENT_RTOL * TABLE_TOL_FRACTION))
                .collect::<Result<_, _>>()
                .map_err(failure)?;
            let marginal: Vec<[f64; 3]> = plans.iter().map(|p| unit_row_law(p).raw_moments()).collect();
            let m2: f64 = marginal.iter().map(|m| m[1]).sum();
            let m3: f64 = marginal.iter().map(|m| m[2]).sum();
            let ratio = (m2 / (support * support)).min(m3 / (support * support * support));
            let z_high = (-normal::quantile(0.01 * MOMENT_RTOL * ratio)).clamp(Z_HIGH_MIN, 37.0);
            let t_active: Vec<f64> = active.iter().map(|&k| edge[k]).collect();
            let c_active: Vec<f64> = active.iter().map(|&k| nodes[k]).collect();
            if comonotone {
                let values = comonotone_event(&units, &t_active, &c_active, z_high);
                let beyond = normal::cdf(-z_high);
                for (a, &k) in active.iter().enumerate() {
                    let c = nodes[k];
                    let trivial = [1.0, support, support * support, support.powi(3), support - c, support * support - c * c, support.powi(3) - c.powi(3)];
                    for q in 0..7 {
                        partial.point[k][q].add(lam * values[a][q]);
                        partial.lower[k][q].add(lam * values[a][q] * (1.0 - TAIL_MARGIN));
                        partial.upper[k][q].add(lam * (values[a][q] * (1.0 + TAIL_MARGIN) + beyond * trivial[q]));
                    }
                }
            } else {
                let (lower, upper, evaluations) = union_bound_event(&units, &t_active, &c_active, z_high, &mut scratch).map_err(failure)?;
                partial.evaluations += evaluations;
                for (a, &k) in active.iter().enumerate() {
                    for q in 0..7 {
                        let up = upper[a][q] * (1.0 + TAIL_MARGIN);
                        partial.point[k][q].add(lam * up);
                        partial.lower[k][q].add(lam * lower[a][q] * (1.0 - TAIL_MARGIN));
                        partial.upper[k][q].add(lam * up);
                    }
                }
            }
        }
        Ok(partial)
    };
    let run = || -> Result<Vec<TailPartial>, Error> { chunks.par_iter().map(process).collect() };
    let partials = match config.threads {
        Some(threads) => rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| Error::ThreadPool(e.to_string()))?
            .install(run)?,
        None => run()?,
    };
    let mut point = vec![[Neumaier::default(); 7]; n_e];
    let mut lower = vec![[Neumaier::default(); 7]; n_e];
    let mut upper = vec![[Neumaier::default(); 7]; n_e];
    let mut top_half = vec![[Neumaier::default(); 4]; n_e];
    let mut rate = Neumaier::default();
    let (mut rows_total, mut valid_events, mut one_unit_events, mut exact_events, mut bounded_events, mut evaluations) = (0, 0, 0, 0, 0, 0);
    for p in partials {
        for k in 0..n_e {
            for q in 0..7 {
                point[k][q].add(p.point[k][q].value());
                lower[k][q].add(p.lower[k][q].value());
                upper[k][q].add(p.upper[k][q].value());
            }
            for q in 0..4 {
                top_half[k][q].add(p.top_half[k][q].value());
            }
        }
        rate.add(p.rate.value());
        rows_total += p.rows;
        valid_events += p.valid_events;
        one_unit_events += p.one_unit_events;
        exact_events += p.exact_events;
        bounded_events += p.bounded_events;
        evaluations += p.evaluations;
    }
    let value7 = |a: &Vec<[Neumaier; 7]>| -> Vec<[f64; 7]> { a.iter().map(|x| std::array::from_fn(|q| x[q].value())).collect() };
    let (point, lower, upper) = (value7(&point), value7(&lower), value7(&upper));
    let width: Vec<f64> = (0..n_e)
        .map(|k| (0..7).filter(|&q| upper[k][q] > 0.0).map(|q| (upper[k][q] - lower[k][q]) / upper[k][q]).fold(0.0, f64::max))
        .collect();
    Ok(SharedTails {
        edge: edge.to_vec(),
        top_node: nodes,
        converged: config.rtol.map(|rtol| width.iter().all(|&w| w <= rtol)),
        point,
        lower,
        upper,
        top_half: top_half.iter().map(|x| std::array::from_fn(|q| x[q].value())).collect(),
        width,
        total_rate: rate.value(),
        input_rows: input.rate.len(),
        valid_rows: rows_total,
        events: n_events,
        valid_events,
        one_unit_events,
        exact_events,
        bounded_events,
        evaluations,
        diagnostics: pass.diagnostics,
    })
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

    // [Eltq-Tail-Queries] Spec test 38: two two-point units against the exact
    // four-outcome law (each outcome at these edges passes both or neither
    // of the lower bound's events, so its rate is exact less the margin);
    // unreachable edges cost nothing; bitwise across thread counts; matched
    // refused.
    #[test]
    fn shared_tails_two_point_pair_and_determinism() {
        let rows = [(0, 0, 1.0, 300.0, 400.0, 300.0, 1000.0), (0, 1, 1.0, 500.0, 600.0, 500.0, 1500.0)];
        let (mut cols, mut codes) = (Vec::new(), Vec::new());
        let edges = [400.0, 1200.0, 2000.0];
        let t = multiunit_tail_moments(&input(&rows, &mut cols, &mut codes), &edges, None, &TailConfig::new()).unwrap();
        let (p, q) = (0.3, 500.0 / 1500.0);
        let rho = (300.0 / 500.0) * (500.0 / 600.0_f64.hypot(500.0));
        let both = normal::bivariate_upper(normal::quantile(1.0 - p), normal::quantile(1.0 - q), rho);
        for (k, exact) in [p + q - both, q, both].into_iter().enumerate() {
            assert!((t.lower[k][0] / exact - 1.0).abs() < 2e-6, "edge {k}: {} vs {exact}", t.lower[k][0]);
            assert!(t.lower[k][0] <= exact && exact <= t.upper[k][0]);
        }
        assert!(t.bounded_events == 1 && t.converged.is_none() && t.evaluations > 0);
        let far = multiunit_tail_moments(&input(&rows, &mut cols, &mut codes), &[2500.0], None, &TailConfig::new()).unwrap();
        assert!(far.upper[0].iter().all(|&v| v == 0.0) && far.evaluations == 0);
        let betas = [(0, 0, 1.0, 200.0, 100.0, 150.0, 1000.0), (0, 1, 1.0, 500.0, 300.0, 200.0, 2000.0), (1, 0, 0.5, 300.0, 50.0, 400.0, 3000.0), (1, 1, 0.5, 40.0, 30.0, 30.0, 400.0)];
        let mut config = TailConfig::new();
        config.threads = Some(1);
        config.chunk_rows = 2;
        let a = multiunit_tail_moments(&input(&betas, &mut cols, &mut codes), &[1000.0, 2500.0], Some(&[900.0, 2000.0]), &config).unwrap();
        config.threads = Some(3);
        let b = multiunit_tail_moments(&input(&betas, &mut cols, &mut codes), &[1000.0, 2500.0], Some(&[900.0, 2000.0]), &config).unwrap();
        for k in 0..2 {
            assert_eq!(a.lower[k].map(f64::to_bits), b.lower[k].map(f64::to_bits));
            assert_eq!(a.upper[k].map(f64::to_bits), b.upper[k].map(f64::to_bits));
            assert!((0..7).all(|q| a.lower[k][q] <= a.upper[k][q]));
        }
        config.coupling = Coupling::Matched;
        assert!(matches!(multiunit_tail_moments(&input(&betas, &mut cols, &mut codes), &[1000.0], None, &config), Err(Error::MatchedRefused)));
    }
}
