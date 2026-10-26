//! Tail measures of a one-row ELT at candidate edges, with no loss grid
//! (spec section 4B): for an edge `t` and a node `c <= t`, the tail rate
//! `R(t)`, the tail moments `T_r(t) = sum lambda P(X^r; X > t)` and the
//! excess moments `D_r(t, c) = sum lambda P(X^r - c^r; X > t)` for `r = 1, 2,
//! 3`, and apart from them the atoms in the final half bucket `(c, t]`,
//! which a grid keeps at `c`.
//!
//! A beta row is exact by its law: on the continued fraction's route the
//! partial-moment identity in recursive form and the centred recurrence of
//! the density's differential equation, both from one incomplete beta and
//! one density value; past `LARGE_PARAMETER_FROM` the narrow kernel's
//! quadrature of the centred moments, and from `NARROW_NORMAL_FROM` the
//! normal law's closed forms. Atoms and two-point rows are exact with the
//! strict inequality of the deposit (`dev/done/plan-model-moments-and-
//! windowed-views.md`, [Eltq-Tail-Queries]).

use rayon::prelude::*;

use crate::beta::beta_params;
use crate::config::ClassifyOptions;
use crate::diagnostics::Diagnostics;
use crate::error::Error;
use crate::incbeta::BetaFn;
use crate::law::BetaLaw;
use crate::moments::{RowLaw, row_law};
use crate::quantize::{EltInput, classify_rows_with};
use crate::sum::Neumaier;

/// Rows per parallel chunk of the tail pass; part of the deterministic
/// contract (partials are folded in chunk order).
pub const TAIL_CHUNK_ROWS: usize = 4096;

/// How the rows of a tail query were evaluated, by count.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TailRegimes {
    /// Betas through the continued fraction, `min(alpha, beta) < 2e5`.
    pub incomplete_beta: usize,
    /// Betas through the narrow kernel's quadrature.
    pub narrow_kernel: usize,
    /// Betas through the normal law, `min(alpha, beta) >= 1e16`.
    pub normal_law: usize,
    pub atoms: usize,
    pub two_point: usize,
}

/// Tail measures at each edge, rate-weighted; `[f64; 3]` entries are orders
/// 1 to 3. An order whose value leaves double precision at some edge is
/// infinite there and flagged in `moment_overflow`.
#[derive(Debug, Clone)]
pub struct TailMoments {
    pub edge: Vec<f64>,
    pub top_node: Vec<f64>,
    /// `R(t)`.
    pub rate_above: Vec<f64>,
    /// `T_r(t)`.
    pub moments_above: Vec<[f64; 3]>,
    /// `D_r(t, c)`.
    pub excess_above: Vec<[f64; 3]>,
    /// The rate of the atoms in `(c, t]`.
    pub top_half_rate: Vec<f64>,
    /// `sum lambda (x^r - c^r)` over the atoms in `(c, t]`.
    pub top_half_excess: Vec<[f64; 3]>,
    pub total_rate: f64,
    pub regimes: TailRegimes,
    pub moment_overflow: [bool; 3],
    pub input_rows: usize,
    pub valid_rows: usize,
    pub diagnostics: Diagnostics,
}

/// One row's contribution at one edge: `R`, `T_1..3`, `D_1..3`, the
/// top-half rate and `sum (x^r - c^r)` there, per unit of rate.
type Contribution = [f64; 11];

/// `x^r - c^r` for `r = 1, 2, 3` with `c <= x`, as `(x - c)` times a sum of
/// nonnegative terms, so a short excess keeps its digits.
fn power_excess(x: f64, c: f64) -> [f64; 3] {
    let d = x - c;
    [d, d * (x + c), d * (x * x + x * c + c * c)]
}

/// An atom of unit rate at `x`, at edge `t` and node `c`.
fn atom_contribution(x: f64, t: f64, c: f64) -> Contribution {
    let mut out = [0.0; 11];
    if x > t {
        let excess = power_excess(x, c);
        out[0] = 1.0;
        out[1] = x;
        out[2] = x * x;
        out[3] = x * x * x;
        out[4..7].copy_from_slice(&excess);
    } else if x > c {
        let excess = power_excess(x, c);
        out[7] = 1.0;
        out[8..11].copy_from_slice(&excess);
    }
    out
}

/// `T_r` and `D_r` per unit of rate from the centred tail moments `K_k =
/// P((Y - c)^k; Y > y)` of the share, `k = 0..3`, at `chat = c / M`:
/// `D_r = M^r sum_{k=1}^r C(r, k) chat^(r-k) K_k` and `T_r` the same with
/// `k` from 0, each a sum of nonnegative terms.
fn from_centred(k: [f64; 4], chat: f64, m: f64) -> ([f64; 3], [f64; 3]) {
    let k = k.map(|v| v.max(0.0));
    let d1 = k[1];
    let d2 = 2.0 * chat * k[1] + k[2];
    let d3 = 3.0 * chat * chat * k[1] + 3.0 * chat * k[2] + k[3];
    let t1 = chat * k[0] + d1;
    let t2 = chat * chat * k[0] + d2;
    let t3 = chat * chat * chat * k[0] + d3;
    let (m2, m3) = (m * m, m * m * m);
    ([m * t1, m2 * t2, m3 * t3], [m * d1, m2 * d2, m3 * d3])
}

/// The tail of a beta share through the continued fraction at `y` (`0 <= y
/// < 1`) and `chat <= y`: `(K_0..3 at chat, G_1..3)` with `G_r = P(Y^r; Y >
/// y)`.
///
/// Notes
/// -----
/// The density satisfies `d[u (1 - u) f] = (alpha - (alpha + beta) u) f du`,
/// and integrating `(u - chat)^k` against it over `(y, 1)` by parts gives
/// `(alpha + beta + k) K_(k+1) = (y - chat)^k y (1 - y) f(y) + [alpha -
/// (alpha + beta) chat + k (1 - 2 chat)] K_k + k chat (1 - chat) K_(k-1)`
/// (spec section 4B). At `chat = 0` every term is nonnegative and `G_r` is
/// the identity `P(Y^r; Y > y) = P(Y^r) I-bar_y(alpha + r, beta)` in
/// recursive form; the four orders cost one incomplete beta and one density
/// value, `sf_and_density_times_y`.
pub fn beta_tail(bf: &BetaFn, y: f64, chat: f64) -> ([f64; 4], [f64; 3]) {
    let (a, b) = (bf.alpha, bf.beta);
    let (sf, y_f) = bf.sf_and_density_times_y(y);
    // y (1 - y) f(y), finite at y = 0 for every alpha > 0.
    let boundary = (1.0 - y) * y_f;
    let g1 = (boundary + a * sf) / (a + b);
    let g2 = (y * boundary + (a + 1.0) * g1) / (a + b + 1.0);
    let g3 = (y * y * boundary + (a + 2.0) * g2) / (a + b + 2.0);
    let lead = a - (a + b) * chat;
    let spread = 1.0 - 2.0 * chat;
    let cc = chat * (1.0 - chat);
    let d = y - chat;
    let k1 = (boundary + lead * sf) / (a + b);
    let k2 = (d * boundary + (lead + spread) * k1 + cc * sf) / (a + b + 1.0);
    let k3 = (d * d * boundary + (lead + 2.0 * spread) * k2 + 2.0 * cc * k1) / (a + b + 2.0);
    ([sf, k1, k2, k3], [g1, g2, g3])
}

/// A contributing row's law with its beta kernel built once.
enum RowTail {
    Atom(f64),
    TwoPoint { m: f64, p: f64 },
    Regular { bf: BetaFn, m: f64 },
    Narrow { law: BetaLaw, m: f64 },
}

impl RowTail {
    fn new(law: RowLaw) -> Result<RowTail, Error> {
        Ok(match law {
            RowLaw::Atom { mu } => RowTail::Atom(mu),
            RowLaw::TwoPoint { mu, max_loss } => RowTail::TwoPoint { m: max_loss, p: mu / max_loss },
            RowLaw::Beta { mu, sd, max_loss } => {
                let (alpha, beta) = beta_params(max_loss, mu, sd);
                match BetaLaw::new(alpha, beta).ok_or(Error::KernelFailure { row: 0, alpha, beta })? {
                    BetaLaw::Regular(bf) => RowTail::Regular { bf, m: max_loss },
                    narrow => RowTail::Narrow { law: narrow, m: max_loss },
                }
            }
        })
    }

    /// The row's contribution per unit of rate at edge `t` and node `c`.
    fn at(&self, t: f64, c: f64) -> Contribution {
        match *self {
            RowTail::Atom(x) => atom_contribution(x, t, c),
            // The atom at zero exceeds no edge and lies in no half bucket.
            RowTail::TwoPoint { m, p } => atom_contribution(m, t, c).map(|v| v * p),
            RowTail::Regular { ref bf, m } => {
                let mut out = [0.0; 11];
                if t >= m {
                    return out;
                }
                let (y, chat) = (t / m, c / m);
                let (k, g) = beta_tail(bf, y, chat);
                let (_, d) = from_centred(k, chat, m);
                out[0] = k[0].max(0.0);
                out[1] = m * g[0].max(0.0);
                out[2] = m * m * g[1].max(0.0);
                out[3] = m * m * m * g[2].max(0.0);
                out[4..7].copy_from_slice(&d);
                out
            }
            RowTail::Narrow { ref law, m } => {
                let mut out = [0.0; 11];
                let BetaLaw::Narrow(narrow) = law else { unreachable!() };
                if t >= m {
                    return out;
                }
                let chat = c / m;
                let k = narrow.centred_tail(t / m, chat);
                let (tr, d) = from_centred(k, chat, m);
                out[0] = k[0];
                out[1..4].copy_from_slice(&tr);
                out[4..7].copy_from_slice(&d);
                out
            }
        }
    }
}

/// Tail measures of a one-row ELT at `edge` with nodes `top_node` (the edge
/// itself when None), classified exactly as the grid route classifies the
/// same input (spec section 4B). Rows are processed in fixed chunks of
/// `TAIL_CHUNK_ROWS` folded in chunk order, so the result is bitwise
/// identical for any `threads`.
pub fn tail_moments(
    input: &EltInput<'_>,
    options: &ClassifyOptions,
    edge: &[f64],
    top_node: Option<&[f64]>,
    threads: Option<usize>,
) -> Result<TailMoments, Error> {
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
    if threads == Some(0) {
        return Err(Error::BadThreads(0));
    }
    let classified = classify_rows_with(input, options)?;
    let n_rows = input.rate.len();
    let n_edges = edge.len();
    // Edges in ascending order, so a row whose survival has underflowed
    // stops there; results are written back in the caller's order.
    let mut order: Vec<usize> = (0..n_edges).collect();
    order.sort_by(|&i, &j| edge[i].total_cmp(&edge[j]));
    let law_of = |row: usize| {
        row_law(classified.kinds[row], classified.mu_eff[row], classified.sd_eff[row], input.max_loss[row])
    };
    let n_chunks = n_rows.div_ceil(TAIL_CHUNK_ROWS);
    let process = |chunk: usize| -> Result<(Vec<Neumaier>, Neumaier, TailRegimes), Error> {
        let mut acc = vec![Neumaier::default(); n_edges * 11];
        let mut rate = Neumaier::default();
        let mut regimes = TailRegimes::default();
        let end = ((chunk + 1) * TAIL_CHUNK_ROWS).min(n_rows);
        for row in chunk * TAIL_CHUNK_ROWS..end {
            let Some(law) = law_of(row) else { continue };
            let lam = input.rate[row];
            rate.add(lam);
            let tail = RowTail::new(law).map_err(|_| match law {
                RowLaw::Beta { mu, sd, max_loss } => {
                    let (alpha, beta) = beta_params(max_loss, mu, sd);
                    Error::KernelFailure { row, alpha, beta }
                }
                _ => Error::KernelFailure { row, alpha: f64::NAN, beta: f64::NAN },
            })?;
            match &tail {
                RowTail::Atom(_) => regimes.atoms += 1,
                RowTail::TwoPoint { .. } => regimes.two_point += 1,
                RowTail::Regular { .. } => regimes.incomplete_beta += 1,
                RowTail::Narrow { law: BetaLaw::Narrow(narrow), .. } if narrow.is_normal() => regimes.normal_law += 1,
                RowTail::Narrow { .. } => regimes.narrow_kernel += 1,
            }
            let continuous = matches!(tail, RowTail::Regular { .. } | RowTail::Narrow { .. });
            for &e in &order {
                let contribution = tail.at(edge[e], nodes[e]);
                // A beta has no atom for a half bucket, and once its survival
                // is zero at an edge it is zero at every larger one; an atom
                // can still fall in a later edge's half bucket.
                if continuous && contribution.iter().all(|&v| v == 0.0) {
                    break;
                }
                for (slot, v) in acc[e * 11..(e + 1) * 11].iter_mut().zip(contribution) {
                    if v != 0.0 {
                        slot.add(lam * v);
                    }
                }
            }
        }
        Ok((acc, rate, regimes))
    };
    let run = || -> Result<Vec<_>, Error> { (0..n_chunks).into_par_iter().map(process).collect() };
    let partials = match threads {
        Some(threads) => rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| Error::ThreadPool(e.to_string()))?
            .install(run)?,
        None => run()?,
    };
    let mut totals = vec![Neumaier::default(); n_edges * 11];
    let mut total_rate = Neumaier::default();
    let mut regimes = TailRegimes::default();
    for (acc, rate, r) in partials {
        for (t, a) in totals.iter_mut().zip(&acc) {
            t.add(a.value());
        }
        total_rate.add(rate.value());
        regimes.incomplete_beta += r.incomplete_beta;
        regimes.narrow_kernel += r.narrow_kernel;
        regimes.normal_law += r.normal_law;
        regimes.atoms += r.atoms;
        regimes.two_point += r.two_point;
    }
    let value = |e: usize, i: usize| totals[e * 11 + i].value();
    let mut moment_overflow = [false; 3];
    let mut triple = |e: usize, start: usize| -> [f64; 3] {
        let mut out = [value(e, start), value(e, start + 1), value(e, start + 2)];
        for (r, v) in out.iter_mut().enumerate() {
            if !v.is_finite() {
                moment_overflow[r] = true;
                *v = f64::INFINITY;
            }
        }
        out
    };
    let moments_above = (0..n_edges).map(|e| triple(e, 1)).collect();
    let excess_above = (0..n_edges).map(|e| triple(e, 4)).collect();
    let top_half_excess = (0..n_edges).map(|e| triple(e, 8)).collect();
    let valid_rows = (0..n_rows).filter(|&row| law_of(row).is_some()).count();
    Ok(TailMoments {
        edge: edge.to_vec(),
        top_node: nodes,
        rate_above: (0..n_edges).map(|e| value(e, 0)).collect(),
        moments_above,
        excess_above,
        top_half_rate: (0..n_edges).map(|e| value(e, 7)).collect(),
        top_half_excess,
        total_rate: total_rate.value(),
        regimes,
        moment_overflow,
        input_rows: n_rows,
        valid_rows,
        diagnostics: classified.diagnostics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_row(rate: f64, m: f64, mu: f64, sd: f64) -> [Vec<f64>; 5] {
        [vec![rate], vec![m], vec![mu], vec![sd], vec![0.0]]
    }

    fn query(cols: &[Vec<f64>; 5], edge: &[f64], node: Option<&[f64]>) -> TailMoments {
        let input = EltInput { rate: &cols[0], max_loss: &cols[1], mean_loss: &cols[2], sd_indep: &cols[3], sd_corr: &cols[4] };
        tail_moments(&input, &ClassifyOptions::default(), edge, node, None).unwrap()
    }

    // [Eltq-Tail-Queries] Spec test 37: Beta(2, 1000) on [0, 1] at the top
    // edge 0.00195 and node 0.0019 of test 18's window: the overflow rate and
    // the excess moment about the node are 0.41891026155683486 and
    // 5.796480662291177e-4 (SciPy), which the grid route reaches only to its
    // curve's 1e-4.
    #[test]
    fn beta_window_figures() {
        let (a, b) = (2.0_f64, 1000.0_f64);
        let p = a / (a + b);
        let sd = (p * (1.0 - p) / (a + b + 1.0)).sqrt();
        let tail = query(&one_row(1.0, 1.0, p, sd), &[0.00195], Some(&[0.0019]));
        assert!((tail.rate_above[0] - 0.41891026155683486).abs() < 1e-14);
        assert!((tail.excess_above[0][0] - 5.796480662291177e-4).abs() < 1e-14);
        assert_eq!(tail.regimes.incomplete_beta, 1);
    }

    // The recurrence against the partial-moment identity P(Y^r; Y > y) =
    // P(Y^r) I-bar_y(alpha + r, beta), and the centred moments against their
    // binomial expansion where nothing cancels (c well below the tail).
    #[test]
    fn recurrence_matches_the_identity() {
        for (a, b) in [(0.3, 7.0), (2.5, 2.5), (40.0, 3.0), (1e-3, 900.0), (5.0, 1e6)] {
            let bf = BetaFn::new(a, b);
            let mean = a / (a + b);
            for y in [0.25 * mean, mean, 3.0 * mean.min(0.3)] {
                let (k, g) = beta_tail(&bf, y, 0.0);
                let mut moment = 1.0;
                for r in 1..=3 {
                    moment *= (a + (r - 1) as f64) / (a + b + (r - 1) as f64);
                    let want = moment * BetaFn::new(a + r as f64, b).sf(y);
                    // Both sides are continued fractions: about 1e-13 apart,
                    // and above the switch point the formulation's residual
                    // of order eps * beta (spec section 5A).
                    let tolerance = 5e-13 + 2e-16 * b;
                    assert!((g[r - 1] - want).abs() <= tolerance * want, "a={a} b={b} y={y} r={r}: {} vs {want}", g[r - 1]);
                    assert!((k[r] - g[r - 1]).abs() <= 1e-15 * g[r - 1].max(1e-300));
                }
                let c = 0.5 * y;
                let (kc, _) = beta_tail(&bf, y, c);
                let naive = [k[1] - c * k[0], k[2] - 2.0 * c * k[1] + c * c * k[0]];
                assert!((kc[1] - naive[0]).abs() <= 1e-12 * naive[0], "a={a} b={b} y={y}");
                assert!((kc[2] - naive[1]).abs() <= 1e-11 * naive[1], "a={a} b={b} y={y}");
            }
        }
    }

    // Atoms with the deposit's strict inequality: one exactly at the edge is
    // not above it, one at the next float is; one in (c, t] is in the half
    // bucket only; a two-point row's atom at M likewise, with rate p.
    #[test]
    fn atoms_at_the_edge() {
        let t = 10.0_f64;
        let above = f64::from_bits(t.to_bits() + 1);
        let at = query(&one_row(0.5, 20.0, t, 0.0), &[t], Some(&[9.5]));
        assert_eq!((at.rate_above[0], at.top_half_rate[0]), (0.0, 0.5));
        assert_eq!(at.top_half_excess[0][0], 0.5 * 0.5);
        let next = query(&one_row(0.5, 20.0, above, 0.0), &[t], Some(&[9.5]));
        assert_eq!((next.rate_above[0], next.top_half_rate[0]), (0.5, 0.0));
        // Two-point: mu = 4, M = t, rate 1: p = 0.4 at M; excluded at t = M
        // and included one unit in the last place below.
        let cols = one_row(1.0, t, 4.0, 9.0);
        let tail = query(&cols, &[t, f64::from_bits(t.to_bits() - 1)], None);
        assert_eq!(tail.regimes.two_point, 1);
        assert_eq!(tail.rate_above[0], 0.0);
        assert!((tail.rate_above[1] - 0.4).abs() < 1e-16);
    }

    // At the threshold 2e5 the continued fraction and the narrow kernel give
    // the same tail; the query is monotone in the edge.
    #[test]
    fn narrow_handover_and_monotone() {
        let m = 1.0;
        let mut previous: Option<TailMoments> = None;
        for scale in [1.0_f64 - 1e-9, 1.0 + 1e-9] {
            let (a, b): (f64, f64) = (2e5 * scale, 6e5 * scale);
            let p = a / (a + b);
            let sd = (p * (1.0 - p) / (a + b + 1.0)).sqrt();
            let edges: Vec<f64> = (-4..=6).map(|k| p + k as f64 * sd).collect();
            let tail = query(&one_row(1.0, m, p, sd), &edges, None);
            for w in tail.rate_above.windows(2) {
                assert!(w[1] <= w[0]);
            }
            if let Some(prev) = &previous {
                assert_eq!((prev.regimes.incomplete_beta, tail.regimes.narrow_kernel), (1, 1));
                for e in 0..edges.len() {
                    assert!((tail.rate_above[e] - prev.rate_above[e]).abs() < 1e-8, "edge {e}");
                    let (x, y) = (tail.excess_above[e][0], prev.excess_above[e][0]);
                    assert!((x - y).abs() <= 1e-7 * y.max(1e-30), "edge {e}: {x} vs {y}");
                }
            }
            previous = Some(tail);
        }
    }
}
