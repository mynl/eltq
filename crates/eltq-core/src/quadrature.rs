//! Shared-factor rules for integrating out `Z_c`.
//!
//! Node rules: `Midpoint`, quantile nodes `z_q = Phi^{-1}((q + 1/2) / Q)`
//! with equal weights, the midpoint-quantile style of the one-unit
//! specification; `GaussHermite`, probabilists' Gauss-Hermite nodes and
//! weights by Golub-Welsch (eigenvalues of the symmetric tridiagonal Jacobi
//! matrix with off-diagonal sqrt(k)); only the first eigenvector component
//! is tracked through the QL iteration, so the rule costs O(Q^2) and is
//! stable at any Q. Weights are normalized to sum to one. Both rules return
//! ascending nodes, which the comonotone quantile walk relies on.
//!
//! The stratified rule (2.6.0, spec section 4A): strata of `Z_c` in
//! probability, `Q` of equal width over `[0, STRATA_BODY_TOP]` and a
//! geometric ladder of ratio `STRATA_TAIL_RATIO` from there to the floor
//! `STRATA_TAIL_FLOOR` in the upper tail. A unit's conditional law is
//! averaged over each stratum exactly, through the bivariate normal cdf for
//! a loading below one and through the restriction of its marginal to the
//! stratum at loading one, so every unit's marginal is exact at any `Q`;
//! the units are treated as independent given the stratum. That dependence
//! error is governed by the width of the stratum in which a tail level
//! lies: with equal strata the whole upper tail of an event total sits in
//! one stratum and the exceedance at the 1e-4 and 1e-5 levels was short by
//! 50% to 99% even at 128 strata, while the ladder holds it to the ratio's
//! own figure at every level, 3e-3 at the square root of two
//! (`benchmarks/z-convergence-2.6.0.md`). Every view is nondecreasing in the
//! loss, so every exceedance lives in the upper tail of the factor and the
//! lower tail keeps the body's width. [`FactorRule`] carries either form to
//! the conditional matrices.

use crate::error::Error;
use crate::normal;

/// Probability of the factor covered by the `Q` equal body strata; the
/// upper tail above it is the ladder. Mirrors `eltq._meta.STRATA_BODY_TOP`.
pub const STRATA_BODY_TOP: f64 = 0.75;
/// Ratio between successive tail strata in probability. Mirrors
/// `eltq._meta.STRATA_TAIL_RATIO`.
pub const STRATA_TAIL_RATIO: f64 = std::f64::consts::SQRT_2;
/// The ladder stops at the first tail probability at or below this; the
/// last stratum reaches 1. Mirrors `eltq._meta.STRATA_TAIL_FLOOR`.
pub const STRATA_TAIL_FLOOR: f64 = 5.960_464_477_539_063e-8; // 2^-24

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZRule {
    Midpoint,
    GaussHermite,
    Stratified,
}

impl ZRule {
    pub fn parse(name: &str) -> Result<Self, Error> {
        match name {
            "midpoint" => Ok(ZRule::Midpoint),
            "gauss_hermite" => Ok(ZRule::GaussHermite),
            "stratified" => Ok(ZRule::Stratified),
            other => Err(Error::BadZRule(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ZRule::Midpoint => "midpoint",
            ZRule::GaussHermite => "gauss_hermite",
            ZRule::Stratified => "stratified",
        }
    }
}

/// The shared factor's rule in the form the conditional matrices consume:
/// `Q` nodes with weights (the node rules), or strata with their boundaries
/// and widths (the stratified rule).
#[derive(Debug, Clone, PartialEq)]
pub struct FactorRule {
    pub rule: ZRule,
    /// Nodes `z_q`, ascending; empty under the stratified rule.
    pub nodes: Vec<f64>,
    /// Weights summing to one: the rule's weights, or the strata's widths in
    /// probability.
    pub weights: Vec<f64>,
    /// Stratum boundaries, `len() + 1` values: in probability from 0 to 1
    /// and in `z` from `-inf` to `+inf`; empty under the node rules.
    pub edges_u: Vec<f64>,
    pub edges_z: Vec<f64>,
}

impl FactorRule {
    /// The rule with `q` nodes, or `q` body strata plus the tail ladder.
    pub fn new(q: usize, rule: ZRule) -> Result<Self, Error> {
        if q < 1 {
            return Err(Error::BadZNodes(q));
        }
        match rule {
            ZRule::Stratified => Ok(Self::from_edges(stratum_edges(q))),
            _ => {
                let (nodes, weights) = z_quadrature(q, rule)?;
                Ok(FactorRule {
                    rule,
                    nodes,
                    weights,
                    edges_u: Vec::new(),
                    edges_z: Vec::new(),
                })
            }
        }
    }

    /// The stratified rule on explicit boundaries in probability, ascending
    /// from 0 to 1 (the convergence study's equal strata).
    pub fn from_edges(edges_u: Vec<f64>) -> Self {
        let edges_z = edges_u.iter().map(|&u| normal::quantile(u)).collect();
        let weights = edges_u.windows(2).map(|e| e[1] - e[0]).collect();
        FactorRule {
            rule: ZRule::Stratified,
            nodes: Vec::new(),
            weights,
            edges_u,
            edges_z,
        }
    }
}

/// Boundaries of the default stratified partition: `q` equal strata over
/// `[0, STRATA_BODY_TOP]`, then tail probabilities `(1 - STRATA_BODY_TOP) /
/// STRATA_TAIL_RATIO^k` for `k = 1, 2, ...` down to the floor, then 1.
pub fn stratum_edges(q: usize) -> Vec<f64> {
    let mut edges: Vec<f64> = (0..=q).map(|i| STRATA_BODY_TOP * i as f64 / q as f64).collect();
    let mut k = 1;
    loop {
        let tail = (1.0 - STRATA_BODY_TOP) * STRATA_TAIL_RATIO.powi(-k);
        if tail <= STRATA_TAIL_FLOOR {
            break;
        }
        edges.push(1.0 - tail);
        k += 1;
    }
    edges.push(1.0);
    edges
}

impl FactorRule {
    /// One node at `z = 0` with weight one: the rule for a unit whose law
    /// does not depend on the factor.
    pub fn single() -> Self {
        FactorRule {
            rule: ZRule::Midpoint,
            nodes: vec![0.0],
            weights: vec![1.0],
            edges_u: Vec::new(),
            edges_z: Vec::new(),
        }
    }

    /// Number of nodes or strata, `Q`.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    pub fn is_stratified(&self) -> bool {
        self.rule == ZRule::Stratified
    }
}

/// Nodes and weights for `q` points under a node rule; the stratified rule
/// has no nodes and is refused here (use [`FactorRule::new`]).
pub fn z_quadrature(q: usize, rule: ZRule) -> Result<(Vec<f64>, Vec<f64>), Error> {
    if q < 1 {
        return Err(Error::BadZNodes(q));
    }
    match rule {
        ZRule::Midpoint => {
            let qf = q as f64;
            let nodes = (0..q).map(|i| normal::quantile((i as f64 + 0.5) / qf)).collect();
            Ok((nodes, vec![1.0 / qf; q]))
        }
        ZRule::GaussHermite => Ok(gauss_hermite(q)),
        ZRule::Stratified => Err(Error::BadZRule("stratified (a rule of strata, not nodes)".to_string())),
    }
}

/// Probabilists' Gauss-Hermite rule of `n` points, symmetrized exactly.
fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 1 {
        return (vec![0.0], vec![1.0]);
    }
    let mut d = vec![0.0_f64; n];
    let mut e: Vec<f64> = (1..n).map(|k| (k as f64).sqrt()).collect();
    e.push(0.0);
    let mut first = vec![0.0_f64; n];
    first[0] = 1.0;
    tql_first_row(&mut d, &mut e, &mut first);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap());
    let nodes: Vec<f64> = order.iter().map(|&i| d[i]).collect();
    let weights: Vec<f64> = order.iter().map(|&i| first[i] * first[i]).collect();
    let total: f64 = weights.iter().sum();
    // Nodes come in +/- pairs with equal weights; enforce it bitwise.
    let nodes = (0..n).map(|i| 0.5 * (nodes[i] - nodes[n - 1 - i])).collect();
    let weights = (0..n)
        .map(|i| 0.5 * (weights[i] + weights[n - 1 - i]) / total)
        .collect();
    (nodes, weights)
}

/// Implicit-shift QL on a symmetric tridiagonal matrix (diagonal `d`,
/// subdiagonal `e[i]` between i and i + 1, `e[n - 1]` unused), rotating only
/// the first row of the accumulated eigenvector matrix (`first`). On return
/// `d` holds the eigenvalues and `first[i]` the first component of
/// eigenvector i.
fn tql_first_row(d: &mut [f64], e: &mut [f64], first: &mut [f64]) {
    let n = d.len();
    for l in 0..n {
        let mut iterations = 0;
        loop {
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iterations += 1;
            assert!(iterations <= 200, "QL iteration failed to converge");
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (1.0_f64, 1.0_f64, 0.0_f64);
            let mut i = m;
            let mut early = false;
            while i > l {
                i -= 1;
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    early = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                let a = first[i + 1];
                first[i + 1] = s * first[i] + c * a;
                first[i] = c * first[i] - s * a;
            }
            if early {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments(nodes: &[f64], weights: &[f64]) -> (f64, f64, f64, f64) {
        let m = |k: i32| nodes.iter().zip(weights).map(|(z, w)| w * z.powi(k)).sum::<f64>();
        (m(0), m(1), m(2), m(4))
    }

    #[test]
    fn gauss_hermite_moments_are_exact() {
        for q in [1, 2, 3, 8, 32, 128, 1001] {
            let (z, w) = z_quadrature(q, ZRule::GaussHermite).unwrap();
            let (m0, m1, m2, m4) = moments(&z, &w);
            assert!((m0 - 1.0).abs() < 1e-14, "q={q}");
            assert!(m1.abs() < 1e-13, "q={q}");
            if q >= 2 {
                assert!((m2 - 1.0).abs() < 1e-11, "q={q} m2={m2}");
            }
            if q >= 3 {
                assert!((m4 - 3.0).abs() < 1e-10, "q={q} m4={m4}");
            }
            assert!(z.windows(2).all(|p| p[1] > p[0]));
            for i in 0..q {
                assert_eq!(z[i], -z[q - 1 - i]);
                assert_eq!(w[i], w[q - 1 - i]);
            }
        }
    }

    #[test]
    fn gauss_hermite_matches_closed_forms() {
        let (z, w) = z_quadrature(3, ZRule::GaussHermite).unwrap();
        assert!((z[2] - 3.0_f64.sqrt()).abs() < 1e-14);
        assert!((w[1] - 2.0 / 3.0).abs() < 1e-14);
        let (z, _) = z_quadrature(2, ZRule::GaussHermite).unwrap();
        assert!((z[1] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn midpoint_rule() {
        let (z, w) = z_quadrature(4, ZRule::Midpoint).unwrap();
        assert!(w.iter().all(|&x| x == 0.25));
        assert!((z[0] - normal::quantile(0.125)).abs() < 1e-15);
        assert_eq!(z[0], -z[3]);
        assert!(matches!(z_quadrature(0, ZRule::Midpoint), Err(Error::BadZNodes(0))));
        assert!(matches!(ZRule::parse("simpson"), Err(Error::BadZRule(_))));
    }

    #[test]
    fn stratified_rule() {
        // Q = 3 body strata of width 1/4 over [0, 3/4], then the ladder
        // 1/4 / sqrt(2)^k for k = 1..43 (the 44th, 2^-24, is the floor),
        // then the top stratum to 1: 3 + 43 + 1 = 47 strata.
        let rule = FactorRule::new(3, ZRule::Stratified).unwrap();
        assert_eq!(rule.len(), 47);
        assert!(rule.is_stratified() && rule.nodes.is_empty());
        assert_eq!(rule.edges_u[..4], [0.0, 0.25, 0.5, 0.75]);
        assert!((rule.edges_u[4] - (1.0 - 0.25 / 2f64.sqrt())).abs() < 1e-16);
        assert!((1.0 - rule.edges_u[46] - 0.25 * 2f64.powf(-21.5)).abs() < 1e-18);
        assert_eq!(*rule.edges_u.last().unwrap(), 1.0);
        assert_eq!(rule.edges_z[0], f64::NEG_INFINITY);
        assert_eq!(*rule.edges_z.last().unwrap(), f64::INFINITY);
        assert!(rule.edges_u.windows(2).all(|e| e[1] > e[0]));
        let total: f64 = rule.weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-15);
        assert_eq!(rule.weights[0], 0.25);
        assert_eq!(FactorRule::new(32, ZRule::Stratified).unwrap().len(), 32 + 44);
        // Explicit equal strata (the study's comparison).
        let equal = FactorRule::from_edges((0..=4).map(|i| i as f64 / 4.0).collect());
        assert_eq!(equal.edges_z[2], 0.0);
        assert!(equal.weights.iter().all(|&w| w == 0.25));
        assert!(matches!(z_quadrature(4, ZRule::Stratified), Err(Error::BadZRule(_))));
        assert!(matches!(FactorRule::new(0, ZRule::Stratified), Err(Error::BadZNodes(0))));
        let nodes = FactorRule::new(8, ZRule::GaussHermite).unwrap();
        assert_eq!((nodes.len(), nodes.edges_z.len()), (8, 0));
        assert_eq!(FactorRule::single().nodes, vec![0.0]);
    }
}
