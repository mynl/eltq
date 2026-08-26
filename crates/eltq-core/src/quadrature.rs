//! Shared-factor quadrature rules for integrating out `Z_c`.
//!
//! `Midpoint`: quantile nodes `z_q = Phi^{-1}((q + 1/2) / Q)` with equal
//! weights, the midpoint-quantile style of the one-unit specification.
//! `GaussHermite`: probabilists' Gauss-Hermite nodes and weights by
//! Golub-Welsch (eigenvalues of the symmetric tridiagonal Jacobi matrix with
//! off-diagonal sqrt(k)); only the first eigenvector component is tracked
//! through the QL iteration, so the rule costs O(Q^2) and is stable at any
//! Q. Weights are normalized to sum to one. Both rules return ascending
//! nodes, which the comonotone quantile walk relies on.

use crate::error::Error;
use crate::normal;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZRule {
    Midpoint,
    GaussHermite,
}

impl ZRule {
    pub fn parse(name: &str) -> Result<Self, Error> {
        match name {
            "midpoint" => Ok(ZRule::Midpoint),
            "gauss_hermite" => Ok(ZRule::GaussHermite),
            other => Err(Error::BadZRule(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ZRule::Midpoint => "midpoint",
            ZRule::GaussHermite => "gauss_hermite",
        }
    }
}

/// Nodes and weights for `q` points under `rule`.
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
}
