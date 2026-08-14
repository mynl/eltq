//! Batched per-event midpoint-quantile kernel (the production hot path).
//!
//! Computes all K midpoint quantiles of one beta event with per-event setup
//! done once and warm starts between successive quantiles: from the previous
//! quantile x_k, the next is predicted by a second-order Taylor step along
//! the quantile ODE dQ/du = 1/f(Q) (both derivatives analytic), then refined
//! by one or two safeguarded Halley iterations inside a maintained bracket
//! [x_k, 1), so hard shapes cannot diverge. Expected ~1-2 CDF evaluations
//! per quantile versus ~5-8 for cold scalar solves.
//!
//! Upper-half quantiles are computed on the reflected distribution with
//! exactly formed complement probabilities (Q(a, b; u) = 1 - Q(b, a; 1 - u)),
//! so both tails keep full relative precision. Fast paths: uniform
//! (alpha = beta = 1) is exact, symmetric (alpha = beta) solves half and
//! mirrors.

use crate::incbeta::BetaFn;
use crate::kernel::KernelError;

const MAX_STEP_ITERS: usize = 16;
const MAX_ROBUST_ITERS: usize = 500;

/// Fill `out[k] = Q(alpha, beta; (k + 1/2) / K)` with `K = out.len()`.
pub fn beta_midpoint_quantiles(alpha: f64, beta: f64, out: &mut [f64]) -> Result<(), KernelError> {
    let k = out.len();
    if k == 0 {
        return Ok(());
    }
    if !(alpha > 0.0 && beta > 0.0 && alpha.is_finite() && beta.is_finite()) {
        return Err(KernelError);
    }
    let k_f = k as f64;
    // Uniform: x_k = (k + 1/2) / K exactly.
    if alpha == 1.0 && beta == 1.0 {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = (i as f64 + 0.5) / k_f;
        }
        return Ok(());
    }
    // Symmetric: solve the lower half, mirror the rest about 1/2.
    if alpha == beta {
        let bf = BetaFn::new(alpha, beta);
        let half = k / 2;
        walk(&bf, (0..half).map(|i| (i, (i as f64 + 0.5) / k_f)), out)?;
        if k % 2 == 1 {
            out[half] = 0.5;
        }
        for i in 0..half {
            out[k - 1 - i] = 1.0 - out[i];
        }
        return Ok(());
    }
    let lower = BetaFn::new(alpha, beta);
    let cdf_half = lower.cdf(0.5);
    if !cdf_half.is_finite() {
        return Err(KernelError);
    }
    // u_i <= cdf(1/2) exactly when i < split: those quantiles lie in [0, 1/2].
    let split = ((cdf_half * k_f + 0.5).floor()).clamp(0.0, k_f) as usize;
    walk(&lower, (0..split).map(|i| (i, (i as f64 + 0.5) / k_f)), out)?;
    if split < k {
        let upper = BetaFn::new(beta, alpha);
        // Complements (K - j - 1/2) / K ascend as j descends; formed exactly.
        walk_map(
            &upper,
            (split..k).rev().map(|j| (j, (k_f - j as f64 - 0.5) / k_f)),
            |j, y| out[j] = 1.0 - y,
        )?;
    }
    Ok(())
}

/// Walk ascending targets writing `out[idx] = x` directly.
fn walk<I>(bf: &BetaFn, targets: I, out: &mut [f64]) -> Result<(), KernelError>
where
    I: Iterator<Item = (usize, f64)>,
{
    walk_map(bf, targets, |idx, x| out[idx] = x)
}

/// Walk targets with strictly increasing u, storing via `store(idx, x)`.
fn walk_map<I, S>(bf: &BetaFn, targets: I, mut store: S) -> Result<(), KernelError>
where
    I: Iterator<Item = (usize, f64)>,
    S: FnMut(usize, f64),
{
    let mut prev: Option<(f64, f64)> = None; // (x, u) with cdf(x) ~= u
    for (idx, u) in targets {
        let x = match prev {
            None => robust_solve(bf, u, 0.0, 1.0, cold_start(bf, u))?,
            Some((x0, u0)) => step_solve(bf, x0, u0, u)?,
        };
        store(idx, x);
        prev = Some((x, u));
    }
    Ok(())
}

/// Initial estimate for the smallest target probability.
fn cold_start(bf: &BetaFn, u: f64) -> f64 {
    let ln_tail = bf.ln_lower_tail_start(u);
    if ln_tail < (0.1_f64).ln() {
        // Deep-tail closed form; may underflow to zero, which robust_solve
        // treats as a below-bracket start and recovers from.
        ln_tail.exp().max(f64::MIN_POSITIVE)
    } else {
        let mean = bf.alpha / (bf.alpha + bf.beta);
        mean.clamp(1e-3, 1.0 - 1e-3)
    }
}

/// Predict x at `u` from the previous quantile `(x0, u0)` via a second-order
/// Taylor step along the quantile ODE, then refine with safeguarded Halley.
fn step_solve(bf: &BetaFn, x0: f64, u0: f64, u: f64) -> Result<f64, KernelError> {
    let mut lo = x0;
    let mut hi = 1.0_f64;
    let f0 = bf.pdf(x0);
    if !(f0.is_finite() && f0 > 0.0) {
        // Previous point sits on a density plateau or at zero: no usable ODE
        // predictor; fall back to a fresh robust solve above x0.
        return robust_solve(bf, u, lo, hi, cold_start(bf, u).max(x0));
    }
    let du = u - u0;
    let q1 = 1.0 / f0; // dQ/du
    let g0 = bf.dlnf(x0); // f'/f
    let mut x = x0 + du * q1 * (1.0 - 0.5 * du * g0 * q1); // + O(du^3)
    if !(x > lo && x < hi) {
        let newton_only = x0 + du * q1;
        x = if newton_only > lo && newton_only < hi {
            newton_only
        } else {
            0.5 * (lo + hi)
        };
    }
    let mut best = x;
    let mut best_err = f64::INFINITY;
    for _ in 0..MAX_STEP_ITERS {
        let e = bf.cdf(x) - u;
        if !e.is_finite() {
            return Err(KernelError);
        }
        if e.abs() < best_err {
            best_err = e.abs();
            best = x;
        }
        if e.abs() <= 32.0 * f64::EPSILON * u {
            return Ok(x);
        }
        if e > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        if hi - lo <= 2.0 * f64::EPSILON * hi {
            return Ok(best);
        }
        let f = bf.pdf(x);
        if !(f.is_finite() && f > 0.0) {
            return robust_solve(bf, u, lo, hi, 0.5 * (lo + hi));
        }
        let newton = e / f;
        let halley_factor = 1.0 - 0.5 * newton * bf.dlnf(x);
        let step = if halley_factor.is_finite() && halley_factor > 0.5 {
            newton / halley_factor
        } else {
            newton
        };
        let next = x - step;
        // Cubic convergence: a relative step of 1e-6 leaves an error of
        // order 1e-18 x, so the confirming CDF evaluation is unnecessary.
        if next > lo && next < hi && step.abs() <= 1e-6 * x {
            return Ok(next);
        }
        x = if next > lo && next < hi {
            next
        } else {
            0.5 * (lo + hi)
        };
    }
    robust_solve(bf, u, lo, hi, 0.5 * (lo + hi))
}

/// Bracketed Newton / log-space-Newton / bisection solve; cannot diverge.
/// Same safeguarding as the reference kernel, on the batched CDF.
fn robust_solve(bf: &BetaFn, u: f64, lo0: f64, hi0: f64, start: f64) -> Result<f64, KernelError> {
    let mut lo = lo0;
    let mut hi = hi0;
    let mut x = if start > lo && start < hi {
        start
    } else {
        0.5 * (lo + hi)
    };
    let mut best = x;
    let mut best_err = f64::INFINITY;
    for _ in 0..MAX_ROBUST_ITERS {
        let cdf = bf.cdf(x);
        if !cdf.is_finite() {
            return Err(KernelError);
        }
        let e = cdf - u;
        if e.abs() < best_err {
            best_err = e.abs();
            best = x;
        }
        if e == 0.0 {
            return Ok(x);
        }
        if e > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        if hi - lo <= 2.0 * f64::EPSILON * hi {
            return Ok(best);
        }
        if hi <= f64::MIN_POSITIVE {
            // True quantile underflows f64; the atom at zero is exact.
            return Ok(0.0);
        }
        let f = bf.pdf(x);
        let mut next = f64::NAN;
        if f.is_finite() && f > 0.0 {
            let cand = x - e / f;
            if cand > lo && cand < hi {
                next = cand;
            } else {
                // Log-space Newton handles power-law regions linear in ln x.
                let dt = (e / (f * x)).clamp(-700.0, 700.0);
                let cand = x * (-dt).exp();
                if cand > lo && cand < hi {
                    next = cand;
                }
            }
        }
        if !next.is_finite() || next <= lo || next >= hi {
            next = 0.5 * (lo + hi);
        }
        x = next;
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::midpoint_quantiles_reference;

    fn batched(alpha: f64, beta: f64, k: usize) -> Vec<f64> {
        let mut out = vec![0.0; k];
        beta_midpoint_quantiles(alpha, beta, &mut out).unwrap();
        out
    }

    fn reference(alpha: f64, beta: f64, k: usize) -> Vec<f64> {
        let mut out = vec![0.0; k];
        midpoint_quantiles_reference(alpha, beta, &mut out).unwrap();
        out
    }

    #[test]
    fn uniform_fast_path_is_exact() {
        let xs = batched(1.0, 1.0, 8);
        for (i, &x) in xs.iter().enumerate() {
            assert_eq!(x, (i as f64 + 0.5) / 8.0);
        }
    }

    #[test]
    fn symmetric_mirror_matches_reference() {
        for k in [1, 2, 7, 32] {
            let got = batched(3.5, 3.5, k);
            let want = reference(3.5, 3.5, k);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-13, "k={k}: {g} vs {w}");
            }
            // Exact mirror by construction (upper half is built as 1 - x).
            for i in 0..k / 2 {
                assert_eq!(got[k - 1 - i], 1.0 - got[i]);
            }
        }
    }

    #[test]
    fn agrees_with_reference_across_regimes() {
        let shapes = [
            (2.5, 7.5),
            (7.5, 2.5),
            (0.5, 1.5),
            (300.0, 100.0),
            (40.0, 2.0),
            (2.0, 40.0),
            (0.5, 0.5),
            (500.0, 500.0),
        ];
        for (a, b) in shapes {
            let got = batched(a, b, 32);
            let want = reference(a, b, 32);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-13,
                    "a={a} b={b} k={i}: batched {g} vs reference {w}"
                );
            }
        }
    }

    #[test]
    fn extreme_shapes_agree_loosely_and_stay_sane() {
        // Plateau regions of extreme U-shapes amplify CDF noise into x; the
        // histogram effect stays negligible (weights move by w dx / h).
        for (a, b) in [(6e-4, 1.4e-3), (1e-3, 1.0), (1.0, 1e-3), (5e-4, 5e-4)] {
            let got = batched(a, b, 64);
            let want = reference(a, b, 64);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(g.is_finite() && (0.0..=1.0).contains(g));
                assert!(
                    (g - w).abs() <= 1e-9,
                    "a={a} b={b} k={i}: batched {g} vs reference {w}"
                );
            }
            for pair in got.windows(2) {
                assert!(pair[1] >= pair[0] - 1e-12);
            }
        }
    }

    #[test]
    fn single_quantile_is_the_median_side() {
        let got = batched(2.0, 5.0, 1);
        let want = reference(2.0, 5.0, 1);
        assert!((got[0] - want[0]).abs() < 1e-13);
    }
}
