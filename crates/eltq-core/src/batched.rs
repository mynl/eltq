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
//!
//! When both parameters are at or above `LARGE_PARAMETER_FROM` neither
//! kernel walks: the continued fraction behind the CDF needs about
//! sqrt(min(alpha, beta)) iterations near the mean and stops at 400, and
//! the exponent of its front factor rounds by eps (alpha + beta), which
//! overflows beyond about 5e18. Every quantile then comes from the
//! Cornish-Fisher asymptotic expansion in the beta's cumulants (Abramowitz
//! and Stegun, Handbook of Mathematical Functions, 26.2.49;
//! `dev/done/plan-quantiles-large-parameters.md`).

use crate::incbeta::BetaFn;
use crate::kernel::KernelError;
use crate::normal;

const MAX_STEP_ITERS: usize = 16;
const MAX_ROBUST_ITERS: usize = 500;

/// The smaller parameter from which both kernels take the Cornish-Fisher
/// expansion instead of the walk. Measured against SciPy's `betaincinv`
/// (2026-10-25, 32 midpoint levels, beta / alpha from 1 to 3e4): the walk
/// is within 4e-9 standard deviations up to 3.2e5 and drifts from 5.6e5
/// (3e-7) to 1e-2 at 1e7; the expansion is within 4e-9 at 1.8e5 and
/// improves as min(alpha, beta)^(-3/2). Mirrored by `LARGE_PARAMETER_FROM`
/// in `python/eltq/_meta.py`.
pub const LARGE_PARAMETER_FROM: f64 = 2e5;

/// Whether `(alpha, beta)` takes the expansion route.
#[inline]
pub fn uses_expansion(alpha: f64, beta: f64) -> bool {
    alpha.min(beta) >= LARGE_PARAMETER_FROM
}

/// The beta's mean, standard deviation, skewness and excess kurtosis, from
/// its moments, in forms that stay finite for parameters up to about 1e150
/// (products of two parameters, never of three).
struct Cumulants {
    mean: f64,
    sd: f64,
    skew: f64,
    kurt: f64,
}

impl Cumulants {
    fn of(a: f64, b: f64) -> Self {
        let s = a + b;
        let mean = a / s;
        let sd = (a * b / (s * s * (s + 1.0))).sqrt();
        // (b - a) / sqrt(a b), the shape factor of the odd cumulants.
        let r = (b / a).sqrt() - (a / b).sqrt();
        let skew = 2.0 * r * (s + 1.0).sqrt() / (s + 2.0);
        let kurt = 6.0 * (r * r * (s + 1.0) - (s + 2.0)) / ((s + 2.0) * (s + 3.0));
        Cumulants { mean, sd, skew, kurt }
    }

    /// `Q(u)` to second order in the cumulants, `z = Phi^{-1}(u)`:
    /// `mean + sd [z + g1 (z^2 - 1) / 6 + g2 (z^3 - 3 z) / 24
    /// - g1^2 (2 z^3 - 5 z) / 36]`. Endpoint targets give 0 and 1; the
    /// clip to the unit interval never acts at these parameters (it would
    /// need |z| above sqrt(min(alpha, beta))).
    fn quantile(&self, u: f64) -> f64 {
        if u <= 0.0 {
            return 0.0;
        }
        if u >= 1.0 {
            return 1.0;
        }
        let z = normal::quantile(u);
        let z2 = z * z;
        let w = z + self.skew * (z2 - 1.0) / 6.0 + self.kurt * z * (z2 - 3.0) / 24.0
            - self.skew * self.skew * z * (2.0 * z2 - 5.0) / 36.0;
        (self.mean + self.sd * w).clamp(0.0, 1.0)
    }
}

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
    if uses_expansion(alpha, beta) {
        let cumulants = Cumulants::of(alpha, beta);
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = cumulants.quantile((i as f64 + 0.5) / k_f);
        }
        return Ok(());
    }
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

/// Fill `out[k] = Q(alpha, beta; targets[k])` for non-decreasing targets in
/// [0, 1]; endpoint targets yield 0 and 1 exactly.
///
/// Repeated targets are expected: Gauss-Hermite nodes push `Phi(w_c z)` to
/// exactly 0 or 1 in the tails, and from Q in the hundreds two adjacent
/// nodes in z ~ [8.14, 8.29] both give `Phi(z) = 1 - 2^-53`. Only the first
/// of each run of equal interior targets is solved; the rest of the run
/// copies its quantile, so the walks see strictly increasing targets.
///
/// Same machinery as the midpoint kernel: the lower half is walked
/// directly and the upper half on the reflected distribution with exactly
/// formed complements, so both tails keep full relative precision. A run of
/// equal targets never straddles the split, which is a threshold on u.
pub fn beta_quantiles(alpha: f64, beta: f64, targets: &[f64], out: &mut [f64]) -> Result<(), KernelError> {
    assert_eq!(targets.len(), out.len());
    if targets.is_empty() {
        return Ok(());
    }
    if !(alpha > 0.0 && beta > 0.0 && alpha.is_finite() && beta.is_finite()) {
        return Err(KernelError);
    }
    if targets.iter().any(|u| !(0.0..=1.0).contains(u)) {
        return Err(KernelError);
    }
    if targets.windows(2).any(|w| w[1] < w[0]) {
        return Err(KernelError);
    }
    if uses_expansion(alpha, beta) {
        // Repeated targets repeat their quantile; endpoints give 0 and 1.
        let cumulants = Cumulants::of(alpha, beta);
        for (slot, &u) in out.iter_mut().zip(targets) {
            *slot = cumulants.quantile(u);
        }
        return Ok(());
    }
    let lower = BetaFn::new(alpha, beta);
    let cdf_half = lower.cdf(0.5);
    if !cdf_half.is_finite() {
        return Err(KernelError);
    }
    let split = targets.partition_point(|&u| u <= cdf_half);
    // Interior targets that start a run of equal values: the ones solved.
    let leads = |i: usize| targets[i] > 0.0 && targets[i] < 1.0 && (i == 0 || targets[i - 1] != targets[i]);
    walk(&lower, (0..split).filter(|&i| leads(i)).map(|i| (i, targets[i])), out)?;
    if split < targets.len() {
        let upper = BetaFn::new(beta, alpha);
        walk_map(
            &upper,
            (split..targets.len())
                .rev()
                .filter(|&j| leads(j))
                .map(|j| (j, 1.0 - targets[j])),
            |j, y| out[j] = 1.0 - y,
        )?;
    }
    for i in 0..targets.len() {
        let u = targets[i];
        if u == 0.0 {
            out[i] = 0.0;
        } else if u == 1.0 {
            out[i] = 1.0;
        } else if i > 0 && targets[i - 1] == u {
            out[i] = out[i - 1];
        }
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
/// Callers with repeated targets pass the first of each run and copy the
/// rest (`beta_quantiles`): the warm-started step predicts along du and
/// brackets in [x_prev, 1), which assumes u moves forward.
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
    fn large_parameters_match_scipy() {
        // scipy.special.betaincinv(a, b, u), 2026-10-25, where SciPy is
        // sound (the walk agrees with it to 4e-9 sigma up to 3.2e5 and the
        // expansion to 1e-8 sigma to 1e7; SciPy's own floor rises above
        // 1e-8 sigma from about a = 3e7 with b = 1e10). Generated by
        // `for a, b in ...: for u in (0.5/32, 10.5/32, 31.5/32):
        //  print(betaincinv(a, b, u))`.
        let table = [
            (3.0e5, 3.0e5, 0.015625, 0.4986096819715214),
            (3.0e5, 3.0e5, 0.328125, 0.49971269133255036),
            (3.0e5, 3.0e5, 0.984375, 0.5013903180284787),
            (3.0e5, 9.0e5, 0.015625, 0.24914911242004817),
            (3.0e5, 9.0e5, 0.328125, 0.2498239487843525),
            (3.0e5, 9.0e5, 0.984375, 0.2508518984619453),
            (3.0e5, 9.0e7, 0.015625, 0.0033092296878922947),
            (3.0e5, 9.0e7, 0.328125, 0.003319560913916627),
            (3.0e5, 9.0e7, 0.984375, 0.003335315273308392),
            (1.0e6, 1.0e6, 0.015625, 0.49923849064541),
            (1.0e6, 1.0e6, 0.328125, 0.4998426345986764),
            (1.0e6, 1.0e6, 0.984375, 0.5007615093545901),
            (1.0e6, 3.0e8, 0.015625, 0.0033151193066180266),
            (1.0e6, 3.0e8, 0.328125, 0.0033207819867629947),
            (1.0e6, 3.0e8, 0.984375, 0.0033294069724413957),
            (1.0e7, 3.0e9, 0.015625, 0.003320000458083755),
            (1.0e7, 3.0e9, 0.328125, 0.003321792211207708),
            (1.0e7, 3.0e9, 0.984375, 0.003324518615004973),
            (1.0e8, 1.0e8, 0.015625, 0.49992384903024656),
            (1.0e8, 1.0e8, 0.328125, 0.4999842634614292),
            (1.0e8, 1.0e8, 0.984375, 0.5000761509697534),
        ];
        for &(a, b, u, want) in &table {
            let cumulants = Cumulants::of(a, b);
            let got = cumulants.quantile(u);
            assert!(
                (got - want).abs() <= 1e-8 * cumulants.sd,
                "a={a} b={b} u={u}: {got} vs SciPy {want}, {} sigma",
                (got - want).abs() / cumulants.sd
            );
            // Through the public kernel too.
            let mut out = [0.0];
            beta_quantiles(a, b, &[u], &mut out).unwrap();
            assert_eq!(out[0], got);
        }
    }

    #[test]
    fn large_parameters_continuous_at_the_threshold() {
        // The walk just below the threshold and the expansion just above
        // it agree to 1e-8 sigma; the parameters themselves differ by 2e-9.
        for ratio in [1.0, 3.0, 300.0] {
            let below = LARGE_PARAMETER_FROM * (1.0 - 1e-9);
            let above = LARGE_PARAMETER_FROM * (1.0 + 1e-9);
            assert!(!uses_expansion(below, below * ratio) && uses_expansion(above, above * ratio));
            let walk = batched(below, below * ratio, 32);
            let expansion = batched(above, above * ratio, 32);
            let sd = Cumulants::of(above, above * ratio).sd;
            for (i, (w, e)) in walk.iter().zip(&expansion).enumerate() {
                assert!(
                    (w - e).abs() <= 1e-8 * sd,
                    "ratio={ratio} k={i}: walk {w} vs expansion {e}, {} sigma",
                    (w - e).abs() / sd
                );
            }
        }
    }

    #[test]
    fn huge_parameters_stay_finite_and_monotone() {
        // The row of the 2026-10-25 report (exposed value 1e6, mean
        // 250,000, sd 1e-6), a symmetric giant, and two lopsided ones.
        for (a, b) in [(4.6875e22, 1.40625e23), (1e26, 1e26), (2e5, 1e30), (1e30, 2e5)] {
            let xs = batched(a, b, 32);
            let cumulants = Cumulants::of(a, b);
            for pair in xs.windows(2) {
                assert!(pair[1] >= pair[0], "a={a} b={b}: {pair:?}");
            }
            for &x in &xs {
                assert!(x.is_finite());
                assert!((x - cumulants.mean).abs() <= 2.2 * cumulants.sd + 2.0 * f64::EPSILON * cumulants.mean);
            }
        }
    }

    #[test]
    fn arbitrary_targets_above_the_threshold() {
        let (a, b) = (3e5, 2e6);
        let targets = [0.0, 0.0, 1e-12, 0.3, 0.3, 0.5, 0.999, 1.0, 1.0];
        let mut out = vec![0.0; targets.len()];
        beta_quantiles(a, b, &targets, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[7], 1.0);
        assert_eq!(out[8], 1.0);
        assert_eq!(out[3], out[4]);
        for pair in out.windows(2) {
            assert!(pair[1] >= pair[0]);
        }
        assert_eq!(out[5], Cumulants::of(a, b).quantile(0.5));
        assert!(out[2] > 0.0 && out[2] < out[3]);
    }

    #[test]
    fn arbitrary_targets_agree_with_midpoint_kernel() {
        for (a, b) in [(2.5, 7.5), (0.5, 0.5), (300.0, 100.0), (40.0, 2.0)] {
            let k = 32;
            let targets: Vec<f64> = (0..k).map(|i| (i as f64 + 0.5) / k as f64).collect();
            let mut got = vec![0.0; k];
            beta_quantiles(a, b, &targets, &mut got).unwrap();
            let want = batched(a, b, k);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() <= 1e-13, "a={a} b={b}: {g} vs {w}");
            }
            // Irregular ascending targets round-trip through the cdf.
            let bf = BetaFn::new(a, b);
            // Upper targets stop where 1 - x stays representable for every
            // shape (Beta(0.5, 0.5) at u = 1 - 1e-9 rounds to exactly 1).
            let irregular = [1e-9, 0.01, 0.2, 0.5, 0.9, 0.999, 0.999_999];
            let mut xs = vec![0.0; irregular.len()];
            beta_quantiles(a, b, &irregular, &mut xs).unwrap();
            for (x, u) in xs.iter().zip(&irregular) {
                // One ulp of x moves the cdf by pdf(x) ulp(x): steep tails
                // (Beta(0.5, 0.5) near 1) cannot round-trip tighter than that.
                let tolerance = 1e-12 * u.max(1e-3) + 8.0 * f64::EPSILON * x * bf.pdf(*x);
                assert!((bf.cdf(*x) - u).abs() <= tolerance, "a={a} b={b} u={u} x={x}");
            }
        }
        let mut xs = [0.0; 3];
        beta_quantiles(2.0, 3.0, &[0.0, 0.5, 1.0], &mut xs).unwrap();
        assert_eq!(xs[0], 0.0);
        assert_eq!(xs[2], 1.0);
        assert!(beta_quantiles(2.0, 3.0, &[0.5, 0.4], &mut xs[..2]).is_err());
        // Repeated interior targets (Gauss-Hermite nodes whose Phi rounds to
        // the same value) are solved once and copied.
        beta_quantiles(2.0, 3.0, &[0.4, 0.4], &mut xs[..2]).unwrap();
        assert_eq!(xs[0], xs[1]);
        let mut lone = [0.0; 1];
        beta_quantiles(2.0, 3.0, &[0.4], &mut lone).unwrap();
        // Within a run the copies are bitwise equal; against a lone solve
        // they agree to solver precision (the run reaches 0.4 by a warm step
        // from 0.1, the lone call by a cold solve).
        let close = |a: f64, b: f64| (a - b).abs() <= 4.0 * f64::EPSILON * b;
        let mut run = [0.0; 5];
        beta_quantiles(2.0, 3.0, &[0.1, 0.4, 0.4, 0.4, 0.7], &mut run).unwrap();
        assert_eq!(&run[2..4], &[run[1]; 2]);
        assert!(close(run[1], lone[0]), "{} vs {}", run[1], lone[0]);
        assert!(run[0] < run[1] && run[3] < run[4]);
        // The same in the reflected upper half (cdf(1/2) = 0.6875 here),
        // including the value 1 - 2^-53 that Phi(z) takes for z near 8.2.
        let top = 1.0 - f64::EPSILON / 2.0;
        let mut upper = [0.0; 6];
        beta_quantiles(2.0, 3.0, &[0.2, 0.9, 0.9, top, top, 1.0], &mut upper).unwrap();
        let mut lone_upper = [0.0; 2];
        beta_quantiles(2.0, 3.0, &[0.9], &mut lone_upper[..1]).unwrap();
        beta_quantiles(2.0, 3.0, &[top], &mut lone_upper[1..]).unwrap();
        assert!(upper[1] == upper[2] && upper[3] == upper[4]);
        assert!(close(upper[1], lone_upper[0]) && close(upper[3], lone_upper[1]));
        assert!(upper[0] < upper[1] && upper[2] < upper[3] && upper[4] < upper[5]);
        assert_eq!(upper[5], 1.0);
        // Repeated endpoints (Gauss-Hermite tails) are accepted.
        let mut ys = [0.0; 6];
        beta_quantiles(2.0, 1.5, &[0.0, 0.0, 1e-20, 0.5, 1.0, 1.0], &mut ys).unwrap();
        assert_eq!(&ys[..2], &[0.0, 0.0]);
        assert!(ys[2] > 0.0 && ys[2] < ys[3]);
        assert_eq!(&ys[4..], &[1.0, 1.0]);
    }

    #[test]
    fn single_quantile_is_the_median_side() {
        let got = batched(2.0, 5.0, 1);
        let want = reference(2.0, 5.0, 1);
        assert!((got[0] - want[0]).abs() < 1e-13);
    }
}
