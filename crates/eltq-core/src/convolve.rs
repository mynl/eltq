//! Convolution across units: direct (exact, small supports) and FFT (real
//! transforms via `realfft`: a real row of length n has an n / 2 + 1 point
//! spectrum, halving the work of a complex transform; plans cached per
//! worker per size).
//!
//! Both paths produce the same linear convolution; the engine picks per
//! event and view by a cost estimate and tests hold them to agree within
//! floating-point tolerance.

use std::collections::HashMap;
use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;

/// `out = a * b` (linear convolution, length a + b - 1), direct O(|a| |b|).
pub fn convolve_direct(a: &[f64], b: &[f64], out: &mut Vec<f64>) {
    out.clear();
    if a.is_empty() || b.is_empty() {
        return;
    }
    out.resize(a.len() + b.len() - 1, 0.0);
    // Iterate the shorter operand on the outside so the inner axpy is long.
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    for (i, &s) in short.iter().enumerate() {
        if s == 0.0 {
            continue;
        }
        for (o, &l) in out[i..i + long.len()].iter_mut().zip(long) {
            *o += s * l;
        }
    }
}

/// Estimated multiply-add count of chaining direct convolutions over rows
/// of the given lengths.
pub fn direct_cost(lengths: &[usize]) -> f64 {
    let mut acc = 1usize;
    let mut cost = 0.0;
    for &len in lengths {
        cost += (acc * len) as f64;
        acc += len - 1;
    }
    cost
}

/// Rough cost of one real FFT of size n (about 2.5 n log2 n flops).
pub fn fft_cost(n: usize) -> f64 {
    2.5 * n as f64 * (n as f64).log2().max(1.0)
}

/// Next power of two at or above `n` (FFT size).
pub fn fft_size(n: usize) -> usize {
    n.next_power_of_two().max(2)
}

/// Per-worker real FFT plans, cached by size, plus the scratch the
/// transforms need.
pub struct FftCache {
    planner: RealFftPlanner<f64>,
    forward: HashMap<usize, Arc<dyn RealToComplex<f64>>>,
    inverse: HashMap<usize, Arc<dyn ComplexToReal<f64>>>,
    real: Vec<f64>,
    scratch: Vec<Complex<f64>>,
}

impl Default for FftCache {
    fn default() -> Self {
        FftCache {
            planner: RealFftPlanner::new(),
            forward: HashMap::new(),
            inverse: HashMap::new(),
            real: Vec::new(),
            scratch: Vec::new(),
        }
    }
}

impl FftCache {
    fn forward(&mut self, n: usize) -> Arc<dyn RealToComplex<f64>> {
        let planner = &mut self.planner;
        self.forward
            .entry(n)
            .or_insert_with(|| planner.plan_fft_forward(n))
            .clone()
    }

    fn inverse(&mut self, n: usize) -> Arc<dyn ComplexToReal<f64>> {
        let planner = &mut self.planner;
        self.inverse
            .entry(n)
            .or_insert_with(|| planner.plan_fft_inverse(n))
            .clone()
    }

    /// Spectrum length of a real transform of size n.
    pub fn spectrum_len(n: usize) -> usize {
        n / 2 + 1
    }

    /// Forward transform of a real row zero-padded to `n` into `spectrum`
    /// (length n / 2 + 1).
    pub fn spectrum(&mut self, row: &[f64], n: usize, spectrum: &mut Vec<Complex<f64>>) {
        let plan = self.forward(n);
        self.real.clear();
        self.real.resize(n, 0.0);
        self.real[..row.len()].copy_from_slice(row);
        spectrum.clear();
        spectrum.resize(Self::spectrum_len(n), Complex::new(0.0, 0.0));
        self.scratch.resize(plan.get_scratch_len(), Complex::new(0.0, 0.0));
        plan.process_with_scratch(&mut self.real, spectrum, &mut self.scratch)
            .expect("real FFT sizes are consistent");
    }

    /// Inverse transform of `spectrum` (length n / 2 + 1 for size `n`,
    /// consumed) into the first `len` real values of `out`, normalized by n
    /// and clamped below at zero.
    ///
    /// The transformed rows are probability weights. A bin whose exact value
    /// is zero (a gap in a ceded tower, the region below a deductible, the
    /// far tail of a tight beta) comes back as symmetric round-off of order
    /// 1e-17 relative to the largest bin, so about half of such bins would be
    /// negative. Clamping per element moves the mass by at most the sum of
    /// those magnitudes and keeps the result independent of thread count; a
    /// NaN passes through unclamped so a defect upstream stays visible.
    pub fn inverse_into(&mut self, spectrum: &mut [Complex<f64>], n: usize, len: usize, out: &mut Vec<f64>) {
        let plan = self.inverse(n);
        self.real.clear();
        self.real.resize(n, 0.0);
        self.scratch.resize(plan.get_scratch_len(), Complex::new(0.0, 0.0));
        // realfft requires the imaginary parts of the DC and Nyquist bins to
        // vanish; rounding leaves them at ~1e-17 for real convolutions.
        spectrum[0].im = 0.0;
        if n % 2 == 0 {
            spectrum[n / 2].im = 0.0;
        }
        plan.process_with_scratch(spectrum, &mut self.real, &mut self.scratch)
            .expect("real FFT sizes are consistent");
        let scale = 1.0 / n as f64;
        out.clear();
        out.extend(self.real[..len].iter().map(|v| {
            let x = v * scale;
            if x < 0.0 { 0.0 } else { x }
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_matches_hand_result_and_fft() {
        let a = [0.5, 0.25, 0.25];
        let b = [0.1, 0.9];
        let mut out = Vec::new();
        convolve_direct(&a, &b, &mut out);
        assert_eq!(out.len(), 4);
        let want = [0.05, 0.475, 0.25, 0.225];
        for (o, w) in out.iter().zip(&want) {
            assert!((o - w).abs() < 1e-15);
        }
        let mut cache = FftCache::default();
        let n = fft_size(4);
        let mut sa = Vec::new();
        let mut sb = Vec::new();
        cache.spectrum(&a, n, &mut sa);
        cache.spectrum(&b, n, &mut sb);
        for (x, y) in sa.iter_mut().zip(&sb) {
            *x *= y;
        }
        let mut back = Vec::new();
        cache.inverse_into(&mut sa, n, 4, &mut back);
        for (o, w) in back.iter().zip(&want) {
            assert!((o - w).abs() < 1e-14);
        }
    }

    #[test]
    fn costs_are_monotone() {
        assert!(direct_cost(&[100, 100]) < direct_cost(&[100, 100, 100]));
        assert!(fft_cost(256) < fft_cost(512));
        assert_eq!(fft_size(300), 512);
        assert_eq!(fft_size(1), 2);
    }
}
