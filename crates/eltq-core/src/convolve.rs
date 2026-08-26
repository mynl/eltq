//! Convolution across units: direct (exact, small supports) and FFT
//! (rustfft, complex transforms of real rows; real transforms and plan
//! tuning are the 1.11.0 kernel item).
//!
//! Both paths produce the same linear convolution; the engine picks per
//! event and view by a cost estimate and tests hold them to agree within
//! floating-point tolerance.

use std::collections::HashMap;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

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

/// Rough cost of one complex FFT of size n (5 n log2 n flops).
pub fn fft_cost(n: usize) -> f64 {
    5.0 * n as f64 * (n as f64).log2().max(1.0)
}

/// Next power of two at or above `n` (FFT size).
pub fn fft_size(n: usize) -> usize {
    n.next_power_of_two().max(2)
}

/// Per-worker FFT plans, cached by size.
pub struct FftCache {
    planner: FftPlanner<f64>,
    forward: HashMap<usize, Arc<dyn Fft<f64>>>,
    inverse: HashMap<usize, Arc<dyn Fft<f64>>>,
}

impl Default for FftCache {
    fn default() -> Self {
        FftCache {
            planner: FftPlanner::new(),
            forward: HashMap::new(),
            inverse: HashMap::new(),
        }
    }
}

impl FftCache {
    pub fn forward(&mut self, n: usize) -> Arc<dyn Fft<f64>> {
        let planner = &mut self.planner;
        self.forward
            .entry(n)
            .or_insert_with(|| planner.plan_fft_forward(n))
            .clone()
    }

    pub fn inverse(&mut self, n: usize) -> Arc<dyn Fft<f64>> {
        let planner = &mut self.planner;
        self.inverse
            .entry(n)
            .or_insert_with(|| planner.plan_fft_inverse(n))
            .clone()
    }

    /// Forward transform of a real row zero-padded to `n` into `spectrum`.
    pub fn spectrum(&mut self, row: &[f64], n: usize, spectrum: &mut Vec<Complex<f64>>) {
        spectrum.clear();
        spectrum.resize(n, Complex::new(0.0, 0.0));
        for (s, &v) in spectrum.iter_mut().zip(row) {
            s.re = v;
        }
        self.forward(n).process(spectrum);
    }

    /// Inverse transform of `spectrum` (consumed) into the first `len` real
    /// values of `out`, normalized by n.
    pub fn inverse_into(&mut self, spectrum: &mut [Complex<f64>], len: usize, out: &mut Vec<f64>) {
        let n = spectrum.len();
        self.inverse(n).process(spectrum);
        let scale = 1.0 / n as f64;
        out.clear();
        out.extend(spectrum[..len].iter().map(|c| c.re * scale));
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
        cache.inverse_into(&mut sa, 4, &mut back);
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
