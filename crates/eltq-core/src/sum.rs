//! Neumaier compensated summation for validation totals.
//!
//! Used for reference totals (total rate, AAL, quadrature mean) so that
//! summation-order noise stays far below the stated tolerances even over a
//! million rows. Histogram deposits themselves use plain accumulation, as the
//! specification permits.

/// Neumaier (improved Kahan) compensated accumulator.
#[derive(Debug, Default, Clone, Copy)]
pub struct Neumaier {
    sum: f64,
    compensation: f64,
}

impl Neumaier {
    pub fn add(&mut self, value: f64) {
        let t = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.compensation += (self.sum - t) + value;
        } else {
            self.compensation += (value - t) + self.sum;
        }
        self.sum = t;
    }

    pub fn value(&self) -> f64 {
        self.sum + self.compensation
    }
}

/// Compensated sum of a slice.
pub fn sum(values: &[f64]) -> f64 {
    let mut acc = Neumaier::default();
    for &v in values {
        acc.add(v);
    }
    acc.value()
}

/// Compensated dot product of weights against the grid i * h.
pub fn grid_moment(weights: &[f64], bucket_size: f64) -> f64 {
    let mut acc = Neumaier::default();
    for (i, &w) in weights.iter().enumerate() {
        acc.add(w * (i as f64 * bucket_size));
    }
    acc.value()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensated_sum_beats_naive() {
        // 1 + 1e-16 repeated: naive summation loses the small terms.
        let mut values = vec![1.0];
        values.extend(std::iter::repeat_n(1e-16, 1_000_000));
        let exact = 1.0 + 1e-10;
        assert!((sum(&values) - exact).abs() < 1e-15);
    }

    #[test]
    fn grid_moment_matches_manual() {
        let w = [0.5, 0.25, 0.25];
        assert_eq!(grid_moment(&w, 2.0), 0.25 * 2.0 + 0.25 * 4.0);
    }
}
