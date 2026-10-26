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

/// `(sum, grid_moment)` in one pass over the weights, each accumulator in
/// its own order, so both agree with the separate functions to the bit.
pub fn mass_and_moment(weights: &[f64], bucket_size: f64) -> (f64, f64) {
    let mut mass = Neumaier::default();
    let mut moment = Neumaier::default();
    for (i, &w) in weights.iter().enumerate() {
        mass.add(w);
        moment.add(w * (i as f64 * bucket_size));
    }
    (mass.value(), moment.value())
}

#[cfg(test)]
mod tests {
    use super::*;

    // [tidy] The combined pass is the two separate ones to the bit.
    #[test]
    fn combined_pass_matches_the_separate_ones() {
        let weights: Vec<f64> = (0..10_000).map(|i| ((i * 7919) % 1000) as f64 * 1e-7 + 1e-16).collect();
        let (mass, moment) = mass_and_moment(&weights, 2.5);
        assert_eq!(mass, sum(&weights));
        assert_eq!(moment, grid_moment(&weights, 2.5));
    }

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
