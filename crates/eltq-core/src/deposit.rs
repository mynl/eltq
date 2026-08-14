//! Linear mass- and mean-preserving grid deposit.

/// Deposit weight `w` at loss `x` onto the even grid with node spacing
/// `1 / inv_h`, splitting linearly between the two adjacent nodes.
///
/// The caller guarantees grid coverage (`x <= (n - 1) h`), which makes the
/// top-node branch reachable only for x at (or within rounding of) the top
/// node; out-of-bounds indexing is impossible for any nonnegative finite x.
#[inline]
pub fn deposit(hist: &mut [f64], x: f64, w: f64, inv_h: f64) {
    let r = x * inv_h;
    let i = r as usize;
    let n = hist.len();
    if i >= n - 1 {
        hist[n - 1] += w;
    } else {
        let f = r - i as f64;
        hist[i] += w * (1.0 - f);
        hist[i + 1] += w * f;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_node_gets_full_weight() {
        let mut hist = vec![0.0; 5];
        deposit(&mut hist, 0.75, 0.4, 4.0); // h = 0.25, node 3
        assert_eq!(hist, vec![0.0, 0.0, 0.0, 0.4, 0.0]);
    }

    #[test]
    fn off_grid_preserves_mass_and_moment() {
        let mut hist = vec![0.0; 12];
        let (x, w, h) = (3.3567, 0.037, 1.0);
        deposit(&mut hist, x, w, 1.0 / h);
        let mass: f64 = hist.iter().sum();
        let moment: f64 = hist
            .iter()
            .enumerate()
            .map(|(i, &v)| v * i as f64 * h)
            .sum();
        assert!((mass - w).abs() <= 1e-16);
        assert!((moment - w * x).abs() <= 1e-15);
        assert!(hist[3] > 0.0 && hist[4] > 0.0);
    }

    #[test]
    fn top_endpoint_is_explicit() {
        let mut hist = vec![0.0; 5];
        deposit(&mut hist, 1.0, 0.2, 4.0); // top node of a 5-node h = 0.25 grid
        assert_eq!(hist[4], 0.2);
    }

    #[test]
    fn zero_loss_lands_on_node_zero() {
        let mut hist = vec![0.0; 3];
        deposit(&mut hist, 0.0, 0.7, 2.0);
        assert_eq!(hist, vec![0.7, 0.0, 0.0]);
    }

    #[test]
    fn single_node_grid_takes_everything_at_zero() {
        let mut hist = vec![0.0; 1];
        deposit(&mut hist, 0.0, 0.3, 1.0);
        assert_eq!(hist[0], 0.3);
    }
}
