//! Multi-unit secondary uncertainty: unit preparation, conditional pmfs
//! given the shared event factor, and the exact two-unit joint pmf.
//!
//! Model (Burke, Rau-Chaplin, Varghese 2016, section III.E; the packaged
//! SciPy reference `eltq.reference_multiunit` is the oracle): per unit the
//! damage ratio is a beta moment-matched to `(mu / M, (sd_indep + sd_corr) /
//! M)`; one shared standard normal `Z_c` per event occurrence and an
//! independent `Z_i` per unit give `U_i = Phi((sd_corr Z_c + sd_indep Z_i) /
//! t_i)`, `t_i = hypot(sd_indep, sd_corr)`, and `X_i = M_i F_i^{-1}(U_i)`.
//! Conditional on `Z_c = z` the units are independent with
//! `P(X_i <= x | z) = Phi((y(x) - w_c z) / w_I)`, `y(x) = Phi^{-1}(F_i(x /
//! M_i))`.
//!
//! Discretization matches the reference: a beta unit's conditional pmf is
//! bucketed by cdf differences at half-bucket edges (the edge scores `y` are
//! computed once per unit, independent of the z count); atoms and comonotone
//! units' per-z point values use the linear mass- and mean-preserving
//! deposit. The joint of two units is `C_1^T diag(w) C_2` with `C_i` the
//! `(Q x n_i)` conditional matrices: one GEMM.

use crate::batched::beta_quantiles;
use crate::beta::beta_params;
use crate::config::VariancePolicy;
use crate::deposit::deposit;
use crate::error::Error;
use crate::incbeta::BetaFn;
use crate::normal;
use crate::quadrature::{ZRule, z_quadrature};
use crate::sum::{Neumaier, sum};

/// Per-unit ELT fields in loss currency units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitSpec {
    pub mean_loss: f64,
    pub sd_indep: f64,
    pub sd_corr: f64,
    pub max_loss: f64,
}

/// How a unit contributes to an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitKind {
    /// Contributes nothing (dropped by the variance policy).
    Drop,
    /// All conditional mass at one loss value.
    Atom,
    /// sd_indep == 0: loss is a deterministic function of Z_c.
    Comonotone,
    /// General case.
    Beta,
}

/// Validated, derived per-unit parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitPlan {
    pub kind: UnitKind,
    pub mean_loss: f64,
    pub max_loss: f64,
    pub alpha: f64,
    pub beta: f64,
    /// sd_corr / t, the shared-factor loading.
    pub w_corr: f64,
    /// sd_indep / t.
    pub w_indep: f64,
    /// Total sd after any clamp, loss units.
    pub sigma_used: f64,
    /// Diagnostic category names, in the reference's vocabulary.
    pub notes: Vec<&'static str>,
}

impl UnitPlan {
    /// Largest loss the unit can produce.
    pub fn support(&self) -> f64 {
        if self.kind == UnitKind::Atom { self.mean_loss } else { self.max_loss }
    }
}

/// Validate one unit and derive beta parameters and copula weights.
/// Structural problems (nonfinite, negative, mean above max) are errors;
/// inadmissible variance follows `policy`.
pub fn prepare_unit(
    spec: &UnitSpec,
    label: &str,
    policy: VariancePolicy,
) -> Result<UnitPlan, Error> {
    let UnitSpec {
        mean_loss: mu,
        sd_indep: s_i,
        sd_corr: s_c,
        max_loss: m,
    } = *spec;
    let invalid = |message: &str| Error::UnitInvalid {
        unit: label.to_string(),
        message: message.to_string(),
    };
    if !(mu.is_finite() && s_i.is_finite() && s_c.is_finite() && m.is_finite()) {
        return Err(invalid("all fields must be finite"));
    }
    if mu < 0.0 || s_i < 0.0 || s_c < 0.0 || m < 0.0 {
        return Err(invalid("negative field"));
    }
    if mu > m {
        return Err(invalid(&format!("mean_loss {mu} exceeds max_loss {m}")));
    }
    let mut sigma = s_i + s_c;
    let mut plan = UnitPlan {
        kind: UnitKind::Atom,
        mean_loss: mu,
        max_loss: m,
        alpha: f64::NAN,
        beta: f64::NAN,
        w_corr: 0.0,
        w_indep: 0.0,
        sigma_used: sigma,
        notes: Vec::new(),
    };
    if m == 0.0 || sigma == 0.0 || mu == 0.0 || mu == m {
        plan.notes.push("degenerate_atom");
        return Ok(plan);
    }
    let v_max = mu * (m - mu);
    if sigma * sigma >= v_max {
        match policy {
            VariancePolicy::Error => {
                return Err(Error::UnitInadmissibleVariance {
                    unit: label.to_string(),
                    sigma2: sigma * sigma,
                    v_max,
                });
            }
            VariancePolicy::Drop => {
                plan.kind = UnitKind::Drop;
                plan.notes.push("variance_dropped");
                return Ok(plan);
            }
            VariancePolicy::Atom => {
                plan.notes.push("variance_atom");
                return Ok(plan);
            }
            VariancePolicy::ClampSigma { factor } => {
                sigma = factor * v_max.sqrt();
                plan.sigma_used = sigma;
                plan.notes.push("sigma_clamped");
            }
        }
    }
    let (alpha, beta) = beta_params(m, mu, sigma * sigma);
    plan.alpha = alpha;
    plan.beta = beta;
    let t = s_i.hypot(s_c);
    plan.w_corr = s_c / t;
    plan.w_indep = s_i / t;
    plan.kind = if s_i == 0.0 { UnitKind::Comonotone } else { UnitKind::Beta };
    Ok(plan)
}

/// Smallest grid length covering `max_loss`: ceil(M / h) + 1.
pub fn grid_length_for(max_loss: f64, bucket_size: f64) -> usize {
    if max_loss > 0.0 {
        (max_loss / bucket_size).ceil() as usize + 1
    } else {
        1
    }
}

/// Normal scores of the n + 1 half-bucket edges of a beta unit,
/// `y_e = Phi^{-1}(I_{edge_e / M}(alpha, beta))`, `edge_e = (e - 1/2) h`.
/// Edges below 0 map to -inf and edges at or above M to +inf.
pub fn edge_scores(plan: &UnitPlan, bucket_size: f64, n: usize) -> Vec<f64> {
    let bf = BetaFn::new(plan.alpha, plan.beta);
    (0..=n)
        .map(|e| {
            let edge = (e as f64 - 0.5) * bucket_size;
            normal::quantile(bf.cdf(edge / plan.max_loss))
        })
        .collect()
}

/// Conditional bucket pmf of one unit for each shared-factor node, as a
/// row-major `(z.len() x n)` matrix; every row sums to one.
pub fn conditional_matrix(
    plan: &UnitPlan,
    bucket_size: f64,
    n: usize,
    z: &[f64],
) -> Result<Vec<f64>, Error> {
    let q = z.len();
    let inv_h = 1.0 / bucket_size;
    let mut out = vec![0.0_f64; q * n];
    match plan.kind {
        UnitKind::Drop => {}
        UnitKind::Atom => {
            let mut row = vec![0.0_f64; n];
            deposit(&mut row, plan.mean_loss, 1.0, inv_h);
            for r in out.chunks_exact_mut(n) {
                r.copy_from_slice(&row);
            }
        }
        UnitKind::Comonotone => {
            // U = Phi(w_corr z), an atom per node; the nodes ascend, so the
            // targets do too and one warm-started walk serves them all.
            let targets: Vec<f64> = z.iter().map(|&zq| normal::cdf(plan.w_corr * zq)).collect();
            let mut quantiles = vec![0.0_f64; q];
            beta_quantiles(plan.alpha, plan.beta, &targets, &mut quantiles).map_err(|_| {
                Error::KernelFailure {
                    row: 0,
                    alpha: plan.alpha,
                    beta: plan.beta,
                }
            })?;
            for (r, &x) in out.chunks_exact_mut(n).zip(&quantiles) {
                deposit(r, (plan.max_loss * x).min(plan.max_loss), 1.0, inv_h);
            }
        }
        UnitKind::Beta => {
            let y = edge_scores(plan, bucket_size, n);
            conditional_rows_from_scores(&y, plan.w_corr, plan.w_indep, z, &mut out);
        }
    }
    Ok(out)
}

/// Fill `out` (row-major `(z.len() x (y.len() - 1))`) with the adjacent
/// differences of `Phi((y_e - w_corr z_q) / w_indep)`.
pub fn conditional_rows_from_scores(y: &[f64], w_corr: f64, w_indep: f64, z: &[f64], out: &mut [f64]) {
    let n = y.len() - 1;
    let inv_w = 1.0 / w_indep;
    for (r, &zq) in out.chunks_exact_mut(n).zip(z) {
        let shift = w_corr * zq;
        let mut prev = 0.0_f64; // Phi(-inf) at edge 0
        for (k, slot) in r.iter_mut().enumerate() {
            let next = normal::cdf((y[k + 1] - shift) * inv_w);
            *slot = next - prev;
            prev = next;
        }
    }
}

/// Configuration of the two-unit joint.
#[derive(Debug, Clone)]
pub struct JointConfig {
    pub bucket_size: f64,
    pub n1: usize,
    pub n2: usize,
    pub z_nodes: usize,
    pub z_rule: ZRule,
    pub variance_policy: VariancePolicy,
}

impl JointConfig {
    pub fn new(bucket_size: f64, n1: usize, n2: usize) -> Self {
        JointConfig {
            bucket_size,
            n1,
            n2,
            z_nodes: 1001,
            z_rule: ZRule::Midpoint,
            variance_policy: VariancePolicy::default(),
        }
    }
}

/// Parameters and moment diagnostics of a two-unit joint pmf (mirrors the
/// Python `JointMeta`).
#[derive(Debug, Clone)]
pub struct JointMeta {
    pub bucket_size: f64,
    pub z_nodes: usize,
    pub z_rule: ZRule,
    pub alpha: [f64; 2],
    pub beta: [f64; 2],
    pub sigma_used: [f64; 2],
    pub copula_normal_correlation: f64,
    pub total_mass: f64,
    pub grid_mean: [f64; 2],
    pub analytic_mean: [f64; 2],
    pub grid_sd: [f64; 2],
    pub grid_correlation: f64,
    pub notes: [Vec<&'static str>; 2],
}

/// Row-major `(n1 x n2)` joint pmf plus metadata.
#[derive(Debug, Clone)]
pub struct JointResult {
    pub pmf: Vec<f64>,
    pub n1: usize,
    pub n2: usize,
    pub meta: JointMeta,
}

/// Exact bucketed bivariate pmf of one event across two units.
pub fn joint_two_units(units: [UnitSpec; 2], config: &JointConfig) -> Result<JointResult, Error> {
    let h = config.bucket_size;
    if !h.is_finite() || h <= 0.0 {
        return Err(Error::BadBucketSize(h));
    }
    if let VariancePolicy::ClampSigma { factor } = config.variance_policy
        && !(factor.is_finite() && factor > 0.0 && factor < 1.0)
    {
        return Err(Error::BadClampFactor(factor));
    }
    if config.variance_policy == VariancePolicy::Drop {
        return Err(Error::BadJointPolicy);
    }
    let (z, w) = z_quadrature(config.z_nodes, config.z_rule)?;
    let lengths = [config.n1, config.n2];
    let mut plans = Vec::with_capacity(2);
    for (i, (spec, &n)) in units.iter().zip(&lengths).enumerate() {
        if n < 1 {
            return Err(Error::BadGridLength(n));
        }
        let plan = prepare_unit(spec, &format!("unit {}", i + 1), config.variance_policy)?;
        let top = (n - 1) as f64 * h;
        if plan.max_loss > top {
            return Err(Error::GridOverflow {
                grid_length: n,
                bucket_size: h,
                top,
                max_loss: plan.max_loss,
                required: grid_length_for(plan.max_loss, h),
                example_rows: vec![i],
            });
        }
        plans.push(plan);
    }
    let (n1, n2) = (config.n1, config.n2);
    let mut c1 = conditional_matrix(&plans[0], h, n1, &z)?;
    let c2 = conditional_matrix(&plans[1], h, n2, &z)?;
    // Fold the weights into C1's rows, then pmf = C1^T C2.
    for (row, &wq) in c1.chunks_exact_mut(n1).zip(&w) {
        for v in row.iter_mut() {
            *v *= wq;
        }
    }
    let q = z.len();
    let mut pmf = vec![0.0_f64; n1 * n2];
    // SAFETY: shapes are (q x n1)^T . (q x n2) -> (n1 x n2), all buffers
    // sized exactly and non-overlapping; strides describe row-major
    // storage with C1 read transposed.
    unsafe {
        matrixmultiply::dgemm(
            n1,
            q,
            n2,
            1.0,
            c1.as_ptr(),
            1,
            n1 as isize,
            c2.as_ptr(),
            n2 as isize,
            1,
            0.0,
            pmf.as_mut_ptr(),
            n2 as isize,
            1,
        );
    }

    // Moments of the returned pmf.
    let mut marg1 = vec![0.0_f64; n1];
    let mut marg2 = vec![0.0_f64; n2];
    for (i, row) in pmf.chunks_exact(n2).enumerate() {
        for (j, &p) in row.iter().enumerate() {
            marg1[i] += p;
            marg2[j] += p;
        }
    }
    let moment = |m: &[f64], power: i32| -> f64 {
        let mut acc = Neumaier::default();
        for (i, &p) in m.iter().enumerate() {
            acc.add(p * (i as f64 * h).powi(power));
        }
        acc.value()
    };
    let mean = [moment(&marg1, 1), moment(&marg2, 1)];
    let sd = [
        (moment(&marg1, 2) - mean[0] * mean[0]).max(0.0).sqrt(),
        (moment(&marg2, 2) - mean[1] * mean[1]).max(0.0).sqrt(),
    ];
    let mut cross = Neumaier::default();
    for (i, row) in pmf.chunks_exact(n2).enumerate() {
        let x1 = i as f64 * h;
        for (j, &p) in row.iter().enumerate() {
            cross.add(p * x1 * (j as f64 * h));
        }
    }
    let correlation = if sd[0] > 0.0 && sd[1] > 0.0 {
        (cross.value() - mean[0] * mean[1]) / (sd[0] * sd[1])
    } else {
        f64::NAN
    };
    let meta = JointMeta {
        bucket_size: h,
        z_nodes: config.z_nodes,
        z_rule: config.z_rule,
        alpha: [plans[0].alpha, plans[1].alpha],
        beta: [plans[0].beta, plans[1].beta],
        sigma_used: [plans[0].sigma_used, plans[1].sigma_used],
        copula_normal_correlation: plans[0].w_corr * plans[1].w_corr,
        total_mass: sum(&pmf),
        grid_mean: mean,
        analytic_mean: [plans[0].mean_loss, plans[1].mean_loss],
        grid_sd: sd,
        grid_correlation: correlation,
        notes: [plans[0].notes.clone(), plans[1].notes.clone()],
    };
    Ok(JointResult { pmf, n1, n2, meta })
}

#[cfg(test)]
mod tests {
    use super::*;

    const U1: UnitSpec = UnitSpec {
        mean_loss: 200.0,
        sd_indep: 100.0,
        sd_corr: 150.0,
        max_loss: 1000.0,
    };
    const U2: UnitSpec = UnitSpec {
        mean_loss: 500.0,
        sd_indep: 300.0,
        sd_corr: 200.0,
        max_loss: 2000.0,
    };

    #[test]
    fn prepare_unit_kinds_and_policies() {
        let plan = prepare_unit(&U1, "u", VariancePolicy::default()).unwrap();
        assert_eq!(plan.kind, UnitKind::Beta);
        assert!((plan.sigma_used - 250.0).abs() < 1e-12);
        assert!((plan.w_corr.hypot(plan.w_indep) - 1.0).abs() < 1e-15);
        let como = prepare_unit(
            &UnitSpec {
                sd_indep: 0.0,
                ..U1
            },
            "u",
            VariancePolicy::default(),
        )
        .unwrap();
        assert_eq!(como.kind, UnitKind::Comonotone);
        assert_eq!(como.w_corr, 1.0);
        let atom = prepare_unit(
            &UnitSpec {
                sd_indep: 0.0,
                sd_corr: 0.0,
                ..U1
            },
            "u",
            VariancePolicy::default(),
        )
        .unwrap();
        assert_eq!(atom.kind, UnitKind::Atom);
        assert_eq!(atom.notes, vec!["degenerate_atom"]);
        let bad = UnitSpec {
            mean_loss: 30.0,
            sd_indep: 50.0,
            sd_corr: 30.0,
            max_loss: 100.0,
        };
        assert!(matches!(
            prepare_unit(&bad, "u", VariancePolicy::Error),
            Err(Error::UnitInadmissibleVariance { .. })
        ));
        let clamped = prepare_unit(&bad, "u", VariancePolicy::default()).unwrap();
        assert_eq!(clamped.notes, vec!["sigma_clamped"]);
        assert!(clamped.sigma_used < (30.0_f64 * 70.0).sqrt());
        assert_eq!(prepare_unit(&bad, "u", VariancePolicy::Atom).unwrap().kind, UnitKind::Atom);
        assert_eq!(prepare_unit(&bad, "u", VariancePolicy::Drop).unwrap().kind, UnitKind::Drop);
        assert!(matches!(
            prepare_unit(
                &UnitSpec {
                    mean_loss: f64::NAN,
                    ..U1
                },
                "u",
                VariancePolicy::default()
            ),
            Err(Error::UnitInvalid { .. })
        ));
    }

    #[test]
    fn conditional_rows_are_distributions() {
        let (z, _) = z_quadrature(16, ZRule::Midpoint).unwrap();
        for spec in [
            U1,
            UnitSpec {
                sd_indep: 0.0,
                ..U1
            },
            UnitSpec {
                sd_indep: 0.0,
                sd_corr: 0.0,
                ..U1
            },
        ] {
            let plan = prepare_unit(&spec, "u", VariancePolicy::default()).unwrap();
            let n = grid_length_for(plan.support(), 10.0);
            let c = conditional_matrix(&plan, 10.0, n, &z).unwrap();
            for row in c.chunks_exact(n) {
                let mass: f64 = row.iter().sum();
                assert!((mass - 1.0).abs() < 1e-12, "{spec:?}: mass {mass}");
                assert!(row.iter().all(|&p| p >= -1e-16));
            }
        }
    }

    #[test]
    fn joint_matches_naive_product_and_is_outer_for_independent_units() {
        let mut config = JointConfig::new(10.0, 101, 201);
        config.z_nodes = 8;
        let result = joint_two_units([U1, U2], &config).unwrap();
        assert!((result.meta.total_mass - 1.0).abs() < 1e-12);
        // Naive triple loop reproduces the GEMM.
        let (z, w) = z_quadrature(8, ZRule::Midpoint).unwrap();
        let p1 = prepare_unit(&U1, "u", VariancePolicy::default()).unwrap();
        let p2 = prepare_unit(&U2, "u", VariancePolicy::default()).unwrap();
        let c1 = conditional_matrix(&p1, 10.0, 101, &z).unwrap();
        let c2 = conditional_matrix(&p2, 10.0, 201, &z).unwrap();
        for i in 0..101 {
            for j in 0..201 {
                let naive: f64 = (0..8).map(|q| w[q] * c1[q * 101 + i] * c2[q * 201 + j]).sum();
                assert!((result.pmf[i * 201 + j] - naive).abs() < 1e-15);
            }
        }
        let independent = joint_two_units(
            [
                UnitSpec {
                    sd_corr: 0.0,
                    ..U1
                },
                UnitSpec {
                    sd_corr: 0.0,
                    ..U2
                },
            ],
            &config,
        )
        .unwrap();
        let mut m1 = vec![0.0; 101];
        let mut m2 = vec![0.0; 201];
        for i in 0..101 {
            for j in 0..201 {
                m1[i] += independent.pmf[i * 201 + j];
                m2[j] += independent.pmf[i * 201 + j];
            }
        }
        for i in 0..101 {
            for j in 0..201 {
                assert!((independent.pmf[i * 201 + j] - m1[i] * m2[j]).abs() < 1e-15);
            }
        }
        assert_eq!(independent.meta.copula_normal_correlation, 0.0);
    }

    #[test]
    fn joint_errors() {
        let config = JointConfig::new(10.0, 50, 201);
        assert!(matches!(
            joint_two_units([U1, U2], &config),
            Err(Error::GridOverflow { required: 101, .. })
        ));
        let mut config = JointConfig::new(10.0, 101, 201);
        config.z_nodes = 0;
        assert!(matches!(joint_two_units([U1, U2], &config), Err(Error::BadZNodes(0))));
        config.z_nodes = 4;
        config.variance_policy = VariancePolicy::Drop;
        assert!(matches!(joint_two_units([U1, U2], &config), Err(Error::BadJointPolicy)));
    }
}
