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
use crate::diagnostics::{Category, Diagnostics};
use crate::error::Error;
use crate::incbeta::BetaFn;
use crate::law::{BetaLaw, NarrowLaw};
use crate::normal;
use crate::quadrature::{FactorRule, ZRule};
use crate::sum::{Neumaier, sum};
use crate::terms::{Clause, UnitTerms, View};
use crate::validate::ATOM_RESOLUTION;

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
    /// Inadmissible variance under the Bernoulli policy: the two-point law
    /// on 0 and M with `P(X = M) = mu / M` (spec section 4A). Given `Z_c =
    /// z` the unit loses M with probability `1 - Phi((Phi^{-1}(1 - p) -
    /// w_corr z) / w_indep)`, a step at `z = Phi^{-1}(1 - p)` when `w_corr
    /// = 1`.
    TwoPoint,
}

/// The coupling rule (spec section 4A): the loading every unit of an event
/// takes on the common factor, or Homer and Li's roll-up in its place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coupling {
    /// `w_i = sd_corr / hypot(sd_indep, sd_corr)` (Burke 2016, step iv).
    Burke,
    /// Every loading one: one uniform per occurrence (Mitchell-Wallace et
    /// al. 2017, section 1.10.5.9); computed exactly, with no factor rule.
    Comonotone,
    /// Every loading zero.
    Independent,
    /// Homer and Li 2017, section 3.2.2: the event's rows rolled into one,
    /// its loss allocated pro rata to the means, uncapped as published.
    Homer,
    /// Burke's loadings scaled by a common factor per event so that the
    /// event total's variance equals the roll-up convention's `(sd_I,R +
    /// sd_C,R)^2`; saturated at the largest loading one when the target is
    /// out of reach (the multi-unit article; no published source).
    Matched,
}

impl Coupling {
    pub fn parse(name: &str) -> Result<Self, Error> {
        match name {
            "burke" => Ok(Coupling::Burke),
            "comonotone" => Ok(Coupling::Comonotone),
            "independent" => Ok(Coupling::Independent),
            "homer" => Ok(Coupling::Homer),
            "matched" => Ok(Coupling::Matched),
            other => Err(Error::BadCoupling(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Coupling::Burke => "burke",
            Coupling::Comonotone => "comonotone",
            Coupling::Independent => "independent",
            Coupling::Homer => "homer",
            Coupling::Matched => "matched",
        }
    }
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

    /// `P(X = M) = mu / M` of a two-point unit.
    pub fn two_point_p(&self) -> f64 {
        self.mean_loss / self.max_loss
    }

    /// Whether the unit's conditional law varies with the shared factor.
    pub fn depends_on_factor(&self) -> bool {
        match self.kind {
            UnitKind::Comonotone => true,
            UnitKind::Beta | UnitKind::TwoPoint => self.w_corr > 0.0,
            UnitKind::Atom | UnitKind::Drop => false,
        }
    }

    /// The plan under a coupling: Burke's loadings as prepared, or every
    /// loading forced to one (`Comonotone`) or zero (`Independent`). A beta
    /// unit at loading one takes the `Comonotone` kind and at loading zero
    /// the `Beta` kind with no factor dependence; atoms, dropped rows and
    /// the `Homer` coupling, which rolls the event up instead, leave the
    /// plan alone.
    pub fn couple(mut self, coupling: Coupling) -> UnitPlan {
        match coupling {
            Coupling::Comonotone => match self.kind {
                UnitKind::Beta | UnitKind::TwoPoint => {
                    if self.kind == UnitKind::Beta {
                        self.kind = UnitKind::Comonotone;
                    }
                    self.w_corr = 1.0;
                    self.w_indep = 0.0;
                }
                _ => {}
            },
            Coupling::Independent => match self.kind {
                UnitKind::Beta | UnitKind::Comonotone | UnitKind::TwoPoint => {
                    if self.kind == UnitKind::Comonotone {
                        self.kind = UnitKind::Beta;
                    }
                    self.w_corr = 0.0;
                    self.w_indep = 1.0;
                }
                _ => {}
            },
            // Matched: the scale is found per event in the engine
            // (`matched_scale`) and applied through `with_loading`.
            Coupling::Burke | Coupling::Homer | Coupling::Matched => {}
        }
        self
    }

    /// The plan at loading `w` on the common factor (`w_indep = sqrt(1 -
    /// w^2)`), the kind following the loading for beta units.
    pub fn with_loading(mut self, w: f64) -> UnitPlan {
        if matches!(self.kind, UnitKind::Beta | UnitKind::Comonotone | UnitKind::TwoPoint) {
            let w = w.clamp(0.0, 1.0);
            self.w_corr = w;
            self.w_indep = ((1.0 - w) * (1.0 + w)).sqrt();
            if self.kind != UnitKind::TwoPoint {
                self.kind = if w >= 1.0 { UnitKind::Comonotone } else { UnitKind::Beta };
            }
        }
        self
    }
}

/// The roll-up convention's standard deviation of the event total,
/// `sqrt(sum sd_indep^2) + sum sd_corr` (Homer and Li 2017; Mitchell-Wallace
/// et al. 2017), the target of the `matched` coupling.
pub fn roll_up_sd(specs: &[UnitSpec]) -> f64 {
    let r = rolled_spec(specs);
    r.sd_indep + r.sd_corr
}

/// Variance of the event total of the units' bucketed conditional laws under
/// the rule, from the conditional means and variances alone: `sum_s w_s
/// (sum_i Var(X_i | s) + (sum_i E[X_i | s])^2) - (sum_i E[X_i])^2`, the units
/// independent given the node or stratum.
pub fn total_variance(plans: &[UnitPlan], bucket_size: f64, rule: &FactorRule) -> Result<f64, Error> {
    let q = rule.len();
    let mut cond_var = vec![0.0_f64; q];
    let mut cond_mean = vec![0.0_f64; q];
    let mut mean = 0.0;
    for plan in plans {
        let n = grid_length_for(plan.support(), bucket_size);
        let unit_rule = if plan.depends_on_factor() { rule } else { &FactorRule::single() };
        let matrix = conditional_matrix(plan, bucket_size, n, unit_rule)?;
        for s in 0..q {
            let row = if unit_rule.len() == 1 { &matrix[..n] } else { &matrix[s * n..(s + 1) * n] };
            let m1: f64 = row.iter().enumerate().map(|(k, &p)| p * k as f64 * bucket_size).sum();
            let m2: f64 = row.iter().enumerate().map(|(k, &p)| p * (k as f64 * bucket_size).powi(2)).sum();
            cond_mean[s] += m1;
            cond_var[s] += (m2 - m1 * m1).max(0.0);
            mean += rule.weights[s] * m1;
        }
    }
    let second: f64 = (0..q).map(|s| rule.weights[s] * (cond_var[s] + cond_mean[s] * cond_mean[s])).sum();
    Ok((second - mean * mean).max(0.0))
}

/// The common scale of the `matched` coupling for one event.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchedScale {
    /// The factor `c` on Burke's loadings, `w_i = min(c w_i^B, 1)`.
    pub scale: f64,
    pub sd_target: f64,
    pub sd_achieved: f64,
    /// The target was out of reach: `scale` saturates the largest loading at
    /// one (or every loading is zero), and `sd_achieved` falls short.
    pub saturated: bool,
}

/// Bracket `c` on `[0, 1 / max_i w_i]` for `Var(sum X_i) = sd_target^2`, the
/// variance from `total_variance` increasing continuously in `c`; bisection
/// to `1e-7` relative in the variance. Returns the saturated end when the
/// target is not reached before the largest loading is one, or `c = 0` when
/// the independent variance already exceeds it.
pub fn matched_scale(plans: &[UnitPlan], bucket_size: f64, rule: &FactorRule, sd_target: f64) -> Result<MatchedScale, Error> {
    let w_max = plans.iter().map(|p| p.w_corr).fold(0.0_f64, f64::max);
    let target = sd_target * sd_target;
    let at = |c: f64| -> Result<f64, Error> {
        let scaled: Vec<UnitPlan> = plans.iter().map(|p| p.clone().with_loading(c * p.w_corr)).collect();
        total_variance(&scaled, bucket_size, rule)
    };
    let (mut lo, mut hi) = (0.0_f64, if w_max > 0.0 { 1.0 / w_max } else { 0.0 });
    let var_lo = at(lo)?;
    if var_lo >= target {
        return Ok(MatchedScale { scale: 0.0, sd_target, sd_achieved: var_lo.sqrt(), saturated: var_lo > target });
    }
    let var_hi = at(hi)?;
    if var_hi < target {
        return Ok(MatchedScale { scale: hi, sd_target, sd_achieved: var_hi.sqrt(), saturated: true });
    }
    let mut achieved = var_hi;
    let mut c = hi;
    for _ in 0..60 {
        c = 0.5 * (lo + hi);
        achieved = at(c)?;
        if (achieved - target).abs() <= 1e-7 * target {
            break;
        }
        if achieved < target {
            lo = c;
        } else {
            hi = c;
        }
    }
    Ok(MatchedScale { scale: c, sd_target, sd_achieved: achieved.sqrt(), saturated: false })
}

/// The comonotone sum of bucketed marginals on one grid. Every unit is a
/// nondecreasing function of one uniform `U`, so between two consecutive
/// breakpoints of the merged cumulative probabilities each unit sits in one
/// bucket, and the interval's length is the probability of that
/// combination; `deposit(indices, mass)` receives each combination once.
/// Each row's last cumulative value is taken as exactly one, so the masses
/// sum to one whatever the rows' rounding.
pub fn comonotone_merge(rows: &[&[f64]], mut deposit: impl FnMut(&[usize], f64)) {
    let k = rows.len();
    let mut idx = vec![0usize; k];
    let mut cum: Vec<f64> = rows.iter().map(|r| if r.len() == 1 { 1.0 } else { r[0] }).collect();
    let mut u = 0.0_f64;
    loop {
        let next = cum.iter().cloned().fold(1.0_f64, f64::min);
        deposit(&idx, (next - u).max(0.0));
        if next >= 1.0 {
            return;
        }
        u = next;
        for j in 0..k {
            while cum[j] <= u && idx[j] + 1 < rows[j].len() {
                idx[j] += 1;
                cum[j] = if idx[j] + 1 == rows[j].len() { 1.0 } else { cum[j] + rows[j][idx[j]] };
            }
        }
    }
}

/// The comonotone sum's bucket pmf: the indices add, since the rows share
/// the grid; `out` is resized to the sum of the rows' spans plus one.
pub fn comonotone_sum(rows: &[&[f64]], out: &mut Vec<f64>) {
    let total = rows.iter().map(|r| r.len().saturating_sub(1)).sum::<usize>() + 1;
    out.clear();
    out.resize(total, 0.0);
    comonotone_merge(rows, |idx, mass| out[idx.iter().sum::<usize>()] += mass);
}

/// Homer and Li 2017, section 3.2.2 (the same rule as Mitchell-Wallace et
/// al. 2017, section 1.10.5.10): the event's rows rolled into one, means and
/// exposures and correlated sds added, independent sds in quadrature.
pub fn rolled_spec(specs: &[UnitSpec]) -> UnitSpec {
    UnitSpec {
        mean_loss: specs.iter().map(|s| s.mean_loss).sum(),
        sd_indep: specs.iter().map(|s| s.sd_indep * s.sd_indep).sum::<f64>().sqrt(),
        sd_corr: specs.iter().map(|s| s.sd_corr).sum(),
        max_loss: specs.iter().map(|s| s.max_loss).sum(),
    }
}

/// Unit `k`'s terms expressed on the rolled loss: with the allocation
/// `X_k = s X_R`, `s = mu_k / mu_R`, `term_k(s x) = s term_k'(x)` where every
/// monetary breakpoint of `term_k'` is the original divided by `s`.
pub fn terms_on_rolled(terms: &UnitTerms, share: f64) -> UnitTerms {
    UnitTerms {
        unit_id: terms.unit_id.clone(),
        limit: terms.limit / share,
        ded: terms.ded / share,
        clauses: terms
            .clauses
            .iter()
            .map(|c| Clause {
                share: c.share,
                limit: c.limit / share,
                attach: c.attach / share,
            })
            .collect(),
    }
}

/// The value of a view of the event total under the `homer` coupling at
/// each node of the rolled grid: `sum_k term_k(s_k x)` with `s_k` the
/// units' shares of the rolled mean (zero-mean units receive nothing).
pub fn homer_view_values(unit_terms: &[&UnitTerms], shares: &[f64], view: View, bucket_size: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64 * bucket_size;
            unit_terms
                .iter()
                .zip(shares)
                .filter(|(_, s)| **s > 0.0)
                .map(|(t, &s)| t.view(view, s * x))
                .sum()
        })
        .collect()
}

/// The expected allocation over the exposures under the `homer` coupling,
/// `sum_k P((s_k X_R - M_k)^+)`, from the rolled row's limited expected
/// values: the one figure that shows how far the published rule allocates
/// beyond a unit's exposed value (the multi-unit article, section on the
/// rules).
pub fn homer_excess(rolled: &LayerMeans<'_>, shares: &[f64], max_losses: &[f64]) -> f64 {
    shares
        .iter()
        .zip(max_losses)
        .filter(|(s, _)| **s > 0.0)
        .map(|(&s, &m)| s * rolled.layer_mean(m / s, f64::INFINITY))
        .sum()
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
    // Narrower than the standardized coordinate resolves: an atom at mu,
    // as `validate::classify_row` makes it (spec sections 4 and 4A).
    if sigma < mu * ATOM_RESOLUTION {
        plan.notes.push("atom_by_resolution");
        return Ok(plan);
    }
    // In ratios, as `validate::classify_row` (peer review, finding I3).
    let bound = (m - mu) / mu;
    if (sigma / mu) * (sigma / mu) >= bound {
        match policy {
            VariancePolicy::Error => {
                return Err(Error::UnitInadmissibleVariance {
                    unit: label.to_string(),
                    sigma2: sigma * sigma,
                    v_max: mu * (m - mu),
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
                sigma = factor * mu * bound.sqrt();
                plan.sigma_used = sigma;
                plan.notes.push("sigma_clamped");
            }
            VariancePolicy::Bernoulli => {
                // The variance maximizer on [0, M]; its sd in ratios.
                plan.sigma_used = mu * bound.sqrt();
                plan.notes.push("variance_bernoulli");
                plan.kind = UnitKind::TwoPoint;
                set_loadings(&mut plan, s_i, s_c);
                return Ok(plan);
            }
        }
    }
    let (alpha, beta) = beta_params(m, mu, sigma);
    plan.alpha = alpha;
    plan.beta = beta;
    set_loadings(&mut plan, s_i, s_c);
    plan.kind = if s_i == 0.0 { UnitKind::Comonotone } else { UnitKind::Beta };
    Ok(plan)
}

/// The copula loadings `w_corr = sd_corr / t`, `w_indep = sd_indep / t`,
/// `t = hypot(sd_indep, sd_corr)` (Burke 2016, step iv); `t > 0` here since
/// a unit with both sds zero is an atom.
fn set_loadings(plan: &mut UnitPlan, s_i: f64, s_c: f64) {
    let t = s_i.hypot(s_c);
    plan.w_corr = s_c / t;
    plan.w_indep = s_i / t;
}

/// Row classification for the multi-unit ELT route: the outcome of
/// [`classify_unit_row`], enough to rebuild a [`UnitPlan`] in the hot pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowClass {
    pub kind: UnitKind,
    /// Effective mean.
    pub mu: f64,
    /// Effective total sd after any clamp.
    pub sigma: f64,
}

const DROPPED: RowClass = RowClass {
    kind: UnitKind::Drop,
    mu: 0.0,
    sigma: 0.0,
};

/// Classify one unit row, recording diagnostics under the one-unit
/// categories (structural problems drop the row instead of erroring, as in
/// `quantize`); only the strict variance policy can fail.
pub fn classify_unit_row(
    row: usize,
    spec: &UnitSpec,
    policy: VariancePolicy,
    diagnostics: &mut Diagnostics,
) -> Result<RowClass, Error> {
    let UnitSpec {
        mean_loss: mu,
        sd_indep: s_i,
        sd_corr: s_c,
        max_loss: m,
    } = *spec;
    if !(mu.is_finite() && s_i.is_finite() && s_c.is_finite() && m.is_finite()) {
        diagnostics.note(Category::Nonfinite, row);
        return Ok(DROPPED);
    }
    if mu < 0.0 || s_i < 0.0 || s_c < 0.0 || m < 0.0 {
        diagnostics.note(Category::NegativeField, row);
        return Ok(DROPPED);
    }
    if mu > m {
        diagnostics.note(Category::MeanExceedsMax, row);
        return Ok(DROPPED);
    }
    let sigma = s_i + s_c;
    if m == 0.0 || sigma == 0.0 || mu == 0.0 || mu == m {
        diagnostics.note(Category::DegenerateAtom, row);
        return Ok(RowClass {
            kind: UnitKind::Atom,
            mu,
            sigma,
        });
    }
    // Each unit row is classified exactly as a one-row ELT row (spec section
    // 4A), so a standard deviation below 2^-48 of the mean is an atom at mu
    // here too, under its own category (before 2.10.1 it was a beta, and one
    // that narrow is beyond every kernel).
    if sigma < mu * ATOM_RESOLUTION {
        diagnostics.note(Category::AtomByResolution, row);
        return Ok(RowClass {
            kind: UnitKind::Atom,
            mu,
            sigma,
        });
    }
    // In ratios, as `validate::classify_row` (peer review, finding I3).
    let bound = (m - mu) / mu;
    let nondegenerate = if s_i == 0.0 { UnitKind::Comonotone } else { UnitKind::Beta };
    if (sigma / mu) * (sigma / mu) >= bound {
        return match policy {
            VariancePolicy::Error => Err(Error::InadmissibleVariance {
                row,
                sigma2: sigma * sigma,
                v_max: mu * (m - mu),
            }),
            VariancePolicy::Drop => {
                diagnostics.note(Category::VarianceDropped, row);
                Ok(DROPPED)
            }
            VariancePolicy::Atom => {
                diagnostics.note(Category::VarianceAtom, row);
                Ok(RowClass {
                    kind: UnitKind::Atom,
                    mu,
                    sigma,
                })
            }
            VariancePolicy::ClampSigma { factor } => {
                diagnostics.note(Category::SigmaClamped, row);
                Ok(RowClass {
                    kind: nondegenerate,
                    mu,
                    sigma: factor * mu * bound.sqrt(),
                })
            }
            VariancePolicy::Bernoulli => {
                diagnostics.note(Category::VarianceBernoulli, row);
                Ok(RowClass {
                    kind: UnitKind::TwoPoint,
                    mu,
                    sigma: mu * bound.sqrt(),
                })
            }
        };
    }
    Ok(RowClass {
        kind: nondegenerate,
        mu,
        sigma,
    })
}

impl UnitPlan {
    /// Rebuild the plan of a classified row (beta parameters and copula
    /// weights are cheap; only the classification is stored between passes).
    pub fn from_row(class: &RowClass, spec: &UnitSpec) -> UnitPlan {
        let mut plan = UnitPlan {
            kind: class.kind,
            mean_loss: class.mu,
            max_loss: spec.max_loss,
            alpha: f64::NAN,
            beta: f64::NAN,
            w_corr: 0.0,
            w_indep: 0.0,
            sigma_used: class.sigma,
            notes: Vec::new(),
        };
        if matches!(class.kind, UnitKind::Beta | UnitKind::Comonotone) {
            let (alpha, beta) = beta_params(spec.max_loss, class.mu, class.sigma);
            plan.alpha = alpha;
            plan.beta = beta;
        }
        if matches!(class.kind, UnitKind::Beta | UnitKind::Comonotone | UnitKind::TwoPoint) {
            set_loadings(&mut plan, spec.sd_indep, spec.sd_corr);
        }
        plan
    }
}

/// The law of a beta or comonotone unit's loss share at its parameters: the
/// incomplete beta, or the narrow kernel from `LARGE_PARAMETER_FROM` (spec
/// section 5A), whose continued fraction would return NaN there; a law that
/// cannot be evaluated is the typed kernel error, never a NaN.
pub fn unit_law(plan: &UnitPlan) -> Result<BetaLaw, Error> {
    BetaLaw::new(plan.alpha, plan.beta).ok_or(Error::KernelFailure {
        row: 0,
        alpha: plan.alpha,
        beta: plan.beta,
    })
}

/// Unconditional bucket pmf of one unit: beta cdf differences at
/// half-bucket edges, the linear deposit of an atom, or the two atoms of a
/// two-point unit.
pub fn marginal_pmf(plan: &UnitPlan, bucket_size: f64, n: usize) -> Result<Vec<f64>, Error> {
    let mut row = vec![0.0_f64; n];
    if plan.kind == UnitKind::Atom {
        deposit(&mut row, plan.mean_loss, 1.0, 1.0 / bucket_size);
        return Ok(row);
    }
    if plan.kind == UnitKind::TwoPoint {
        let p = plan.two_point_p();
        deposit(&mut row, 0.0, 1.0 - p, 1.0 / bucket_size);
        deposit(&mut row, plan.max_loss, p, 1.0 / bucket_size);
        return Ok(row);
    }
    if plan.kind == UnitKind::Drop {
        return Ok(row);
    }
    let law = unit_law(plan)?;
    let mut prev = 0.0;
    for (k, slot) in row.iter_mut().enumerate() {
        let edge = (k as f64 + 0.5) * bucket_size;
        let next = law.cdf(edge / plan.max_loss);
        *slot = next - prev;
        prev = next;
    }
    Ok(row)
}

/// Analytic layer means of one unit's marginal (the LEV identity
/// `E[min(Y, y)] = y (1 - I_y(a, b)) + a / (a + b) I_y(a + 1, b)`), with the
/// two incomplete-beta bundles built once per unit.
pub struct LayerMeans<'a> {
    plan: &'a UnitPlan,
    law: Option<LayerLaw>,
}

/// The limited mean's kernel: the two incomplete betas, or the narrow
/// kernel's moment from `LARGE_PARAMETER_FROM`.
enum LayerLaw {
    Regular(BetaFn, BetaFn),
    Narrow(NarrowLaw),
}

impl<'a> LayerMeans<'a> {
    pub fn new(plan: &'a UnitPlan) -> Result<Self, Error> {
        let law = if matches!(plan.kind, UnitKind::Beta | UnitKind::Comonotone) {
            Some(match unit_law(plan)? {
                BetaLaw::Regular(bf) => LayerLaw::Regular(bf, BetaFn::new(plan.alpha + 1.0, plan.beta)),
                BetaLaw::Narrow(law) => LayerLaw::Narrow(law),
            })
        } else {
            None
        };
        Ok(LayerMeans { plan, law })
    }

    fn limited_mean(&self, y: f64) -> f64 {
        let mean = self.plan.alpha / (self.plan.alpha + self.plan.beta);
        if y <= 0.0 {
            return 0.0;
        }
        if y >= 1.0 {
            return mean;
        }
        match self.law.as_ref().expect("beta unit") {
            LayerLaw::Regular(bf, bf_plus) => y * (1.0 - bf.cdf(y)) + mean * bf_plus.cdf(y),
            LayerLaw::Narrow(law) => law.limited_mean(y),
        }
    }

    /// `E[min(max(X - attach, 0), limit)]`.
    pub fn layer_mean(&self, attach: f64, limit: f64) -> f64 {
        if limit <= 0.0 {
            return 0.0;
        }
        match self.plan.kind {
            UnitKind::Atom => (self.plan.mean_loss - attach).max(0.0).min(limit),
            // The loss is M with probability p and 0 otherwise.
            UnitKind::TwoPoint => self.plan.two_point_p() * (self.plan.max_loss - attach).max(0.0).min(limit),
            UnitKind::Drop => 0.0,
            _ => {
                let m = self.plan.max_loss;
                m * (self.limited_mean((attach + limit) / m) - self.limited_mean(attach / m))
            }
        }
    }

    /// Analytic mean of a view of the unit's loss.
    pub fn view_mean(&self, terms: &UnitTerms, view: View) -> f64 {
        if view == View::GroundUp {
            return self.plan.mean_loss;
        }
        let (ded, lim) = (terms.ded, terms.limit);
        let gross = self.layer_mean(ded, lim.min(self.plan.support()));
        if view == View::Gross {
            return gross;
        }
        // A clause on gross is a layer on X: the attachment shifts by ded and
        // the limit is cut by whatever of the policy limit remains above it.
        let ceded: f64 = terms
            .clauses
            .iter()
            .map(|c| {
                let effective = c.limit.min((lim - c.attach).max(0.0));
                c.share * self.layer_mean(ded + c.attach, effective)
            })
            .sum();
        match view {
            View::Ceded => ceded,
            _ => gross - ceded,
        }
    }
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
pub fn edge_scores(plan: &UnitPlan, bucket_size: f64, n: usize) -> Result<Vec<f64>, Error> {
    let law = unit_law(plan)?;
    Ok((0..=n)
        .map(|e| {
            let edge = (e as f64 - 0.5) * bucket_size;
            normal::quantile(law.cdf(edge / plan.max_loss))
        })
        .collect())
}

/// The unit's marginal distribution function at the `n + 1` half-bucket
/// edges `(e - 1/2) h`: 0 below zero, 1 at or above M; the beta's
/// incomplete beta or the two-point step `1 - p` in between.
pub fn edge_cdf(plan: &UnitPlan, bucket_size: f64, n: usize) -> Result<Vec<f64>, Error> {
    match plan.kind {
        UnitKind::TwoPoint => {
            let p = plan.two_point_p();
            Ok((0..=n)
                .map(|e| {
                    let edge = (e as f64 - 0.5) * bucket_size;
                    if edge < 0.0 {
                        0.0
                    } else if edge < plan.max_loss {
                        1.0 - p
                    } else {
                        1.0
                    }
                })
                .collect())
        }
        _ => {
            let law = unit_law(plan)?;
            Ok((0..=n)
                .map(|e| {
                    let edge = (e as f64 - 0.5) * bucket_size;
                    law.cdf(edge / plan.max_loss)
                })
                .collect())
        }
    }
}

/// Conditional bucket pmf of one unit for each node or stratum of the
/// shared-factor rule, as a row-major `(rule.len() x n)` matrix; every row
/// sums to one.
///
/// Under a node rule row `q` is the law given `Z_c = z_q`. Under the
/// stratified rule row `s` is the law given `Phi(Z_c)` in the stratum
/// `[a_s, b_s]`: for a loading below one the average of the conditional
/// distribution function over the stratum is `(Phi_2(y, z_{s+1}; w) -
/// Phi_2(y, z_s; w)) / (b_s - a_s)` at each edge score `y`, and the rows
/// telescope to the marginal exactly; at loading one the stratum's law is
/// the marginal restricted to it, `(clip(F(x), a_s, b_s) - a_s) / (b_s -
/// a_s)`.
pub fn conditional_matrix(
    plan: &UnitPlan,
    bucket_size: f64,
    n: usize,
    rule: &FactorRule,
) -> Result<Vec<f64>, Error> {
    let q = rule.len();
    let inv_h = 1.0 / bucket_size;
    let mut out = vec![0.0_f64; q * n];
    if plan.kind == UnitKind::Drop {
        return Ok(out);
    }
    if plan.kind == UnitKind::Atom {
        let mut row = vec![0.0_f64; n];
        deposit(&mut row, plan.mean_loss, 1.0, inv_h);
        for r in out.chunks_exact_mut(n) {
            r.copy_from_slice(&row);
        }
        return Ok(out);
    }
    if rule.is_stratified() {
        stratified_rows(plan, bucket_size, n, rule, &mut out)?;
        return Ok(out);
    }
    let z = &rule.nodes;
    match plan.kind {
        UnitKind::Drop | UnitKind::Atom => unreachable!(),
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
            let y = edge_scores(plan, bucket_size, n)?;
            conditional_rows_from_scores(&y, plan.w_corr, plan.w_indep, z, &mut out);
        }
        UnitKind::TwoPoint => {
            let p = plan.two_point_p();
            for (r, &zq) in out.chunks_exact_mut(n).zip(z) {
                let at_max = two_point_conditional(p, plan.w_corr, plan.w_indep, zq);
                deposit(r, 0.0, 1.0 - at_max, inv_h);
                deposit(r, plan.max_loss, at_max, inv_h);
            }
        }
    }
    Ok(out)
}

/// `P(X = M | Z_c = z)` of a two-point unit: `1 - Phi((Phi^{-1}(1 - p) -
/// w_corr z) / w_indep)`, the step `1{z > Phi^{-1}(1 - p)}` at loading one.
pub fn two_point_conditional(p: f64, w_corr: f64, w_indep: f64, z: f64) -> f64 {
    let y0 = normal::quantile(1.0 - p);
    if w_indep == 0.0 {
        if z > y0 { 1.0 } else { 0.0 }
    } else {
        1.0 - normal::cdf((y0 - w_corr * z) / w_indep)
    }
}

/// The stratified rule's rows for a beta, comonotone or two-point unit.
fn stratified_rows(plan: &UnitPlan, bucket_size: f64, n: usize, rule: &FactorRule, out: &mut [f64]) -> Result<(), Error> {
    let f = edge_cdf(plan, bucket_size, n)?;
    if plan.kind == UnitKind::Comonotone || plan.w_indep == 0.0 {
        // Loading one: the marginal restricted to the stratum, exact.
        for (s, r) in out.chunks_exact_mut(n).enumerate() {
            let (a, b) = (rule.edges_u[s], rule.edges_u[s + 1]);
            let inv_width = 1.0 / rule.weights[s];
            let mut prev = f[0].clamp(a, b);
            for (k, slot) in r.iter_mut().enumerate() {
                let next = f[k + 1].clamp(a, b);
                *slot = inv_width * (next - prev);
                prev = next;
            }
        }
        return Ok(());
    }
    // Loading below one: the stratum average of Phi((y - w z) / w_indep) at
    // each edge score y is (U(y, z_s) - U(y, z_{s+1})) over the stratum's
    // width, with U(y, z) = P(V <= y, Z_c > z) = P(-V > -y, Z_c > z), the
    // upper tail of the bivariate normal of (-V, Z_c) with correlation -w.
    // The upper-tail form keeps the narrow tail strata relatively accurate,
    // where the cdf's complement loses them to cancellation. U(y, -inf) =
    // Phi(y), U(y, +inf) = 0; the edge at or above M has y = +inf and U =
    // Phi(-z), the edge below zero y = -inf and U = 0.
    let y: Vec<f64> = f.iter().map(|&v| normal::quantile(v)).collect();
    let bvn = normal::BivariateUpper::new(-plan.w_corr);
    let upper = |ye: f64, zs: f64| -> f64 {
        if zs == f64::NEG_INFINITY {
            normal::cdf(ye)
        } else if zs == f64::INFINITY {
            0.0
        } else if ye == f64::INFINITY {
            normal::cdf(-zs)
        } else {
            bvn.eval(-ye, zs)
        }
    };
    let mut prev_col: Vec<f64> = y.iter().map(|&ye| upper(ye, rule.edges_z[0])).collect();
    let mut next_col = vec![0.0_f64; n + 1];
    for (s, r) in out.chunks_exact_mut(n).enumerate() {
        let zb = rule.edges_z[s + 1];
        let inv_width = 1.0 / rule.weights[s];
        for (e, &ye) in y.iter().enumerate() {
            next_col[e] = upper(ye, zb);
        }
        let mut prev = 0.0_f64; // the edge below zero
        for (k, slot) in r.iter_mut().enumerate() {
            let next = inv_width * (prev_col[k + 1] - next_col[k + 1]);
            *slot = (next - prev).max(0.0);
            prev = next;
        }
        std::mem::swap(&mut prev_col, &mut next_col);
    }
    Ok(())
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
    pub coupling: Coupling,
}

impl JointConfig {
    pub fn new(bucket_size: f64, n1: usize, n2: usize) -> Self {
        JointConfig {
            bucket_size,
            n1,
            n2,
            z_nodes: 32,
            z_rule: ZRule::Stratified,
            variance_policy: VariancePolicy::default(),
            coupling: Coupling::Burke,
        }
    }
}

/// Parameters and moment diagnostics of a two-unit joint pmf (mirrors the
/// Python `JointMeta`).
#[derive(Debug, Clone)]
pub struct JointMeta {
    pub bucket_size: f64,
    pub coupling: Coupling,
    /// The `matched` coupling's scale and saturation for this event.
    pub matched: Option<MatchedScale>,
    pub z_nodes: usize,
    pub z_rule: ZRule,
    /// Nodes or strata the rule actually used (`z_nodes` body strata plus
    /// the tail ladder under the stratified rule).
    pub z_strata: usize,
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
    let rule = FactorRule::new(config.z_nodes, config.z_rule)?;
    let w = &rule.weights;
    let lengths = [config.n1, config.n2];
    let mut plans = Vec::with_capacity(2);
    for (i, (spec, &n)) in units.iter().zip(&lengths).enumerate() {
        if n < 1 {
            return Err(Error::BadGridLength(n));
        }
        let plan = prepare_unit(spec, &format!("unit {}", i + 1), config.variance_policy)?.couple(config.coupling);
        plans.push(plan);
    }
    let matched = if config.coupling == Coupling::Matched {
        let scale = matched_scale(&plans, h, &rule, roll_up_sd(&units))?;
        for p in plans.iter_mut() {
            *p = p.clone().with_loading(scale.scale * p.w_corr);
        }
        Some(scale)
    } else {
        None
    };
    // Under the homer coupling unit k receives s_k X_R up to s_k M_R, which
    // exceeds M_k whenever its share of the mean exceeds its share of the
    // exposure; its grid must cover that.
    let rolled = rolled_spec(&units);
    let rolled_plan = if config.coupling == Coupling::Homer {
        Some(prepare_unit(&rolled, "rolled", config.variance_policy)?)
    } else {
        None
    };
    let shares: [f64; 2] = if rolled.mean_loss > 0.0 {
        [units[0].mean_loss / rolled.mean_loss, units[1].mean_loss / rolled.mean_loss]
    } else {
        [0.0, 0.0]
    };
    for i in 0..2 {
        let n = lengths[i];
        let reach = match &rolled_plan {
            Some(r) => shares[i] * r.support(),
            None => plans[i].max_loss,
        };
        let top = (n - 1) as f64 * h;
        if reach > top {
            return Err(Error::GridOverflow {
                grid_length: n,
                bucket_size: h,
                top,
                max_loss: reach,
                required: grid_length_for(reach, h),
                example_rows: vec![i],
            });
        }
    }
    let (n1, n2) = (config.n1, config.n2);
    let mut pmf = vec![0.0_f64; n1 * n2];
    match (config.coupling, &rolled_plan) {
        (Coupling::Homer, Some(rolled_plan)) => {
            // Each bucket of the rolled row at the pair of its allocations,
            // deposited linearly on each axis.
            let n_r = grid_length_for(rolled_plan.support(), h);
            let base = marginal_pmf(rolled_plan, h, n_r)?;
            let inv_h = 1.0 / h;
            let mut row1 = vec![0.0_f64; n1];
            let mut row2 = vec![0.0_f64; n2];
            for (i, &p) in base.iter().enumerate() {
                if p == 0.0 {
                    continue;
                }
                let x = i as f64 * h;
                row1.iter_mut().for_each(|v| *v = 0.0);
                row2.iter_mut().for_each(|v| *v = 0.0);
                deposit(&mut row1, shares[0] * x, 1.0, inv_h);
                deposit(&mut row2, shares[1] * x, 1.0, inv_h);
                for (a, &pa) in row1.iter().enumerate() {
                    if pa == 0.0 {
                        continue;
                    }
                    for (b, &pb) in row2.iter().enumerate() {
                        if pb != 0.0 {
                            pmf[a * n2 + b] += p * pa * pb;
                        }
                    }
                }
            }
        }
        (Coupling::Comonotone, _) => {
            // Exact: the merged breakpoints of the two bucketed marginals.
            let m1 = marginal_pmf(&plans[0], h, n1)?;
            let m2 = marginal_pmf(&plans[1], h, n2)?;
            comonotone_merge(&[&m1, &m2], |idx, mass| pmf[idx[0] * n2 + idx[1]] += mass);
        }
        _ => {
            let mut c1 = conditional_matrix(&plans[0], h, n1, &rule)?;
            let c2 = conditional_matrix(&plans[1], h, n2, &rule)?;
            // Fold the weights into C1's rows, then pmf = C1^T C2.
            for (row, &wq) in c1.chunks_exact_mut(n1).zip(w) {
                for v in row.iter_mut() {
                    *v *= wq;
                }
            }
            let q = rule.len();
            // SAFETY: shapes are (q x n1)^T . (q x n2) -> (n1 x n2), all
            // buffers sized exactly and non-overlapping; strides describe
            // row-major storage with C1 read transposed.
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
        }
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
        coupling: config.coupling,
        matched,
        z_nodes: config.z_nodes,
        z_rule: config.z_rule,
        z_strata: rule.len(),
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
        let clamped = prepare_unit(&bad, "u", VariancePolicy::clamp_default()).unwrap();
        assert_eq!(clamped.notes, vec!["sigma_clamped"]);
        assert!(clamped.sigma_used < (30.0_f64 * 70.0).sqrt());
        // The default (2.5.0): the two-point law with the maximizer's sd and
        // Burke's loadings.
        let two_point = prepare_unit(&bad, "u", VariancePolicy::default()).unwrap();
        assert_eq!(two_point.kind, UnitKind::TwoPoint);
        assert_eq!(two_point.notes, vec!["variance_bernoulli"]);
        assert!((two_point.sigma_used - (30.0_f64 * 70.0).sqrt()).abs() < 1e-12);
        assert!((two_point.w_corr - 30.0 / 50.0_f64.hypot(30.0)).abs() < 1e-15);
        assert!((two_point.two_point_p() - 0.3).abs() < 1e-15);
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

    // [Eltq-Multiunit-Narrow-Units] Spec test 34: a unit narrower than 2^-48
    // of its mean is an atom by resolution, on both classifiers, and a unit
    // past the fraction's regime is bucketed by the narrow kernel, half on
    // each side of its mean, with no NaN.
    #[test]
    fn narrow_units_are_atoms_or_narrow_kernels() {
        let tiny = UnitSpec { mean_loss: 0.5, sd_indep: 1e-16, sd_corr: 0.0, max_loss: 1.0 };
        let plan = prepare_unit(&tiny, "u", VariancePolicy::default()).unwrap();
        assert_eq!((plan.kind, plan.notes.clone()), (UnitKind::Atom, vec!["atom_by_resolution"]));
        let mut diag = Diagnostics::default();
        let class = classify_unit_row(0, &tiny, VariancePolicy::default(), &mut diag).unwrap();
        assert_eq!(class.kind, UnitKind::Atom);
        assert_eq!(diag.get(Category::AtomByResolution).count, 1);
        let narrow = UnitSpec { sd_indep: 1e-5, ..tiny };
        let plan = prepare_unit(&narrow, "u", VariancePolicy::default()).unwrap();
        assert!(unit_law(&plan).unwrap().is_narrow());
        let pmf = marginal_pmf(&plan, 0.2, 6).unwrap();
        assert!(pmf.iter().all(|p| p.is_finite()));
        assert!((pmf[2] - 0.5).abs() < 1e-12 && (pmf[3] - 0.5).abs() < 1e-12);
        let strata = FactorRule::new(32, ZRule::Stratified).unwrap();
        let rows = conditional_matrix(&plan, 0.2, 6, &strata).unwrap();
        assert!(rows.iter().all(|p| p.is_finite() && *p >= 0.0));
        assert!((LayerMeans::new(&plan).unwrap().layer_mean(0.0, 0.4) - 0.4).abs() < 1e-12);
    }

    /// Inadmissible: 80^2 = 6400 >= 30 * 70.
    const BAD: UnitSpec = UnitSpec {
        mean_loss: 30.0,
        sd_indep: 50.0,
        sd_corr: 30.0,
        max_loss: 100.0,
    };

    #[test]
    fn conditional_rows_are_distributions() {
        let rule = FactorRule::new(16, ZRule::Midpoint).unwrap();
        let strata = FactorRule::new(16, ZRule::Stratified).unwrap();
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
            BAD,
            UnitSpec {
                sd_indep: 0.0,
                sd_corr: 80.0,
                ..BAD
            },
        ] {
            let plan = prepare_unit(&spec, "u", VariancePolicy::default()).unwrap();
            let n = grid_length_for(plan.support(), 10.0);
            for r in [&rule, &strata] {
                let c = conditional_matrix(&plan, 10.0, n, r).unwrap();
                for row in c.chunks_exact(n) {
                    let mass: f64 = row.iter().sum();
                    assert!((mass - 1.0).abs() < 1e-12, "{spec:?}: mass {mass}");
                    assert!(row.iter().all(|&p| p >= 0.0));
                }
            }
        }
    }

    // [z-rule] Spec test 28: under the stratified rule the mixture of a
    // unit's rows is its direct bucketing to rounding, for every kind and
    // every loading, at any Q.
    #[test]
    fn stratified_rows_mix_to_the_marginal() {
        let specs = [
            U1,
            UnitSpec { sd_indep: 0.0, ..U1 },                 // beta at loading one
            UnitSpec { sd_indep: 2.0, sd_corr: 248.0, ..U1 }, // loading 0.99997
            UnitSpec { sd_corr: 0.0, ..U1 },                  // independent
            BAD,
            UnitSpec { sd_indep: 0.0, sd_corr: 80.0, ..BAD }, // two-point at loading one
            UnitSpec { sd_indep: 80.0, sd_corr: 0.0, ..BAD },
        ];
        for q in [1, 4, 32] {
            let strata = FactorRule::new(q, ZRule::Stratified).unwrap();
            for spec in specs {
                let plan = prepare_unit(&spec, "u", VariancePolicy::default()).unwrap();
                let n = grid_length_for(plan.support(), 10.0);
                let c = conditional_matrix(&plan, 10.0, n, &strata).unwrap();
                let direct = marginal_pmf(&plan, 10.0, n).unwrap();
                let mut mixed = vec![0.0; n];
                for (row, &wq) in c.chunks_exact(n).zip(&strata.weights) {
                    for (m, &p) in mixed.iter_mut().zip(row) {
                        *m += wq * p;
                    }
                }
                for (k, (m, d)) in mixed.iter().zip(&direct).enumerate() {
                    assert!((m - d).abs() < 2e-15, "{spec:?} q={q} k={k}: {m} vs {d}");
                }
            }
        }
        // Under a node rule the same unit at loading one is Q atoms.
        let plan = prepare_unit(&UnitSpec { sd_indep: 0.0, ..U1 }, "u", VariancePolicy::default()).unwrap();
        let nodes = FactorRule::new(8, ZRule::Midpoint).unwrap();
        let c = conditional_matrix(&plan, 10.0, 101, &nodes).unwrap();
        for row in c.chunks_exact(101) {
            assert!(row.iter().filter(|&&p| p > 0.0).count() <= 2);
        }
    }

    // [bernoulli] Spec test 26: the two-point conditional law. Its mixture
    // over Gauss-Hermite nodes is the marginal (smooth in z for w < 1), the
    // atoms sit at 0 and M only, and the step at loading one resolves to the
    // nodes' cumulative weight.
    #[test]
    fn two_point_conditional_law() {
        let plan = prepare_unit(&BAD, "u", VariancePolicy::default()).unwrap();
        let rule = FactorRule::new(64, ZRule::GaussHermite).unwrap();
        let (z, w) = (&rule.nodes, &rule.weights);
        let n = grid_length_for(plan.support(), 10.0);
        let c = conditional_matrix(&plan, 10.0, n, &rule).unwrap();
        let mut marginal = vec![0.0; n];
        for (row, &wq) in c.chunks_exact(n).zip(w) {
            for (m, &p) in marginal.iter_mut().zip(row) {
                *m += wq * p;
            }
        }
        assert!((marginal[0] - 0.7).abs() < 1e-12);
        assert!((marginal[n - 1] - 0.3).abs() < 1e-12);
        assert!(marginal[1..n - 1].iter().all(|&p| p == 0.0));
        let direct = marginal_pmf(&plan, 10.0, n).unwrap();
        assert_eq!((direct[0], direct[n - 1]), (0.7, 0.3));
        // Each row's mass at M is the closed form.
        for (row, &zq) in c.chunks_exact(n).zip(z) {
            let want = 1.0 - normal::cdf((normal::quantile(0.7) - plan.w_corr * zq) / plan.w_indep);
            assert!((row[n - 1] - want).abs() < 1e-15);
        }
        // Layer means: the loss is M with probability p.
        let lm = LayerMeans::new(&plan).unwrap();
        assert!((lm.layer_mean(0.0, f64::INFINITY) - 30.0).abs() < 1e-12);
        assert!((lm.layer_mean(40.0, 50.0) - 0.3 * 50.0).abs() < 1e-12);
        // Loading one: a step at Phi^{-1}(1 - p); at the midpoint nodes the
        // mixture gives p to the nearest cumulative node weight.
        let step = prepare_unit(
            &UnitSpec {
                sd_indep: 0.0,
                sd_corr: 80.0,
                ..BAD
            },
            "u",
            VariancePolicy::default(),
        )
        .unwrap();
        assert_eq!((step.kind, step.w_corr), (UnitKind::TwoPoint, 1.0));
        let rule = FactorRule::new(1000, ZRule::Midpoint).unwrap();
        let c = conditional_matrix(&step, 10.0, n, &rule).unwrap();
        let at_max: f64 = c.chunks_exact(n).zip(&rule.weights).map(|(row, &wq)| wq * row[n - 1]).sum();
        assert!((at_max - 0.3).abs() <= 1.0 / 1000.0 + 1e-12);
        assert!(c.chunks_exact(n).all(|row| row[n - 1] == 0.0 || row[n - 1] == 1.0));
    }

    #[test]
    fn joint_matches_naive_product_and_is_outer_for_independent_units() {
        let mut config = JointConfig::new(10.0, 101, 201);
        config.z_nodes = 8;
        config.z_rule = ZRule::Midpoint;
        let result = joint_two_units([U1, U2], &config).unwrap();
        assert!((result.meta.total_mass - 1.0).abs() < 1e-12);
        // Naive triple loop reproduces the GEMM.
        let rule = FactorRule::new(8, ZRule::Midpoint).unwrap();
        let w = &rule.weights;
        let p1 = prepare_unit(&U1, "u", VariancePolicy::default()).unwrap();
        let p2 = prepare_unit(&U2, "u", VariancePolicy::default()).unwrap();
        let c1 = conditional_matrix(&p1, 10.0, 101, &rule).unwrap();
        let c2 = conditional_matrix(&p2, 10.0, 201, &rule).unwrap();
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
