//! Multi-unit per-event engine: total event severity by view, rate
//! accumulated across events.
//!
//! Input rows are event x unit, grouped contiguously by event (validated,
//! never sorted for the caller) with the rate constant within an event and
//! each unit at most once per event. Per event the engine
//!
//! 1. classifies the units (atom, comonotone, beta, or dropped) and builds
//!    each unit's base conditional rows given the shared factor: Q rows for
//!    correlated units, a single row for independent units and atoms;
//! 2. per view, maps every unit's rows through its compiled [`ViewMap`]
//!    (identity maps reuse the base rows), convolves the independent block
//!    once, and folds the correlated units in per z node, either by direct
//!    convolution or by FFT with the spectra accumulated over z so only one
//!    inverse transform is needed per view (chosen by cost, or forced);
//! 3. deposits the event's per-view pmf into the chunk's histograms weighted
//!    by the event rate, reconciles the pipeline view means against the
//!    analytic beta layer means, and (optionally) compares the exact
//!    ground-up total against the rolled single-row beta.
//!
//! Parallelism follows `quantize`: fixed chunks of whole events, private
//! per-chunk partials, a sequential fold in chunk order, so results are
//! bitwise identical for any thread count. The z loop lives inside the
//! event; no cross-thread reduction ever mixes floating-point orders.
//! Mathematics: `eltq.reference_multiunit` is the oracle.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;
use rustfft::num_complex::Complex;

use crate::config::{DEFAULT_CHUNK_ROWS, VariancePolicy};
use crate::convolve::{FftCache, convolve_direct, direct_cost, fft_cost, fft_size};
use crate::diagnostics::{Category, Diagnostics, MAX_EXAMPLES};
use crate::error::Error;
use crate::multiunit::{
    LayerMeans, RowClass, UnitKind, UnitPlan, UnitSpec, classify_unit_row, conditional_matrix,
    grid_length_for, marginal_pmf, prepare_unit,
};
use crate::quadrature::{ZRule, z_quadrature};
use crate::sum::{Neumaier, grid_moment, sum};
use crate::terms::{UnitTerms, View, ViewMap, check_grid_alignment};

/// Structure-of-arrays multi-unit ELT. `unit_code` indexes the terms table;
/// `event_code` only needs to be constant within an event and unique across
/// events (any integer coding). Row identifiers stay outside the hot path.
#[derive(Debug, Clone, Copy)]
pub struct MultiInput<'a> {
    pub event_code: &'a [i64],
    pub unit_code: &'a [i64],
    pub rate: &'a [f64],
    pub mean_loss: &'a [f64],
    pub sd_indep: &'a [f64],
    pub sd_corr: &'a [f64],
    pub max_loss: &'a [f64],
}

impl MultiInput<'_> {
    fn validated_len(&self) -> Result<usize, Error> {
        let expected = self.event_code.len();
        for (name, len) in [
            ("unit_id", self.unit_code.len()),
            ("rate", self.rate.len()),
            ("mean_loss", self.mean_loss.len()),
            ("sd_indep", self.sd_indep.len()),
            ("sd_corr", self.sd_corr.len()),
            ("max_loss", self.max_loss.len()),
        ] {
            if len != expected {
                return Err(Error::ColumnLengthMismatch {
                    name,
                    got: len,
                    expected,
                });
            }
        }
        Ok(expected)
    }

    fn spec(&self, row: usize) -> UnitSpec {
        UnitSpec {
            mean_loss: self.mean_loss[row],
            sd_indep: self.sd_indep[row],
            sd_corr: self.sd_corr[row],
            max_loss: self.max_loss[row],
        }
    }
}

/// How units are convolved across an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Convolution {
    /// Direct or FFT per event and view by estimated cost.
    Auto,
    Direct,
    Fft,
}

impl Convolution {
    pub fn parse(name: &str) -> Result<Self, Error> {
        match name {
            "auto" => Ok(Convolution::Auto),
            "direct" => Ok(Convolution::Direct),
            "fft" => Ok(Convolution::Fft),
            other => Err(Error::BadConvolution(other.to_string())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Convolution::Auto => "auto",
            Convolution::Direct => "direct",
            Convolution::Fft => "fft",
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultiConfig {
    pub bucket_size: f64,
    pub grid_length: Option<usize>,
    /// Shared-factor quadrature nodes Q.
    pub z_nodes: usize,
    pub z_rule: ZRule,
    /// Views to compute, in output order.
    pub views: Vec<View>,
    pub normalize: bool,
    pub variance_policy: VariancePolicy,
    pub threads: Option<usize>,
    /// Rows per parallel chunk (rounded up to whole events); part of the
    /// deterministic contract.
    pub chunk_rows: usize,
    pub roll_up_diagnostic: bool,
    pub convolution: Convolution,
}

impl MultiConfig {
    pub fn new(bucket_size: f64) -> Self {
        MultiConfig {
            bucket_size,
            grid_length: None,
            z_nodes: 32,
            z_rule: ZRule::Midpoint,
            views: View::ALL.to_vec(),
            normalize: false,
            variance_policy: VariancePolicy::default(),
            threads: None,
            chunk_rows: DEFAULT_CHUNK_ROWS,
            roll_up_diagnostic: false,
            convolution: Convolution::Auto,
        }
    }

    pub fn validate(&self) -> Result<(), Error> {
        if !self.bucket_size.is_finite() || self.bucket_size <= 0.0 {
            return Err(Error::BadBucketSize(self.bucket_size));
        }
        if self.z_nodes < 1 {
            return Err(Error::BadZNodes(self.z_nodes));
        }
        if let Some(n) = self.grid_length
            && n < 1
        {
            return Err(Error::BadGridLength(n));
        }
        if let VariancePolicy::ClampSigma { factor } = self.variance_policy
            && !(factor.is_finite() && factor > 0.0 && factor < 1.0)
        {
            return Err(Error::BadClampFactor(factor));
        }
        if self.views.is_empty() {
            return Err(Error::NoViews);
        }
        if self.chunk_rows < 1 {
            return Err(Error::BadChunkRows(self.chunk_rows));
        }
        if self.threads == Some(0) {
            return Err(Error::BadThreads(0));
        }
        Ok(())
    }
}

/// Exact ground-up total versus the rolled single-row beta for one event.
#[derive(Debug, Clone, PartialEq)]
pub struct RollUpEntry {
    /// First row of the event (callers translate to the event id).
    pub event_row: usize,
    pub units: usize,
    pub sd_exact: f64,
    pub sd_rolled: f64,
    pub sd_ratio: f64,
    pub q99_exact: f64,
    pub q99_rolled: f64,
    pub q99_ratio: f64,
}

/// Metadata accompanying the per-view distributions; per-view vectors are
/// in `views` order. Masses and means describe the arrays returned
/// (normalized when `normalized`); `total_rate`, `analytic_aal`,
/// `analytic_view_mean` and `view_mean_error` are always rate-weighted.
#[derive(Debug, Clone)]
pub struct MultiMeta {
    pub bucket_size: f64,
    pub grid_length: usize,
    pub z_nodes: usize,
    pub z_rule: ZRule,
    pub views: Vec<View>,
    pub normalized: bool,
    pub total_rate: f64,
    pub analytic_aal: f64,
    pub view_mass: Vec<f64>,
    pub view_mean: Vec<f64>,
    pub analytic_view_mean: Vec<f64>,
    pub view_mean_error: Vec<f64>,
    pub input_rows: usize,
    pub valid_rows: usize,
    pub events: usize,
    pub valid_events: usize,
    pub diagnostics: Diagnostics,
    pub roll_up: Option<Vec<RollUpEntry>>,
    /// Largest |mass - 1| of any per-event view pmf (conditional exactness).
    pub max_mass_error: f64,
    /// Event-views convolved by each path.
    pub direct_paths: usize,
    pub fft_paths: usize,
}

#[derive(Debug, Clone)]
pub struct MultiResult {
    /// One rate-weight array per view, in `meta.views` order.
    pub views: Vec<Vec<f64>>,
    pub meta: MultiMeta,
}

/// Private per-chunk partial result.
struct ChunkPartial {
    views: Vec<Vec<f64>>,
    rate: f64,
    aal: f64,
    analytic: Vec<f64>,
    valid_rows: usize,
    valid_events: usize,
    max_mass_error: f64,
    roll_up: Vec<RollUpEntry>,
    direct_paths: usize,
    fft_paths: usize,
}

/// One contributing unit of an event inside the worker.
struct LiveUnit {
    row: usize,
    plan: UnitPlan,
    /// Grid length of the unit's own support.
    n: usize,
    /// Number of base rows: Q when the unit depends on the shared factor,
    /// 1 otherwise (independent beta units and atoms).
    rows: usize,
    base: Vec<f64>,
    /// Index of the unit's slot in the worker's view-row scratch.
    slot: usize,
}

/// The current view's rows of a unit: its scratch slot when a map was
/// applied, the base rows otherwise (identity map).
fn rows_for<'u>(view_rows: &'u [Vec<f64>], unit: &'u LiveUnit) -> &'u [f64] {
    let slot = &view_rows[unit.slot];
    if slot.is_empty() { &unit.base } else { slot }
}

/// Per-worker state: quadrature, FFT plans, compiled view maps, scratch.
struct Worker<'a> {
    z: &'a [f64],
    w: &'a [f64],
    h: f64,
    fft: FftCache,
    maps: HashMap<(usize, usize, View), ViewMap>,
    convolution: Convolution,
    // Scratch buffers reused across events.
    view_rows: Vec<Vec<f64>>,
    indep: Vec<f64>,
    acc: Vec<f64>,
    tmp: Vec<f64>,
    pmf: Vec<f64>,
    spec_indep: Vec<Complex<f64>>,
    spec_acc: Vec<Complex<f64>>,
    spec_prod: Vec<Complex<f64>>,
    spec_tmp: Vec<Complex<f64>>,
}

impl Worker<'_> {
    fn view_map(&mut self, unit_code: usize, terms: &UnitTerms, view: View, n: usize) -> &ViewMap {
        if self.maps.len() > 4096 {
            self.maps.clear();
        }
        self.maps
            .entry((unit_code, n, view))
            .or_insert_with(|| ViewMap::compile(terms, view, self.h, n))
    }

    /// Convolve the live units' view rows into `self.pmf` (length `total`),
    /// averaging over the shared factor. Returns whether the FFT path ran.
    fn convolve_event(&mut self, units: &[LiveUnit], lens: &[usize], total: usize) -> bool {
        let q = self.z.len();
        // The view-row scratch leaves `self` for the duration so the other
        // scratch buffers can be borrowed mutably alongside it.
        let view_rows = std::mem::take(&mut self.view_rows);
        // Independent block: single-row units, convolved once.
        self.indep.clear();
        self.indep.push(1.0);
        for (u, &len) in units.iter().zip(lens) {
            if u.rows == 1 {
                let row = &rows_for(&view_rows, u)[..len];
                convolve_direct(&self.indep, row, &mut self.tmp);
                std::mem::swap(&mut self.indep, &mut self.tmp);
            }
        }
        let correlated: Vec<usize> = (0..units.len()).filter(|&i| units[i].rows > 1).collect();
        self.pmf.clear();
        self.pmf.resize(total, 0.0);
        if correlated.is_empty() {
            self.pmf[..self.indep.len()].copy_from_slice(&self.indep);
            self.view_rows = view_rows;
            return false;
        }
        let use_fft = match self.convolution {
            Convolution::Direct => false,
            Convolution::Fft => true,
            Convolution::Auto => {
                let mut chain: Vec<usize> = correlated.iter().map(|&i| lens[i]).collect();
                chain.push(self.indep.len());
                let direct = q as f64 * direct_cost(&chain);
                let n = fft_size(total);
                let fft = fft_cost(n) * (q * correlated.len() + 2) as f64 + (q * n) as f64;
                fft < direct
            }
        };
        if use_fft {
            let n = fft_size(total);
            self.fft.spectrum(&self.indep.clone(), n, &mut self.spec_indep);
            self.spec_acc.clear();
            self.spec_acc.resize(n, Complex::new(0.0, 0.0));
            for zi in 0..q {
                self.spec_prod.clear();
                self.spec_prod.extend_from_slice(&self.spec_indep);
                for &i in &correlated {
                    let len = lens[i];
                    let row = &rows_for(&view_rows, &units[i])[zi * len..(zi + 1) * len];
                    self.fft.spectrum(row, n, &mut self.spec_tmp);
                    for (p, t) in self.spec_prod.iter_mut().zip(&self.spec_tmp) {
                        *p *= t;
                    }
                }
                let wq = self.w[zi];
                for (a, p) in self.spec_acc.iter_mut().zip(&self.spec_prod) {
                    *a += wq * p;
                }
            }
            let mut spec = std::mem::take(&mut self.spec_acc);
            self.fft.inverse_into(&mut spec, total, &mut self.pmf);
            self.spec_acc = spec;
        } else {
            for zi in 0..q {
                let first = correlated[0];
                let len = lens[first];
                self.acc.clear();
                self.acc
                    .extend_from_slice(&rows_for(&view_rows, &units[first])[zi * len..(zi + 1) * len]);
                for &i in &correlated[1..] {
                    let len = lens[i];
                    let row = &rows_for(&view_rows, &units[i])[zi * len..(zi + 1) * len];
                    convolve_direct(&self.acc, row, &mut self.tmp);
                    std::mem::swap(&mut self.acc, &mut self.tmp);
                }
                if self.indep.len() > 1 {
                    convolve_direct(&self.acc, &self.indep, &mut self.tmp);
                    std::mem::swap(&mut self.acc, &mut self.tmp);
                }
                let wq = self.w[zi];
                for (p, a) in self.pmf.iter_mut().zip(&self.acc) {
                    *p += wq * a;
                }
            }
        }
        self.view_rows = view_rows;
        use_fft
    }
}

/// Chunk boundaries as event index ranges, each holding whole events and at
/// least `chunk_rows` rows (except the last).
fn event_chunks(events: &[(usize, usize)], chunk_rows: usize) -> Vec<(usize, usize)> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut rows = 0;
    for (i, &(s, e)) in events.iter().enumerate() {
        rows += e - s;
        if rows >= chunk_rows {
            chunks.push((start, i + 1));
            start = i + 1;
            rows = 0;
        }
    }
    if start < events.len() {
        chunks.push((start, events.len()));
    }
    chunks
}

fn quantile_of(pmf: &[f64], bucket_size: f64, p: f64) -> f64 {
    let total: f64 = pmf.iter().sum();
    let target = p * total;
    let mut acc = 0.0;
    for (i, &v) in pmf.iter().enumerate() {
        acc += v;
        if acc >= target {
            return i as f64 * bucket_size;
        }
    }
    (pmf.len().saturating_sub(1)) as f64 * bucket_size
}

fn sd_of(pmf: &[f64], bucket_size: f64) -> f64 {
    let mean = grid_moment(pmf, bucket_size);
    let mut acc = Neumaier::default();
    for (i, &p) in pmf.iter().enumerate() {
        let x = i as f64 * bucket_size;
        acc.add(p * x * x);
    }
    (acc.value() - mean * mean).max(0.0).sqrt()
}

/// Total event severity by view for a multi-unit ELT.
pub fn multiunit_views(
    input: &MultiInput<'_>,
    terms: &[UnitTerms],
    config: &MultiConfig,
) -> Result<MultiResult, Error> {
    config.validate()?;
    let n_rows = input.validated_len()?;
    let h = config.bucket_size;
    for t in terms {
        check_grid_alignment(t, h)?;
    }
    let (z, w) = z_quadrature(config.z_nodes, config.z_rule)?;

    // Pass 1: structure, classification, event supports.
    let mut diagnostics = Diagnostics::default();
    let mut classes = vec![
        RowClass {
            kind: UnitKind::Drop,
            mu: 0.0,
            sigma: 0.0
        };
        n_rows
    ];
    let mut events: Vec<(usize, usize)> = Vec::new();
    let mut seen: HashSet<i64> = HashSet::new();
    let mut start = 0;
    while start < n_rows {
        let code = input.event_code[start];
        let mut end = start + 1;
        while end < n_rows && input.event_code[end] == code {
            end += 1;
        }
        if !seen.insert(code) {
            return Err(Error::MultiUnitStructure(format!(
                "event code {code} rows are not contiguous (reappears at row {start})"
            )));
        }
        events.push((start, end));
        start = end;
    }
    let mut event_support = vec![0.0_f64; events.len()];
    let mut event_live = vec![false; events.len()];
    let mut units_here: Vec<i64> = Vec::new();
    for (e, &(start, end)) in events.iter().enumerate() {
        let rate = input.rate[start];
        units_here.clear();
        for row in start..end {
            let code = input.unit_code[row];
            if code < 0 || code as usize >= terms.len() {
                return Err(Error::UnitCodeOutOfRange {
                    row,
                    code,
                    available: terms.len(),
                });
            }
            if units_here.contains(&code) {
                return Err(Error::MultiUnitStructure(format!(
                    "event at row {start}: unit code {code} appears twice"
                )));
            }
            units_here.push(code);
            if input.rate[row].to_bits() != rate.to_bits() && input.rate[row] != rate {
                return Err(Error::MultiUnitStructure(format!(
                    "event at row {start}: rate varies across its rows"
                )));
            }
        }
        if !rate.is_finite() || rate < 0.0 {
            let category = if rate.is_finite() { Category::NegativeField } else { Category::Nonfinite };
            for row in start..end {
                diagnostics.note(category, row);
            }
            continue;
        }
        if rate == 0.0 {
            for row in start..end {
                diagnostics.note(Category::ZeroRate, row);
            }
            continue;
        }
        let mut support = 0.0;
        for row in start..end {
            let class = classify_unit_row(row, &input.spec(row), config.variance_policy, &mut diagnostics)?;
            classes[row] = class;
            if class.kind != UnitKind::Drop {
                event_live[e] = true;
                support += if class.kind == UnitKind::Atom { class.mu } else { input.max_loss[row] };
            }
        }
        event_support[e] = support;
    }
    let max_support = event_support.iter().cloned().fold(0.0_f64, f64::max);
    let auto_length = grid_length_for(max_support, h);
    let n = match config.grid_length {
        None => auto_length,
        Some(n) => {
            let top = (n - 1) as f64 * h;
            if max_support > top {
                let example_rows = events
                    .iter()
                    .enumerate()
                    .filter(|(e, _)| event_live[*e] && event_support[*e] > top)
                    .map(|(_, &(start, _))| start)
                    .take(MAX_EXAMPLES)
                    .collect();
                return Err(Error::GridOverflow {
                    grid_length: n,
                    bucket_size: h,
                    top,
                    max_loss: max_support,
                    required: auto_length,
                    example_rows,
                });
            }
            n
        }
    };

    // Views computed per event: the requested ones plus ground_up for the
    // roll-up diagnostic when not requested itself.
    let mut computed: Vec<View> = config.views.clone();
    let roll_up_slot = if config.roll_up_diagnostic {
        match computed.iter().position(|&v| v == View::GroundUp) {
            Some(i) => Some(i),
            None => {
                computed.push(View::GroundUp);
                Some(computed.len() - 1)
            }
        }
    } else {
        None
    };
    let n_views = config.views.len();
    let chunks = event_chunks(&events, config.chunk_rows);

    let process_chunk = |chunk_index: usize| -> Result<ChunkPartial, Error> {
        let (first, last) = chunks[chunk_index];
        let mut partial = ChunkPartial {
            views: (0..n_views).map(|_| vec![0.0; n]).collect(),
            rate: 0.0,
            aal: 0.0,
            analytic: vec![0.0; n_views],
            valid_rows: 0,
            valid_events: 0,
            max_mass_error: 0.0,
            roll_up: Vec::new(),
            direct_paths: 0,
            fft_paths: 0,
        };
        let mut rate_acc = Neumaier::default();
        let mut aal_acc = Neumaier::default();
        let mut analytic_acc: Vec<Neumaier> = vec![Neumaier::default(); n_views];
        let mut worker = Worker {
            z: &z,
            w: &w,
            h,
            fft: FftCache::default(),
            maps: HashMap::new(),
            convolution: config.convolution,
            view_rows: Vec::new(),
            indep: Vec::new(),
            acc: Vec::new(),
            tmp: Vec::new(),
            pmf: Vec::new(),
            spec_indep: Vec::new(),
            spec_acc: Vec::new(),
            spec_prod: Vec::new(),
            spec_tmp: Vec::new(),
        };
        for e in first..last {
            if !event_live[e] {
                continue;
            }
            let (start, end) = events[e];
            let lam = input.rate[start];
            partial.valid_events += 1;
            rate_acc.add(lam);
            // Live units with their base conditional rows.
            let mut units: Vec<LiveUnit> = Vec::new();
            let mut mu_sum = 0.0;
            for row in start..end {
                let class = classes[row];
                if class.kind == UnitKind::Drop {
                    continue;
                }
                let spec = input.spec(row);
                let plan = UnitPlan::from_row(&class, &spec);
                let n_unit = grid_length_for(plan.support(), h);
                let correlated = matches!(plan.kind, UnitKind::Comonotone)
                    || (plan.kind == UnitKind::Beta && plan.w_corr > 0.0);
                let nodes: &[f64] = if correlated { &z } else { &[0.0] };
                let base = conditional_matrix(&plan, h, n_unit, nodes).map_err(|_| Error::KernelFailure {
                    row,
                    alpha: plan.alpha,
                    beta: plan.beta,
                })?;
                mu_sum += plan.mean_loss;
                units.push(LiveUnit {
                    row,
                    plan,
                    n: n_unit,
                    rows: nodes.len(),
                    base,
                    slot: units.len(),
                });
            }
            partial.valid_rows += units.len();
            aal_acc.add(lam * mu_sum);
            let layer_means: Vec<LayerMeans> = units.iter().map(|u| LayerMeans::new(&u.plan)).collect();
            worker.view_rows.resize_with(units.len(), Vec::new);
            for (vi, &view) in computed.iter().enumerate() {
                // Transform base rows through the view maps; identity maps
                // leave the slot empty so the base rows are used directly.
                let mut lens = Vec::with_capacity(units.len());
                for (i, u) in units.iter().enumerate() {
                    let code = input.unit_code[u.row] as usize;
                    let map = worker.view_map(code, &terms[code], view, u.n).clone();
                    lens.push(map.support_len);
                    let slot = &mut worker.view_rows[i];
                    slot.clear();
                    if !map.identity {
                        slot.resize(u.rows * map.support_len, 0.0);
                        for (r, dst) in u.base.chunks_exact(u.n).zip(slot.chunks_exact_mut(map.support_len)) {
                            map.apply(r, dst);
                        }
                    }
                }
                let total = lens.iter().map(|l| l - 1).sum::<usize>() + 1;
                let used_fft = worker.convolve_event(&units, &lens, total);
                if used_fft {
                    partial.fft_paths += 1;
                } else {
                    partial.direct_paths += 1;
                }
                let mass: f64 = worker.pmf.iter().sum();
                partial.max_mass_error = partial.max_mass_error.max((mass - 1.0).abs());
                if vi < n_views {
                    for (dst, &p) in partial.views[vi].iter_mut().zip(&worker.pmf) {
                        *dst += lam * p;
                    }
                    let analytic: f64 = units
                        .iter()
                        .zip(&layer_means)
                        .map(|(u, lm)| lm.view_mean(&terms[input.unit_code[u.row] as usize], view))
                        .sum();
                    analytic_acc[vi].add(lam * analytic);
                }
                if roll_up_slot == Some(vi) && units.len() >= 2 {
                    let rolled = UnitSpec {
                        mean_loss: units.iter().map(|u| input.mean_loss[u.row]).sum(),
                        sd_indep: units
                            .iter()
                            .map(|u| input.sd_indep[u.row] * input.sd_indep[u.row])
                            .sum::<f64>()
                            .sqrt(),
                        sd_corr: units.iter().map(|u| input.sd_corr[u.row]).sum(),
                        max_loss: units.iter().map(|u| input.max_loss[u.row]).sum(),
                    };
                    let rolled_plan = prepare_unit(&rolled, "rolled", VariancePolicy::default())?;
                    let rolled_pmf = marginal_pmf(&rolled_plan, h, total);
                    let (sd_exact, sd_rolled) = (sd_of(&worker.pmf, h), sd_of(&rolled_pmf, h));
                    let (q_exact, q_rolled) =
                        (quantile_of(&worker.pmf, h, 0.99), quantile_of(&rolled_pmf, h, 0.99));
                    partial.roll_up.push(RollUpEntry {
                        event_row: start,
                        units: units.len(),
                        sd_exact,
                        sd_rolled,
                        sd_ratio: if sd_rolled > 0.0 { sd_exact / sd_rolled } else { f64::NAN },
                        q99_exact: q_exact,
                        q99_rolled: q_rolled,
                        q99_ratio: if q_rolled > 0.0 { q_exact / q_rolled } else { f64::NAN },
                    });
                }
            }
        }
        partial.rate = rate_acc.value();
        partial.aal = aal_acc.value();
        partial.analytic = analytic_acc.iter().map(Neumaier::value).collect();
        Ok(partial)
    };

    // Pass 2: parallel chunks, sequential fold in chunk order.
    let mut views: Vec<Vec<f64>> = (0..n_views).map(|_| vec![0.0; n]).collect();
    let mut total_rate = Neumaier::default();
    let mut analytic_aal = Neumaier::default();
    let mut analytic: Vec<Neumaier> = vec![Neumaier::default(); n_views];
    let mut valid_rows = 0;
    let mut valid_events = 0;
    let mut max_mass_error = 0.0_f64;
    let mut roll_up: Vec<RollUpEntry> = Vec::new();
    let (mut direct_paths, mut fft_paths) = (0, 0);
    let mut fold = |partials: Vec<Result<ChunkPartial, Error>>| -> Result<(), Error> {
        for partial in partials {
            let partial = partial?;
            for (dst, src) in views.iter_mut().zip(&partial.views) {
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += s;
                }
            }
            total_rate.add(partial.rate);
            analytic_aal.add(partial.aal);
            for (acc, v) in analytic.iter_mut().zip(&partial.analytic) {
                acc.add(*v);
            }
            valid_rows += partial.valid_rows;
            valid_events += partial.valid_events;
            max_mass_error = max_mass_error.max(partial.max_mass_error);
            roll_up.extend(partial.roll_up);
            direct_paths += partial.direct_paths;
            fft_paths += partial.fft_paths;
        }
        Ok(())
    };
    let n_chunks = chunks.len();
    let run_waves = |fold: &mut dyn FnMut(Vec<Result<ChunkPartial, Error>>) -> Result<(), Error>|
     -> Result<(), Error> {
        let wave = rayon::current_num_threads().max(1) * 2;
        let mut next = 0;
        while next < n_chunks {
            let end = (next + wave).min(n_chunks);
            let partials: Vec<Result<ChunkPartial, Error>> =
                (next..end).into_par_iter().map(process_chunk).collect();
            fold(partials)?;
            next = end;
        }
        Ok(())
    };
    match config.threads {
        Some(threads) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| Error::ThreadPool(e.to_string()))?;
            pool.install(|| run_waves(&mut fold))?;
        }
        None => run_waves(&mut fold)?,
    }

    let total_rate = total_rate.value();
    if config.normalize {
        if total_rate <= 0.0 {
            return Err(Error::ZeroTotalRate);
        }
        for v in views.iter_mut() {
            for x in v.iter_mut() {
                *x /= total_rate;
            }
        }
    }
    let scale = if config.normalize { total_rate } else { 1.0 };
    let view_mass: Vec<f64> = views.iter().map(|v| sum(v)).collect();
    let view_mean: Vec<f64> = views.iter().map(|v| grid_moment(v, h)).collect();
    let analytic_view_mean: Vec<f64> = analytic.iter().map(Neumaier::value).collect();
    let view_mean_error = view_mean
        .iter()
        .zip(&analytic_view_mean)
        .map(|(m, a)| m * scale - a)
        .collect();
    let meta = MultiMeta {
        bucket_size: h,
        grid_length: n,
        z_nodes: config.z_nodes,
        z_rule: config.z_rule,
        views: config.views.clone(),
        normalized: config.normalize,
        total_rate,
        analytic_aal: analytic_aal.value(),
        view_mass,
        view_mean,
        analytic_view_mean,
        view_mean_error,
        input_rows: n_rows,
        valid_rows,
        events: events.len(),
        valid_events,
        diagnostics,
        roll_up: if config.roll_up_diagnostic { Some(roll_up) } else { None },
        max_mass_error,
        direct_paths,
        fft_paths,
    };
    Ok(MultiResult { views, meta })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terms::parse_terms_line;

    struct Rows {
        event: Vec<i64>,
        unit: Vec<i64>,
        rate: Vec<f64>,
        mu: Vec<f64>,
        si: Vec<f64>,
        sc: Vec<f64>,
        m: Vec<f64>,
    }

    fn rows(spec: &[(i64, i64, f64, f64, f64, f64, f64)]) -> Rows {
        Rows {
            event: spec.iter().map(|r| r.0).collect(),
            unit: spec.iter().map(|r| r.1).collect(),
            rate: spec.iter().map(|r| r.2).collect(),
            mu: spec.iter().map(|r| r.3).collect(),
            si: spec.iter().map(|r| r.4).collect(),
            sc: spec.iter().map(|r| r.5).collect(),
            m: spec.iter().map(|r| r.6).collect(),
        }
    }

    fn input(r: &Rows) -> MultiInput<'_> {
        MultiInput {
            event_code: &r.event,
            unit_code: &r.unit,
            rate: &r.rate,
            mean_loss: &r.mu,
            sd_indep: &r.si,
            sd_corr: &r.sc,
            max_loss: &r.m,
        }
    }

    const TOWER: &str = "unit a 750 xs 250 occurrence net of 50% po 400 xs 100 and 500 xs 500";

    #[test]
    fn atom_event_view_values() {
        let r = rows(&[(1, 0, 0.02, 1000.0, 0.0, 0.0, 1000.0)]);
        let terms = vec![parse_terms_line(TOWER).unwrap()];
        let result = multiunit_views(&input(&r), &terms, &MultiConfig::new(10.0)).unwrap();
        // ground_up 1000, gross 750, ceded 450, net 300.
        for (vi, loss) in [100usize, 75, 45, 30].iter().enumerate() {
            let v = &result.views[vi];
            assert_eq!(v[*loss], 0.02, "view {vi}");
            assert_eq!(v.iter().sum::<f64>(), 0.02);
        }
        assert_eq!(result.meta.grid_length, 101);
        assert_eq!(result.meta.max_mass_error, 0.0);
        assert!((result.meta.analytic_view_mean[2] - 0.02 * 450.0).abs() < 1e-12);
    }

    #[test]
    fn independent_event_is_direct_convolution_and_mass_is_rate() {
        let r = rows(&[
            (7, 0, 0.03, 200.0, 250.0, 0.0, 1000.0),
            (7, 1, 0.03, 500.0, 500.0, 0.0, 2000.0),
        ]);
        let terms = vec![UnitTerms::bare("a"), UnitTerms::bare("b")];
        let mut config = MultiConfig::new(10.0);
        config.views = vec![View::GroundUp];
        config.z_nodes = 4;
        let result = multiunit_views(&input(&r), &terms, &config).unwrap();
        let p1 = marginal_pmf(&UnitPlan::from_row(&classify_unit_row(0, &input(&r).spec(0), VariancePolicy::default(), &mut Diagnostics::default()).unwrap(), &input(&r).spec(0)), 10.0, 101);
        let p2 = marginal_pmf(&UnitPlan::from_row(&classify_unit_row(1, &input(&r).spec(1), VariancePolicy::default(), &mut Diagnostics::default()).unwrap(), &input(&r).spec(1)), 10.0, 201);
        let mut want = Vec::new();
        convolve_direct(&p1, &p2, &mut want);
        assert_eq!(result.meta.grid_length, 301);
        for (g, w) in result.views[0].iter().zip(&want) {
            assert!((g - 0.03 * w).abs() < 1e-16);
        }
        assert!((result.meta.view_mass[0] - 0.03).abs() < 1e-15);
        assert_eq!(result.meta.fft_paths, 0);
    }

    fn mixed_rows() -> Rows {
        rows(&[
            (1, 0, 0.010, 200.0, 100.0, 150.0, 1000.0),
            (1, 1, 0.010, 500.0, 300.0, 200.0, 2000.0),
            (1, 2, 0.010, 300.0, 50.0, 100.0, 800.0),
            (2, 0, 0.020, 250.0, 0.0, 200.0, 1000.0),
            (2, 2, 0.020, 100.0, 120.0, 0.0, 500.0),
            (3, 1, 0.005, 400.0, 80.0, 60.0, 1200.0),
            (4, 0, 0.003, 300.0, 0.0, 0.0, 1000.0),
            (4, 1, 0.003, 600.0, 500.0, 500.0, 2000.0), // 1000^2 >= 600 * 1400: clamped
            (4, 2, 0.003, 150.0, 60.0, 40.0, 600.0),
            (5, 0, 0.0, 200.0, 50.0, 50.0, 1000.0),
            (6, 2, 0.007, 700.0, 10.0, 10.0, 600.0),
            (6, 1, 0.007, 200.0, 40.0, 30.0, 900.0),
        ])
    }

    fn mixed_terms() -> Vec<UnitTerms> {
        vec![
            parse_terms_line(TOWER).unwrap(),
            parse_terms_line("unit b 1000 xs 0 occurrence net of 100 po 400 xs 100").unwrap(),
            UnitTerms::bare("c"),
        ]
    }

    #[test]
    fn direct_and_fft_paths_agree_and_masses_are_rates() {
        let r = mixed_rows();
        let terms = mixed_terms();
        let mut config = MultiConfig::new(10.0);
        config.z_nodes = 8;
        config.roll_up_diagnostic = true;
        config.convolution = Convolution::Direct;
        let direct = multiunit_views(&input(&r), &terms, &config).unwrap();
        config.convolution = Convolution::Fft;
        let fft = multiunit_views(&input(&r), &terms, &config).unwrap();
        assert!(direct.meta.fft_paths == 0 && fft.meta.direct_paths == 0);
        for (d, f) in direct.views.iter().zip(&fft.views) {
            for (x, y) in d.iter().zip(f) {
                assert!((x - y).abs() < 1e-14, "{x} vs {y}");
            }
        }
        let meta = &direct.meta;
        let total_rate = 0.010 + 0.020 + 0.005 + 0.003 + 0.007;
        assert!((meta.total_rate - total_rate).abs() < 1e-15);
        for mass in &meta.view_mass {
            assert!((mass - total_rate).abs() < 1e-13);
        }
        assert!(meta.max_mass_error < 1e-12);
        assert_eq!((meta.events, meta.valid_events, meta.valid_rows), (6, 5, 10));
        assert_eq!(meta.diagnostics.get(Category::ZeroRate).count, 1);
        assert_eq!(meta.diagnostics.get(Category::MeanExceedsMax).count, 1);
        assert_eq!(meta.diagnostics.get(Category::SigmaClamped).count, 1);
        let roll_up = meta.roll_up.as_ref().unwrap();
        assert_eq!(roll_up.iter().map(|e| e.event_row).collect::<Vec<_>>(), vec![0, 3, 6]);
        assert!(roll_up.iter().all(|e| e.sd_ratio.is_finite() && e.sd_ratio > 0.0));
        // Analytic means reconcile within the Q = 8 midpoint quadrature error.
        for vi in 0..4 {
            let rel = meta.view_mean_error[vi].abs() / meta.analytic_view_mean[vi].abs();
            assert!(rel < 5e-2, "view {vi}: {rel}");
        }
    }

    #[test]
    fn threads_and_chunking_are_bitwise_invariant() {
        let r = mixed_rows();
        let terms = mixed_terms();
        let mut config = MultiConfig::new(10.0);
        config.z_nodes = 8;
        config.chunk_rows = 3;
        let base = multiunit_views(&input(&r), &terms, &config).unwrap();
        for threads in [1, 2, 5] {
            config.threads = Some(threads);
            let other = multiunit_views(&input(&r), &terms, &config).unwrap();
            assert_eq!(base.views, other.views, "threads={threads}");
            assert_eq!(base.meta.total_rate, other.meta.total_rate);
            assert_eq!(base.meta.analytic_view_mean, other.meta.analytic_view_mean);
        }
        assert_eq!(event_chunks(&[(0, 3), (3, 5), (5, 6), (6, 9)], 3), vec![(0, 1), (1, 3), (3, 4)]);
    }

    #[test]
    fn structural_and_grid_errors() {
        let terms = vec![UnitTerms::bare("a"), UnitTerms::bare("b")];
        let row = |e, u| (e, u, 0.01, 300.0, 40.0, 30.0, 1000.0);
        let regrouped = rows(&[row(1, 0), row(2, 0), row(1, 1)]);
        assert!(matches!(
            multiunit_views(&input(&regrouped), &terms, &MultiConfig::new(10.0)),
            Err(Error::MultiUnitStructure(_))
        ));
        let mut varying = rows(&[row(1, 0), row(1, 1)]);
        varying.rate[1] = 0.02;
        assert!(matches!(
            multiunit_views(&input(&varying), &terms, &MultiConfig::new(10.0)),
            Err(Error::MultiUnitStructure(_))
        ));
        let duplicate = rows(&[row(1, 0), row(1, 0)]);
        assert!(matches!(
            multiunit_views(&input(&duplicate), &terms, &MultiConfig::new(10.0)),
            Err(Error::MultiUnitStructure(_))
        ));
        let out_of_range = rows(&[row(1, 5)]);
        assert!(matches!(
            multiunit_views(&input(&out_of_range), &terms, &MultiConfig::new(10.0)),
            Err(Error::UnitCodeOutOfRange { row: 0, code: 5, .. })
        ));
        let pair = rows(&[row(1, 0), row(1, 1)]);
        let mut config = MultiConfig::new(10.0);
        config.grid_length = Some(150);
        assert!(matches!(
            multiunit_views(&input(&pair), &terms, &config),
            Err(Error::GridOverflow { required: 201, .. })
        ));
        config.grid_length = None;
        config.views.clear();
        assert!(matches!(multiunit_views(&input(&pair), &terms, &config), Err(Error::NoViews)));
        let misaligned = vec![parse_terms_line("unit a 755 xs 0").unwrap(), UnitTerms::bare("b")];
        assert!(matches!(
            multiunit_views(&input(&pair), &misaligned, &MultiConfig::new(10.0)),
            Err(Error::Terms(_))
        ));
    }
}
