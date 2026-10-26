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
//! 2. materializes only the distinct row sets a unit's terms produce (base
//!    for ground_up, gross after policy terms, ceded and net after
//!    reinsurance; a unit without policy terms has gross = ground_up, one
//!    without reinsurance has net = gross and a ceded atom at zero that
//!    drops out of the ceded convolution entirely);
//! 3. convolves across units per view, sharing work between views: bare
//!    units (no terms at all) form one block convolved once per z node and
//!    reused by ground_up, gross and net, and on the FFT path every (unit,
//!    row set, z) spectrum is transformed once at one event-wide size and
//!    multiplied into each view, with the spectra accumulated over z so one
//!    inverse per view suffices. Direct or FFT is chosen per event from the
//!    operations each path would actually run (or forced);
//! 4. deposits the event's per-view pmf into the chunk's histograms weighted
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

use crate::config::VariancePolicy;
use crate::convolve::{FftCache, convolve_direct, fft_cost, fft_size};
use crate::diagnostics::{Category, Diagnostics, MAX_EXAMPLES};
use crate::error::Error;
use crate::multiunit::{
    LayerMeans, RowClass, UnitKind, UnitPlan, UnitSpec, classify_unit_row, conditional_matrix,
    grid_length_for, marginal_pmf, prepare_unit,
};
use crate::quadrature::{ZRule, z_quadrature};
use crate::sum::{Neumaier, grid_moment, sum};
use crate::terms::{UnitTerms, View, ViewMap, check_grid_alignment};

/// Default rows per parallel chunk for the multi-unit route: whole events,
/// small enough that a 100k-row portfolio still spreads across every core.
pub const DEFAULT_MULTI_CHUNK_ROWS: usize = 2048;

/// Measured ratio of FFT-path wall time to its flop estimate relative to the
/// direct path's (the vectorized axpy chains run near peak, the FFT path
/// pays for scratch, complex products and plan dispatch); calibrated on the
/// 1.11.0 benchmark so `Auto` tracks the faster path.
const FFT_COST_FACTOR: f64 = 3.0;

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
    /// Direct or FFT per event by estimated cost.
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
            chunk_rows: DEFAULT_MULTI_CHUNK_ROWS,
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
    /// Events convolved by each path.
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

/// Row-set ids: the distinct transformed rows a unit can need.
const SET_BASE: usize = 0;
const SET_GROSS: usize = 1;
const SET_CEDED: usize = 2;
const SET_NET: usize = 3;

/// Row set a view uses for a unit with the given terms; None is the ceded
/// atom at zero, which contributes nothing to the convolution.
fn rowset_for(view: View, terms: &UnitTerms) -> Option<usize> {
    match view {
        View::GroundUp => Some(SET_BASE),
        View::Gross => Some(if terms.has_policy_terms() { SET_GROSS } else { SET_BASE }),
        View::Ceded => {
            if terms.has_reinsurance() {
                Some(SET_CEDED)
            } else {
                None
            }
        }
        View::Net => Some(if terms.has_reinsurance() {
            SET_NET
        } else if terms.has_policy_terms() {
            SET_GROSS
        } else {
            SET_BASE
        }),
    }
}

/// Whether the event needs the shared bare block: some unit is bare and some
/// computed view other than ceded (which never uses bare units) reads it.
/// A view's own `uses_shared` is true only when this is.
fn uses_shared_block(units: &[LiveUnit], computed: &[View]) -> bool {
    units.iter().any(|u| u.bare) && computed.iter().any(|&v| v != View::Ceded)
}

fn view_of_set(set: usize) -> View {
    match set {
        SET_GROSS => View::Gross,
        SET_CEDED => View::Ceded,
        SET_NET => View::Net,
        _ => View::GroundUp,
    }
}

/// One contributing unit of an event inside the worker.
struct LiveUnit {
    row: usize,
    code: usize,
    plan: UnitPlan,
    /// Number of base rows: Q when the unit depends on the shared factor,
    /// 1 otherwise (independent beta units and atoms).
    rows: usize,
    /// Row sets by id (base, gross, ceded, net); unused sets stay empty.
    sets: [Vec<f64>; 4],
    lens: [usize; 4],
    /// No terms at all: identical rows in ground_up, gross and net.
    bare: bool,
}

impl LiveUnit {
    fn set(&self, id: usize) -> &[f64] {
        &self.sets[id]
    }

    fn row(&self, id: usize, zi: usize) -> &[f64] {
        let len = self.lens[id];
        let q = if self.rows == 1 { 0 } else { zi };
        &self.sets[id][q * len..(q + 1) * len]
    }
}

/// Spectra of a unit's row sets: `rows x spectrum_len` complex per set.
#[derive(Default)]
struct UnitSpectra {
    sets: [Vec<Complex<f64>>; 4],
}

impl UnitSpectra {
    fn row(&self, id: usize, rows: usize, zi: usize, len: usize) -> &[Complex<f64>] {
        let q = if rows == 1 { 0 } else { zi };
        &self.sets[id][q * len..(q + 1) * len]
    }
}

/// Per-worker state: quadrature, FFT plans, compiled view maps, scratch.
struct Worker<'a> {
    z: &'a [f64],
    w: &'a [f64],
    h: f64,
    fft: FftCache,
    maps: HashMap<(usize, usize, View), ViewMap>,
    // Scratch reused across events.
    acc: Vec<f64>,
    tmp: Vec<f64>,
    pmf: Vec<f64>,
    shared_indep: Vec<f64>,
    shared_corr: Vec<Vec<f64>>,
    spectra: Vec<UnitSpectra>,
    shared_spec: Vec<Vec<Complex<f64>>>,
    spec_acc: Vec<Complex<f64>>,
    spec_prod: Vec<Complex<f64>>,
    spec_tmp: Vec<Complex<f64>>,
}

/// Cost of chaining direct convolutions over rows of the given lengths,
/// starting from a row of length `start`; returns (cost, final length).
fn chain_cost(start: usize, lengths: impl Iterator<Item = usize>) -> (f64, usize) {
    let mut acc = start;
    let mut cost = 0.0;
    for len in lengths {
        cost += (acc * len) as f64;
        acc += len - 1;
    }
    (cost, acc)
}

impl Worker<'_> {
    fn view_map(&mut self, unit_code: usize, terms: &UnitTerms, view: View, n: usize) -> ViewMap {
        if self.maps.len() > 4096 {
            self.maps.clear();
        }
        self.maps
            .entry((unit_code, n, view))
            .or_insert_with(|| ViewMap::compile(terms, view, self.h, n))
            .clone()
    }

    /// Materialize the row sets the computed views need for every unit.
    fn materialize(&mut self, units: &mut [LiveUnit], terms: &[UnitTerms], computed: &[View]) {
        for u in units.iter_mut() {
            let t = &terms[u.code];
            let mut needed = [false; 4];
            for &view in computed {
                if let Some(id) = rowset_for(view, t) {
                    needed[id] = true;
                }
            }
            for id in [SET_GROSS, SET_CEDED, SET_NET] {
                if !needed[id] {
                    u.sets[id].clear();
                    u.lens[id] = 0;
                    continue;
                }
                let map = self.view_map(u.code, t, view_of_set(id), u.lens[SET_BASE]);
                let len = map.support_len;
                u.lens[id] = len;
                let n = u.lens[SET_BASE];
                let (base, set) = {
                    let (left, right) = u.sets.split_at_mut(id);
                    (&left[SET_BASE], &mut right[0])
                };
                set.clear();
                set.resize(u.rows * len, 0.0);
                for (src, dst) in base.chunks_exact(n).zip(set.chunks_exact_mut(len)) {
                    map.apply(src, dst);
                }
            }
        }
    }

    /// Estimated multiply-add counts of the direct and FFT paths for the
    /// computed views (mirrors the loops in `direct_views` / `fft_views`).
    fn estimate(&self, units: &[LiveUnit], terms: &[UnitTerms], computed: &[View]) -> (f64, f64) {
        let q = self.z.len();
        let bare_exists = units.iter().any(|u| u.bare);
        let shared_needed = uses_shared_block(units, computed);
        let mut direct = 0.0;
        // Shared bare block, charged only when some computed view reads it.
        let (c_ind, shared_ind_len) = chain_cost(
            1,
            units.iter().filter(|u| u.bare && u.rows == 1).map(|u| u.lens[SET_BASE]),
        );
        let (c_corr, shared_corr_len) = chain_cost(
            1,
            units.iter().filter(|u| u.bare && u.rows > 1).map(|u| u.lens[SET_BASE]),
        );
        if shared_needed {
            direct += c_ind + q as f64 * c_corr;
        }
        let mut max_total = 1;
        let mut forward = 0usize;
        let mut products = 0.0;
        for u in units {
            let t = &terms[u.code];
            let mut seen = [false; 4];
            for &view in computed {
                if let Some(id) = rowset_for(view, t)
                    && !seen[id]
                {
                    seen[id] = true;
                    forward += u.rows;
                }
            }
        }
        for &view in computed {
            let uses_shared = bare_exists && view != View::Ceded;
            let pairs: Vec<(usize, usize)> = units
                .iter()
                .enumerate()
                .filter(|(_, u)| !u.bare)
                .filter_map(|(i, u)| rowset_for(view, &terms[u.code]).map(|id| (i, id)))
                .collect();
            let total = pairs.iter().map(|&(i, id)| units[i].lens[id] - 1).sum::<usize>()
                + if uses_shared { shared_ind_len - 1 + shared_corr_len - 1 } else { 0 }
                + 1;
            max_total = max_total.max(total);
            let start = if uses_shared { shared_ind_len } else { 1 };
            let (c, ind_len) =
                chain_cost(start, pairs.iter().filter(|&&(i, _)| units[i].rows == 1).map(|&(i, id)| units[i].lens[id]));
            direct += c;
            let corr: Vec<usize> = pairs
                .iter()
                .filter(|&&(i, _)| units[i].rows > 1)
                .map(|&(i, id)| units[i].lens[id])
                .collect();
            let any_corr = !corr.is_empty() || (uses_shared && shared_corr_len > 1);
            if any_corr {
                let (c1, l1) = chain_cost(1, corr.iter().cloned());
                let mut per_z = c1;
                let mut len = l1;
                if uses_shared && shared_corr_len > 1 {
                    per_z += (len * shared_corr_len) as f64;
                    len += shared_corr_len - 1;
                }
                per_z += (len * ind_len) as f64;
                direct += q as f64 * per_z;
                products += (q * (pairs.len() + usize::from(uses_shared))) as f64;
            } else {
                products += (pairs.len() + usize::from(uses_shared)) as f64;
            }
        }
        let n = fft_size(max_total);
        let half = (n / 2 + 1) as f64;
        let shared_products = if shared_needed {
            (units.iter().filter(|u| u.bare).count() * q) as f64
        } else {
            0.0
        };
        let fft = (forward + computed.len()) as f64 * fft_cost(n) + (products + shared_products) * half * 3.0;
        (direct, fft)
    }

    /// Direct path: fill `pmfs[v]` for every computed view.
    fn direct_views(&mut self, units: &[LiveUnit], terms: &[UnitTerms], computed: &[View], pmfs: &mut [Vec<f64>]) {
        let q = self.z.len();
        let bare_exists = units.iter().any(|u| u.bare);
        let shared_corr_units: Vec<usize> = (0..units.len()).filter(|&i| units[i].bare && units[i].rows > 1).collect();
        // Shared bare block: independent part once, correlated part per z;
        // skipped when no computed view reads it (ceded alone).
        if uses_shared_block(units, computed) {
            self.shared_indep.clear();
            self.shared_indep.push(1.0);
            for u in units.iter().filter(|u| u.bare && u.rows == 1) {
                convolve_direct(&self.shared_indep, u.set(SET_BASE), &mut self.tmp);
                std::mem::swap(&mut self.shared_indep, &mut self.tmp);
            }
            self.shared_corr.resize_with(q, Vec::new);
            if shared_corr_units.is_empty() {
                for s in self.shared_corr.iter_mut() {
                    s.clear();
                }
            } else {
                for zi in 0..q {
                    self.acc.clear();
                    self.acc.push(1.0);
                    for &i in &shared_corr_units {
                        convolve_direct(&self.acc, units[i].row(SET_BASE, zi), &mut self.tmp);
                        std::mem::swap(&mut self.acc, &mut self.tmp);
                    }
                    std::mem::swap(&mut self.shared_corr[zi], &mut self.acc);
                }
            }
        }
        for (vi, &view) in computed.iter().enumerate() {
            let uses_shared = bare_exists && view != View::Ceded;
            let pairs: Vec<(usize, usize)> = units
                .iter()
                .enumerate()
                .filter(|(_, u)| !u.bare)
                .filter_map(|(i, u)| rowset_for(view, &terms[u.code]).map(|id| (i, id)))
                .collect();
            let total = pairs.iter().map(|&(i, id)| units[i].lens[id] - 1).sum::<usize>()
                + if uses_shared {
                    self.shared_indep.len() - 1 + self.shared_corr[0].len().max(1) - 1
                } else {
                    0
                }
                + 1;
            // Independent chain once.
            self.tmp.clear();
            if uses_shared {
                self.pmf.clear();
                self.pmf.extend_from_slice(&self.shared_indep);
            } else {
                self.pmf.clear();
                self.pmf.push(1.0);
            }
            for &(i, id) in pairs.iter().filter(|&&(i, _)| units[i].rows == 1) {
                convolve_direct(&self.pmf, units[i].set(id), &mut self.tmp);
                std::mem::swap(&mut self.pmf, &mut self.tmp);
            }
            let indep_len = self.pmf.len();
            let corr: Vec<(usize, usize)> = pairs.iter().cloned().filter(|&(i, _)| units[i].rows > 1).collect();
            let shared_corr_live = uses_shared && !shared_corr_units.is_empty();
            let out = &mut pmfs[vi];
            out.clear();
            out.resize(total, 0.0);
            if corr.is_empty() && !shared_corr_live {
                out[..indep_len].copy_from_slice(&self.pmf);
                continue;
            }
            for zi in 0..q {
                self.acc.clear();
                self.acc.push(1.0);
                for &(i, id) in &corr {
                    convolve_direct(&self.acc, units[i].row(id, zi), &mut self.tmp);
                    std::mem::swap(&mut self.acc, &mut self.tmp);
                }
                if shared_corr_live {
                    convolve_direct(&self.acc, &self.shared_corr[zi], &mut self.tmp);
                    std::mem::swap(&mut self.acc, &mut self.tmp);
                }
                if indep_len > 1 {
                    convolve_direct(&self.acc, &self.pmf, &mut self.tmp);
                    std::mem::swap(&mut self.acc, &mut self.tmp);
                }
                let wq = self.w[zi];
                for (o, a) in out.iter_mut().zip(&self.acc) {
                    *o += wq * a;
                }
            }
        }
    }

    /// FFT path: one event-wide size, every (unit, row set, z) spectrum
    /// transformed once, spectra accumulated over z, one inverse per view.
    fn fft_views(&mut self, units: &[LiveUnit], terms: &[UnitTerms], computed: &[View], pmfs: &mut [Vec<f64>]) {
        let q = self.z.len();
        let bare_exists = units.iter().any(|u| u.bare);
        // Event-wide size from the largest view total.
        let mut max_total = 1;
        let mut totals = Vec::with_capacity(computed.len());
        for &view in computed {
            let total = units
                .iter()
                .filter_map(|u| rowset_for(view, &terms[u.code]).map(|id| u.lens[id] - 1))
                .sum::<usize>()
                + 1;
            totals.push(total);
            max_total = max_total.max(total);
        }
        let n = fft_size(max_total);
        let half = FftCache::spectrum_len(n);
        // Spectra of every needed row set.
        self.spectra.resize_with(units.len(), UnitSpectra::default);
        for (i, u) in units.iter().enumerate() {
            let t = &terms[u.code];
            let mut needed = [false; 4];
            for &view in computed {
                if let Some(id) = rowset_for(view, t) {
                    needed[id] = true;
                }
            }
            for id in 0..4 {
                let set = &mut self.spectra[i].sets[id];
                set.clear();
                if !needed[id] {
                    continue;
                }
                let len = u.lens[id];
                for r in 0..u.rows {
                    self.fft.spectrum(&u.sets[id][r * len..(r + 1) * len], n, &mut self.spec_tmp);
                    set.extend_from_slice(&self.spec_tmp);
                }
            }
        }
        // Shared bare block spectrum per z, only when some computed view
        // reads it: with ceded alone the bare units' spectra were never built.
        let shared_rows = if units.iter().any(|u| u.bare && u.rows > 1) { q } else { 1 };
        self.shared_spec.resize_with(shared_rows, Vec::new);
        if uses_shared_block(units, computed) {
            for zi in 0..shared_rows {
                let s = &mut self.shared_spec[zi];
                s.clear();
                s.resize(half, Complex::new(1.0, 0.0));
                for (i, u) in units.iter().enumerate().filter(|(_, u)| u.bare) {
                    for (a, b) in s.iter_mut().zip(self.spectra[i].row(SET_BASE, u.rows, zi, half)) {
                        *a *= b;
                    }
                }
            }
        }
        for (vi, &view) in computed.iter().enumerate() {
            let uses_shared = bare_exists && view != View::Ceded;
            let pairs: Vec<(usize, usize)> = units
                .iter()
                .enumerate()
                .filter(|(_, u)| !u.bare)
                .filter_map(|(i, u)| rowset_for(view, &terms[u.code]).map(|id| (i, id)))
                .collect();
            let any_corr = pairs.iter().any(|&(i, _)| units[i].rows > 1) || (uses_shared && shared_rows > 1);
            let nodes = if any_corr { q } else { 1 };
            self.spec_acc.clear();
            self.spec_acc.resize(half, Complex::new(0.0, 0.0));
            for zi in 0..nodes {
                self.spec_prod.clear();
                if uses_shared {
                    self.spec_prod.extend_from_slice(&self.shared_spec[if shared_rows > 1 { zi } else { 0 }]);
                } else {
                    self.spec_prod.resize(half, Complex::new(1.0, 0.0));
                }
                for &(i, id) in &pairs {
                    for (a, b) in self.spec_prod.iter_mut().zip(self.spectra[i].row(id, units[i].rows, zi, half)) {
                        *a *= b;
                    }
                }
                let wq = if any_corr { self.w[zi] } else { 1.0 };
                for (a, p) in self.spec_acc.iter_mut().zip(&self.spec_prod) {
                    *a += wq * p;
                }
            }
            let mut spec = std::mem::take(&mut self.spec_acc);
            self.fft.inverse_into(&mut spec, n, totals[vi], &mut pmfs[vi]);
            self.spec_acc = spec;
        }
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

    // Pass 1: structure, classification, event spans.
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
    // Event span in buckets: each live unit's pmf has
    // `grid_length_for(support, h)` nodes, so the convolution across units
    // has span + 1. Summing supports in money first undercounts whenever a
    // support is off the grid.
    let mut event_span = vec![0usize; events.len()];
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
        let mut span = 0;
        for row in start..end {
            let class = classify_unit_row(row, &input.spec(row), config.variance_policy, &mut diagnostics)?;
            classes[row] = class;
            if class.kind != UnitKind::Drop {
                event_live[e] = true;
                // Same support as `UnitPlan::support`, which sizes the unit's rows in pass 2.
                let support = if class.kind == UnitKind::Atom { class.mu } else { input.max_loss[row] };
                span += grid_length_for(support, h) - 1;
            }
        }
        event_span[e] = span;
    }
    let max_span = event_span.iter().copied().max().unwrap_or(0);
    let auto_length = max_span + 1;
    let n = match config.grid_length {
        None => auto_length,
        Some(n) => {
            if auto_length > n {
                let example_rows = events
                    .iter()
                    .enumerate()
                    .filter(|(e, _)| event_live[*e] && event_span[*e] + 1 > n)
                    .map(|(_, &(start, _))| start)
                    .take(MAX_EXAMPLES)
                    .collect();
                return Err(Error::GridOverflow {
                    grid_length: n,
                    bucket_size: h,
                    top: (n - 1) as f64 * h,
                    max_loss: max_span as f64 * h,
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
            acc: Vec::new(),
            tmp: Vec::new(),
            pmf: Vec::new(),
            shared_indep: Vec::new(),
            shared_corr: Vec::new(),
            spectra: Vec::new(),
            shared_spec: Vec::new(),
            spec_acc: Vec::new(),
            spec_prod: Vec::new(),
            spec_tmp: Vec::new(),
        };
        let mut pmfs: Vec<Vec<f64>> = (0..computed.len()).map(|_| Vec::new()).collect();
        let mut units: Vec<LiveUnit> = Vec::new();
        for e in first..last {
            if !event_live[e] {
                continue;
            }
            let (start, end) = events[e];
            let lam = input.rate[start];
            partial.valid_events += 1;
            rate_acc.add(lam);
            // Live units with their base conditional rows.
            units.clear();
            let mut mu_sum = 0.0;
            for row in start..end {
                let class = classes[row];
                if class.kind == UnitKind::Drop {
                    continue;
                }
                let spec = input.spec(row);
                let plan = UnitPlan::from_row(&class, &spec);
                let n_unit = grid_length_for(plan.support(), h);
                let nodes: &[f64] = if plan.depends_on_factor() { &z } else { &[0.0] };
                let base = conditional_matrix(&plan, h, n_unit, nodes).map_err(|_| Error::KernelFailure {
                    row,
                    alpha: plan.alpha,
                    beta: plan.beta,
                })?;
                mu_sum += plan.mean_loss;
                let code = input.unit_code[row] as usize;
                let t = &terms[code];
                units.push(LiveUnit {
                    row,
                    code,
                    plan,
                    rows: nodes.len(),
                    sets: [base, Vec::new(), Vec::new(), Vec::new()],
                    lens: [n_unit, 0, 0, 0],
                    bare: !t.has_policy_terms() && !t.has_reinsurance(),
                });
            }
            partial.valid_rows += units.len();
            aal_acc.add(lam * mu_sum);
            worker.materialize(&mut units, terms, &computed);
            let use_fft = match config.convolution {
                Convolution::Direct => false,
                Convolution::Fft => true,
                Convolution::Auto => {
                    let (direct, fft) = worker.estimate(&units, terms, &computed);
                    fft * FFT_COST_FACTOR < direct
                }
            };
            if use_fft {
                worker.fft_views(&units, terms, &computed, &mut pmfs);
                partial.fft_paths += 1;
            } else {
                worker.direct_views(&units, terms, &computed, &mut pmfs);
                partial.direct_paths += 1;
            }
            let layer_means: Vec<LayerMeans> = units.iter().map(|u| LayerMeans::new(&u.plan)).collect();
            for (vi, &view) in computed.iter().enumerate() {
                let pmf = &pmfs[vi];
                let mass: f64 = pmf.iter().sum();
                partial.max_mass_error = partial.max_mass_error.max((mass - 1.0).abs());
                if vi < n_views {
                    // Pass 1 sized the grid to every event's span; a longer
                    // pmf is a sizing bug and must not lose its tail silently.
                    assert!(
                        pmf.len() <= n,
                        "event at row {start}: pmf of {} nodes exceeds the grid of {n}",
                        pmf.len()
                    );
                    for (dst, &p) in partial.views[vi].iter_mut().zip(pmf) {
                        *dst += lam * p;
                    }
                    let analytic: f64 = units
                        .iter()
                        .zip(&layer_means)
                        .map(|(u, lm)| lm.view_mean(&terms[u.code], view))
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
                    let rolled_pmf = marginal_pmf(&rolled_plan, h, pmf.len());
                    let (sd_exact, sd_rolled) = (sd_of(pmf, h), sd_of(&rolled_pmf, h));
                    let (q_exact, q_rolled) = (quantile_of(pmf, h, 0.99), quantile_of(&rolled_pmf, h, 0.99));
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
        let plan = |row: usize| {
            let spec = input(&r).spec(row);
            let class = classify_unit_row(row, &spec, VariancePolicy::clamp_default(), &mut Diagnostics::default()).unwrap();
            UnitPlan::from_row(&class, &spec)
        };
        let p1 = marginal_pmf(&plan(0), 10.0, 101);
        let p2 = marginal_pmf(&plan(1), 10.0, 201);
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
            (1, 3, 0.010, 150.0, 40.0, 0.0, 600.0),
            (2, 0, 0.020, 250.0, 0.0, 200.0, 1000.0),
            (2, 2, 0.020, 100.0, 120.0, 0.0, 500.0),
            (3, 1, 0.005, 400.0, 80.0, 60.0, 1200.0),
            (4, 0, 0.003, 300.0, 0.0, 0.0, 1000.0),
            (4, 1, 0.003, 600.0, 500.0, 500.0, 2000.0), // 1000^2 >= 600 * 1400: two-point
            (4, 2, 0.003, 150.0, 60.0, 40.0, 600.0),
            (4, 3, 0.003, 200.0, 30.0, 70.0, 700.0),
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
            UnitTerms::bare("d"),
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
            assert!(f.iter().all(|&y| y >= 0.0), "negative FFT weight");
        }
        let meta = &direct.meta;
        let total_rate = 0.010 + 0.020 + 0.005 + 0.003 + 0.007;
        assert!((meta.total_rate - total_rate).abs() < 1e-15);
        for mass in &meta.view_mass {
            assert!((mass - total_rate).abs() < 1e-13);
        }
        assert!(meta.max_mass_error < 1e-12);
        assert_eq!((meta.events, meta.valid_events, meta.valid_rows), (6, 5, 12));
        assert_eq!(meta.diagnostics.get(Category::ZeroRate).count, 1);
        assert_eq!(meta.diagnostics.get(Category::MeanExceedsMax).count, 1);
        assert_eq!(meta.diagnostics.get(Category::VarianceBernoulli).count, 1);
        assert_eq!(meta.diagnostics.get(Category::SigmaClamped).count, 0);
        let roll_up = meta.roll_up.as_ref().unwrap();
        assert_eq!(roll_up.iter().map(|e| e.event_row).collect::<Vec<_>>(), vec![0, 4, 7]);
        assert!(roll_up.iter().all(|e| e.sd_ratio.is_finite() && e.sd_ratio > 0.0));
        // Analytic means reconcile within the Q = 8 midpoint quadrature error.
        for vi in 0..4 {
            let rel = meta.view_mean_error[vi].abs() / meta.analytic_view_mean[vi].abs();
            assert!(rel < 5e-2, "view {vi}: {rel}");
        }
        // Auto picks one of the two and reproduces it.
        config.convolution = Convolution::Auto;
        let auto = multiunit_views(&input(&r), &terms, &config).unwrap();
        for (a, d) in auto.views.iter().zip(&direct.views) {
            for (x, y) in a.iter().zip(d) {
                assert!((x - y).abs() < 1e-14);
            }
        }
        // The clamp stays selectable (deprecated, removed at 3.0.0).
        config.variance_policy = VariancePolicy::clamp_default();
        let clamped = multiunit_views(&input(&r), &terms, &config).unwrap();
        assert_eq!(clamped.meta.diagnostics.get(Category::SigmaClamped).count, 1);
        assert_eq!(clamped.meta.diagnostics.get(Category::VarianceBernoulli).count, 0);
    }

    #[test]
    fn ceded_only_fft_skips_the_unused_bare_block() {
        // Ceded never uses bare units, so their spectra are never built; the
        // FFT path must not form the shared block from them (1.11.2).
        let r = rows(&[
            (1, 0, 0.01, 300.0, 40.0, 30.0, 1000.0),
            (1, 1, 0.01, 200.0, 50.0, 60.0, 800.0),
            (1, 2, 0.01, 250.0, 0.0, 120.0, 900.0),
        ]);
        let terms = vec![
            parse_terms_line("unit a 1000 xs 0 occurrence net of 50% po 500 xs 0").unwrap(),
            UnitTerms::bare("b"),
            UnitTerms::bare("c"),
        ];
        let mut config = MultiConfig::new(10.0);
        config.views = vec![View::Ceded];
        config.z_nodes = 8;
        config.convolution = Convolution::Fft;
        let fft = multiunit_views(&input(&r), &terms, &config).unwrap();
        config.convolution = Convolution::Direct;
        let direct = multiunit_views(&input(&r), &terms, &config).unwrap();
        assert_eq!((fft.meta.fft_paths, direct.meta.direct_paths), (1, 1));
        for (x, y) in fft.views[0].iter().zip(&direct.views[0]) {
            assert!((x - y).abs() < 1e-15, "{x} vs {y}");
            assert!(*x >= 0.0, "negative FFT weight {x}");
        }
        assert!((fft.meta.view_mass[0] - 0.01).abs() < 1e-15);
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
