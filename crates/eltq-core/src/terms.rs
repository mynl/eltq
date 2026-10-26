//! DecL-style unit terms: grammar, parsing, view maps, and their compiled
//! grid form. Rust twin of `eltq.terms` (the Python module documents the
//! semantics; the parse goldens in `fixtures/golden/terms_parse.json` hold
//! both parsers to the same behavior).
//!
//! ```text
//! unit <unit_id> [<limit> xs <ded>] [occurrence net of <clause> (and <clause>)*]
//! clause := [<share> po] <limit> xs <attach>
//! share  := <number> | <number>%
//! ```
//!
//! Views of the ground-up loss X: `ground_up` = X; `gross` = min(max(X -
//! ded, 0), limit); `ceded` = sum over clauses of share * min(max(gross -
//! attach, 0), limit); `net` = gross - ceded. A percent share is the
//! fraction of the layer, a plain-number share an amount of the layer
//! limit. Clauses may overlap only while their combined share stays at or
//! below 100% in every band. Keywords match `aggregate`'s `decl.lark`
//! (`xs`, `po` = part of, `occurrence net of`, `and`).
//!
//! A compiled [`ViewMap`] turns a view into a gather-scatter pass on the
//! grid: source bucket k (loss k h) deposits its mass at v(k h) by the
//! linear mass- and mean-preserving rule, at most two `(index, weight)`
//! pairs. Breakpoints must be multiples of h, so gross is a pure index map
//! and only fractional shares land off grid.

use crate::error::Error;

/// Keyword vocabulary (also written to the parse goldens for alignment).
pub const KEYWORDS: [&str; 7] = ["and", "occurrence", "of", "po", "net", "unit", "xs"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum View {
    GroundUp,
    Gross,
    Ceded,
    Net,
}

impl View {
    pub const ALL: [View; 4] = [View::GroundUp, View::Gross, View::Ceded, View::Net];

    pub fn parse(name: &str) -> Result<Self, Error> {
        match name {
            "ground_up" => Ok(View::GroundUp),
            "gross" => Ok(View::Gross),
            "ceded" => Ok(View::Ceded),
            "net" => Ok(View::Net),
            other => Err(Error::Terms(format!(
                "unknown view {other:?}; expected ground_up, gross, ceded, or net"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            View::GroundUp => "ground_up",
            View::Gross => "gross",
            View::Ceded => "ceded",
            View::Net => "net",
        }
    }
}

/// One occurrence reinsurance layer applied to the unit's gross loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clause {
    /// Fraction of the layer in (0, 1].
    pub share: f64,
    pub limit: f64,
    pub attach: f64,
}

impl Clause {
    pub fn new(share: f64, limit: f64, attach: f64) -> Result<Self, Error> {
        if !(share > 0.0 && share <= 1.0) {
            return Err(Error::Terms(format!("clause share must be in (0, 1], got {share}")));
        }
        if !(limit.is_finite() && limit > 0.0) {
            return Err(Error::Terms(format!(
                "clause limit must be finite and positive, got {limit}"
            )));
        }
        if !(attach.is_finite() && attach >= 0.0) {
            return Err(Error::Terms(format!(
                "clause attachment must be finite and >= 0, got {attach}"
            )));
        }
        Ok(Clause {
            share,
            limit,
            attach,
        })
    }

    #[inline]
    pub fn apply(&self, gross: f64) -> f64 {
        self.share * (gross - self.attach).max(0.0).min(self.limit)
    }

    fn text(&self) -> String {
        let share = if self.share == 1.0 {
            String::new()
        } else {
            format!("{}% po ", fmt_g15(100.0 * self.share))
        };
        format!("{share}{} xs {}", fmt_g15(self.limit), fmt_g15(self.attach))
    }
}

/// Policy layer plus occurrence reinsurance clauses for one unit.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitTerms {
    pub unit_id: String,
    /// Policy limit; infinite when absent.
    pub limit: f64,
    pub ded: f64,
    pub clauses: Vec<Clause>,
}

impl UnitTerms {
    pub fn new(unit_id: &str, limit: f64, ded: f64, clauses: Vec<Clause>) -> Result<Self, Error> {
        if unit_id.is_empty() {
            return Err(Error::Terms("unit_id must be nonempty".to_string()));
        }
        if limit.is_nan() || limit <= 0.0 {
            return Err(Error::Terms(format!("unit {unit_id}: limit must be positive, got {limit}")));
        }
        if !(ded.is_finite() && ded >= 0.0) {
            return Err(Error::Terms(format!(
                "unit {unit_id}: ded must be finite and >= 0, got {ded}"
            )));
        }
        let terms = UnitTerms {
            unit_id: unit_id.to_string(),
            limit,
            ded,
            clauses,
        };
        terms.check_overlap()?;
        Ok(terms)
    }

    /// No policy terms and no reinsurance.
    pub fn bare(unit_id: &str) -> Self {
        UnitTerms {
            unit_id: unit_id.to_string(),
            limit: f64::INFINITY,
            ded: 0.0,
            clauses: Vec::new(),
        }
    }

    /// Combined share must not exceed 100% of gross in any band; beyond
    /// that ceded exceeds gross and net decreases or goes negative.
    fn check_overlap(&self) -> Result<(), Error> {
        let mut points: Vec<f64> = self
            .clauses
            .iter()
            .flat_map(|c| [c.attach, c.attach + c.limit])
            .collect();
        points.sort_by(|a, b| a.partial_cmp(b).unwrap());
        points.dedup();
        for pair in points.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            let share: f64 = self
                .clauses
                .iter()
                .filter(|c| c.attach <= lo && hi <= c.attach + c.limit)
                .map(|c| c.share)
                .sum();
            if share > 1.0 + 1e-12 {
                return Err(Error::Terms(format!(
                    "unit {}: clauses cede {}% of gross between {} and {}; \
                     combined share must not exceed 100%",
                    self.unit_id,
                    fmt_g15(100.0 * share),
                    fmt_g15(lo),
                    fmt_g15(hi)
                )));
            }
        }
        Ok(())
    }

    pub fn has_policy_terms(&self) -> bool {
        self.ded > 0.0 || self.limit.is_finite()
    }

    pub fn has_reinsurance(&self) -> bool {
        !self.clauses.is_empty()
    }

    #[inline]
    pub fn gross(&self, x: f64) -> f64 {
        (x - self.ded).max(0.0).min(self.limit)
    }

    #[inline]
    pub fn ceded(&self, x: f64) -> f64 {
        let g = self.gross(x);
        self.clauses.iter().map(|c| c.apply(g)).sum()
    }

    #[inline]
    pub fn net(&self, x: f64) -> f64 {
        self.gross(x) - self.ceded(x)
    }

    #[inline]
    pub fn view(&self, view: View, x: f64) -> f64 {
        match view {
            View::GroundUp => x,
            View::Gross => self.gross(x),
            View::Ceded => self.ceded(x),
            View::Net => self.net(x),
        }
    }

    /// Finite monetary breakpoints that must sit on the grid.
    pub fn breakpoints(&self) -> Vec<f64> {
        let mut points = vec![self.ded];
        if self.limit.is_finite() {
            points.push(self.limit);
        }
        for c in &self.clauses {
            points.push(c.limit);
            points.push(c.attach);
        }
        points
    }

    /// Canonical one-line rendering in the grammar (matches `eltq.terms`).
    pub fn text(&self) -> String {
        let mut parts = vec![format!("unit {}", self.unit_id)];
        if self.has_policy_terms() {
            parts.push(format!("{} xs {}", fmt_g15(self.limit), fmt_g15(self.ded)));
        }
        if !self.clauses.is_empty() {
            let clauses: Vec<String> = self.clauses.iter().map(Clause::text).collect();
            parts.push(format!("occurrence net of {}", clauses.join(" and ")));
        }
        parts.join(" ")
    }
}

/// Require every monetary breakpoint to be a multiple of `bucket_size`.
pub fn check_grid_alignment(terms: &UnitTerms, bucket_size: f64) -> Result<(), Error> {
    for value in terms.breakpoints() {
        let ratio = value / bucket_size;
        if (ratio - ratio.round()).abs() > 1e-9 * ratio.abs().max(1.0) {
            return Err(Error::Terms(format!(
                "unit {}: breakpoint {} is not a multiple of bucket_size {}",
                terms.unit_id,
                fmt_g15(value),
                fmt_g15(bucket_size)
            )));
        }
    }
    Ok(())
}

/// Python's `f"{value:.15g}"` with `inf` for infinity.
pub fn fmt_g15(value: f64) -> String {
    if value.is_infinite() {
        return if value > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let sci = format!("{:.14e}", value); // d.dddddddddddddde[-]x
    let (mantissa, exponent) = sci.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    if (-4..15).contains(&exponent) {
        let decimals = (14 - exponent).max(0) as usize;
        let fixed = format!("{:.*}", decimals, value);
        strip_zeros(&fixed)
    } else {
        let mantissa = strip_zeros(mantissa);
        let sign = if exponent < 0 { "-" } else { "+" };
        format!("{mantissa}e{sign}{:02}", exponent.abs())
    }
}

fn strip_zeros(text: &str) -> String {
    if text.contains('.') {
        let trimmed = text.trim_end_matches('0').trim_end_matches('.');
        trimmed.to_string()
    } else {
        text.to_string()
    }
}

fn number(token: &str, what: &str, line: &str) -> Result<f64, Error> {
    let value: f64 = token
        .parse()
        .map_err(|_| Error::Terms(format!("expected a number for {what}, got {token:?} in: {line}")))?;
    if value.is_nan() {
        return Err(Error::Terms(format!("{what} must not be NaN in: {line}")));
    }
    Ok(value)
}

/// Parse one `unit ...` line.
pub fn parse_terms_line(line: &str) -> Result<UnitTerms, Error> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let err = |message: String| Error::Terms(format!("{message} in: {line}"));
    if tokens.first() != Some(&"unit") {
        return Err(Error::Terms(format!("terms line must start with 'unit': {line}")));
    }
    let unit_id = *tokens.get(1).ok_or_else(|| err("missing unit_id".into()))?;
    if KEYWORDS.contains(&unit_id) {
        return Err(err(format!("unit_id {unit_id:?} is a keyword")));
    }
    let mut pos = 2;
    let (mut limit, mut ded) = (f64::INFINITY, 0.0);
    if pos < tokens.len() && tokens[pos] != "occurrence" {
        if pos + 2 >= tokens.len() || tokens[pos + 1] != "xs" {
            return Err(err("expected '<limit> xs <ded>' after unit_id".into()));
        }
        limit = number(tokens[pos], "policy limit", line)?;
        ded = number(tokens[pos + 2], "deductible", line)?;
        pos += 3;
    }
    let mut clauses = Vec::new();
    if pos < tokens.len() {
        if tokens.get(pos..pos + 3) != Some(&["occurrence", "net", "of"][..]) {
            return Err(err("expected 'occurrence net of'".into()));
        }
        pos += 3;
        loop {
            let (clause, next) = parse_clause(&tokens, pos, line)?;
            clauses.push(clause);
            pos = next;
            if pos == tokens.len() {
                break;
            }
            if tokens[pos] != "and" {
                return Err(err("expected 'and' between clauses".into()));
            }
            pos += 1;
            if pos == tokens.len() {
                return Err(err("dangling 'and'".into()));
            }
        }
    }
    UnitTerms::new(unit_id, limit, ded, clauses).map_err(|e| Error::Terms(format!("{e} in: {line}")))
}

/// Parse `[<share> po] <limit> xs <attach>` starting at `pos`.
fn parse_clause(tokens: &[&str], mut pos: usize, line: &str) -> Result<(Clause, usize), Error> {
    let err = |message: String| Error::Terms(format!("{message} in: {line}"));
    let mut share_token = None;
    if pos + 1 < tokens.len() && tokens[pos + 1] == "po" {
        share_token = Some(tokens[pos]);
        pos += 2;
    }
    if pos + 2 >= tokens.len() || tokens[pos + 1] != "xs" {
        return Err(err("expected '<limit> xs <attach>' in clause".into()));
    }
    let limit = number(tokens[pos], "clause limit", line)?;
    let attach = number(tokens[pos + 2], "clause attachment", line)?;
    pos += 3;
    if !(limit.is_finite() && limit > 0.0) {
        return Err(err("clause limit must be finite and positive".into()));
    }
    let share = match share_token {
        None => 1.0,
        Some(token) if token.ends_with('%') => {
            let percent = number(&token[..token.len() - 1], "percent share", line)?;
            if !(percent > 0.0 && percent <= 100.0) {
                return Err(err(format!("percent share must be in (0, 100], got {percent}")));
            }
            percent / 100.0
        }
        Some(token) => {
            let amount = number(token, "share amount", line)?;
            if !(amount > 0.0 && amount <= limit) {
                return Err(err(format!(
                    "share amount must be in (0, limit], got {amount} for limit {limit}"
                )));
            }
            amount / limit
        }
    };
    let clause = Clause::new(share, limit, attach).map_err(|e| Error::Terms(format!("{e} in: {line}")))?;
    Ok((clause, pos))
}

/// Parse terms text: one `unit` line per unit, blank lines and `#` comments
/// ignored, a unit id given twice is an error. Returns the units in order.
pub fn parse_terms(text: &str) -> Result<Vec<UnitTerms>, Error> {
    let mut parsed: Vec<UnitTerms> = Vec::new();
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let terms = parse_terms_line(line)?;
        if parsed.iter().any(|t| t.unit_id == terms.unit_id) {
            return Err(Error::Terms(format!("unit {:?} appears twice in terms", terms.unit_id)));
        }
        parsed.push(terms);
    }
    Ok(parsed)
}

/// A view compiled onto a grid of spacing h with n source buckets: source
/// bucket k deposits its mass at `index[2k]` and `index[2k + 1]` with
/// weights `weight[2k]`, `weight[2k + 1]` (the second pair carries zero
/// weight when the value sits on a node).
#[derive(Debug, Clone, PartialEq)]
pub struct ViewMap {
    pub view: View,
    pub n: usize,
    pub index: Vec<usize>,
    pub weight: Vec<f64>,
    /// Number of destination buckets actually reachable (max index + 1).
    pub support_len: usize,
    /// Every source bucket maps to itself.
    pub identity: bool,
}

impl ViewMap {
    /// Compile `view` of `terms` for the first `n` buckets of the grid.
    pub fn compile(terms: &UnitTerms, view: View, bucket_size: f64, n: usize) -> Self {
        let values: Vec<f64> = (0..n).map(|k| terms.view(view, k as f64 * bucket_size)).collect();
        Self::from_values(view, &values, bucket_size)
    }

    /// Compile a map from the view's value at each source node, `values[k]`
    /// at loss `k h` (the `homer` coupling's composed allocation map, 2.8.0).
    pub fn from_values(view: View, values: &[f64], bucket_size: f64) -> Self {
        let n = values.len();
        let mut index = Vec::with_capacity(2 * n);
        let mut weight = Vec::with_capacity(2 * n);
        let mut support_len = 0;
        let mut identity = true;
        for (k, &value) in values.iter().enumerate() {
            let r = value / bucket_size;
            let i = (r as usize).min(n.saturating_sub(1));
            let f = r - i as f64;
            // Snap values within rounding of a node onto it.
            let (i0, w0, i1, w1) = if f.abs() <= 1e-9 || i + 1 >= n {
                (i, 1.0, i, 0.0)
            } else if (1.0 - f).abs() <= 1e-9 {
                (i + 1, 1.0, i + 1, 0.0)
            } else {
                (i, 1.0 - f, i + 1, f)
            };
            identity &= i0 == k && w0 == 1.0;
            support_len = support_len.max(if w1 > 0.0 { i1 + 1 } else { i0 + 1 });
            index.push(i0);
            index.push(i1);
            weight.push(w0);
            weight.push(w1);
        }
        ViewMap {
            view,
            n,
            index,
            weight,
            support_len: support_len.max(1),
            identity,
        }
    }

    /// Scatter `src` (length n) into `dst` (zeroed by this call, length at
    /// least `support_len`).
    pub fn apply(&self, src: &[f64], dst: &mut [f64]) {
        debug_assert_eq!(src.len(), self.n);
        for v in dst.iter_mut() {
            *v = 0.0;
        }
        if self.identity {
            dst[..self.n].copy_from_slice(src);
            return;
        }
        for (k, &p) in src.iter().enumerate() {
            if p == 0.0 {
                continue;
            }
            dst[self.index[2 * k]] += p * self.weight[2 * k];
            let w1 = self.weight[2 * k + 1];
            if w1 != 0.0 {
                dst[self.index[2 * k + 1]] += p * w1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOWER: &str = "unit plant_a 750 xs 250 occurrence net of 50% po 400 xs 100 and 500 xs 500";

    #[test]
    fn parses_full_and_partial_lines() {
        let t = parse_terms_line(TOWER).unwrap();
        assert_eq!(t.unit_id, "plant_a");
        assert_eq!((t.limit, t.ded), (750.0, 250.0));
        assert_eq!(t.clauses, vec![Clause::new(0.5, 400.0, 100.0).unwrap(), Clause::new(1.0, 500.0, 500.0).unwrap()]);
        assert_eq!(parse_terms_line("unit fleet").unwrap(), UnitTerms::bare("fleet"));
        let only = parse_terms_line("unit d occurrence net of 300 xs 100").unwrap();
        assert!(only.limit.is_infinite() && only.clauses.len() == 1);
        assert_eq!(parse_terms_line("unit x inf xs 100").unwrap().ded, 100.0);
    }

    #[test]
    fn share_conventions_and_bounds() {
        let share = |line: &str| parse_terms_line(line).unwrap().clauses[0].share;
        assert_eq!(share("unit a occurrence net of 50% po 400 xs 100"), 0.5);
        assert_eq!(share("unit a occurrence net of 100 po 400 xs 100"), 0.25);
        assert_eq!(share("unit a occurrence net of .2 po 5 xs 1"), 0.2 / 5.0);
        for bad in [
            "unit a occurrence net of 150% po 400 xs 100",
            "unit a occurrence net of 500 po 400 xs 100",
            "unit a occurrence net of 0 po 400 xs 100",
            "unit a occurrence net of 300 xs 100 and 200 xs 200",
        ] {
            assert!(matches!(parse_terms_line(bad), Err(Error::Terms(_))), "{bad}");
        }
    }

    #[test]
    fn canonical_text_round_trips() {
        for line in [TOWER, "unit fleet", "unit e 600 xs 50", "unit y 1e6 xs 2.5e5"] {
            let t = parse_terms_line(line).unwrap();
            assert_eq!(parse_terms_line(&t.text()).unwrap(), t);
        }
        assert_eq!(
            parse_terms_line("unit b 1000 xs 0 occurrence net of 100 po 400 xs 100").unwrap().text(),
            "unit b 1000 xs 0 occurrence net of 25% po 400 xs 100"
        );
        assert_eq!(fmt_g15(1e6), "1000000");
        assert_eq!(fmt_g15(0.125), "0.125");
        assert_eq!(fmt_g15(1e-5), "1e-05");
        assert_eq!(fmt_g15(1e16), "1e+16");
        assert_eq!(fmt_g15(2.5e5), "250000");
    }

    #[test]
    fn text_parsing_and_alignment() {
        let all = parse_terms(&format!("# c\n\n{TOWER}\nunit fleet # x\n")).unwrap();
        assert_eq!(all.len(), 2);
        assert!(parse_terms(&format!("{TOWER}\n{TOWER}")).is_err());
        let t = parse_terms_line(TOWER).unwrap();
        assert!(check_grid_alignment(&t, 50.0).is_ok());
        assert!(check_grid_alignment(&t, 300.0).is_err());
    }

    #[test]
    fn view_values_and_maps() {
        let t = parse_terms_line(TOWER).unwrap();
        assert_eq!((t.gross(1000.0), t.ceded(1000.0), t.net(1000.0)), (750.0, 450.0, 300.0));
        let h = 10.0;
        let n = 101;
        let src: Vec<f64> = (0..n).map(|k| 1.0 / n as f64 + k as f64 * 1e-4).collect();
        for view in View::ALL {
            let map = ViewMap::compile(&t, view, h, n);
            let mut dst = vec![0.0; n];
            map.apply(&src, &mut dst);
            let mass: f64 = dst.iter().sum();
            let want_mass: f64 = src.iter().sum();
            assert!((mass - want_mass).abs() < 1e-14);
            let mean: f64 = dst.iter().enumerate().map(|(i, p)| p * i as f64 * h).sum();
            let want_mean: f64 = src.iter().enumerate().map(|(k, p)| p * t.view(view, k as f64 * h)).sum();
            assert!((mean - want_mean).abs() < 1e-10, "{view:?}");
            assert!(dst[map.support_len..].iter().all(|&v| v == 0.0));
            assert_eq!(map.identity, view == View::GroundUp);
        }
        let ceded = ViewMap::compile(&t, View::Ceded, h, n);
        assert_eq!(ceded.support_len, 46); // max ceded 450 at index 45
        let bare = ViewMap::compile(&UnitTerms::bare("u"), View::Gross, h, n);
        assert!(bare.identity);
        let zero = ViewMap::compile(&UnitTerms::bare("u"), View::Ceded, h, n);
        assert_eq!(zero.support_len, 1);
    }
}
