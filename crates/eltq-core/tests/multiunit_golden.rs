//! Multi-unit views golden (1.10.0): `fixtures/multiunit_small_elt.csv` +
//! `fixtures/multiunit_small_terms.txt` against
//! `fixtures/golden/multiunit_small.*` from the packaged SciPy reference.

use std::collections::HashMap;
use std::path::PathBuf;

use eltq_core::{Category, MultiConfig, MultiInput, UnitTerms, View, ZRule, multiunit_views, parse_terms};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// Dense codes by first appearance plus the ids in that order.
fn factorize(values: &[String]) -> (Vec<i64>, Vec<String>) {
    let mut table: HashMap<&str, i64> = HashMap::new();
    let mut ids = Vec::new();
    let codes = values
        .iter()
        .map(|v| {
            *table.entry(v.as_str()).or_insert_with(|| {
                ids.push(v.clone());
                (ids.len() - 1) as i64
            })
        })
        .collect();
    (codes, ids)
}

#[test]
fn multiunit_small_golden() {
    let text = std::fs::read_to_string(fixtures_dir().join("multiunit_small_elt.csv")).unwrap();
    let mut lines = text.lines();
    assert_eq!(
        lines.next().unwrap().trim(),
        "event_id,unit_id,rate,mean_loss,sd_indep,sd_corr,max_loss"
    );
    let (mut event_id, mut unit_id) = (Vec::new(), Vec::new());
    let mut numeric: [Vec<f64>; 5] = Default::default();
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 7, "bad fixture line: {line}");
        event_id.push(fields[0].to_string());
        unit_id.push(fields[1].to_string());
        for (col, field) in numeric.iter_mut().zip(&fields[2..]) {
            col.push(field.parse().unwrap());
        }
    }
    let (event_code, _) = factorize(&event_id);
    let (unit_code, unit_ids) = factorize(&unit_id);
    let terms_text = std::fs::read_to_string(fixtures_dir().join("multiunit_small_terms.txt")).unwrap();
    let parsed = parse_terms(&terms_text).unwrap();
    let terms: Vec<UnitTerms> = unit_ids
        .iter()
        .map(|id| {
            parsed
                .iter()
                .find(|t| &t.unit_id == id)
                .cloned()
                .unwrap_or_else(|| UnitTerms::bare(id))
        })
        .collect();

    let meta_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures_dir().join("golden/multiunit_small.meta.json")).unwrap(),
    )
    .unwrap();
    let params = &meta_json["params"];
    let mut config = MultiConfig::new(params["bucket_size"].as_f64().unwrap());
    config.z_nodes = params["z_nodes"].as_u64().unwrap() as usize;
    config.z_rule = ZRule::parse(params["z_rule"].as_str().unwrap()).unwrap();
    config.views = meta_json["views"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| View::parse(v.as_str().unwrap()).unwrap())
        .collect();
    let input = MultiInput {
        event_code: &event_code,
        unit_code: &unit_code,
        rate: &numeric[0],
        mean_loss: &numeric[1],
        sd_indep: &numeric[2],
        sd_corr: &numeric[3],
        max_loss: &numeric[4],
    };
    let result = multiunit_views(&input, &terms, &config).unwrap();
    let meta = &result.meta;

    let golden_text =
        std::fs::read_to_string(fixtures_dir().join("golden/multiunit_small.arrays.csv")).unwrap();
    let mut golden_lines = golden_text.lines();
    let header: Vec<&str> = golden_lines.next().unwrap().split(',').collect();
    assert_eq!(header, config.views.iter().map(|v| v.as_str()).collect::<Vec<_>>());
    let mut golden: Vec<Vec<f64>> = vec![Vec::new(); header.len()];
    for line in golden_lines.filter(|l| !l.trim().is_empty()) {
        for (col, cell) in golden.iter_mut().zip(line.split(',')) {
            col.push(cell.trim().parse().unwrap());
        }
    }
    let total_rate = meta_json["total_rate"].as_f64().unwrap();
    assert_eq!(meta.grid_length, meta_json["grid_length"].as_u64().unwrap() as usize);
    let atol = 1e-13 * total_rate;
    for (vi, view) in config.views.iter().enumerate() {
        let (got, want) = (&result.views[vi], &golden[vi]);
        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= atol + 1e-9 * w.abs(),
                "{}[{i}]: got {g}, want {w}",
                view.as_str()
            );
        }
        let want_mean = meta_json["analytic_view_mean"][view.as_str()].as_f64().unwrap();
        assert!((meta.analytic_view_mean[vi] - want_mean).abs() <= 1e-12 * want_mean.abs());
    }
    let rel = |got: f64, want: f64| (got - want).abs() / want.abs().max(f64::MIN_POSITIVE);
    assert!(rel(meta.total_rate, total_rate) < 1e-12);
    assert!(rel(meta.analytic_aal, meta_json["analytic_aal"].as_f64().unwrap()) < 1e-12);
    assert_eq!(meta.valid_rows, meta_json["valid_rows"].as_u64().unwrap() as usize);
    assert_eq!(meta.events, meta_json["events"].as_u64().unwrap() as usize);
    assert_eq!(meta.valid_events, meta_json["valid_events"].as_u64().unwrap() as usize);
    let golden_diag = meta_json["diagnostics"].as_object().unwrap();
    for category in Category::ALL {
        let want = golden_diag
            .get(category.as_str())
            .map(|v| v["count"].as_u64().unwrap())
            .unwrap_or(0);
        assert_eq!(meta.diagnostics.get(category).count, want, "category {}", category.as_str());
    }
    assert!(meta.max_mass_error < 1e-12);
}
