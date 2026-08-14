//! Streaming CSV input route.
//!
//! Reads the six canonical columns (names configurable) into
//! structure-of-arrays buffers. Unparseable or empty numeric cells become NaN
//! so the affected rows flow into the `nonfinite` diagnostic category with
//! bounded examples, instead of aborting a million-row run; structural CSV
//! problems (missing columns, ragged rows, IO errors) are hard typed errors.

use std::path::Path;

use crate::error::Error;
use crate::quantize::EltInput;

/// Column-name mapping; defaults are the canonical schema.
#[derive(Debug, Clone)]
pub struct CsvColumns {
    /// Optional identifier column; read when present, never in the hot path.
    pub event_id: String,
    pub rate: String,
    pub max_loss: String,
    pub mean_loss: String,
    pub sd_indep: String,
    pub sd_corr: String,
}

impl Default for CsvColumns {
    fn default() -> Self {
        CsvColumns {
            event_id: "event_id".to_string(),
            rate: "rate".to_string(),
            max_loss: "max_loss".to_string(),
            mean_loss: "mean_loss".to_string(),
            sd_indep: "sd_indep".to_string(),
            sd_corr: "sd_corr".to_string(),
        }
    }
}

/// Owned structure-of-arrays ELT columns.
#[derive(Debug, Default)]
pub struct EltColumns {
    pub rate: Vec<f64>,
    pub max_loss: Vec<f64>,
    pub mean_loss: Vec<f64>,
    pub sd_indep: Vec<f64>,
    pub sd_corr: Vec<f64>,
    /// Present when the file has an event-id column.
    pub event_ids: Option<Vec<String>>,
}

impl EltColumns {
    pub fn input(&self) -> EltInput<'_> {
        EltInput {
            rate: &self.rate,
            max_loss: &self.max_loss,
            mean_loss: &self.mean_loss,
            sd_indep: &self.sd_indep,
            sd_corr: &self.sd_corr,
        }
    }
}

/// Read an ELT CSV with headers into column buffers.
pub fn read_elt_csv(path: &Path, columns: &CsvColumns) -> Result<EltColumns, Error> {
    let csv_err = |e: csv::Error| Error::Csv(e.to_string());
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(csv_err)?;
    let headers = reader.headers().map_err(csv_err)?.clone();
    let find = |name: &str| headers.iter().position(|h| h == name);
    let require =
        |name: &str| find(name).ok_or_else(|| Error::MissingColumn(name.to_string()));
    let rate_idx = require(&columns.rate)?;
    let max_loss_idx = require(&columns.max_loss)?;
    let mean_loss_idx = require(&columns.mean_loss)?;
    let sd_indep_idx = require(&columns.sd_indep)?;
    let sd_corr_idx = require(&columns.sd_corr)?;
    let id_idx = find(&columns.event_id);

    let mut out = EltColumns {
        event_ids: id_idx.map(|_| Vec::new()),
        ..EltColumns::default()
    };
    let mut record = csv::StringRecord::new();
    while reader.read_record(&mut record).map_err(csv_err)? {
        let parse = |idx: usize| -> f64 {
            record
                .get(idx)
                .and_then(|s| s.trim().parse::<f64>().ok())
                .unwrap_or(f64::NAN)
        };
        out.rate.push(parse(rate_idx));
        out.max_loss.push(parse(max_loss_idx));
        out.mean_loss.push(parse(mean_loss_idx));
        out.sd_indep.push(parse(sd_indep_idx));
        out.sd_corr.push(parse(sd_corr_idx));
        if let (Some(idx), Some(ids)) = (id_idx, out.event_ids.as_mut()) {
            ids.push(record.get(idx).unwrap_or("").to_string());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, contents: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("eltq-core-test-{name}"));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        path
    }

    #[test]
    fn reads_canonical_schema() {
        let path = write_temp(
            "canonical.csv",
            "event_id,rate,max_loss,mean_loss,sd_indep,sd_corr\n\
             ev1,0.01,100.0,40.0,15.0,10.0\n\
             ev2,0.02,50.0,20.0,5.0,0.0\n",
        );
        let cols = read_elt_csv(&path, &CsvColumns::default()).unwrap();
        assert_eq!(cols.rate, vec![0.01, 0.02]);
        assert_eq!(cols.max_loss, vec![100.0, 50.0]);
        assert_eq!(cols.event_ids.as_deref(), Some(&["ev1".to_string(), "ev2".to_string()][..]));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn renamed_columns_and_missing_id() {
        let path = write_temp(
            "renamed.csv",
            "lam,M,mu,s1,s2\n0.01,100.0,40.0,15.0,10.0\n",
        );
        let columns = CsvColumns {
            rate: "lam".into(),
            max_loss: "M".into(),
            mean_loss: "mu".into(),
            sd_indep: "s1".into(),
            sd_corr: "s2".into(),
            ..CsvColumns::default()
        };
        let cols = read_elt_csv(&path, &columns).unwrap();
        assert_eq!(cols.rate, vec![0.01]);
        assert!(cols.event_ids.is_none());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn missing_column_is_typed_error() {
        let path = write_temp(
            "missing.csv",
            "event_id,rate,max_loss,mean_loss,sd_indep\nev1,0.01,100.0,40.0,15.0\n",
        );
        match read_elt_csv(&path, &CsvColumns::default()) {
            Err(Error::MissingColumn(name)) => assert_eq!(name, "sd_corr"),
            other => panic!("expected MissingColumn, got {other:?}"),
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn bad_cells_become_nan() {
        let path = write_temp(
            "badcell.csv",
            "event_id,rate,max_loss,mean_loss,sd_indep,sd_corr\n\
             ev1,not_a_number,100.0,40.0,15.0,10.0\n\
             ev2,0.02,50.0,,5.0,0.0\n",
        );
        let cols = read_elt_csv(&path, &CsvColumns::default()).unwrap();
        assert!(cols.rate[0].is_nan());
        assert!(cols.mean_loss[1].is_nan());
        std::fs::remove_file(path).ok();
    }
}
