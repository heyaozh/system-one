//! Recording every decision to JSONL (and optionally Parquet).
//!
//! A recording is the raw material for three things:
//!
//! 1. **Replay** — re-run a backtest or a test suite without paying for or
//!    waiting on a backend ([`crate::backend::Replay`]).
//! 2. **Calibration** — join the recorded probabilities with what actually
//!    happened and measure how honest the backend was ([`crate::calibration`]).
//! 3. **Training data** — the same rows, with outcomes attached, are a
//!    labelled dataset for distilling a small local model.
//!
//! One JSON object per line:
//!
//! ```json
//! {"ts":"2026-09-21T10:00:00Z","backend":"jev-http:jev-latest","state_hash":"…",
//!  "schema_hash":"…","state":{…},"schema":{…},"answers":{…},"outcome":null,"tag":null}
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::answer::RawAnswers;
use crate::error::Result;
use crate::hash::hash_json;
use crate::schema::QuestionSchema;

/// One recorded decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub ts: DateTime<Utc>,
    pub backend: String,
    pub state_hash: String,
    pub schema_hash: String,
    pub state: serde_json::Value,
    pub schema: QuestionSchema,
    pub answers: RawAnswers,
    /// What actually happened, attached later with [`attach_outcomes`].
    /// Free-form: `{"is_urgent": true, "department": "billing"}`.
    #[serde(default)]
    pub outcome: Option<serde_json::Value>,
    /// Optional free-form tag (`"shadow"`, `"backtest-2026-09"`, …).
    #[serde(default)]
    pub tag: Option<String>,
}

impl Record {
    pub fn new(backend: &str, state: &serde_json::Value, schema: &QuestionSchema, answers: &RawAnswers) -> Self {
        Self {
            ts: Utc::now(),
            backend: backend.to_string(),
            state_hash: hash_json(state),
            schema_hash: schema.hash(),
            state: state.clone(),
            schema: schema.clone(),
            answers: answers.clone(),
            outcome: None,
            tag: None,
        }
    }
}

/// Append-only JSONL writer, safe to share across tasks.
pub struct Recorder {
    path: PathBuf,
    file: Mutex<std::fs::File>,
    tag: Option<String>,
}

impl Recorder {
    /// Open (or create) a JSONL file for appending.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            path,
            file: Mutex::new(file),
            tag: None,
        })
    }

    /// Tag every record written by this recorder.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Write one record.
    pub fn write(&self, mut record: Record) -> Result<()> {
        if record.tag.is_none() {
            record.tag = self.tag.clone();
        }
        let line = serde_json::to_string(&record)?;
        let mut f = self.file.lock().expect("recorder poisoned");
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(())
    }
}

/// Read every record of a JSONL file. Malformed lines are skipped.
pub fn read_records(path: impl AsRef<Path>) -> Result<Vec<Record>> {
    let f = std::fs::File::open(path)?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(r) = serde_json::from_str::<Record>(&line) {
            out.push(r);
        }
    }
    Ok(out)
}

/// Attach outcomes to records by state hash and rewrite the file.
///
/// `outcomes` maps `state_hash -> outcome JSON`. Records without a matching
/// hash are left untouched. Returns the number of records updated.
pub fn attach_outcomes(
    path: impl AsRef<Path>,
    outcomes: &std::collections::HashMap<String, serde_json::Value>,
) -> Result<usize> {
    let path = path.as_ref();
    let mut records = read_records(path)?;
    let mut n = 0;
    for r in &mut records {
        if let Some(o) = outcomes.get(&r.state_hash) {
            r.outcome = Some(o.clone());
            n += 1;
        }
    }
    let tmp = path.with_extension("jsonl.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        for r in &records {
            f.write_all(serde_json::to_string(r)?.as_bytes())?;
            f.write_all(b"\n")?;
        }
    }
    std::fs::rename(tmp, path)?;
    Ok(n)
}

/// Export a JSONL recording to Parquet (feature `parquet`).
///
/// Nested JSON columns (`state`, `schema`, `answers`, `outcome`) are stored as
/// JSON strings so the file loads anywhere; flatten them in pandas/polars.
#[cfg(feature = "parquet")]
pub fn export_parquet(jsonl: impl AsRef<Path>, out: impl AsRef<Path>) -> Result<usize> {
    use arrow::array::{ArrayRef, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let records = read_records(jsonl)?;
    let col = |f: &dyn Fn(&Record) -> String| -> ArrayRef {
        Arc::new(StringArray::from(records.iter().map(f).collect::<Vec<_>>()))
    };
    let opt_col = |f: &dyn Fn(&Record) -> Option<String>| -> ArrayRef {
        Arc::new(StringArray::from(records.iter().map(f).collect::<Vec<_>>()))
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Utf8, false),
        Field::new("backend", DataType::Utf8, false),
        Field::new("state_hash", DataType::Utf8, false),
        Field::new("schema_hash", DataType::Utf8, false),
        Field::new("state", DataType::Utf8, false),
        Field::new("schema", DataType::Utf8, false),
        Field::new("answers", DataType::Utf8, false),
        Field::new("outcome", DataType::Utf8, true),
        Field::new("tag", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            col(&|r| r.ts.to_rfc3339()),
            col(&|r| r.backend.clone()),
            col(&|r| r.state_hash.clone()),
            col(&|r| r.schema_hash.clone()),
            col(&|r| r.state.to_string()),
            col(&|r| serde_json::to_string(&r.schema).unwrap_or_default()),
            col(&|r| serde_json::to_string(&r.answers).unwrap_or_default()),
            opt_col(&|r| r.outcome.as_ref().map(|o| o.to_string())),
            opt_col(&|r| r.tag.clone()),
        ],
    )
    .map_err(|e| crate::error::Error::Backend {
        backend: "parquet".into(),
        message: e.to_string(),
    })?;

    let file = std::fs::File::create(out)?;
    let mut w = ArrowWriter::try_new(file, schema, None).map_err(|e| crate::error::Error::Backend {
        backend: "parquet".into(),
        message: e.to_string(),
    })?;
    w.write(&batch).map_err(|e| crate::error::Error::Backend {
        backend: "parquet".into(),
        message: e.to_string(),
    })?;
    w.close().map_err(|e| crate::error::Error::Backend {
        backend: "parquet".into(),
        message: e.to_string(),
    })?;
    Ok(records.len())
}
