//! Replay answers from a recording — no network, no cost, fully reproducible.

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;

use crate::answer::RawAnswers;
use crate::backend::DecisionBackend;
use crate::error::{Error, Result};
use crate::hash::hash_json;
use crate::record::read_records;
use crate::schema::QuestionSchema;

/// Looks up `(state_hash, schema_hash)` in a recording.
///
/// Use it to re-run a backtest against answers you already paid for, or to
/// make an integration test deterministic. Unknown states return
/// [`Error::NotRecorded`], unless a fallback backend is attached.
pub struct Replay {
    table: HashMap<(String, String), RawAnswers>,
    fallback: Option<Box<dyn DecisionBackend>>,
    source: String,
}

impl Replay {
    /// Load a JSONL recording. Later records win on duplicate keys.
    pub fn from_jsonl(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut table = HashMap::new();
        for r in read_records(path)? {
            table.insert((r.state_hash, r.schema_hash), r.answers);
        }
        Ok(Self { table, fallback: None, source: path.display().to_string() })
    }

    /// Answer unknown states with another backend (e.g. live Jev) instead of
    /// failing — a simple read-through cache across runs.
    pub fn with_fallback<B: DecisionBackend + 'static>(mut self, backend: B) -> Self {
        self.fallback = Some(Box::new(backend));
        self
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
}

#[async_trait]
impl DecisionBackend for Replay {
    fn id(&self) -> String {
        format!("replay:{}", self.source)
    }

    async fn decide(&self, state: &serde_json::Value, schema: &QuestionSchema) -> Result<RawAnswers> {
        let key = (hash_json(state), schema.hash());
        if let Some(a) = self.table.get(&key) {
            return Ok(a.clone());
        }
        match &self.fallback {
            Some(b) => b.decide(state, schema).await,
            None => Err(Error::NotRecorded { state_hash: key.0, schema_hash: key.1 }),
        }
    }
}
