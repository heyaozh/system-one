//! Run two backends side by side: return the primary's answer, record both.
//!
//! This is how you A/B a new model against the one in production without
//! touching business logic:
//!
//! ```ignore
//! let backend = Shadow::new(JevHttp::from_env().unwrap(), LocalLogprob::new(...))
//!     .with_recorder(Recorder::open("runs/shadow.jsonl")?);
//! ```
//!
//! Shadow failures never fail the primary call; they are recorded as a
//! `tag = "shadow-error"` line so you can count them.

use async_trait::async_trait;
use std::sync::Arc;

use crate::answer::RawAnswers;
use crate::backend::DecisionBackend;
use crate::error::Result;
use crate::record::{Record, Recorder};
use crate::schema::QuestionSchema;

pub struct Shadow<P, S> {
    primary: P,
    shadow: S,
    recorder: Option<Arc<Recorder>>,
}

impl<P: DecisionBackend, S: DecisionBackend> Shadow<P, S> {
    pub fn new(primary: P, shadow: S) -> Self {
        Self { primary, shadow, recorder: None }
    }

    /// Record both answers (tagged `primary` / `shadow`) to this recorder.
    pub fn with_recorder(mut self, recorder: Recorder) -> Self {
        self.recorder = Some(Arc::new(recorder));
        self
    }

    pub fn with_shared_recorder(mut self, recorder: Arc<Recorder>) -> Self {
        self.recorder = Some(recorder);
        self
    }
}

#[async_trait]
impl<P: DecisionBackend, S: DecisionBackend> DecisionBackend for Shadow<P, S> {
    fn id(&self) -> String {
        format!("shadow({} | {})", self.primary.id(), self.shadow.id())
    }

    async fn decide(&self, state: &serde_json::Value, schema: &QuestionSchema) -> Result<RawAnswers> {
        let (p, s) = tokio::join!(self.primary.decide(state, schema), self.shadow.decide(state, schema));
        let primary = p?;
        if let Some(rec) = &self.recorder {
            let mut r = Record::new(&self.primary.id(), state, schema, &primary);
            r.tag = Some("primary".into());
            let _ = rec.write(r);
            match s {
                Ok(sa) => {
                    let mut r = Record::new(&self.shadow.id(), state, schema, &sa);
                    r.tag = Some("shadow".into());
                    let _ = rec.write(r);
                }
                Err(_) => {
                    let mut r = Record::new(&self.shadow.id(), state, schema, &RawAnswers::default());
                    r.tag = Some("shadow-error".into());
                    let _ = rec.write(r);
                }
            }
        }
        Ok(primary)
    }
}
