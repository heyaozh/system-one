//! The front door: [`Engine`] wraps a backend with typing, caching, batching
//! and recording.

use futures::{stream, StreamExt};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::answer::RawAnswers;
use crate::backend::DecisionBackend;
use crate::error::Result;
use crate::hash::hash_json;
use crate::record::{Record, Recorder};
use crate::schema::QuestionSchema;
use crate::traits::JevQuestions;

/// Ask typed questions against any backend.
///
/// ```ignore
/// let engine = Engine::new(JevHttp::from_env().unwrap())
///     .with_recorder(Recorder::open("runs/today.jsonl")?)
///     .with_cache();
///
/// let t: Triage = engine.ask(&ticket_text).await?;
/// ```
pub struct Engine<B: DecisionBackend> {
    backend: B,
    recorder: Option<Arc<Recorder>>,
    cache: Option<Mutex<HashMap<(String, String), RawAnswers>>>,
    concurrency: usize,
}

impl<B: DecisionBackend> Engine<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            recorder: None,
            cache: None,
            concurrency: 16,
        }
    }

    /// Record every answer to a JSONL file.
    pub fn with_recorder(mut self, recorder: Recorder) -> Self {
        self.recorder = Some(Arc::new(recorder));
        self
    }

    /// Share a recorder with other engines / a [`crate::backend::Shadow`].
    pub fn with_shared_recorder(mut self, recorder: Arc<Recorder>) -> Self {
        self.recorder = Some(recorder);
        self
    }

    /// In-memory content cache keyed by `(state hash, schema hash)`. Identical
    /// states are never sent twice in one process.
    pub fn with_cache(mut self) -> Self {
        self.cache = Some(Mutex::new(HashMap::new()));
        self
    }

    /// Max in-flight requests for [`Engine::ask_many`] (default 16).
    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = n.max(1);
        self
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Ask the questions of `Q` against `state` and get a typed `Q` back.
    pub async fn ask<Q: JevQuestions, S: Serialize + ?Sized>(&self, state: &S) -> Result<Q> {
        let raw = self.ask_raw(state, &Q::schema()).await?;
        Q::from_raw(&raw)
    }

    /// Untyped variant: hand-built schema, raw answers.
    pub async fn ask_raw<S: Serialize + ?Sized>(&self, state: &S, schema: &QuestionSchema) -> Result<RawAnswers> {
        let state_json = serde_json::to_value(state)?;
        let key = (hash_json(&state_json), schema.hash());

        if let Some(c) = &self.cache {
            if let Some(hit) = c.lock().expect("cache poisoned").get(&key) {
                return Ok(hit.clone());
            }
        }

        let raw = self.backend.decide(&state_json, schema).await?;

        if let Some(c) = &self.cache {
            c.lock().expect("cache poisoned").insert(key, raw.clone());
        }
        if let Some(r) = &self.recorder {
            r.write(Record::new(&self.backend.id(), &state_json, schema, &raw))?;
        }
        Ok(raw)
    }

    /// Ask the same questions against many states, `concurrency` at a time.
    ///
    /// Results are returned in input order. One failed state does not abort
    /// the batch: each element is its own `Result`.
    pub async fn ask_many<Q, S, I>(&self, states: I) -> Vec<Result<Q>>
    where
        Q: JevQuestions,
        S: Serialize + Send + Sync,
        I: IntoIterator<Item = S>,
    {
        let states: Vec<S> = states.into_iter().collect();
        stream::iter(states.iter())
            .map(|s| self.ask::<Q, S>(s))
            .buffered(self.concurrency)
            .collect()
            .await
    }

    /// Same as [`Engine::ask_many`] but returns the state alongside its answer.
    pub async fn ask_many_with<Q, S, I>(&self, states: I) -> Vec<(S, Result<Q>)>
    where
        Q: JevQuestions,
        S: Serialize + Send + Sync,
        I: IntoIterator<Item = S>,
    {
        let states: Vec<S> = states.into_iter().collect();
        let answers: Vec<Result<Q>> = stream::iter(states.iter())
            .map(|s| self.ask::<Q, S>(s))
            .buffered(self.concurrency)
            .collect()
            .await;
        states.into_iter().zip(answers).collect()
    }
}
