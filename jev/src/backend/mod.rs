//! Backends: anything that can turn `(state, schema)` into raw answers.
//!
//! The crate ships:
//!
//! | Backend | Purpose |
//! |---|---|
//! | [`JevHttp`] | The official TypeSafe AI Jev API (`api.typesafe.ai/v1/systemone`). |
//! | [`Mock`] | Deterministic answers for unit tests and offline development. |
//! | [`Replay`] | Answers from a JSONL recording — free, reproducible backtests. |
//! | [`LocalLogprob`] | Your own model behind an OpenAI-compatible server (vLLM, llama.cpp, …), read out via log-probs. |
//! | [`Shadow`] | Run a primary and a shadow backend side by side; return the primary, record both. |
//!
//! Implement [`DecisionBackend`] for anything else (a trained classifier, a
//! rules engine, a remote gRPC service).

use async_trait::async_trait;

use crate::answer::RawAnswers;
use crate::error::Result;
use crate::schema::QuestionSchema;

mod mock;
mod replay;
mod shadow;
#[cfg(feature = "http")]
mod http;
#[cfg(feature = "http")]
mod logprob;

pub use mock::{answers, Mock};
pub use replay::Replay;
pub use shadow::Shadow;
#[cfg(feature = "http")]
pub use http::JevHttp;
#[cfg(feature = "http")]
pub use logprob::{LocalLogprob, PromptStyle};

/// The one trait every backend implements.
#[async_trait]
pub trait DecisionBackend: Send + Sync {
    /// Human-readable identifier stored in recordings (e.g. `jev-http`,
    /// `local:updown-v3`).
    fn id(&self) -> String;

    /// Answer every question in `schema` against `state`.
    ///
    /// `state` is already serialised to JSON; a plain string state arrives as
    /// a JSON string value, a struct as an object.
    async fn decide(&self, state: &serde_json::Value, schema: &QuestionSchema) -> Result<RawAnswers>;
}

// Allow `Arc<dyn DecisionBackend>` and `Box<dyn DecisionBackend>` to be used
// wherever a backend is expected.
#[async_trait]
impl<T: DecisionBackend + ?Sized> DecisionBackend for std::sync::Arc<T> {
    fn id(&self) -> String {
        (**self).id()
    }
    async fn decide(&self, state: &serde_json::Value, schema: &QuestionSchema) -> Result<RawAnswers> {
        (**self).decide(state, schema).await
    }
}

#[async_trait]
impl<T: DecisionBackend + ?Sized> DecisionBackend for Box<T> {
    fn id(&self) -> String {
        (**self).id()
    }
    async fn decide(&self, state: &serde_json::Value, schema: &QuestionSchema) -> Result<RawAnswers> {
        (**self).decide(state, schema).await
    }
}
