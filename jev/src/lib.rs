//! # jev
//!
//! Typed, calibrated decisions for Rust.
//!
//! `jev` is a backend-agnostic client for *System One* style models — models
//! that do not generate text but answer a fixed set of questions with
//! calibrated probabilities. The first backend is TypeSafe AI's **Jev** API;
//! the same interface serves a mock, a recording, or your own fine-tuned
//! model behind an OpenAI-compatible server.
//!
//! ```ignore
//! use jev::prelude::*;
//!
//! #[derive(JevChoice)]
//! enum Department { Billing, Technical, Sales, Spam }
//!
//! #[derive(JevQuestions)]
//! struct Triage {
//!     #[jev("Which department should handle this ticket?")]
//!     department: Choice<Department>,
//!     #[jev("Is the customer asking for a refund?")]
//!     wants_refund: Noul,
//!     #[jev("How urgent is this?", levels = ["Can wait", "Today", "Right now"])]
//!     urgency: Score,
//! }
//!
//! let engine = Engine::new(JevHttp::from_env().expect("JEV_API_KEY"));
//! let t: Triage = engine.ask("Hi, my payouts have failed for 3 days…").await?;
//! if t.wants_refund.decide(/*false positive*/ 1.0, /*false negative*/ 5.0) { /* … */ }
//! ```
//!
//! The pieces:
//!
//! * **Types** — [`Noul`], [`Choice`], [`Score`] carry full distributions
//!   plus decision helpers (cost-matrix decisions, entropy, sampling).
//! * **Derive** — `#[derive(JevChoice)]` on an enum, `#[derive(JevQuestions)]`
//!   on a struct; the schema is generated, answers are parsed back.
//! * **Backends** — [`backend::JevHttp`], [`backend::Mock`], [`backend::Replay`],
//!   [`backend::LocalLogprob`], [`backend::Shadow`], or your own
//!   [`backend::DecisionBackend`].
//! * **Engine** — [`Engine`] adds caching, batching and recording.
//! * **Recording & calibration** — [`record`] writes JSONL (Parquet with the
//!   `parquet` feature); [`calibration::CalibrationReport`] tells you whether
//!   a 0.8 really meant 80 %.

#![forbid(unsafe_code)]
#![allow(clippy::len_without_is_empty)]

pub mod answer;
pub mod backend;
pub mod calibration;
pub mod engine;
pub mod error;
pub mod hash;
pub mod record;
pub mod schema;
pub mod traits;

pub use answer::{Choice, Noul, RawAnswer, RawAnswers, Score, Usage};
pub use engine::Engine;
pub use error::{Error, Result};
pub use record::{Record, Recorder};
pub use schema::{NoulCriteria, QuestionSchema, QuestionSpec};
pub use traits::{JevChoice, JevQuestions};

#[cfg(feature = "derive")]
pub use jev_derive::{JevChoice, JevQuestions};

/// Everything you need in scope for typical use.
pub mod prelude {
    pub use crate::answer::{Choice, Noul, Score};
    pub use crate::backend::{DecisionBackend, Mock, Replay, Shadow};
    #[cfg(feature = "http")]
    pub use crate::backend::{JevHttp, LocalLogprob};
    pub use crate::calibration::CalibrationReport;
    pub use crate::engine::Engine;
    pub use crate::error::{Error, Result};
    pub use crate::record::{Record, Recorder};
    pub use crate::schema::{QuestionSchema, QuestionSpec};
    pub use crate::traits::{JevChoice as JevChoiceTrait, JevQuestions as JevQuestionsTrait};
    #[cfg(feature = "derive")]
    pub use jev_derive::{JevChoice, JevQuestions};
}

// Used by the derive macros so generated code does not depend on the
// caller's imports.
#[doc(hidden)]
pub mod __private {
    pub use crate::answer::{Choice, Noul, RawAnswers, Score};
    pub use crate::error::{Error, Result};
    pub use crate::schema::{NoulCriteria, QuestionSchema, QuestionSpec};
    pub use crate::traits::{JevChoice, JevQuestions};
}
