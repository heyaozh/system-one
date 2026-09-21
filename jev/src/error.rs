//! Error type shared by every layer of the crate.

use thiserror::Error;

/// Everything that can go wrong while asking a question.
#[derive(Debug, Error)]
pub enum Error {
    /// The backend answered, but the answer does not match the schema we sent.
    /// With the official Jev API this should never happen (outputs are schema
    /// constrained); it protects you against buggy custom backends.
    #[error("answer for question `{question}` does not match schema: {reason}")]
    SchemaMismatch { question: String, reason: String },

    /// The backend returned no answer for a question that was in the schema.
    #[error("backend returned no answer for question `{0}`")]
    MissingAnswer(String),

    /// A schema was invalid before it was even sent (e.g. a `Choice` with
    /// more than 255 options, or a `Score` with fewer than 2 levels).
    #[error("invalid schema: {0}")]
    InvalidSchema(String),

    /// The state could not be serialised.
    #[error("state serialisation failed: {0}")]
    State(#[from] serde_json::Error),

    /// HTTP transport or non-success status from a remote backend.
    #[error("backend `{backend}` failed: {message}")]
    Backend { backend: String, message: String },

    /// The request was rejected with 429/529 and retries were exhausted.
    #[error("backend `{backend}` is rate limited / overloaded after {attempts} attempts")]
    RateLimited { backend: String, attempts: u32 },

    /// A `Replay` backend has no recorded answer for this (state, schema) pair.
    #[error("no recorded answer for state hash {state_hash} / schema hash {schema_hash}")]
    NotRecorded { state_hash: String, schema_hash: String },

    /// I/O while recording or replaying.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
