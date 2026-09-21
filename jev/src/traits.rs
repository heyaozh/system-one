//! Traits implemented by the derive macros.
//!
//! You normally never implement these by hand:
//!
//! ```ignore
//! #[derive(JevChoice)]
//! enum Department { Billing, Technical, Sales }
//!
//! #[derive(JevQuestions)]
//! struct Triage {
//!     #[jev("Which department should handle this?")]
//!     department: Choice<Department>,
//!     #[jev("Does the customer ask for a refund?")]
//!     wants_refund: Noul,
//!     #[jev("How urgent is this?", levels = ["Can wait", "Today", "Right now"])]
//!     urgency: Score,
//! }
//! ```

use crate::answer::RawAnswers;
use crate::error::Result;
use crate::schema::QuestionSchema;

/// An enum whose variants are the options of a `choice` question.
///
/// Implemented by `#[derive(JevChoice)]`. Variant keys default to the
/// `snake_case` variant name; override with `#[jev(key = "...")]`. A doc
/// comment or `#[jev(desc = "...")]` becomes the option description sent to
/// the backend.
pub trait JevChoice: Copy + PartialEq + std::fmt::Debug + Send + Sync + 'static {
    /// Every variant, in declaration order.
    fn all() -> &'static [Self];
    /// Wire key of this variant.
    fn key(&self) -> &'static str;
    /// Optional description of this variant.
    fn description(&self) -> Option<&'static str>;
    /// Parse a wire key back into a variant.
    fn from_key(key: &str) -> Option<Self>;

    /// `(key, description)` pairs, ready for a [`crate::schema::QuestionSpec::Choice`].
    fn criteria() -> Vec<(String, Option<String>)> {
        Self::all()
            .iter()
            .map(|v| (v.key().to_string(), v.description().map(str::to_string)))
            .collect()
    }
}

/// A struct whose fields are the questions asked against one state.
///
/// Implemented by `#[derive(JevQuestions)]`. Field types must be
/// [`crate::Noul`], [`crate::Choice<E>`] or [`crate::Score`].
pub trait JevQuestions: Sized + Send + 'static {
    /// The schema sent to the backend.
    fn schema() -> QuestionSchema;
    /// Fill the struct from raw answers, validating against the schema.
    fn from_raw(raw: &RawAnswers) -> Result<Self>;
}
