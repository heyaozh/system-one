//! Wire-level description of a set of questions.
//!
//! A [`QuestionSchema`] is what a backend receives. It serialises 1:1 to the
//! `questions` object of the official Jev API, so the derive macro only has to
//! build this structure and the HTTP backend can send it verbatim.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::{Error, Result};

/// Maximum number of options a `choice` question may carry (API limit).
pub const MAX_CHOICE_OPTIONS: usize = 255;
/// Minimum / maximum number of levels of a `score` question (API limits).
pub const MIN_SCORE_LEVELS: usize = 2;
pub const MAX_SCORE_LEVELS: usize = 10;

/// One question, in the shape the API expects.
///
/// `type` is serialised from the enum variant; the remaining fields are the
/// variant's payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum QuestionSpec {
    /// A yes/no question. The answer is `P(yes)`.
    Noul {
        instructions: String,
        /// Optional explicit criteria for the `true` and `false` sides.
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    /// Pick one of up to 255 named options. The answer is a distribution
    /// over the option keys.
    Choice {
        instructions: String,
        /// `option_key -> optional description`. Insertion order is kept by
        /// using a `Vec`; the API accepts an object, which we build on send.
        criteria: Vec<(String, Option<String>)>,
    },
    /// An ordered scale with 2–10 described levels. The answer is a
    /// distribution over level indices plus the probability-weighted score.
    Score {
        instructions: String,
        /// Level descriptions, index 0 = lowest.
        criteria: Vec<String>,
    },
}

/// Explicit descriptions of what counts as `true` / `false` for a `noul`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(rename = "true")]
    pub yes: String,
    #[serde(rename = "false")]
    pub no: String,
}

impl QuestionSpec {
    /// Validate API limits before sending.
    pub fn validate(&self, name: &str) -> Result<()> {
        match self {
            QuestionSpec::Noul { .. } => Ok(()),
            QuestionSpec::Choice { criteria, .. } => {
                if criteria.is_empty() {
                    return Err(Error::InvalidSchema(format!("choice `{name}` has no options")));
                }
                if criteria.len() > MAX_CHOICE_OPTIONS {
                    return Err(Error::InvalidSchema(format!(
                        "choice `{name}` has {} options, max is {MAX_CHOICE_OPTIONS}",
                        criteria.len()
                    )));
                }
                Ok(())
            }
            QuestionSpec::Score { criteria, .. } => {
                if criteria.len() < MIN_SCORE_LEVELS || criteria.len() > MAX_SCORE_LEVELS {
                    return Err(Error::InvalidSchema(format!(
                        "score `{name}` has {} levels, must be {MIN_SCORE_LEVELS}..={MAX_SCORE_LEVELS}",
                        criteria.len()
                    )));
                }
                Ok(())
            }
        }
    }

    /// The JSON the API expects for this question (choice criteria become an
    /// object keyed by option).
    pub fn to_api_json(&self) -> serde_json::Value {
        use serde_json::{json, Map, Value};
        match self {
            QuestionSpec::Noul { instructions, criteria } => {
                let mut m = Map::new();
                m.insert("type".into(), json!("noul"));
                m.insert("instructions".into(), json!(instructions));
                if let Some(c) = criteria {
                    m.insert("criteria".into(), json!({ "true": c.yes, "false": c.no }));
                }
                Value::Object(m)
            }
            QuestionSpec::Choice { instructions, criteria } => {
                let mut opts = Map::new();
                for (k, d) in criteria {
                    opts.insert(k.clone(), d.clone().map(Value::String).unwrap_or(Value::Null));
                }
                json!({ "type": "choice", "instructions": instructions, "criteria": opts })
            }
            QuestionSpec::Score { instructions, criteria } => {
                json!({ "type": "score", "instructions": instructions, "criteria": criteria })
            }
        }
    }
}

/// An ordered set of named questions asked against one state.
///
/// Questions are evaluated *in parallel* by the backend, so asking ten costs
/// about the same as asking one.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QuestionSchema {
    /// `question_name -> spec`. A `BTreeMap` gives a stable order, which keeps
    /// the schema hash (used for caching and replay) deterministic.
    pub questions: BTreeMap<String, QuestionSpec>,
}

impl QuestionSchema {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style insert.
    pub fn with(mut self, name: impl Into<String>, spec: QuestionSpec) -> Self {
        self.questions.insert(name.into(), spec);
        self
    }

    pub fn insert(&mut self, name: impl Into<String>, spec: QuestionSpec) {
        self.questions.insert(name.into(), spec);
    }

    pub fn len(&self) -> usize {
        self.questions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.questions.is_empty()
    }

    /// Validate every question against API limits.
    pub fn validate(&self) -> Result<()> {
        if self.questions.is_empty() {
            return Err(Error::InvalidSchema("schema has no questions".into()));
        }
        for (name, q) in &self.questions {
            q.validate(name)?;
        }
        Ok(())
    }

    /// The `questions` object as the API expects it.
    pub fn to_api_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        for (name, q) in &self.questions {
            m.insert(name.clone(), q.to_api_json());
        }
        serde_json::Value::Object(m)
    }

    /// Stable content hash of the schema (hex SHA-256 of canonical JSON).
    pub fn hash(&self) -> String {
        crate::hash::hash_json(&self.to_api_json())
    }
}
