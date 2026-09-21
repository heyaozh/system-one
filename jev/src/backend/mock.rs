//! Deterministic backend for tests and offline development.

use async_trait::async_trait;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::answer::{RawAnswer, RawAnswers};
use crate::backend::DecisionBackend;
use crate::error::Result;
use crate::schema::{QuestionSchema, QuestionSpec};

/// Signature of a custom answer rule: `(state, question_name, spec) -> answer`.
pub type Rule = dyn Fn(&serde_json::Value, &str, &QuestionSpec) -> Option<RawAnswer> + Send + Sync;

/// A backend that answers from a rule, or with a uniform distribution when
/// the rule returns `None`.
///
/// ```ignore
/// let mock = Mock::uniform();                 // every option equally likely
/// let mock = Mock::with_rule(|state, name, spec| { ... });
/// ```
#[derive(Clone)]
pub struct Mock {
    rule: Option<Arc<Rule>>,
    id: String,
}

impl Mock {
    /// Uniform probabilities for every question — enough to exercise
    /// schemas, decoding and control flow.
    pub fn uniform() -> Self {
        Self { rule: None, id: "mock:uniform".into() }
    }

    /// Answer with a custom rule. Questions the rule does not answer fall
    /// back to uniform.
    pub fn with_rule<F>(rule: F) -> Self
    where
        F: Fn(&serde_json::Value, &str, &QuestionSpec) -> Option<RawAnswer> + Send + Sync + 'static,
    {
        Self { rule: Some(Arc::new(rule)), id: "mock:rule".into() }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Uniform answer for a spec. Public so custom rules can fall back to it.
    pub fn uniform_answer(spec: &QuestionSpec) -> RawAnswer {
        match spec {
            QuestionSpec::Noul { .. } => RawAnswer::Noul { noul: 0.5, confidence: None },
            QuestionSpec::Choice { criteria, .. } => {
                let p = 1.0 / criteria.len() as f64;
                let probabilities: BTreeMap<String, f64> =
                    criteria.iter().map(|(k, _)| (k.clone(), p)).collect();
                RawAnswer::Choice {
                    choice: criteria[0].0.clone(),
                    probabilities,
                    confidence: p,
                }
            }
            QuestionSpec::Score { criteria, .. } => {
                let n = criteria.len();
                let p = 1.0 / n as f64;
                let probabilities: BTreeMap<String, f64> =
                    (0..n).map(|i| (i.to_string(), p)).collect();
                let legend: BTreeMap<String, String> =
                    criteria.iter().enumerate().map(|(i, l)| (i.to_string(), l.clone())).collect();
                RawAnswer::Score {
                    score: (n as f64 - 1.0) / 2.0,
                    legend,
                    probabilities,
                    confidence: p,
                }
            }
        }
    }
}

#[async_trait]
impl DecisionBackend for Mock {
    fn id(&self) -> String {
        self.id.clone()
    }

    async fn decide(&self, state: &serde_json::Value, schema: &QuestionSchema) -> Result<RawAnswers> {
        schema.validate()?;
        let mut answers = BTreeMap::new();
        for (name, spec) in &schema.questions {
            let a = self
                .rule
                .as_ref()
                .and_then(|r| r(state, name, spec))
                .unwrap_or_else(|| Self::uniform_answer(spec));
            answers.insert(name.clone(), a);
        }
        Ok(RawAnswers { answers, model: self.id.clone(), usage: Default::default() })
    }
}

/// Helpers for writing mock rules tersely.
pub mod answers {
    use super::*;

    /// A `noul` answer.
    pub fn noul(p: f64) -> RawAnswer {
        RawAnswer::Noul { noul: p, confidence: None }
    }

    /// A `choice` answer from `(key, probability)` pairs; the arg-max is
    /// chosen automatically and probabilities are normalised.
    pub fn choice<'a>(pairs: impl IntoIterator<Item = (&'a str, f64)>) -> RawAnswer {
        let mut probabilities: BTreeMap<String, f64> =
            pairs.into_iter().map(|(k, p)| (k.to_string(), p)).collect();
        let total: f64 = probabilities.values().sum();
        if total > 0.0 {
            for p in probabilities.values_mut() {
                *p /= total;
            }
        }
        let (choice, conf) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, p)| (k.clone(), *p))
            .unwrap_or_default();
        RawAnswer::Choice { choice, probabilities, confidence: conf }
    }

    /// A `score` answer from per-level probabilities (index 0 = lowest).
    pub fn score(level_probs: &[f64], legend: &[&str]) -> RawAnswer {
        let total: f64 = level_probs.iter().sum();
        let probs: Vec<f64> = level_probs.iter().map(|p| if total > 0.0 { p / total } else { *p }).collect();
        let score: f64 = probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
        let confidence = probs.iter().cloned().fold(0.0, f64::max);
        RawAnswer::Score {
            score,
            legend: legend.iter().enumerate().map(|(i, l)| (i.to_string(), l.to_string())).collect(),
            probabilities: probs.iter().enumerate().map(|(i, p)| (i.to_string(), *p)).collect(),
            confidence,
        }
    }
}
