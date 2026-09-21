//! Raw (wire) answers and the typed answers you actually work with.
//!
//! * [`RawAnswer`] / [`RawAnswers`] mirror the API response.
//! * [`Noul`], [`Choice`], [`Score`] are the typed views a
//!   `#[derive(JevQuestions)]` struct is filled with. They carry the full
//!   probability distribution, not just the arg-max, so downstream code can
//!   make cost-aware decisions.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::traits::JevChoice;

// ---------------------------------------------------------------------------
// Wire level
// ---------------------------------------------------------------------------

/// One answer as returned by a backend, before typing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RawAnswer {
    Noul {
        /// `P(yes)`.
        noul: f64,
        /// Not documented for `noul` by the API; kept optional for custom backends.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        confidence: Option<f64>,
    },
    Choice {
        /// Arg-max option key.
        choice: String,
        /// `option_key -> probability`.
        probabilities: BTreeMap<String, f64>,
        #[serde(default)]
        confidence: f64,
    },
    Score {
        /// Probability-weighted level index; may land between levels.
        score: f64,
        /// `"0" -> "Calm"`, … as returned by the API.
        #[serde(default)]
        legend: BTreeMap<String, String>,
        /// `"0" -> probability`, …
        probabilities: BTreeMap<String, f64>,
        #[serde(default)]
        confidence: f64,
    },
}

/// All answers for one request, keyed by question name.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RawAnswers {
    pub answers: BTreeMap<String, RawAnswer>,
    /// Model identifier reported by the backend (e.g. `jev-1.13.0`).
    #[serde(default)]
    pub model: String,
    /// Token usage if the backend reports it.
    #[serde(default)]
    pub usage: Usage,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
}

impl RawAnswers {
    pub fn get(&self, name: &str) -> Result<&RawAnswer> {
        self.answers
            .get(name)
            .ok_or_else(|| Error::MissingAnswer(name.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Typed level
// ---------------------------------------------------------------------------

/// A calibrated yes/no answer: `P(yes)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Noul {
    /// Probability that the answer is *yes*.
    pub p: f64,
}

impl Noul {
    pub fn new(p: f64) -> Self {
        Self { p: p.clamp(0.0, 1.0) }
    }

    /// Plain threshold.
    pub fn is_likely(&self, threshold: f64) -> bool {
        self.p >= threshold
    }

    /// Cost-aware decision. Returns `true` (act as if *yes*) when the expected
    /// cost of acting is lower than the expected cost of not acting.
    ///
    /// * `cost_false_positive` — cost of acting when the truth is *no*.
    /// * `cost_false_negative` — cost of not acting when the truth is *yes*.
    ///
    /// The break-even threshold is `fp / (fp + fn)`. A wrong block that costs
    /// 1 and a missed catch that costs 9 gives a 0.1 threshold, i.e. act
    /// aggressively; the reverse gives 0.9.
    pub fn decide(&self, cost_false_positive: f64, cost_false_negative: f64) -> bool {
        let expected_cost_act = (1.0 - self.p) * cost_false_positive;
        let expected_cost_skip = self.p * cost_false_negative;
        expected_cost_act < expected_cost_skip
    }

    /// Binary entropy in bits. 0 when certain, 1 when `p == 0.5`.
    pub fn entropy(&self) -> f64 {
        binary_entropy(self.p)
    }

    /// Draw a boolean from the distribution.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        rng.gen::<f64>() < self.p
    }

    #[doc(hidden)]
    pub fn from_raw(name: &str, raw: &RawAnswer) -> Result<Self> {
        match raw {
            RawAnswer::Noul { noul, .. } => Ok(Self::new(*noul)),
            other => Err(Error::SchemaMismatch {
                question: name.into(),
                reason: format!("expected noul, got {}", kind_of(other)),
            }),
        }
    }
}

/// A calibrated distribution over the variants of a `#[derive(JevChoice)]` enum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Choice<E: JevChoice> {
    /// Arg-max variant as reported by the backend.
    pub chosen: E,
    /// Probability of every variant, in the enum's declaration order.
    pub probabilities: Vec<(E, f64)>,
    /// Backend-reported confidence in the arg-max.
    pub confidence: f64,
}

impl<E: JevChoice> Choice<E> {
    /// The most likely variant.
    pub fn argmax(&self) -> E {
        self.chosen
    }

    /// Probability of a specific variant.
    pub fn p(&self, variant: E) -> f64 {
        self.probabilities
            .iter()
            .find(|(v, _)| *v == variant)
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    }

    /// Shannon entropy of the distribution, in bits.
    pub fn entropy(&self) -> f64 {
        entropy_bits(self.probabilities.iter().map(|(_, p)| *p))
    }

    /// Variants sorted by descending probability.
    pub fn ranked(&self) -> Vec<(E, f64)> {
        let mut v = self.probabilities.clone();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    /// Draw a variant from the distribution. Useful for stochastic agents in
    /// simulations: the same state yields naturally varied behaviour.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> E {
        let mut u = rng.gen::<f64>();
        for (v, p) in &self.probabilities {
            if u < *p {
                return *v;
            }
            u -= p;
        }
        self.chosen
    }

    /// Pick the *action* that minimises expected cost under this belief.
    ///
    /// `cost(truth, action)` is the loss when the true variant is `truth` and
    /// we choose `action`. Actions are drawn from the same enum. This turns a
    /// probability vector into a decision without hand-picking thresholds.
    pub fn decide<F: Fn(E, E) -> f64>(&self, cost: F) -> E {
        let mut best = self.chosen;
        let mut best_cost = f64::INFINITY;
        for &action in E::all() {
            let expected: f64 = self
                .probabilities
                .iter()
                .map(|(truth, p)| p * cost(*truth, action))
                .sum();
            if expected < best_cost {
                best_cost = expected;
                best = action;
            }
        }
        best
    }

    #[doc(hidden)]
    pub fn from_raw(name: &str, raw: &RawAnswer) -> Result<Self> {
        match raw {
            RawAnswer::Choice { choice, probabilities, confidence } => {
                let chosen = E::from_key(choice).ok_or_else(|| Error::SchemaMismatch {
                    question: name.into(),
                    reason: format!("unknown option `{choice}`"),
                })?;
                let mut probs = Vec::with_capacity(E::all().len());
                for &v in E::all() {
                    let p = probabilities.get(v.key()).copied().unwrap_or(0.0);
                    probs.push((v, p));
                }
                Ok(Self { chosen, probabilities: probs, confidence: *confidence })
            }
            other => Err(Error::SchemaMismatch {
                question: name.into(),
                reason: format!("expected choice, got {}", kind_of(other)),
            }),
        }
    }
}

/// A calibrated distribution over an ordered scale of 2–10 levels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Score {
    /// Probability-weighted level index (`Σ i·pᵢ`). May fall between levels.
    pub value: f64,
    /// Probability of each level, index 0 = lowest.
    pub probabilities: Vec<f64>,
    /// Level descriptions as sent in the schema / echoed by the backend.
    pub legend: Vec<String>,
    pub confidence: f64,
}

impl Score {
    /// Most likely level index.
    pub fn argmax(&self) -> usize {
        self.probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Description of the most likely level.
    pub fn argmax_label(&self) -> &str {
        self.legend.get(self.argmax()).map(String::as_str).unwrap_or("")
    }

    /// `P(level >= idx)` — handy for "at least Frustrated" style gates.
    pub fn p_at_least(&self, idx: usize) -> f64 {
        self.probabilities.iter().skip(idx).sum()
    }

    /// Normalised value in `[0, 1]` regardless of the number of levels.
    pub fn normalised(&self) -> f64 {
        let n = self.probabilities.len();
        if n <= 1 {
            0.0
        } else {
            self.value / (n as f64 - 1.0)
        }
    }

    /// Shannon entropy in bits.
    pub fn entropy(&self) -> f64 {
        entropy_bits(self.probabilities.iter().copied())
    }

    #[doc(hidden)]
    pub fn from_raw(name: &str, raw: &RawAnswer, levels: &[String]) -> Result<Self> {
        match raw {
            RawAnswer::Score { score, legend, probabilities, confidence } => {
                let n = levels.len();
                let mut probs = vec![0.0; n];
                for (k, p) in probabilities {
                    let i: usize = k.parse().map_err(|_| Error::SchemaMismatch {
                        question: name.into(),
                        reason: format!("non-integer level key `{k}`"),
                    })?;
                    if i >= n {
                        return Err(Error::SchemaMismatch {
                            question: name.into(),
                            reason: format!("level {i} out of range (schema has {n})"),
                        });
                    }
                    probs[i] = *p;
                }
                let legend_vec = if legend.is_empty() {
                    levels.to_vec()
                } else {
                    (0..n)
                        .map(|i| legend.get(&i.to_string()).cloned().unwrap_or_else(|| levels[i].clone()))
                        .collect()
                };
                Ok(Self { value: *score, probabilities: probs, legend: legend_vec, confidence: *confidence })
            }
            other => Err(Error::SchemaMismatch {
                question: name.into(),
                reason: format!("expected score, got {}", kind_of(other)),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn kind_of(a: &RawAnswer) -> &'static str {
    match a {
        RawAnswer::Noul { .. } => "noul",
        RawAnswer::Choice { .. } => "choice",
        RawAnswer::Score { .. } => "score",
    }
}

/// Binary entropy in bits.
pub fn binary_entropy(p: f64) -> f64 {
    entropy_bits([p, 1.0 - p].into_iter())
}

/// Shannon entropy in bits of an (approximately normalised) distribution.
pub fn entropy_bits<I: Iterator<Item = f64>>(probs: I) -> f64 {
    -probs
        .filter(|p| *p > 0.0)
        .map(|p| p * p.log2())
        .sum::<f64>()
}
