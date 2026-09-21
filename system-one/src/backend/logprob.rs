//! Your own model as a Jev-style backend, via an OpenAI-compatible server.
//!
//! Works with vLLM, llama.cpp `llama-server`, LM Studio, Ollama (with the
//! OpenAI-compatible endpoint), TGI, … — anything that returns `logprobs`.
//!
//! How it works: each question is rendered as a multiple-choice prompt whose
//! options are labelled `A`, `B`, `C`, …; the model is asked for exactly one
//! token; the log-probabilities of the label tokens are read out and
//! renormalised into a distribution. No text is generated, no parsing can
//! fail, and the result has the same shape as an official Jev answer.
//!
//! This is the *read-out* half of a System One model. The *training* half —
//! fine-tuning the model with a proper scoring rule so those probabilities
//! are calibrated — is up to you; the crate's recorder and calibration report
//! give you the data and the measurement.
//!
//! Limits: a `choice` may have at most 26 options with this backend (one
//! letter per option). `score` levels (2–10) and `noul` are always fine.

use async_trait::async_trait;
use std::collections::BTreeMap;
use std::time::Duration;

use crate::answer::{RawAnswer, RawAnswers};
use crate::backend::DecisionBackend;
use crate::error::{Error, Result};
use crate::schema::{QuestionSchema, QuestionSpec};

const LABELS: &[&str] = &[
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
    "X", "Y", "Z",
];

/// Which OpenAI-compatible endpoint to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptStyle {
    /// `POST {base}/chat/completions` with `logprobs: true, top_logprobs: N`.
    Chat,
    /// `POST {base}/completions` with `logprobs: N` (legacy / base models).
    Completion,
}

/// Log-prob read-out backend for a local or self-hosted model.
#[derive(Clone)]
pub struct LocalLogprob {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
    style: PromptStyle,
    top_logprobs: u32,
    system_prompt: String,
}

impl LocalLogprob {
    /// `base_url` like `http://localhost:8000/v1`; `model` as the server names it.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("reqwest client"),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            api_key: None,
            style: PromptStyle::Chat,
            top_logprobs: 20,
            system_prompt: "You are a calibrated classifier. Read the state, then answer the question \
                            with exactly one option label (a single letter) and nothing else."
                .into(),
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_style(mut self, style: PromptStyle) -> Self {
        self.style = style;
        self
    }

    /// How many top log-probs to request (default 20; must cover the labels).
    pub fn with_top_logprobs(mut self, n: u32) -> Self {
        self.top_logprobs = n;
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Render one question as a labelled multiple-choice prompt.
    /// Returns `(prompt, option_keys_in_label_order)`.
    fn render(&self, state: &serde_json::Value, name: &str, spec: &QuestionSpec) -> Result<(String, Vec<String>)> {
        let state_text = match state {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string_pretty(other)?,
        };
        let (instructions, options): (String, Vec<(String, Option<String>)>) = match spec {
            QuestionSpec::Noul { instructions, criteria } => {
                let yes = criteria.as_ref().map(|c| c.yes.clone());
                let no = criteria.as_ref().map(|c| c.no.clone());
                (instructions.clone(), vec![("yes".into(), yes), ("no".into(), no)])
            }
            QuestionSpec::Choice { instructions, criteria } => (instructions.clone(), criteria.clone()),
            QuestionSpec::Score { instructions, criteria } => (
                format!("{instructions} (ordered scale, lowest first)"),
                criteria
                    .iter()
                    .enumerate()
                    .map(|(i, l)| (i.to_string(), Some(l.clone())))
                    .collect(),
            ),
        };
        if options.len() > LABELS.len() {
            return Err(Error::InvalidSchema(format!(
                "question `{name}` has {} options; LocalLogprob supports at most {}",
                options.len(),
                LABELS.len()
            )));
        }
        let mut prompt = format!("State:\n{state_text}\n\nQuestion ({name}): {instructions}\nOptions:\n");
        let mut keys = Vec::with_capacity(options.len());
        for (i, (key, desc)) in options.iter().enumerate() {
            match desc {
                Some(d) if !d.is_empty() => prompt.push_str(&format!("{}) {key} — {d}\n", LABELS[i])),
                _ => prompt.push_str(&format!("{}) {key}\n", LABELS[i])),
            }
            keys.push(key.clone());
        }
        prompt.push_str("\nAnswer with the option letter only.\nAnswer:");
        Ok((prompt, keys))
    }

    /// Query the server and return `label -> logprob` for the first token.
    async fn first_token_logprobs(&self, prompt: &str) -> Result<BTreeMap<String, f64>> {
        let (url, body) = match self.style {
            PromptStyle::Chat => (
                format!("{}/chat/completions", self.base_url),
                serde_json::json!({
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1,
                    "temperature": 0,
                    "logprobs": true,
                    "top_logprobs": self.top_logprobs,
                }),
            ),
            PromptStyle::Completion => (
                format!("{}/completions", self.base_url),
                serde_json::json!({
                    "model": self.model,
                    "prompt": format!("{}\n\n{}", self.system_prompt, prompt),
                    "max_tokens": 1,
                    "temperature": 0,
                    "logprobs": self.top_logprobs,
                }),
            ),
        };
        let mut req = self.client.post(&url).json(&body);
        if let Some(k) = &self.api_key {
            req = req.bearer_auth(k);
        }
        let resp = req.send().await.map_err(|e| Error::Backend {
            backend: self.id(),
            message: e.to_string(),
        })?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Backend {
                backend: self.id(),
                message: format!("HTTP {status}: {text}"),
            });
        }
        let v: serde_json::Value = resp.json().await.map_err(|e| Error::Backend {
            backend: self.id(),
            message: e.to_string(),
        })?;

        let mut out = BTreeMap::new();
        match self.style {
            PromptStyle::Chat => {
                // choices[0].logprobs.content[0].top_logprobs: [{token, logprob}]
                if let Some(list) = v
                    .pointer("/choices/0/logprobs/content/0/top_logprobs")
                    .and_then(|x| x.as_array())
                {
                    for item in list {
                        if let (Some(tok), Some(lp)) = (item["token"].as_str(), item["logprob"].as_f64()) {
                            out.insert(normalise_token(tok), lp);
                        }
                    }
                }
            }
            PromptStyle::Completion => {
                // choices[0].logprobs.top_logprobs[0]: {token: logprob}
                if let Some(map) = v
                    .pointer("/choices/0/logprobs/top_logprobs/0")
                    .and_then(|x| x.as_object())
                {
                    for (tok, lp) in map {
                        if let Some(lp) = lp.as_f64() {
                            out.insert(normalise_token(tok), lp);
                        }
                    }
                }
            }
        }
        if out.is_empty() {
            return Err(Error::Backend {
                backend: self.id(),
                message: "server returned no logprobs; enable logprobs on the server".into(),
            });
        }
        Ok(out)
    }

    async fn answer_one(&self, state: &serde_json::Value, name: &str, spec: &QuestionSpec) -> Result<RawAnswer> {
        let (prompt, keys) = self.render(state, name, spec)?;
        let lps = self.first_token_logprobs(&prompt).await?;
        // Read out P(label) for each option, renormalise over the option set.
        let mut probs: Vec<f64> = keys
            .iter()
            .enumerate()
            .map(|(i, _)| lps.get(LABELS[i]).map(|lp| lp.exp()).unwrap_or(0.0))
            .collect();
        let total: f64 = probs.iter().sum();
        if total <= 0.0 {
            // None of the labels appeared in the top-k: fall back to uniform
            // rather than failing — and make it visible via confidence = 1/n.
            let n = probs.len() as f64;
            probs.iter_mut().for_each(|p| *p = 1.0 / n);
        } else {
            probs.iter_mut().for_each(|p| *p /= total);
        }
        let (argmax, conf) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, p)| (i, *p))
            .unwrap_or((0, 0.0));

        Ok(match spec {
            QuestionSpec::Noul { .. } => RawAnswer::Noul {
                noul: probs[0],
                confidence: Some(conf),
            },
            QuestionSpec::Choice { .. } => RawAnswer::Choice {
                choice: keys[argmax].clone(),
                probabilities: keys.iter().cloned().zip(probs.iter().copied()).collect(),
                confidence: conf,
            },
            QuestionSpec::Score { criteria, .. } => RawAnswer::Score {
                score: probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum(),
                legend: criteria
                    .iter()
                    .enumerate()
                    .map(|(i, l)| (i.to_string(), l.clone()))
                    .collect(),
                probabilities: probs.iter().enumerate().map(|(i, p)| (i.to_string(), *p)).collect(),
                confidence: conf,
            },
        })
    }
}

fn normalise_token(tok: &str) -> String {
    tok.trim()
        .trim_matches(|c: char| c == ')' || c == '.' || c == ':')
        .to_uppercase()
}

#[async_trait]
impl DecisionBackend for LocalLogprob {
    fn id(&self) -> String {
        format!("local-logprob:{}", self.model)
    }

    async fn decide(&self, state: &serde_json::Value, schema: &QuestionSchema) -> Result<RawAnswers> {
        schema.validate()?;
        // Questions are independent: fire them concurrently, like Jev does.
        let futs = schema.questions.iter().map(|(name, spec)| async move {
            let a = self.answer_one(state, name, spec).await?;
            Ok::<_, Error>((name.clone(), a))
        });
        let pairs = futures::future::try_join_all(futs).await?;
        Ok(RawAnswers {
            answers: pairs.into_iter().collect(),
            model: self.model.clone(),
            usage: Default::default(),
        })
    }
}
