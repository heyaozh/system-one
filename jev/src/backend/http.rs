//! Official TypeSafe AI Jev backend.
//!
//! ```text
//! POST https://api.typesafe.ai/v1/systemone
//! Authorization: Bearer <JEV_API_KEY>
//! { "model": "jev-latest", "state": ..., "questions": { ... } }
//! ```

use async_trait::async_trait;
use std::time::Duration;

use crate::answer::RawAnswers;
use crate::backend::DecisionBackend;
use crate::error::{Error, Result};
use crate::schema::QuestionSchema;

/// Default endpoint of the System One API.
pub const DEFAULT_ENDPOINT: &str = "https://api.typesafe.ai/v1/systemone";
/// Default model route.
pub const DEFAULT_MODEL: &str = "jev-latest";

/// Client for the official Jev API.
#[derive(Clone)]
pub struct JevHttp {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
    max_retries: u32,
    base_backoff: Duration,
}

impl JevHttp {
    /// Build a client with an explicit key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("reqwest client"),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            max_retries: 4,
            base_backoff: Duration::from_millis(250),
        }
    }

    /// Read `JEV_API_KEY` (and optionally `JEV_ENDPOINT`, `JEV_MODEL`) from the
    /// environment. Returns `None` when no key is set so callers can fall back
    /// to a [`crate::backend::Mock`].
    pub fn from_env() -> Option<Self> {
        let key = std::env::var("JEV_API_KEY").ok()?;
        let mut me = Self::new(key);
        if let Ok(ep) = std::env::var("JEV_ENDPOINT") {
            me.endpoint = ep;
        }
        if let Ok(m) = std::env::var("JEV_MODEL") {
            me.model = m;
        }
        Some(me)
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Number of retries on 429 / 529 / transport errors (default 4).
    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    fn body(&self, state: &serde_json::Value, schema: &QuestionSchema) -> serde_json::Value {
        serde_json::json!({
            "model": self.model,
            "state": state,
            "questions": schema.to_api_json(),
        })
    }
}

#[async_trait]
impl DecisionBackend for JevHttp {
    fn id(&self) -> String {
        format!("jev-http:{}", self.model)
    }

    async fn decide(&self, state: &serde_json::Value, schema: &QuestionSchema) -> Result<RawAnswers> {
        schema.validate()?;
        let body = self.body(state, schema);
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let resp = self
                .client
                .post(&self.endpoint)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .await;

            match resp {
                Ok(r) => {
                    let status = r.status();
                    if status.is_success() {
                        return r.json::<RawAnswers>().await.map_err(|e| Error::Backend {
                            backend: self.id(),
                            message: format!("invalid response body: {e}"),
                        });
                    }
                    let retryable = status.as_u16() == 429 || status.as_u16() == 529;
                    if retryable && attempt <= self.max_retries {
                        tokio::time::sleep(self.base_backoff * 2u32.pow(attempt - 1)).await;
                        continue;
                    }
                    if retryable {
                        return Err(Error::RateLimited { backend: self.id(), attempts: attempt });
                    }
                    let text = r.text().await.unwrap_or_default();
                    return Err(Error::Backend {
                        backend: self.id(),
                        message: format!("HTTP {status}: {text}"),
                    });
                }
                Err(e) if attempt <= self.max_retries && (e.is_timeout() || e.is_connect()) => {
                    tokio::time::sleep(self.base_backoff * 2u32.pow(attempt - 1)).await;
                }
                Err(e) => {
                    return Err(Error::Backend { backend: self.id(), message: e.to_string() });
                }
            }
        }
    }
}
