//! Example 4 — the data loop: batch-label, record, attach outcomes, measure
//! calibration.
//!
//! This is the "training-set engine" in miniature:
//!
//! 1. `ask_many` labels a stream of states concurrently (here: synthetic
//!    news headlines) and records every answer to JSONL.
//! 2. Later, real outcomes arrive (here: simulated) and are attached by
//!    state hash.
//! 3. `CalibrationReport` shows whether "0.8" actually meant 80 %.
//! 4. With `--features parquet` the recording is exported to Parquet for
//!    pandas / polars / your trainer.
//!
//! Run:  `cargo run --example batch_and_calibrate`
//!       `cargo run --example batch_and_calibrate --features parquet`

use jev::backend::{answers, DecisionBackend, Mock};
use jev::prelude::*;
use jev::record::{attach_outcomes, read_records};
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Clone, Serialize)]
struct Headline {
    id: u32,
    text: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, JevChoice)]
enum Topic {
    Earnings,
    Regulation,
    Macro,
    Other,
}

#[derive(Debug, JevQuestions)]
#[allow(dead_code)]
struct NewsFeatures {
    #[jev("Does the headline describe an event that was not previously scheduled or expected?")]
    is_surprise: Noul,
    #[jev("Is the headline about a merger, acquisition or takeover?")]
    is_m_and_a: Noul,
    #[jev("What is the main topic of the headline?")]
    topic: Choice<Topic>,
}

/// A mock whose `is_surprise` probability is deliberately *over-confident*
/// (a real 60 % case is reported as 85 %), so the calibration report has
/// something to show. Replace with `JevHttp::from_env()` for the real thing.
fn backend() -> Box<dyn DecisionBackend> {
    if let Some(b) = JevHttp::from_env() {
        return Box::new(b);
    }
    Box::new(Mock::with_rule(|state, name, spec| {
        let text = state["text"].as_str().unwrap_or("").to_lowercase();
        match name {
            "is_surprise" => Some(answers::noul(
                if text.contains("unexpected") || text.contains("shock") {
                    0.85
                } else {
                    0.15
                },
            )),
            "is_m_and_a" => Some(answers::noul(if text.contains("acquire") || text.contains("merger") {
                0.9
            } else {
                0.05
            })),
            "topic" => Some(answers::choice([
                (
                    "earnings",
                    if text.contains("earnings") || text.contains("guidance") {
                        0.8
                    } else {
                        0.05
                    },
                ),
                (
                    "regulation",
                    if text.contains("regulator") || text.contains("fine") {
                        0.8
                    } else {
                        0.05
                    },
                ),
                (
                    "macro",
                    if text.contains("rate") || text.contains("inflation") {
                        0.8
                    } else {
                        0.05
                    },
                ),
                ("other", 0.1),
            ])),
            _ => Some(Mock::uniform_answer(spec)),
        }
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let path = "runs/news_features.jsonl";
    let _ = std::fs::remove_file(path);
    let engine = Engine::new(backend())
        .with_recorder(Recorder::open(path)?)
        .with_concurrency(8)
        .with_cache();

    // 1. Synthetic headline stream.
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let templates = [
        "{co} reports earnings above guidance",
        "Regulator fines {co} over disclosure lapses",
        "{co} in talks to acquire rival, sources say",
        "Central bank holds rates, signals patience on inflation",
        "Unexpected CEO departure at {co} shocks investors",
        "{co} announces scheduled dividend, unchanged",
    ];
    let cos = ["Acme", "Globex", "Initech", "Umbrella", "Vandelay"];
    let headlines: Vec<Headline> = (0..300)
        .map(|i| Headline {
            id: i,
            text: templates[rng.gen_range(0..templates.len())].replace("{co}", cos[rng.gen_range(0..cos.len())]),
        })
        .collect();

    // 2. Label them all, 8 in flight at a time.
    let results = engine.ask_many_with::<NewsFeatures, _, _>(headlines.clone()).await;
    let ok = results.iter().filter(|(_, r)| r.is_ok()).count();
    println!("labelled {ok}/{} headlines -> {path}", results.len());

    // 3. Outcomes arrive later. Simulate a world where a "surprise" headline
    //    is genuinely surprising only 60 % of the time, and non-surprise
    //    headlines are surprising 10 % of the time — then attach by state hash.
    let mut outcomes: HashMap<String, serde_json::Value> = HashMap::new();
    for (h, _) in &results {
        let looks_surprising = h.text.contains("Unexpected") || h.text.contains("shock");
        let truly_surprising = rng.gen::<f64>() < if looks_surprising { 0.6 } else { 0.1 };
        let truly_m_and_a = h.text.contains("acquire");
        let topic = if h.text.contains("earnings") || h.text.contains("dividend") {
            "earnings"
        } else if h.text.contains("Regulator") {
            "regulation"
        } else if h.text.contains("rates") {
            "macro"
        } else {
            "other"
        };
        let state_hash = jev::hash::hash_json(&serde_json::to_value(h)?);
        outcomes.insert(
            state_hash,
            serde_json::json!({ "is_surprise": truly_surprising, "is_m_and_a": truly_m_and_a, "topic": topic }),
        );
    }
    let n = attach_outcomes(path, &outcomes)?;
    println!("attached outcomes to {n} records\n");

    // 4. Calibration per question.
    let records = read_records(path)?;
    for q in ["is_surprise", "is_m_and_a", "topic"] {
        let report = CalibrationReport::from_records(&records, q, 10);
        println!("== {q} ==\n{}", report.render());
    }

    #[cfg(feature = "parquet")]
    {
        let out = "runs/news_features.parquet";
        let n = jev::record::export_parquet(path, out)?;
        println!("exported {n} rows to {out}");
    }
    Ok(())
}
