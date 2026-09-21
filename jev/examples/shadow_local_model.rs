//! Example 5 — run your own model in the shadow of the official API.
//!
//! `Shadow` answers from the primary backend and, in parallel, asks the
//! shadow backend the same question. Both answers are recorded with a
//! `primary` / `shadow` tag. Business logic only ever sees the primary.
//! When the shadow's calibration report looks better, swap the two — one
//! line, nothing else changes.
//!
//! The shadow here is `LocalLogprob`: any model behind an OpenAI-compatible
//! server (vLLM, llama.cpp `llama-server`, LM Studio, Ollama). It reads the
//! log-probabilities of the option labels instead of generating text.
//!
//! Run:
//!   JEV_API_KEY=...  LOCAL_LLM_URL=http://localhost:8000/v1  LOCAL_LLM_MODEL=qwen2.5-7b \
//!   cargo run --example shadow_local_model
//!
//! Without those variables the example explains what it would do and exits.

use jev::backend::Mock;
use jev::prelude::*;
use jev::record::read_records;

#[derive(Clone, Copy, PartialEq, Eq, Debug, JevChoice)]
enum Direction {
    Up,
    Down,
    Flat,
}

#[derive(Debug, JevQuestions)]
struct HeadlineView {
    #[jev("Given only this headline, which direction is the named company's share price more likely to move over the next hour?")]
    direction: Choice<Direction>,
    #[jev("Is the information in the headline likely already priced in (widely expected beforehand)?")]
    priced_in: Noul,
}

#[tokio::main]
async fn main() -> Result<()> {
    let Some(primary) = JevHttp::from_env() else {
        eprintln!("JEV_API_KEY not set — showing the wiring with a Mock primary instead.");
        return demo_with_mock().await;
    };
    let (Ok(url), Ok(model)) = (std::env::var("LOCAL_LLM_URL"), std::env::var("LOCAL_LLM_MODEL")) else {
        eprintln!("set LOCAL_LLM_URL and LOCAL_LLM_MODEL to point at an OpenAI-compatible server");
        return Ok(());
    };
    let shadow = LocalLogprob::new(url, model);
    run(Shadow::new(primary, shadow)).await
}

async fn demo_with_mock() -> Result<()> {
    let primary = Mock::uniform().with_id("mock:primary");
    let shadow = Mock::uniform().with_id("mock:shadow");
    run(Shadow::new(primary, shadow)).await
}

async fn run<B: jev::backend::DecisionBackend>(backend: B) -> Result<()> {
    let path = "runs/shadow.jsonl";
    let _ = std::fs::remove_file(path);
    let backend = jev::backend::Shadow::new(backend, Mock::uniform().with_id("mock:noop"))
        .with_recorder(Recorder::open(path)?);
    // ^ In real use you would pass the Shadow directly; wrapping again here
    //   only serves to keep this function generic over any backend.
    let engine = Engine::new(backend);

    let headlines = [
        "Acme beats earnings estimates, raises full-year guidance",
        "Globex hit with record regulatory fine",
        "Initech to be acquired by Umbrella at 30% premium",
    ];
    for h in headlines {
        let v: HeadlineView = engine.ask(h).await?;
        println!("{h}\n  direction={:?} ({:.2})  priced_in={:.2}", v.direction.argmax(), v.direction.confidence, v.priced_in.p);
    }

    let records = read_records(path)?;
    let primary = records.iter().filter(|r| r.tag.as_deref() == Some("primary")).count();
    let shadow = records.iter().filter(|r| r.tag.as_deref() == Some("shadow")).count();
    println!("\nrecorded {primary} primary + {shadow} shadow answers to {path}");
    println!("attach outcomes later with jev::record::attach_outcomes, then compare\n  CalibrationReport::from_records(primary_only, \"direction\", 10)\nagainst the shadow subset.");
    Ok(())
}
