<p align="center">
  <img src="docs/logo.svg" alt="system-one" width="120"/>
</p>

# system-one

**English** · [中文](README.zh-CN.md)

[![CI](https://github.com/heyaozh/system-one/actions/workflows/ci.yml/badge.svg)](https://github.com/heyaozh/system-one/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Typed, calibrated decisions for Rust.**
A backend-agnostic client for *System One* models — models that answer a fixed set of questions with calibrated probabilities instead of generating text. The first backend is [TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev); the same interface serves a mock, a recording, or **your own fine-tuned model**.

---

## Why

Every "smart if-statement" has the same shape:

```
state (anything serialisable) + questions (bounded answer spaces) → answers with probabilities
```

Ticket routing, a pre-trade risk gate, an NPC's reflex layer, labelling a million headlines — same shape. `system-one` fixes that shape once and stacks the tools you need on top:

| Layer | What you get |
|---|---|
| **Types** | `Noul` (P(yes)), `Choice<E>` (distribution over an enum), `Score` (distribution over an ordered scale). Full distributions, not just arg-max. |
| **Derive** | `#[derive(AsChoice)]` on an enum, `#[derive(AsQuestions)]` on a struct → schema generated, answers parsed back. Compile-time checks (≤255 options, 2–10 levels, field types). |
| **Decision helpers** | `decide(cost)` minimises expected loss with a cost matrix; `entropy()` for active learning; `sample()` for stochastic agents. |
| **Backends** | `JevHttp` (official API), `Mock`, `Replay`, `LocalLogprob` (your model via vLLM / llama.cpp), `Shadow` (A/B two backends). Or implement `DecisionBackend` yourself. |
| **Engine** | Content-hash cache, concurrent `ask_many`, JSONL recording (Parquet with `--features parquet`). |
| **Calibration** | Attach real outcomes to recordings; get Brier, log-loss, ECE and a reliability table per question. |

The point is the **loop**, not the wrapper: *decide → record → attach outcomes → measure calibration → train a small model → swap the backend*. Business code never changes.

![architecture](docs/architecture.svg)

## Quick start

Not on crates.io yet, so take it from git:

```toml
[dependencies]
system-one = { git = "https://github.com/heyaozh/system-one" }
tokio = { version = "1", features = ["full"] }
```

Inside a checkout of this repo, `system-one = { path = "system-one" }` works too.

```rust
use system_one::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug, AsChoice)]
enum Department {
    /// Invoices, payouts, charges, refunds.
    Billing,
    /// Bugs, integrations, API errors.
    Technical,
    Sales,
    Spam,
}

#[derive(Debug, AsQuestions)]
struct Triage {
    #[ask("Which department should handle this ticket?")]
    department: Choice<Department>,
    #[ask("Is the customer explicitly asking for money back?")]
    wants_refund: Noul,
    #[ask("How urgent is this?", levels = ["Can wait a week", "Within a day", "Right now"])]
    urgency: Score,
}

#[tokio::main]
async fn main() -> system_one::Result<()> {
    // JEV_API_KEY=... in the environment; see examples for a Mock fallback.
    let engine = Engine::new(JevHttp::from_env().expect("JEV_API_KEY"))
        .with_recorder(Recorder::open("runs/triage.jsonl")?);

    let t: Triage = engine.ask("Hi, my payouts have failed for 3 days, I want the fees back.").await?;

    // Cost-aware: misrouting a real customer to Spam costs 5, other mistakes 1.
    let route = t.department.decide(|truth, action| match (truth, action) {
        (a, b) if a == b => 0.0,
        (_, Department::Spam) => 5.0,
        _ => 1.0,
    });
    // A missed refund request costs 4, a false flag costs 1 → threshold 0.2.
    let flag = t.wants_refund.decide(1.0, 4.0);

    println!("{route:?} refund={flag} urgency={}", t.urgency.argmax_label());
    Ok(())
}
```

Three questions, one call, ~100 ms, and the routing rule is an explicit cost matrix rather than a threshold someone guessed.

### Try one question first

Before writing any struct, try a question from the shell. `so-ask` is a small
binary shipped with the crate; it builds the schema at runtime, so there is no
Rust to write:

```bash
cargo install --path system-one        # once — puts `so-ask` on your PATH
export JEV_API_KEY=...          # without it you get a uniform Mock answer, and a note saying so

so-ask "Is this customer asking for a refund?" \
  "my payouts failed for 3 days, I want the fees back"

so-ask --choice billing,technical,sales,spam \
  "Which department should handle this?" "getting a 500 from /v1/orders"

so-ask --levels "Can wait a week,Within a day,Right now" \
  "How urgent is this?" "my payouts have failed for 3 days"
```

Without installing, from inside this repo: `cargo run --bin so-ask -- <args>`.

It prints the full distribution, not just the arg-max — which is the whole point.

## The three primitives

| Rust type | API type | Answer |
|---|---|---|
| `Noul` | `noul` | `p: f64` — probability of *yes* |
| `Choice<E>` | `choice` | `probabilities: Vec<(E, f64)>`, `chosen`, `confidence` |
| `Score` | `score` | `probabilities: Vec<f64>` over levels, `value` (probability-weighted index), `legend`, `confidence` |

`#[ask(...)]` field attributes: positional `"instructions"` (or a `///` doc comment), `levels = [...]` for `Score`, optional `yes = "…", no = "…"` criteria for `Noul`, `name = "…"` to override the wire name. Enum variants take `#[ask(key = "…", desc = "…")]` or a doc comment.

## Backends

```rust
// Official API (reads JEV_API_KEY, JEV_ENDPOINT, JEV_MODEL)
let b = JevHttp::from_env().unwrap();

// Deterministic mock for tests — a rule, or uniform
let b = Mock::with_rule(|state, question, spec| /* Option<RawAnswer> */ None);

// Replay a recording: no network, no cost, reproducible
let b = Replay::from_jsonl("runs/2026-09.jsonl")?.with_fallback(JevHttp::from_env().unwrap());

// Your own model behind an OpenAI-compatible server (vLLM, llama.cpp, LM Studio, Ollama)
let b = LocalLogprob::new("http://localhost:8000/v1", "my-finetune-v3");

// A/B: answer from primary, record both
let b = Shadow::new(JevHttp::from_env().unwrap(), LocalLogprob::new(url, model))
    .with_recorder(Recorder::open("runs/shadow.jsonl")?);
```

All of them implement one trait:

```rust
#[async_trait]
pub trait DecisionBackend: Send + Sync {
    fn id(&self) -> String;
    async fn decide(&self, state: &serde_json::Value, schema: &QuestionSchema) -> Result<RawAnswers>;
}
```

so `Engine<Box<dyn DecisionBackend>>` lets you pick the backend at runtime and keep the rest of the code identical.

**`LocalLogprob` is how your own model plugs in.** It renders each question as a labelled multiple-choice prompt, asks for one token, and reads the log-probabilities of the option labels — no generation, no parsing, a distribution every time. Fine-tune any open model with a proper scoring rule (cross-entropy on real outcomes) and it becomes a domain-specific System One model behind the same interface.

## The data loop

```rust
// 1. label a stream, 8 in flight, everything recorded
let engine = Engine::new(backend).with_recorder(Recorder::open("runs/news.jsonl")?).with_concurrency(8);
let results = engine.ask_many::<NewsFeatures, _, _>(headlines).await;

// 2. later, when reality has happened: join outcomes by state hash
let mut outcomes = HashMap::new();
outcomes.insert(system_one::hash::hash_json(&serde_json::to_value(&h)?), json!({ "is_surprise": true }));
system_one::record::attach_outcomes("runs/news.jsonl", &outcomes)?;

// 3. did 0.85 mean 85 %?
let records = system_one::record::read_records("runs/news.jsonl")?;
println!("{}", CalibrationReport::from_records(&records, "is_surprise", 10).render());
```

```
n=300  brier=0.1228  log_loss=0.4111  ece=0.0890  acc@0.5=0.857  base_rate=0.180
  bin          count   mean_p   frac_pos   |gap|
  [0.10,0.20)    249   0.150    0.092   0.058  ##
  [0.80,0.90)     51   0.850    0.608   0.242  ############
```

Those rows, with outcomes attached, are also a labelled training set. `--features parquet` exports them for pandas / polars.

## Examples

The `so-ask` binary above covers one-off questions. The examples are the full loops:

| Example | Shows |
|---|---|
| `ticket_routing` | The canonical smart if-statement; cost-matrix routing. |
| `trade_risk_gate` | Structured state (a proposed order + mandate) → pre-trade compliance gate with hard overrides. Generic trading, no exchange specifics. |
| `chat_reflex` | Reflex layer for a conversational character: response tier, animation clip (sampled, not arg-max), memory need, jailbreak risk. |
| `batch_and_calibrate` | `ask_many` → JSONL → `attach_outcomes` → calibration report (→ Parquet). |
| `shadow_local_model` | Official API as primary, your local model as shadow, both recorded. |

```bash
cargo run --example ticket_routing               # Mock backend, runs offline
JEV_API_KEY=... cargo run --example ticket_routing  # real API
cargo run --example batch_and_calibrate --features parquet
cargo test
```

Every example falls back to a rule-based `Mock` when `JEV_API_KEY` is unset, so they run anywhere.

## API key

A key comes from [TypeSafe AI](https://typesafe.ai) — this crate is a third-party client and issues none of its own.
`JevHttp::from_env()` reads `JEV_API_KEY`, and optionally `JEV_ENDPOINT` and `JEV_MODEL`.
Nothing is read from a config file, so the key never sits in the repo — `.env` is in `.gitignore` for the same reason.

On macOS, keep it in the Keychain and export it only in the shell that needs it:

```bash
# store (or replace) — -U updates an existing item; prompts for the value,
# so it stays out of shell history
security add-generic-password -U -a "$USER" -s JEV_API_KEY -w

# export per shell / per script
export JEV_API_KEY="$(security find-generic-password -a "$USER" -s JEV_API_KEY -w)"
```

On Linux use your secret store (`pass`, `keyctl`, systemd credentials); in CI use the runner's secret mechanism.
Exporting the key from a login profile (`~/.zshrc`, `~/.bash_profile`) works but hands it to every process you ever start — including editors, language servers and coding agents.

## Feature flags

* `http` (default) — `JevHttp` and `LocalLogprob` (pulls in `reqwest` with rustls).
* `derive` (default) — the derive macros.
* `parquet` — `system_one::record::export_parquet`.

## Limits worth knowing

* The official API allows ≤255 `choice` options and 2–10 `score` levels; the derive macro enforces this at compile time.
* `LocalLogprob` supports ≤26 options per `choice` (one letter per option) in this version.
* Jev returns probabilities but no rationale; a `Shadow` with an LLM judge, sampled occasionally, is the usual audit pattern.
* Calibration is a property of a distribution: a backend calibrated on its training data can be confidently wrong on yours. Measure before you trust — that is what the recorder is for.

## Status

`0.1.0`, not published to crates.io yet.

**What is verified:** the type system, derive macros, schema wire format, cost-matrix
decisions, cache, recorder, replay, shadow and the calibration maths — 11 integration
tests, all offline against `Mock`, plus every feature combination compiling on its own.

**What is not:** the automated suite never calls the live API — it runs entirely on
`Mock`, so it stays fast, free and deterministic. `JevHttp` has been exercised against
the real endpoint by hand (September 2026) and the wire format held, but that is one
person on one day, not a regression test. Point `so-ask` at your own key before you
trust it with anything that matters; it is one command and it tells you immediately.

Not affiliated with TypeSafe AI. *Jev* is their model; this is one unofficial client for it — and
for anything else that answers the same shape of question. That is why the crate is not named
after it.

## License

MIT **or** Apache-2.0, at your option — the Rust ecosystem default. See
[LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
