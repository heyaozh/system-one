//! `jev-ask` — ask *one* question from the command line.
//!
//! The smallest possible loop: a question, a piece of state, a calibrated
//! answer. No structs, no derive — the schema is built at runtime, so this
//! doubles as a scratchpad for trying a question out before you commit it to
//! a `#[derive(JevQuestions)]` type.
//!
//! Install it once (`cargo install --path jev`), then:
//!
//! ```text
//! # yes/no (a `noul`)
//! jev-ask "Is this customer asking for a refund?" "my payouts failed, I want the fees back"
//!
//! # pick one (a `choice`)
//! jev-ask --choice billing,technical,sales,spam \
//!     "Which department should handle this?" "getting a 500 from /v1/orders"
//!
//! # ordered scale (a `score`)
//! jev-ask --levels "Can wait a week,Within a day,Right now" \
//!     "How urgent is this?" "my payouts have failed for 3 days"
//! ```
//!
//! From inside this repo without installing: `cargo run --bin jev-ask -- <args>`.
//!
//! With `JEV_API_KEY` set the question goes to the real API; without it you
//! get a uniform mock answer so the plumbing still runs (and says so).

use jev::backend::{DecisionBackend, Mock};
use jev::prelude::*;
use jev::schema::{QuestionSchema, QuestionSpec};

const NAME: &str = "q";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();

    // Optional leading flag: --choice a,b,c  |  --levels "low,mid,high"
    let spec_from_flag = match args.first().map(String::as_str) {
        Some("--choice") => {
            let opts = args.remove(1);
            args.remove(0);
            Some(Box::new(move |instructions: String| QuestionSpec::Choice {
                instructions,
                criteria: opts.split(',').map(|o| (o.trim().to_string(), None)).collect(),
            }) as Box<dyn FnOnce(String) -> QuestionSpec>)
        }
        Some("--levels") => {
            let levels = args.remove(1);
            args.remove(0);
            Some(Box::new(move |instructions: String| QuestionSpec::Score {
                instructions,
                criteria: levels.split(',').map(|l| l.trim().to_string()).collect(),
            }) as Box<dyn FnOnce(String) -> QuestionSpec>)
        }
        _ => None,
    };

    let (question, state) = match (args.first(), args.get(1)) {
        (Some(q), Some(s)) => (q.clone(), s.clone()),
        _ => {
            eprintln!("usage: jev-ask [--choice a,b,c | --levels \"low,high\"] <question> <state>");
            std::process::exit(2);
        }
    };

    let spec = match spec_from_flag {
        Some(build) => build(question),
        None => QuestionSpec::Noul {
            instructions: question,
            criteria: None,
        },
    };
    spec.validate(NAME)?;
    let schema = QuestionSchema::new().with(NAME, spec);

    let backend: Box<dyn DecisionBackend> = match JevHttp::from_env() {
        Some(b) => Box::new(b),
        None => {
            eprintln!("note: JEV_API_KEY is not set — answering with a uniform Mock, not the real model.");
            Box::new(Mock::uniform())
        }
    };

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let raw = rt.block_on(Engine::new(backend).ask_raw(&state, &schema))?;

    println!("state:    {state}");
    println!("model:    {}", if raw.model.is_empty() { "<mock>" } else { &raw.model });
    match raw.get(NAME)? {
        jev::answer::RawAnswer::Noul { noul, .. } => {
            let n = Noul::new(*noul);
            println!("P(yes):   {:.3}", n.p);
            println!("decide:   yes={} at a 1:1 cost ratio", n.decide(1.0, 1.0));
        }
        jev::answer::RawAnswer::Choice {
            choice,
            probabilities,
            confidence,
        } => {
            let mut ps: Vec<_> = probabilities.iter().collect();
            ps.sort_by(|a, b| b.1.total_cmp(a.1));
            println!("argmax:   {choice}  (confidence {confidence:.3})");
            for (k, p) in ps {
                println!("  {p:>6.3}  {k}");
            }
        }
        jev::answer::RawAnswer::Score {
            score,
            legend,
            probabilities,
            confidence,
        } => {
            println!("score:    {score:.3}  (confidence {confidence:.3})");
            let mut ps: Vec<_> = probabilities.iter().collect();
            ps.sort_by_key(|(k, _)| k.parse::<usize>().unwrap_or(0));
            for (k, p) in ps {
                let label = legend.get(k).map(String::as_str).unwrap_or("");
                println!("  {p:>6.3}  [{k}] {label}");
            }
        }
    }
    Ok(())
}
