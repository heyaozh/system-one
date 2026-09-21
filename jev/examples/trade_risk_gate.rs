//! Example 2 — pre-trade risk gate for an automated trading agent.
//!
//! An LLM (or any strategy) proposes an order. Before it reaches the
//! exchange, a System One model answers a fixed set of compliance questions
//! about the *structured* proposal plus the mandate text. The gate then
//! decides with an explicit cost matrix — because "block a good trade" and
//! "let a bad trade through" do not cost the same.
//!
//! The state is a struct: any `Serialize` type works as state.
//!
//! Run:  `cargo run --example trade_risk_gate`

use jev::backend::{answers, DecisionBackend, Mock};
use jev::prelude::*;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct ProposedOrder {
    symbol: String,
    side: String,
    quantity: f64,
    limit_price: Option<f64>,
    notional_usd: f64,
    rationale: String,
}

#[derive(Debug, Serialize)]
struct GateState {
    mandate: &'static str,
    position_limit_usd: f64,
    current_position_usd: f64,
    recent_fills_last_10min: u32,
    proposed: ProposedOrder,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, JevChoice)]
enum Verdict {
    Allow,
    /// Send to a human before execution.
    Review,
    Block,
}

#[derive(Debug, JevQuestions)]
struct RiskGate {
    #[jev("Is the proposed order consistent with the stated mandate?")]
    within_mandate: Noul,
    #[jev("Does the order look like wash trading or self-matching (buying and selling the same instrument in quick succession without economic purpose)?")]
    wash_like: Noul,
    #[jev("How severely would this order breach the position limit after execution?",
          levels = ["No breach", "Minor breach (<10% over)", "Serious breach (>10% over)"])]
    breach: Score,
    #[jev("Overall, what should the risk desk do with this order?")]
    verdict: Choice<Verdict>,
}

fn backend() -> Box<dyn DecisionBackend> {
    if let Some(b) = JevHttp::from_env() {
        return Box::new(b);
    }
    // Mock rule that reads the structured state — good enough to show the
    // control flow without an API key.
    Box::new(Mock::with_rule(|state, name, spec| {
        let notional = state["proposed"]["notional_usd"].as_f64().unwrap_or(0.0);
        let pos = state["current_position_usd"].as_f64().unwrap_or(0.0);
        let limit = state["position_limit_usd"].as_f64().unwrap_or(f64::INFINITY);
        let fills = state["recent_fills_last_10min"].as_u64().unwrap_or(0);
        let rationale = state["proposed"]["rationale"].as_str().unwrap_or("").to_lowercase();
        let over = (pos + notional) / limit;
        match name {
            "within_mandate" => Some(answers::noul(if rationale.contains("meme") { 0.15 } else { 0.9 })),
            "wash_like" => Some(answers::noul(if fills > 20 { 0.7 } else { 0.05 })),
            "breach" => Some(answers::score(
                if over <= 1.0 { &[0.9, 0.08, 0.02] } else if over <= 1.1 { &[0.1, 0.75, 0.15] } else { &[0.02, 0.18, 0.8] },
                &["No breach", "Minor breach (<10% over)", "Serious breach (>10% over)"],
            )),
            "verdict" => Some(answers::choice([
                ("allow", if over <= 1.0 && fills <= 20 { 0.8 } else { 0.1 }),
                ("review", 0.15),
                ("block", if over > 1.1 || fills > 20 { 0.75 } else { 0.05 }),
            ])),
            _ => Some(Mock::uniform_answer(spec)),
        }
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let engine = Engine::new(backend()).with_recorder(Recorder::open("runs/risk_gate.jsonl")?);

    let proposals = vec![
        GateState {
            mandate: "Long-only US large-cap equities; max 5% of NAV per name; no intraday round trips.",
            position_limit_usd: 1_000_000.0,
            current_position_usd: 400_000.0,
            recent_fills_last_10min: 2,
            proposed: ProposedOrder {
                symbol: "MSFT".into(),
                side: "buy".into(),
                quantity: 500.0,
                limit_price: Some(410.0),
                notional_usd: 205_000.0,
                rationale: "Earnings beat, momentum continuation.".into(),
            },
        },
        GateState {
            mandate: "Long-only US large-cap equities; max 5% of NAV per name; no intraday round trips.",
            position_limit_usd: 1_000_000.0,
            current_position_usd: 950_000.0,
            recent_fills_last_10min: 31,
            proposed: ProposedOrder {
                symbol: "DOGE-USD".into(),
                side: "buy".into(),
                quantity: 2_000_000.0,
                limit_price: None,
                notional_usd: 300_000.0,
                rationale: "meme rally, fomo".into(),
            },
        },
    ];

    for st in &proposals {
        let g: RiskGate = engine.ask(st).await?;

        // Cost matrix over (truth, action): letting a blockable order through is
        // 10x worse than reviewing a fine one; blocking a fine one costs 3.
        let action = g.verdict.decide(|truth, action| match (truth, action) {
            (t, a) if t == a => 0.0,
            (Verdict::Block, Verdict::Allow) => 10.0,
            (Verdict::Block, Verdict::Review) => 2.0,
            (Verdict::Allow, Verdict::Block) => 3.0,
            (Verdict::Allow, Verdict::Review) => 1.0,
            (Verdict::Review, _) => 1.0,
            _ => 1.0,
        });

        // Independent hard gates layered on top: any of these overrides.
        let hard_block = g.wash_like.decide(/*false alarm*/ 1.0, /*missed*/ 10.0) || g.breach.p_at_least(2) > 0.5;

        println!(
            "{} {} {:.0} USD  -> {:?}{}   P(mandate)={:.2} P(wash)={:.2} breach={} ({:.2})",
            st.proposed.side,
            st.proposed.symbol,
            st.proposed.notional_usd,
            action,
            if hard_block { " [HARD BLOCK]" } else { "" },
            g.within_mandate.p,
            g.wash_like.p,
            g.breach.argmax_label(),
            g.breach.value
        );
    }
    println!("\nrecorded to runs/risk_gate.jsonl");
    Ok(())
}
