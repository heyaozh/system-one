//! Example 1 — support-ticket triage: the canonical "smart if-statement".
//!
//! Three questions are asked in one call: which department, is a refund
//! being requested, how urgent. The answers are *distributions*, so the
//! routing logic can be cost-aware instead of threshold-by-gut-feeling.
//!
//! Run:  `cargo run --example ticket_routing`
//! With the real API: `JEV_API_KEY=... cargo run --example ticket_routing`

use system_one::backend::{answers, DecisionBackend, Mock};
use system_one::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug, AsChoice)]
enum Department {
    /// Invoices, payouts, charges, refunds.
    Billing,
    /// Bugs, integrations, API errors.
    Technical,
    /// Pricing, upgrades, demos.
    Sales,
    /// Unsolicited or irrelevant.
    Spam,
}

#[derive(Debug, AsQuestions)]
struct Triage {
    #[ask("Which department should handle this ticket?")]
    department: Choice<Department>,
    #[ask("Is the customer explicitly asking for money back?")]
    wants_refund: Noul,
    #[ask("How urgent is this for the customer?", levels = ["Can wait a week", "Within a day", "Right now"])]
    urgency: Score,
}

/// Use the real API when `JEV_API_KEY` is set, otherwise a mock with a
/// keyword rule — so the example runs anywhere.
fn backend() -> Box<dyn DecisionBackend> {
    if let Some(b) = JevHttp::from_env() {
        return Box::new(b);
    }
    Box::new(Mock::with_rule(|state, name, spec| {
        let text = state.as_str().unwrap_or_default().to_lowercase();
        match name {
            "department" => Some(answers::choice([
                (
                    "billing",
                    if text.contains("payout") || text.contains("refund") {
                        0.85
                    } else {
                        0.05
                    },
                ),
                (
                    "technical",
                    if text.contains("error") || text.contains("api") {
                        0.85
                    } else {
                        0.05
                    },
                ),
                (
                    "sales",
                    if text.contains("pricing") || text.contains("upgrade") {
                        0.85
                    } else {
                        0.05
                    },
                ),
                ("spam", if text.contains("crypto giveaway") { 0.9 } else { 0.05 }),
            ])),
            "wants_refund" => Some(answers::noul(if text.contains("refund") { 0.93 } else { 0.08 })),
            "urgency" => Some(answers::score(
                if text.contains("3 days") || text.contains("urgent") {
                    &[0.05, 0.25, 0.70]
                } else {
                    &[0.6, 0.3, 0.1]
                },
                &["Can wait a week", "Within a day", "Right now"],
            )),
            _ => Some(Mock::uniform_answer(spec)),
        }
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let engine = Engine::new(backend());

    let tickets = [
        "Hi, my payouts have been failing for 3 days and I need a refund of the fees.",
        "Getting a 500 error from the /v1/orders API since this morning.",
        "What does the enterprise pricing look like if we upgrade next quarter?",
        "CRYPTO GIVEAWAY!!! send 1 coin get 2 back",
    ];

    for text in tickets {
        let t: Triage = engine.ask(text).await?;

        // Cost-aware routing: misrouting *to* spam is expensive (a real
        // customer is ignored), misrouting *from* spam is cheap.
        let route = t.department.decide(|truth, action| match (truth, action) {
            (a, b) if a == b => 0.0,
            (_, Department::Spam) => 5.0,
            _ => 1.0,
        });

        // Refund flag: a missed refund request annoys a customer (cost 4);
        // a false refund flag only costs a human glance (cost 1).
        let flag_refund = t.wants_refund.decide(1.0, 4.0);

        println!(
            "{text}\n  -> route={route:?}  (argmax={:?}, entropy={:.2} bits)",
            t.department.argmax(),
            t.department.entropy()
        );
        println!("     refund_flag={flag_refund}  P(refund)={:.2}", t.wants_refund.p);
        println!(
            "     urgency={} ({:.2}/{})  P(at least 'Within a day')={:.2}\n",
            t.urgency.argmax_label(),
            t.urgency.value,
            t.urgency.legend.len() - 1,
            t.urgency.p_at_least(1)
        );
    }
    Ok(())
}
