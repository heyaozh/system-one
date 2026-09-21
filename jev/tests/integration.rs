//! Integration tests: derive macros, schema wire format, typed decoding,
//! decision helpers, recording, replay, shadow and calibration — all
//! offline via `Mock`.

use jev::backend::{answers, Mock, Replay, Shadow};
use jev::calibration::CalibrationReport;
use jev::prelude::*;
use jev::record::{attach_outcomes, read_records};
use jev::{QuestionSpec, RawAnswer, RawAnswers};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Copy, PartialEq, Eq, Debug, JevChoice)]
enum Dept {
    /// Money things.
    Billing,
    #[jev(key = "tech", desc = "Bugs and APIs")]
    Technical,
    Sales,
}

#[derive(Debug, JevQuestions)]
#[allow(dead_code)]
struct Triage {
    #[jev("Which department?")]
    dept: Choice<Dept>,
    #[jev("Wants a refund?", yes = "asks for money back", no = "anything else")]
    refund: Noul,
    #[jev("Urgency", levels = ["low", "mid", "high"])]
    urgency: Score,
    /// Doc comments work as instructions too.
    #[jev(name = "is_spam_wire")]
    is_spam: Noul,
}

#[test]
fn choice_derive_exposes_keys_and_descriptions() {
    use jev::JevChoice as _;
    assert_eq!(Dept::all(), &[Dept::Billing, Dept::Technical, Dept::Sales]);
    assert_eq!(Dept::Technical.key(), "tech");
    assert_eq!(Dept::Billing.key(), "billing");
    assert_eq!(Dept::Billing.description(), Some("Money things."));
    assert_eq!(Dept::Technical.description(), Some("Bugs and APIs"));
    assert_eq!(Dept::Sales.description(), None);
    assert_eq!(Dept::from_key("tech"), Some(Dept::Technical));
    assert_eq!(Dept::from_key("nope"), None);
}

#[test]
fn schema_matches_api_wire_format() {
    use jev::JevQuestions as _;
    let schema = Triage::schema();
    schema.validate().unwrap();
    let json = schema.to_api_json();

    assert_eq!(json["dept"]["type"], "choice");
    assert_eq!(json["dept"]["instructions"], "Which department?");
    assert_eq!(json["dept"]["criteria"]["billing"], "Money things.");
    assert_eq!(json["dept"]["criteria"]["tech"], "Bugs and APIs");
    assert!(json["dept"]["criteria"]["sales"].is_null());

    assert_eq!(json["refund"]["type"], "noul");
    assert_eq!(json["refund"]["criteria"]["true"], "asks for money back");
    assert_eq!(json["refund"]["criteria"]["false"], "anything else");

    assert_eq!(json["urgency"]["type"], "score");
    assert_eq!(json["urgency"]["criteria"], serde_json::json!(["low", "mid", "high"]));

    // `name = "..."` overrides the wire name; doc comment is the instruction.
    assert!(json.get("is_spam").is_none());
    assert_eq!(
        json["is_spam_wire"]["instructions"],
        "Doc comments work as instructions too."
    );
}

#[test]
fn api_response_example_decodes() {
    use jev::JevQuestions as _;
    // Shapes copied from the official docs.
    let raw: RawAnswers = serde_json::from_value(serde_json::json!({
        "model": "jev-1.13.0",
        "answers": {
            "dept": { "type": "choice", "choice": "tech",
                      "probabilities": { "billing": 0.12, "tech": 0.88, "sales": 0.0 }, "confidence": 0.81 },
            "refund": { "type": "noul", "noul": 0.95 },
            "urgency": { "type": "score", "score": 1.05,
                         "legend": { "0": "low", "1": "mid", "2": "high" },
                         "probabilities": { "0": 0.0, "1": 0.95, "2": 0.05 }, "confidence": 0.92 },
            "is_spam_wire": { "type": "noul", "noul": 0.01 }
        },
        "usage": { "input_tokens": 304, "output_tokens": 18 }
    }))
    .unwrap();
    let t = Triage::from_raw(&raw).unwrap();
    assert_eq!(t.dept.argmax(), Dept::Technical);
    assert!((t.dept.p(Dept::Billing) - 0.12).abs() < 1e-9);
    assert_eq!(t.dept.probabilities.len(), 3);
    assert!((t.refund.p - 0.95).abs() < 1e-9);
    assert!((t.urgency.value - 1.05).abs() < 1e-9);
    assert_eq!(t.urgency.argmax_label(), "mid");
    assert!((t.urgency.p_at_least(1) - 1.0).abs() < 1e-9);
    assert_eq!(t.urgency.legend, vec!["low", "mid", "high"]);
    assert_eq!(raw.usage.input_tokens, 304);
}

#[test]
fn schema_mismatch_is_an_error_not_a_panic() {
    use jev::JevQuestions as _;
    let mut answers = BTreeMap::new();
    answers.insert(
        "dept".to_string(),
        RawAnswer::Noul {
            noul: 0.5,
            confidence: None,
        },
    ); // wrong type
    let raw = RawAnswers {
        answers,
        ..Default::default()
    };
    let err = Triage::from_raw(&raw).unwrap_err();
    assert!(matches!(err, Error::SchemaMismatch { .. }), "{err:?}");
}

#[test]
fn decision_helpers() {
    let n = Noul::new(0.3);
    // Break-even = fp/(fp+fn). 1/(1+1)=0.5 -> 0.3 does not act.
    assert!(!n.decide(1.0, 1.0));
    // 1/(1+9)=0.1 -> 0.3 acts.
    assert!(n.decide(1.0, 9.0));
    assert!((Noul::new(0.5).entropy() - 1.0).abs() < 1e-9);

    let c = Choice::<Dept> {
        chosen: Dept::Billing,
        probabilities: vec![(Dept::Billing, 0.5), (Dept::Technical, 0.4), (Dept::Sales, 0.1)],
        confidence: 0.5,
    };
    // Misrouting to Sales is catastrophic, otherwise unit cost: pick argmax.
    let a = c.decide(|t, a| {
        if t == a {
            0.0
        } else if a == Dept::Sales {
            100.0
        } else {
            1.0
        }
    });
    assert_eq!(a, Dept::Billing);
    // Now make wrongly routing away from Technical very expensive -> pick Technical.
    let a = c.decide(|t, a| {
        if t == a {
            0.0
        } else if t == Dept::Technical {
            10.0
        } else {
            1.0
        }
    });
    assert_eq!(a, Dept::Technical);
    assert_eq!(c.ranked()[0].0, Dept::Billing);

    let mut rng = rand::thread_rng();
    let mut counts = [0usize; 3];
    for _ in 0..2000 {
        match c.sample(&mut rng) {
            Dept::Billing => counts[0] += 1,
            Dept::Technical => counts[1] += 1,
            Dept::Sales => counts[2] += 1,
        }
    }
    assert!(counts[0] > counts[1] && counts[1] > counts[2]);
}

#[tokio::test]
async fn engine_with_mock_rule_and_cache() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let c2 = calls.clone();
    let mock = Mock::with_rule(move |state, name, _| {
        c2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let s = state.as_str().unwrap_or("");
        match name {
            "dept" => Some(answers::choice([("billing", 0.7), ("tech", 0.2), ("sales", 0.1)])),
            "refund" => Some(answers::noul(if s.contains("refund") { 0.9 } else { 0.1 })),
            _ => None,
        }
    });
    let engine = Engine::new(mock).with_cache();
    let t: Triage = engine.ask("please refund me").await.unwrap();
    assert_eq!(t.dept.argmax(), Dept::Billing);
    assert!(t.refund.p > 0.8);
    // Uniform fallback for questions the rule did not answer.
    assert!((t.urgency.probabilities[0] - 1.0 / 3.0).abs() < 1e-9);

    let n_before = calls.load(std::sync::atomic::Ordering::SeqCst);
    let _: Triage = engine.ask("please refund me").await.unwrap(); // cache hit
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), n_before);
}

#[tokio::test]
async fn ask_many_preserves_order() {
    let engine = Engine::new(Mock::with_rule(|state, name, _| {
        let s = state.as_str().unwrap_or("");
        (name == "refund").then(|| answers::noul(if s == "b" { 0.9 } else { 0.1 }))
    }))
    .with_concurrency(4);
    let out = engine.ask_many::<Triage, _, _>(["a", "b", "c"]).await;
    assert_eq!(out.len(), 3);
    assert!(out[1].as_ref().unwrap().refund.p > 0.8);
    assert!(out[0].as_ref().unwrap().refund.p < 0.2);
}

#[tokio::test]
async fn record_replay_and_outcomes_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.jsonl");

    // 1. Record with a mock.
    let engine = Engine::new(Mock::with_rule(|_, name, _| {
        (name == "refund").then(|| answers::noul(0.8))
    }))
    .with_recorder(Recorder::open(&path).unwrap());
    let t: Triage = engine.ask("x").await.unwrap();
    assert!((t.refund.p - 0.8).abs() < 1e-9);

    // 2. Replay must return the same answer, and fail on unknown state.
    let replay = Replay::from_jsonl(&path).unwrap();
    assert_eq!(replay.len(), 1);
    let engine2 = Engine::new(replay);
    let t2: Triage = engine2.ask("x").await.unwrap();
    assert!((t2.refund.p - 0.8).abs() < 1e-9);
    let err = engine2.ask::<Triage, _>("y").await.unwrap_err();
    assert!(matches!(err, Error::NotRecorded { .. }));

    // 3. Attach an outcome and compute calibration.
    let hash = jev::hash::hash_json(&serde_json::json!("x"));
    let mut outcomes = HashMap::new();
    outcomes.insert(hash, serde_json::json!({ "refund": true }));
    assert_eq!(attach_outcomes(&path, &outcomes).unwrap(), 1);
    let records = read_records(&path).unwrap();
    let report = CalibrationReport::from_records(&records, "refund", 5);
    assert_eq!(report.n, 1);
    assert!((report.brier - 0.04).abs() < 1e-9); // (0.8 - 1)^2
}

#[tokio::test]
async fn shadow_returns_primary_and_records_both() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shadow.jsonl");
    let primary = Mock::with_rule(|_, name, _| (name == "refund").then(|| answers::noul(0.9))).with_id("p");
    let shadow = Mock::with_rule(|_, name, _| (name == "refund").then(|| answers::noul(0.2))).with_id("s");
    let engine = Engine::new(Shadow::new(primary, shadow).with_recorder(Recorder::open(&path).unwrap()));
    let t: Triage = engine.ask("x").await.unwrap();
    assert!((t.refund.p - 0.9).abs() < 1e-9);
    let recs = read_records(&path).unwrap();
    assert_eq!(recs.len(), 2);
    assert_eq!(recs[0].tag.as_deref(), Some("primary"));
    assert_eq!(recs[1].tag.as_deref(), Some("shadow"));
    assert_eq!(recs[1].backend, "s");
}

#[test]
fn calibration_math() {
    // Perfectly calibrated coin at 0.5: brier 0.25, ece ~0.
    let pairs: Vec<(f64, bool)> = (0..1000).map(|i| (0.5, i % 2 == 0)).collect();
    let r = CalibrationReport::from_pairs(pairs, 10);
    assert!((r.brier - 0.25).abs() < 1e-9);
    assert!(r.ece < 1e-9);
    assert!((r.base_rate - 0.5).abs() < 1e-9);
    // Over-confident: says 0.9, right 60%.
    let pairs: Vec<(f64, bool)> = (0..1000).map(|i| (0.9, i % 10 < 6)).collect();
    let r = CalibrationReport::from_pairs(pairs, 10);
    assert!((r.ece - 0.3).abs() < 1e-9);
    assert!(r.render().contains("ece=0.3000"));
}

#[test]
fn schema_limits_are_enforced() {
    let mut s = QuestionSchema::new();
    s.insert(
        "x",
        QuestionSpec::Score {
            instructions: "?".into(),
            criteria: vec!["only one".into()],
        },
    );
    assert!(matches!(s.validate().unwrap_err(), Error::InvalidSchema(_)));
    let mut s = QuestionSchema::new();
    s.insert(
        "x",
        QuestionSpec::Choice {
            instructions: "?".into(),
            criteria: (0..256).map(|i| (i.to_string(), None)).collect(),
        },
    );
    assert!(matches!(s.validate().unwrap_err(), Error::InvalidSchema(_)));
    assert!(QuestionSchema::new().validate().is_err());
}
