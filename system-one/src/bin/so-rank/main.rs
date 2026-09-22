//! `so-rank` — rank a folder of markdown notes by how well each one helps
//! answer a question.
//!
//! Every document is judged on its own against the question: one request per
//! document, all in flight together. There is no keyword pre-filter, so a
//! document that shares no words with the question can still come first.
//! For a few hundred notes that costs cents; put a cheap retriever in front
//! once the collection is much larger.
//!
//! ```text
//! so-rank "How do queue positions evolve after a cancel?" notes/
//! so-rank --field title --field tags --lead --section "contribution" \
//!         --strip '<!--.*?-->' --record runs/rank.jsonl  "…" notes/papers notes/concepts
//!
//! so-rank mark runs/rank.jsonl 3fa9c1 07bd22 --no 91ee04     # what you actually used
//! so-rank report runs/rank.jsonl                              # is 0.8 really 80 %?
//! ```
//!
//! With `JEV_API_KEY` set the questions go to the real API; without it a
//! uniform Mock answers, so the plumbing runs and says so.

mod doc;

use regex::Regex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::exit;

use doc::{collect_files, Doc, Extract};
use system_one::answer::RawAnswer;
use system_one::backend::{DecisionBackend, Mock};
use system_one::calibration::CalibrationReport;
use system_one::prelude::*;
use system_one::record::{attach_outcomes, read_records};
use system_one::schema::NoulCriteria;

/// `println!` that ends the process quietly when the reader has gone away
/// (`so-rank … | head`) instead of panicking on a broken pipe.
macro_rules! out {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        if let Err(e) = writeln!(std::io::stdout(), $($arg)*) {
            if e.kind() == std::io::ErrorKind::BrokenPipe {
                std::process::exit(0);
            }
            panic!("writing to stdout: {e}");
        }
    }};
}

const USAGE: &str = "\
usage:
  so-rank [options] <question> <path>...
  so-rank mark <record.jsonl> <id>... [--no <id>...]
  so-rank report <record.jsonl>

options:
  --top N          rows to print (default 10; 0 = all)
  --field KEY      front-matter key to send (repeatable; default: every key)
  --lead           send the text between the first `# ` heading and the next heading
  --section TEXT   send sections whose heading contains TEXT (repeatable)
                   (without --lead or --section the body is sent from the top)
  --max-chars N    body characters per document; 0 = no body (default 2000)
  --strip REGEX    delete matches from the body first (repeatable; (?m) (?s) allowed)
  --show KEY       front-matter key to print next to each row (repeatable)
  --ext EXT        file extension to load (default md)
  --no-role        skip the question about what each document contributes
  --record FILE    append every decision to FILE (JSONL); enables mark and report
  --concurrency N  requests in flight (default 8)
  --dry-run        build every request and estimate tokens; call nothing
  --json           one JSON object per row instead of a table";

/// Command-line errors are this binary's business, not the library's:
/// usage mistakes are plain messages, library errors convert through `?`.
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn fail<T>(msg: impl Into<String>) -> Res<T> {
    Err(msg.into().into())
}

/// Question names on the wire. `mark` and `report` rely on `RELEVANT`.
const RELEVANT: &str = "relevant";
const ROLE: &str = "role";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("mark") => mark(&args[1..]),
        Some("report") => report(&args[1..]),
        Some("-h" | "--help") | None => {
            out!("{USAGE}");
            return;
        }
        _ => rank(&args),
    };
    if let Err(e) = result {
        eprintln!("so-rank: {e}");
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// rank
// ---------------------------------------------------------------------------

struct Opts {
    question: String,
    paths: Vec<PathBuf>,
    extract: Extract,
    top: usize,
    show: Vec<String>,
    ext: String,
    role: bool,
    record: Option<PathBuf>,
    concurrency: usize,
    dry_run: bool,
    json: bool,
}

fn parse_opts(args: &[String]) -> Res<Opts> {
    let mut o = Opts {
        question: String::new(),
        paths: vec![],
        extract: Extract::default(),
        top: 10,
        show: vec![],
        ext: "md".into(),
        role: true,
        record: None,
        concurrency: 8,
        dry_run: false,
        json: false,
    };
    let mut positional = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |flag: &str| it.next().cloned().ok_or_else(|| format!("{flag} needs a value"));
        let num = |flag: &str, v: String| {
            v.parse::<usize>()
                .map_err(|_| format!("{flag} wants a number, got `{v}`"))
        };
        match a.as_str() {
            "--top" => o.top = num("--top", val("--top")?)?,
            "--field" => o.extract.fields.push(val("--field")?),
            "--lead" => o.extract.lead = true,
            "--section" => o.extract.sections.push(val("--section")?),
            "--max-chars" => o.extract.max_chars = num("--max-chars", val("--max-chars")?)?,
            "--strip" => {
                let pat = val("--strip")?;
                let re = Regex::new(&pat).map_err(|e| format!("--strip `{pat}`: {e}"))?;
                o.extract.strip.push(re);
            }
            "--show" => o.show.push(val("--show")?),
            "--ext" => o.ext = val("--ext")?.trim_start_matches('.').to_string(),
            "--no-role" => o.role = false,
            "--record" => o.record = Some(PathBuf::from(val("--record")?)),
            "--concurrency" => o.concurrency = num("--concurrency", val("--concurrency")?)?,
            "--dry-run" => o.dry_run = true,
            "--json" => o.json = true,
            s if s.starts_with("--") => return fail(format!("unknown option {s}\n\n{USAGE}")),
            _ => positional.push(a.clone()),
        }
    }
    if positional.len() < 2 {
        return fail(format!("need a question and at least one path\n\n{USAGE}"));
    }
    o.question = positional.remove(0);
    o.paths = positional.into_iter().map(PathBuf::from).collect();
    Ok(o)
}

/// The questions asked of every document. Worded for any collection of
/// notes: nothing here names a field, a domain or a language.
fn schema(role: bool) -> QuestionSchema {
    let mut s = QuestionSchema::new().with(
        RELEVANT,
        QuestionSpec::Noul {
            instructions: "The reader's question is in `query`. Would reading the document in `document` \
                           help the reader answer that question?"
                .into(),
            criteria: Some(NoulCriteria {
                yes: "The document addresses the question directly: it studies the same phenomenon, \
                      proposes a method the reader could apply to it, or reports evidence that bears on it."
                    .into(),
                no: "The document is about a different topic, or shares vocabulary with the question \
                     without offering anything the reader could use to answer it."
                    .into(),
            }),
        },
    );
    if role {
        s.insert(
            ROLE,
            QuestionSpec::Choice {
                instructions: "What would the document in `document` mainly contribute toward answering \
                               the question in `query`?"
                    .into(),
                criteria: vec![
                    (
                        "method".into(),
                        Some("A model, algorithm or procedure the reader could apply to the question.".into()),
                    ),
                    (
                        "evidence".into(),
                        Some("Empirical results or data that bear directly on the question.".into()),
                    ),
                    (
                        "background".into(),
                        Some("Context, definitions or a survey that frames the question without answering it.".into()),
                    ),
                    (
                        "counterpoint".into(),
                        Some("Results or arguments that cut against the premise of the question.".into()),
                    ),
                    (
                        "unrelated".into(),
                        Some("Nothing the reader could use for this question.".into()),
                    ),
                ],
            },
        );
    }
    s
}

struct Row {
    id: String,
    path: PathBuf,
    title: String,
    p: f64,
    role: Option<(String, f64)>,
    show: Vec<(String, String)>,
}

fn rank(args: &[String]) -> Res<()> {
    let o = parse_opts(args)?;
    let files = collect_files(&o.paths, &o.ext)?;
    if files.is_empty() {
        return fail(format!("no .{} files under the given paths", o.ext));
    }

    // Read one file at a time; only the extracted state is kept.
    let mut docs = Vec::with_capacity(files.len());
    let mut states = Vec::with_capacity(files.len());
    let mut unreadable = Vec::new();
    for f in files {
        match std::fs::read_to_string(&f) {
            Ok(text) => {
                let d = Doc::parse(f, &text);
                states.push(json!({ "query": o.question, "document": d.to_state(&o.extract) }));
                docs.push(Doc {
                    body: String::new(),
                    ..d
                });
            }
            Err(e) => unreadable.push(format!("{}: {e}", f.display())),
        }
    }
    let schema = schema(o.role);
    schema.validate()?;
    warn_bodyless(&o, &docs, &states);

    if o.dry_run {
        return dry_run(&o, &docs, &states, &schema, &unreadable);
    }

    let backend: Box<dyn DecisionBackend> = match JevHttp::from_env() {
        Some(b) => Box::new(b),
        None => {
            eprintln!("note: JEV_API_KEY is not set — a uniform Mock answers, so the order below means nothing.");
            Box::new(Mock::uniform())
        }
    };
    let mut engine = Engine::new(backend).with_concurrency(o.concurrency).with_cache();
    if let Some(path) = &o.record {
        engine = engine.with_recorder(Recorder::open(path)?);
    }

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let answers = rt.block_on(engine.ask_many_raw(&states, &schema));

    let mut rows = Vec::new();
    let mut failed = Vec::new();
    let mut tokens = 0u64;
    let mut model = String::new();
    for ((d, state), ans) in docs.iter().zip(&states).zip(answers) {
        match ans.and_then(|a| to_row(d, state, &a, &o.show).map(|r| (r, a))) {
            Ok((row, a)) => {
                tokens += a.usage.input_tokens;
                if model.is_empty() {
                    model = a.model.clone();
                }
                rows.push(row);
            }
            Err(e) => failed.push(format!("{}: {e}", d.path.display())),
        }
    }
    sort_rows(&mut rows);

    let shown = if o.top == 0 { rows.len() } else { o.top.min(rows.len()) };
    if o.json {
        for (i, r) in rows.iter().take(shown).enumerate() {
            out!("{}", row_json(i + 1, r));
        }
    } else {
        print_table(&rows[..shown], &o.show);
    }

    eprintln!(
        "\n{} documents judged, {} shown · model {} · {} input tokens{}",
        rows.len(),
        shown,
        if model.is_empty() { "<mock>" } else { &model },
        tokens,
        o.record
            .as_ref()
            .map(|p| format!(" · recorded to {}", p.display()))
            .unwrap_or_default()
    );
    // Nothing is dropped silently: every document that did not make it into
    // the ranking is listed with its reason.
    for msg in unreadable.iter().chain(&failed) {
        eprintln!("skipped: {msg}");
    }
    if !failed.is_empty() {
        exit(2);
    }
    Ok(())
}

fn to_row(d: &Doc, state: &Value, a: &system_one::RawAnswers, show: &[String]) -> Result<Row> {
    let p = match a.get(RELEVANT)? {
        RawAnswer::Noul { noul, .. } => *noul,
        other => {
            return Err(Error::SchemaMismatch {
                question: RELEVANT.into(),
                reason: format!("expected a noul, got {other:?}"),
            })
        }
    };
    let role = match a.answers.get(ROLE) {
        Some(RawAnswer::Choice { choice, confidence, .. }) => Some((choice.clone(), *confidence)),
        _ => None,
    };
    Ok(Row {
        id: short_id(&system_one::hash::hash_json(state)),
        path: d.path.clone(),
        title: d.title(),
        p,
        role,
        show: show
            .iter()
            .map(|k| (k.clone(), d.field_text(k).unwrap_or_default()))
            .collect(),
    })
}

/// Say out loud when body text was requested but a document yielded none:
/// usually `--lead` / `--section` do not match how those files are written,
/// and they are being judged on their title alone.
fn warn_bodyless(o: &Opts, docs: &[Doc], states: &[Value]) {
    if o.extract.max_chars == 0 {
        return;
    }
    let bare: Vec<_> = docs
        .iter()
        .zip(states)
        .filter(|(_, s)| s["document"].get("text").is_none())
        .map(|(d, _)| d.path.display().to_string())
        .collect();
    if bare.is_empty() {
        return;
    }
    eprintln!(
        "warning: {} of {} documents have no body text after --lead/--section/--strip \
         and are judged on title and fields only:",
        bare.len(),
        docs.len()
    );
    for p in bare.iter().take(5) {
        eprintln!("  {p}");
    }
    if bare.len() > 5 {
        eprintln!("  … and {} more (see --dry-run --json)", bare.len() - 5);
    }
}

/// Highest probability first; the path breaks ties so output is stable.
fn sort_rows(rows: &mut [Row]) {
    rows.sort_by(|a, b| b.p.total_cmp(&a.p).then_with(|| a.path.cmp(&b.path)));
}

/// The first 8 hex characters of a record's state hash: what `mark` takes.
fn short_id(state_hash: &str) -> String {
    state_hash.chars().take(8).collect()
}

fn print_table(rows: &[Row], show: &[String]) {
    for (i, r) in rows.iter().enumerate() {
        let role = r
            .role
            .as_ref()
            .map(|(c, conf)| format!("{c} ({conf:.2})"))
            .unwrap_or_default();
        out!("{:>3}. {:.3}  {:<20} {}  [{}]", i + 1, r.p, role, r.title, r.id);
        out!("     {}", r.path.display());
        for (k, v) in show.iter().zip(r.show.iter().map(|(_, v)| v)) {
            if !v.is_empty() {
                out!("     {k}: {v}");
            }
        }
    }
}

fn row_json(rank: usize, r: &Row) -> Value {
    let mut v = json!({
        "rank": rank,
        "id": r.id,
        "path": r.path,
        "title": r.title,
        RELEVANT: r.p,
    });
    if let Some((c, conf)) = &r.role {
        v[ROLE] = json!({ "choice": c, "confidence": conf });
    }
    for (k, val) in &r.show {
        v[k.as_str()] = json!(val);
    }
    v
}

fn dry_run(o: &Opts, docs: &[Doc], states: &[Value], schema: &QuestionSchema, unreadable: &[String]) -> Res<()> {
    let questions = schema.to_api_json().to_string();
    let est = |s: &str| -> usize {
        // ~4 ASCII characters per token; other scripts (CJK, …) nearer one per character.
        let ascii = s.bytes().filter(u8::is_ascii).count();
        let other = s.chars().count() - ascii;
        ascii / 4 + other
    };
    let mut total = 0;
    for (d, s) in docs.iter().zip(states) {
        let n = est(&s.to_string()) + est(&questions);
        total += n;
        if o.json {
            out!("{}", json!({ "path": d.path, "approx_tokens": n, "state": s }));
        }
    }
    eprintln!(
        "dry run: {} documents, {} requests, ≈{} input tokens. Nothing was sent.",
        docs.len(),
        states.len(),
        total
    );
    if let Some(first) = states.first() {
        if !o.json {
            out!("first state:\n{}", serde_json::to_string_pretty(first)?);
        }
    }
    for msg in unreadable {
        eprintln!("skipped: {msg}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// mark / report
// ---------------------------------------------------------------------------

/// `mark <file> <id>... [--no <id>...] [--yes <id>...]`
fn mark(args: &[String]) -> Res<()> {
    let Some((file, ids)) = args.split_first() else {
        return fail(format!("mark needs a record file\n\n{USAGE}"));
    };
    let mut labels: Vec<(String, bool)> = Vec::new();
    let mut yes = true;
    for a in ids {
        match a.as_str() {
            "--no" => yes = false,
            "--yes" => yes = true,
            id => labels.push((id.to_lowercase(), yes)),
        }
    }
    if labels.is_empty() {
        return fail("mark needs at least one id");
    }

    let records = read_records(file)?;
    let mut outcomes: HashMap<String, Value> = HashMap::new();
    for (id, value) in &labels {
        let hash = resolve_id(&records, id)?;
        // Merge, so outcomes other tools attached to the same state survive.
        let mut outcome = records
            .iter()
            .find(|r| r.state_hash == hash)
            .and_then(|r| r.outcome.clone())
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({}));
        outcome[RELEVANT] = json!(value);
        outcomes.insert(hash, outcome);
    }
    let n = attach_outcomes(file, &outcomes)?;
    let (y, no) = labels
        .iter()
        .fold((0, 0), |(y, n), (_, v)| if *v { (y + 1, n) } else { (y, n + 1) });
    out!("marked {y} relevant, {no} not relevant ({n} records updated)");
    Ok(())
}

/// Resolve a short id to exactly one full state hash.
fn resolve_id(records: &[Record], id: &str) -> Res<String> {
    let mut hits: Vec<&str> = records
        .iter()
        .map(|r| r.state_hash.as_str())
        .filter(|h| h.starts_with(id))
        .collect();
    hits.sort_unstable();
    hits.dedup();
    match hits.as_slice() {
        [one] => Ok(one.to_string()),
        [] => fail(format!("no record with id `{id}`")),
        many => fail(format!("id `{id}` matches {} records; use more characters", many.len())),
    }
}

fn report(args: &[String]) -> Res<()> {
    let Some(file) = args.first() else {
        return fail(format!("report needs a record file\n\n{USAGE}"));
    };
    let records = read_records(file)?;
    let labelled = records
        .iter()
        .filter(|r| r.outcome.as_ref().and_then(|o| o.get(RELEVANT)).is_some())
        .count();
    out!("{} records, {} labelled", records.len(), labelled);
    out!("{}", CalibrationReport::from_records(&records, RELEVANT, 10).render());
    if labelled > 0 && labelled < 50 {
        out!("note: under 50 labels the bins are mostly noise.");
    }
    out!(
        "note: labels only on rows you chose to read lean towards high probabilities;\n      \
         mark a few low-ranked rows as well or the report flatters the model."
    );
    Ok(())
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn row(path: &str, p: f64) -> Row {
        Row {
            id: String::new(),
            path: PathBuf::from(path),
            title: String::new(),
            p,
            role: None,
            show: vec![],
        }
    }

    #[test]
    fn sort_is_by_probability_then_path() {
        let mut rows = vec![row("b.md", 0.5), row("a.md", 0.5), row("c.md", 0.9)];
        sort_rows(&mut rows);
        let order: Vec<_> = rows.iter().map(|r| r.path.to_str().unwrap()).collect();
        assert_eq!(order, ["c.md", "a.md", "b.md"]);
    }

    #[test]
    fn default_schema_is_valid_and_generic() {
        let s = schema(true);
        s.validate().unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(schema(false).len(), 1);
    }

    #[test]
    fn options_parse() {
        let args: Vec<String> = [
            "--top",
            "3",
            "--field",
            "title",
            "--lead",
            "--strip",
            "<!--.*?-->",
            "--no-role",
            "why?",
            "a",
            "b",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let o = parse_opts(&args).unwrap();
        assert_eq!(o.top, 3);
        assert_eq!(o.question, "why?");
        assert_eq!(o.paths.len(), 2);
        assert!(o.extract.lead && !o.role);
        assert_eq!(o.extract.strip.len(), 1);
        assert!(parse_opts(&["--bogus".to_string()]).is_err());
        assert!(parse_opts(&["only-a-question".to_string()]).is_err());
        assert!(parse_opts(&["--strip".into(), "(".into(), "q".into(), "p".into()]).is_err());
    }

    #[test]
    fn short_ids_resolve_uniquely_or_fail_loudly() {
        let mk = |h: &str| Record {
            ts: chrono::Utc::now(),
            backend: String::new(),
            state_hash: h.into(),
            schema_hash: String::new(),
            state: Value::Null,
            schema: QuestionSchema::new(),
            answers: Default::default(),
            outcome: None,
            tag: None,
        };
        let recs = vec![mk("abc111"), mk("abc222"), mk("def333"), mk("def333")];
        assert_eq!(
            resolve_id(&recs, "def").unwrap(),
            "def333",
            "same state recorded twice is one id"
        );
        assert!(resolve_id(&recs, "abc").is_err(), "ambiguous");
        assert!(resolve_id(&recs, "zzz").is_err(), "missing");
    }
}
