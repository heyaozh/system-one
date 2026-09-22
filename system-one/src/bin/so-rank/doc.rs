//! Load markdown documents and cut them down to the part worth sending.
//!
//! Nothing here knows about any particular notes collection: which
//! front-matter keys, which sections and which patterns to strip all come
//! from the command line.

use regex::Regex;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

/// One document on disk.
#[derive(Debug, Clone)]
pub struct Doc {
    pub path: PathBuf,
    /// Flat front matter: scalars become strings, lists become arrays.
    pub fields: Map<String, Value>,
    /// Markdown after the front matter.
    pub body: String,
}

/// How to turn a [`Doc`] into the text the model sees.
#[derive(Debug, Clone)]
pub struct Extract {
    /// Front-matter keys to send. Empty means every key.
    pub fields: Vec<String>,
    /// Send the text between the first `# ` heading and the next heading.
    pub lead: bool,
    /// Send sections whose heading contains any of these (case-insensitive).
    /// With neither `lead` nor `sections`, the body is sent from the top.
    pub sections: Vec<String>,
    /// Characters of body text per document (chars, not bytes). 0 = no body.
    pub max_chars: usize,
    /// Deleted from the body, in order, before anything else.
    pub strip: Vec<Regex>,
}

impl Default for Extract {
    fn default() -> Self {
        Self {
            fields: vec![],
            lead: false,
            sections: vec![],
            max_chars: 2000,
            strip: vec![],
        }
    }
}

impl Doc {
    pub fn parse(path: PathBuf, text: &str) -> Self {
        let (fields, body) = split_front_matter(text);
        Doc {
            path,
            fields,
            body: body.to_string(),
        }
    }

    /// Front-matter `title`, else the first `# ` heading, else the file stem.
    pub fn title(&self) -> String {
        if let Some(Value::String(t)) = self.fields.get("title") {
            if !t.is_empty() {
                return t.clone();
            }
        }
        for line in self.body.lines() {
            if let Some(h) = line.strip_prefix("# ") {
                return h.trim().to_string();
            }
        }
        self.path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    /// A front-matter value as display text (lists joined with ", ").
    pub fn field_text(&self, key: &str) -> Option<String> {
        match self.fields.get(key)? {
            Value::String(s) => Some(s.clone()),
            Value::Array(a) => Some(
                a.iter()
                    .map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
            other => Some(other.to_string()),
        }
    }

    /// The JSON object sent as `document` in the state.
    pub fn to_state(&self, ex: &Extract) -> Value {
        let mut doc = Map::new();
        doc.insert("title".into(), Value::String(self.title()));

        let mut fields = Map::new();
        for (k, v) in &self.fields {
            if k == "title" {
                continue;
            }
            if ex.fields.is_empty() || ex.fields.iter().any(|f| f == k) {
                fields.insert(k.clone(), v.clone());
            }
        }
        if !fields.is_empty() {
            doc.insert("fields".into(), Value::Object(fields));
        }

        let text = self.text(ex);
        if !text.is_empty() {
            doc.insert("text".into(), Value::String(text));
        }
        Value::Object(doc)
    }

    /// Body text after stripping, section selection and truncation.
    pub fn text(&self, ex: &Extract) -> String {
        if ex.max_chars == 0 {
            return String::new();
        }
        let mut body = self.body.clone();
        for re in &ex.strip {
            body = re.replace_all(&body, "").into_owned();
        }
        let picked = if ex.lead || !ex.sections.is_empty() {
            let mut parts = Vec::new();
            if ex.lead {
                parts.push(lead(&body));
            }
            parts.extend(sections(&body, &ex.sections));
            parts.retain(|p| !p.trim().is_empty());
            parts.join("\n\n")
        } else {
            body
        };
        truncate_chars(&squeeze_blank_lines(&picked), ex.max_chars)
    }
}

/// Split `---`-delimited front matter from the body.
///
/// Deliberately small: `key: value`, quoted values, inline lists `[a, "b"]`
/// and block lists (`key:` followed by `- item` lines). Anything nested
/// deeper is kept as its raw text so nothing is silently dropped.
pub fn split_front_matter(text: &str) -> (Map<String, Value>, &str) {
    let mut fields = Map::new();
    let rest = match text.strip_prefix("---\n").or_else(|| text.strip_prefix("---\r\n")) {
        Some(r) => r,
        None => return (fields, text),
    };
    let Some(end) = find_closing_fence(rest) else {
        return (fields, text);
    };
    let (fm, after) = (&rest[..end.0], &rest[end.1..]);

    let mut current: Option<String> = None;
    let mut block: Vec<String> = Vec::new();
    let flush = |fields: &mut Map<String, Value>, key: &mut Option<String>, block: &mut Vec<String>| {
        if let Some(k) = key.take() {
            let items: Vec<&str> = block.iter().map(|l| l.trim()).collect();
            let value = if !items.is_empty() && items.iter().all(|l| l.starts_with("- ")) {
                Value::Array(items.iter().map(|l| Value::String(unquote(&l[2..]))).collect())
            } else {
                Value::String(items.join("\n"))
            };
            fields.insert(k, value);
            block.clear();
        }
    };

    for line in fm.lines() {
        let indented = line.starts_with(' ') || line.starts_with('\t');
        if !indented {
            if let Some((k, v)) = line.split_once(':') {
                let key = k.trim();
                if !key.is_empty() && !key.contains(' ') {
                    flush(&mut fields, &mut current, &mut block);
                    let v = v.trim();
                    if v.is_empty() {
                        current = Some(key.to_string());
                    } else {
                        fields.insert(key.to_string(), parse_scalar_or_list(v));
                    }
                    continue;
                }
            }
        }
        if current.is_some() && !line.trim().is_empty() {
            block.push(line.to_string());
        }
    }
    flush(&mut fields, &mut current, &mut block);
    (fields, after)
}

/// Byte range of the closing `---` line inside `rest`: (start, end-after-newline).
fn find_closing_fence(rest: &str) -> Option<(usize, usize)> {
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        if line.trim_end() == "---" {
            return Some((offset, offset + line.len()));
        }
        offset += line.len();
    }
    None
}

fn parse_scalar_or_list(v: &str) -> Value {
    if let Some(inner) = v.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let items = split_top_level_commas(inner)
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(|s| Value::String(unquote(&s)))
            .collect();
        return Value::Array(items);
    }
    Value::String(unquote(v))
}

/// Split on commas that are not inside quotes.
fn split_top_level_commas(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for c in s.chars() {
        match (quote, c) {
            (None, '"' | '\'') => {
                quote = Some(c);
                cur.push(c);
            }
            (Some(q), _) if c == q => {
                quote = None;
                cur.push(c);
            }
            (None, ',') => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    for q in ['"', '\''] {
        if s.len() >= 2 && s.starts_with(q) && s.ends_with(q) {
            return s[1..s.len() - 1].to_string();
        }
    }
    s.to_string()
}

fn heading_level(line: &str) -> Option<usize> {
    let n = line.chars().take_while(|c| *c == '#').count();
    (n > 0 && line[n..].starts_with(' ')).then_some(n)
}

/// Text after the first `# ` heading up to the next heading of any level.
/// Without a `# ` heading: everything before the first heading.
fn lead(body: &str) -> String {
    let mut out = Vec::new();
    let mut seen_h1 = !body.lines().any(|l| heading_level(l) == Some(1));
    for line in body.lines() {
        match heading_level(line) {
            Some(1) if !seen_h1 => seen_h1 = true,
            Some(_) => {
                if seen_h1 {
                    break;
                }
            }
            None if seen_h1 => out.push(line),
            None => {}
        }
    }
    out.join("\n").trim().to_string()
}

/// Every section whose heading text contains one of `wanted`
/// (case-insensitive), heading line included, up to the next heading of the
/// same or a higher level.
fn sections(body: &str, wanted: &[String]) -> Vec<String> {
    if wanted.is_empty() {
        return vec![];
    }
    let wanted: Vec<String> = wanted.iter().map(|w| w.to_lowercase()).collect();
    let mut out = Vec::new();
    let mut open: Option<(usize, Vec<&str>)> = None;
    for line in body.lines() {
        if let Some(level) = heading_level(line) {
            if let Some((lv, _)) = &open {
                if level <= *lv {
                    let (_, buf) = open.take().unwrap();
                    out.push(buf.join("\n").trim().to_string());
                }
            }
            if open.is_none() {
                let text = line[level..].trim().to_lowercase();
                if wanted.iter().any(|w| text.contains(w.as_str())) {
                    open = Some((level, vec![line]));
                    continue;
                }
            }
        }
        if let Some((_, buf)) = &mut open {
            buf.push(line);
        }
    }
    if let Some((_, buf)) = open {
        out.push(buf.join("\n").trim().to_string());
    }
    out
}

fn squeeze_blank_lines(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut blank = 0;
    for line in s.lines() {
        if line.trim().is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out.trim().to_string()
}

/// Cut to `max` characters on a char boundary, marking the cut with `…`.
pub fn truncate_chars(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((i, _)) => format!("{}…", s[..i].trim_end()),
        None => s.to_string(),
    }
}

/// Every file under `roots` with extension `ext`, sorted, deduplicated.
pub fn collect_files(roots: &[PathBuf], ext: &str) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for root in roots {
        walk(root, ext, &mut out)?;
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn walk(path: &Path, ext: &str, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let meta = std::fs::metadata(path)?;
    if meta.is_file() {
        if path.extension().is_some_and(|e| e == ext) {
            out.push(path.to_path_buf());
        }
        return Ok(());
    }
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let name = entry.file_name();
        // Hidden directories (.git, .obsidian, …) are never documents.
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        walk(&entry.path(), ext, out)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const NOTE: &str = "---\n\
title: \"Queue-reactive models\"\n\
tags: [lob, queue, 'hawkes, marked']\n\
year: 2015\n\
authors:\n  - Huang\n  - Lehalle\n\
nested:\n  a: 1\n\
---\n\
# Queue-reactive\n\
\n\
> Lead paragraph about queues.\n\
\n\
## Background\n\
Old stuff.\n\
\n\
## Key contribution\n\
New model.[^me-1]\n\
### Detail\n\
Sub detail.\n\
## Methodology\n\
Maths.\n\
\n\
[^me-1]: my private thought <!-- user-note 2026-06-14 -->\n";

    fn doc() -> Doc {
        Doc::parse(PathBuf::from("notes/huang2015queue.md"), NOTE)
    }

    #[test]
    fn front_matter_scalars_lists_and_nested() {
        let d = doc();
        assert_eq!(d.fields["title"], "Queue-reactive models");
        assert_eq!(d.fields["year"], "2015");
        assert_eq!(d.fields["tags"], serde_json::json!(["lob", "queue", "hawkes, marked"]));
        assert_eq!(d.fields["authors"], serde_json::json!(["Huang", "Lehalle"]));
        assert_eq!(
            d.fields["nested"], "a: 1",
            "nested YAML is kept as raw text, not dropped"
        );
        assert!(d.body.starts_with("# Queue-reactive"));
    }

    #[test]
    fn no_front_matter_is_all_body() {
        let d = Doc::parse(PathBuf::from("x/plain.md"), "just text\n");
        assert!(d.fields.is_empty());
        assert_eq!(d.body, "just text\n");
        assert_eq!(d.title(), "plain", "falls back to the file stem");
    }

    #[test]
    fn title_falls_back_to_first_h1() {
        let d = Doc::parse(PathBuf::from("a.md"), "---\nyear: 1\n---\n# From heading\nbody");
        assert_eq!(d.title(), "From heading");
    }

    #[test]
    fn lead_and_sections() {
        let ex = Extract {
            lead: true,
            sections: vec!["key contribution".into()],
            ..Extract::default()
        };
        let t = doc().text(&ex);
        assert!(t.contains("Lead paragraph"));
        assert!(
            t.contains("## Key contribution") && t.contains("Sub detail"),
            "subsections stay"
        );
        assert!(!t.contains("Old stuff") && !t.contains("Maths"));
    }

    #[test]
    fn strip_runs_before_selection() {
        let ex = Extract {
            strip: vec![
                Regex::new(r"(?m)^\[\^me-[^\]]*\]:.*$").unwrap(),
                Regex::new(r"\[\^me-[^\]]*\]").unwrap(),
            ],
            ..Extract::default()
        };
        let t = doc().text(&ex);
        assert!(!t.contains("me-1") && !t.contains("private thought"));
        assert!(t.contains("New model."));
    }

    #[test]
    fn field_selection_and_zero_body() {
        let ex = Extract {
            fields: vec!["tags".into()],
            max_chars: 0,
            ..Extract::default()
        };
        let s = doc().to_state(&ex);
        assert_eq!(s["title"], "Queue-reactive models");
        assert!(s["fields"].get("tags").is_some() && s["fields"].get("year").is_none());
        assert!(s.get("text").is_none(), "max_chars 0 sends no body");
    }

    #[test]
    fn truncation_counts_chars_not_bytes() {
        assert_eq!(truncate_chars("队列反应模型", 2), "队列…");
        assert_eq!(truncate_chars("short", 10), "short");
    }
}
