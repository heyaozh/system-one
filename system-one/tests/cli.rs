//! Black-box tests of the `so-rank` binary: the things only a real process
//! and a real pipe can show.
#![cfg(feature = "cli")]

use std::process::{Command, Stdio};

/// `so-rank … | head` must end quietly when the reader goes away, not panic.
///
/// The output is made larger than a pipe buffer (64 KiB), so the child is
/// guaranteed to write after the read end is closed, whatever the timing.
#[test]
fn closed_stdout_exits_quietly() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..1500 {
        std::fs::write(
            dir.path().join(format!("note-{i:04}.md")),
            format!("---\ntitle: \"A reasonably long title for note number {i}\"\n---\nbody {i}\n"),
        )
        .unwrap();
    }

    let mut child = Command::new(env!("CARGO_BIN_EXE_so-rank"))
        .args(["--top", "0", "--no-role", "any question"])
        .arg(dir.path())
        .env_remove("JEV_API_KEY")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    drop(child.stdout.take()); // the reader leaves before reading anything

    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(0), "a broken pipe is not an error: {status:?}");
}

#[test]
fn usage_errors_exit_nonzero_with_a_message() {
    let out = Command::new(env!("CARGO_BIN_EXE_so-rank"))
        .args(["--bogus"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown option --bogus"));
}
