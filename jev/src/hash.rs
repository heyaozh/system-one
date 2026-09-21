//! Content hashing used for caching, replay and recording.

use sha2::{Digest, Sha256};

/// Hex SHA-256 of the canonical (sorted-key) JSON encoding of a value.
pub fn hash_json(value: &serde_json::Value) -> String {
    // `serde_json::Value::Object` is a `BTreeMap` unless `preserve_order` is
    // enabled, so `to_string` is canonical enough for our purposes.
    let s = serde_json::to_string(value).expect("Value is always serialisable");
    hash_bytes(s.as_bytes())
}

/// Hex SHA-256 of raw bytes.
pub fn hash_bytes(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}
