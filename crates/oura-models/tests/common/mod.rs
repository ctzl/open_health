//! Shared plumbing for the `oura-models` integration tests.
//!
//! Every fixture this suite compares against is captured local data, so it lives in `tests/vectors/` — gitignored, local-only — and
//! never in a test source. Two files matter here:
//!
//! * the golden vectors themselves (`activity_golden.json`, `activity_oracle.json`,
//!   `illness_golden.json`, `illness_inputs.json`, `cva_golden.json`,
//!   `stress_golden.json`), and
//! * `test-env.json`, the small non-golden facts a fixture is pinned to: the ring
//!   serial, the DB state the capture was taken at, and the cva pipeline
//!   fingerprint that no golden vector carries.
//!
//! A test that cannot read what it needs prints `SKIP <test>: <reason>` and
//! returns — the same convention the live-DB tests already use for a missing
//! events DB (`SKIP …: no events DB (set OURA_DB)`). A checkout without the
//! private data therefore stays green and says out loud what it did not check.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The non-golden personal facts, next to the vectors.
pub const TEST_ENV: &str = "test-env.json";

/// tests/vectors/ of this crate (the fixtures are local-only and gitignored).
pub fn vectors_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors")
}

/// Raw text of a vector file, or None when it is not present.
pub fn load_vector(name: &str) -> Option<String> {
    std::fs::read_to_string(vectors_dir().join(name)).ok()
}

/// A vector file parsed as JSON, or None when it is absent or unparsable.
pub fn vector_json(name: &str) -> Option<serde_json::Value> {
    serde_json::from_str(&load_vector(name)?).ok()
}

/// `tests/vectors/test-env.json`, read once per test binary.
pub fn test_env() -> Option<serde_json::Value> {
    static ENV: OnceLock<Option<serde_json::Value>> = OnceLock::new();
    ENV.get_or_init(|| vector_json(TEST_ENV)).clone()
}

/// The ring every fixture here was captured for.
pub fn test_serial() -> Option<String> {
    Some(test_env()?.get("serial")?.as_str()?.to_string())
}

/// The skip convention: name the test, say what is missing, pass vacuously.
pub fn skip(test: &str, reason: impl std::fmt::Display) {
    eprintln!("SKIP {test}: {reason}");
}

/// The fixture file a test wanted is not in this checkout.
pub fn missing_vector(test: &str, name: &str) {
    skip(
        test,
        format!(
            "no {} — local-only fixture, re-capture per the module docs",
            vectors_dir().join(name).display()
        ),
    );
}

/// test-env.json is not in this checkout, so the personal facts are unavailable.
pub fn missing_env(test: &str) {
    missing_vector(test, TEST_ENV);
}

/// Read `{"a": 1, "b": 2}`-shaped facts out of a JSON object.
pub fn ints(obj: &serde_json::Value, keys: &[&str]) -> Option<Vec<i64>> {
    keys.iter()
        .map(|k| obj.get(*k).and_then(|v| v.as_i64()))
        .collect()
}

/// Read `[1, 2]`-shaped facts out of a JSON array.
pub fn int_array(v: &serde_json::Value) -> Option<Vec<i64>> {
    v.as_array()?
        .iter()
        .map(|x| x.as_i64())
        .collect::<Option<Vec<i64>>>()
}
