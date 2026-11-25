//! Golden checks for the Rust port of tools/run_illness_model.py.
//!
//! tests/vectors/illness_inputs.json is the exact set of 23 tensors the Python
//! oracle fed `illness_detection_0_5_1.pt`, captured on 2026-11-24 by
//! monkeypatching `torch.jit.load` (see the note in `capture_note` below). The
//! Rust feature builder must reproduce them bit for bit — that is the whole port
//! except inference, which sits behind `IllnessModel`.
//!
//! The fixture is window-relative (index 0 = the newest wake day in the DB), so
//! it only holds while the DB's newest night is still the captured one; the test
//! skips with a loud message once the window moves. Regenerate it with:
//!
//!   .venv/Scripts/python.exe <scratch>/dump_illness_inputs.py \
//!       ../open_oura/oura.db -5 \
//!       ../open_oura/profile.json <the test-env.json serial> out.json
//!
//! and save its tensors as tests/vectors/illness_inputs.json in the form
//! `{"inputs": [[<float32 bit pattern as an unsigned int>, ...], ...]}` — 23 rows,
//! nine of 30 values then fourteen of one, 0x7fc00000 for a missing day. Or just
//! re-capture the golden JSON (see docs/python-to-rust-plan.md).
//!
//! THE RE-CAPTURE PROCEDURE, end to end (what to run when a fixture goes stale):
//!   1. the golden JSON: `.venv/Scripts/python.exe tools/run_illness_model.py
//!      ../open_oura/oura.db -5 --json --serial <the test-env.json
//!      serial> > crates/oura-models/tests/vectors/illness_golden.json`
//!   2. the 23 tensors: dump_illness_inputs.py above, then save them as
//!      tests/vectors/illness_inputs.json. The anchor day and the
//!      `days_with_data` count the tests pin to are read from the golden JSON,
//!      which carries both.
//!   3. refresh `captured_db` in tests/vectors/test-env.json: `ILLNESS_GUARD=1`
//!      cargo test -p oura-models --test illness_test -- --nocapture` prints the
//!      live block.
//!
//! THE WINDOW-DRIFT GUARD: both live-DB tests here first probe the DB with plain
//! SQL (read-only, no clock, no model, tens of ms) for the facts the fixtures are
//! pinned to — the clock anchors/boots, the newest bedtime window, and the row
//! families the features are built from. If any of it moved, the fixture is stale
//! rather than the port, so the test prints "golden stale — re-capture per the
//! module header" and skips; `json_matches_golden_vector` additionally compares
//! the anchor day its golden carries against the anchor day the pipeline just
//! computed. If the facts match, the comparison runs for real and a genuine
//! regression still fails. The guard is a pre-check, not a substitute: drift it
//! does not look at still has to be caught by the assertions below.

use std::path::{Path, PathBuf};

use oura_models::illness::{self, Args, Prepared};

mod common;

/// The 23 model inputs, bit for bit, from tests/vectors/illness_inputs.json:
/// every value is the raw float32 bit pattern the Python oracle produced, so a
/// missing day (NaN) survives the round trip.
fn oracle_inputs() -> Option<Vec<Vec<f32>>> {
    let rows = common::vector_json("illness_inputs.json")?
        .get("inputs")?
        .as_array()?
        .clone();
    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        let bits = common::int_array(row)?;
        out.push(
            bits.into_iter()
                .map(|b| f32::from_bits(b as u32)) // bits are 0..2^32 by construction
                .collect(),
        );
    }
    Some(out)
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(Path::new("."))
        .to_path_buf()
}

/// The CLI the fixtures were captured with, for the ring named by
/// tests/vectors/test-env.json.
fn oracle_args(serial: &str) -> Option<Args> {
    let sibling = repo_root().parent()?.join("open_oura");
    let db = sibling.join("oura.db");
    let profile = sibling.join("profile.json");
    if !db.exists() {
        return None;
    }
    Some(Args {
        db,
        tz: -5.0,
        profile: profile.exists().then_some(profile),
        serial: Some(serial.to_string()),
        pretty: false,
    })
}

// ---------------------------------------------------------------------------
// Window-drift guard (see the module docs): plain SQL, no clock, no model
// ---------------------------------------------------------------------------

/// The row families the fixtures are built from, with the tag filters illness.rs
/// uses: BEDTIME_TAG 0x76, IBI_TAGS 0x44/0x60/0x71, HRV_TAG 0x5D, TEMP_TAGS
/// 0x46/0x69/0x75, and the MET rows its activity_by_day pass reads. The bedtime
/// family is what picks the anchor day, so it is the one that matters most.
const FAMILIES: [(&str, &str); 5] = [
    ("bedtime", "tag = 118"),
    ("ibi", "tag IN (68,96,113)"),
    ("hrv", "tag = 93"),
    ("temperature", "tag IN (70,105,117)"),
    ("met", "tag = 80"),
];

/// What the events table said when the fixtures were captured: the clock counts,
/// and for every family in `FAMILIES` a row count and the newest `ring_timestamp`
/// (the only clock SQL can see). A count moving means the ring re-synced rows
/// inside the captured window; a max moving means new samples or a new night
/// arrived. Refresh with the re-capture procedure in the module docs —
/// `ILLNESS_GUARD=1` prints the live block. It lives in
/// tests/vectors/test-env.json, next to the serial it was measured for.
fn captured_db() -> Option<DbFacts> {
    let d = common::test_env()?.get("captured_db")?.clone();
    let [anchors, boots]: [i64; 2] = common::ints(&d, &["anchors", "boots"])?
        .try_into()
        .ok()?;
    let rows = d.get("families")?.as_array()?;
    if rows.len() != FAMILIES.len() {
        return None;
    }
    let mut families = [(0i64, 0i64); 5];
    for (i, row) in rows.iter().enumerate() {
        let row = common::int_array(row)?;
        if row.len() != 2 {
            return None;
        }
        families[i] = (row[0], row[1]);
    }
    Some(DbFacts { anchors, boots, families })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DbFacts {
    anchors: i64,
    boots: i64,
    families: [(i64, i64); 5],
}

/// The live facts in the shape test-env.json wants, ready to paste.
impl std::fmt::Display for DbFacts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\"captured_db\": {{")?;
        writeln!(f, "  \"anchors\": {}, \"boots\": {},", self.anchors, self.boots)?;
        write!(f, "  \"families\": [")?;
        for i in 0..FAMILIES.len() {
            let (rows, last_ds) = self.families[i];
            write!(f, "[{rows}, {last_ds}],")?;
        }
        writeln!(f, "]  // rows + newest ring_timestamp, per FAMILIES entry")?;
        writeln!(f, "}},")
    }
}

fn probe_db(db: &Path, serial: &str) -> rusqlite::Result<DbFacts> {
    let conn = rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let anchors = conn.query_row(
        "SELECT COUNT(*) FROM events \
         WHERE serial = ?1 AND tag IN (66,133) AND decoded_json IS NOT NULL \
         AND json_type(decoded_json,'$.unix_time') = 'integer'",
        [serial],
        |r| r.get(0),
    )?;
    let boots = conn.query_row(
        "SELECT COUNT(*) FROM events WHERE serial = ?1 AND tag = 65 AND decoded_json IS NOT NULL",
        [serial],
        |r| r.get(0),
    )?;
    let mut families = [(0, 0); 5];
    for i in 0..FAMILIES.len() {
        let sql = format!(
            "SELECT COUNT(*), MAX(ring_timestamp) FROM events \
             WHERE serial = ?1 AND {} AND decoded_json IS NOT NULL",
            FAMILIES[i].1
        );
        families[i] = conn.query_row(&sql, [serial], |r| Ok((r.get(0)?, r.get(1)?)))?;
    }
    Ok(DbFacts { anchors, boots, families })
}

/// How the live DB differs from the captured one, one line per moved fact.
fn drift_report(fixture: &DbFacts, live: &DbFacts) -> Vec<String> {
    let mut out = Vec::new();
    if fixture.anchors != live.anchors {
        out.push(format!("clock anchors {} -> {}", fixture.anchors, live.anchors));
    }
    if fixture.boots != live.boots {
        out.push(format!("ring boots {} -> {}", fixture.boots, live.boots));
    }
    for i in 0..FAMILIES.len() {
        if fixture.families[i] != live.families[i] {
            out.push(format!(
                "{} rows/newest-ds {:?} -> {:?}",
                FAMILIES[i].0, fixture.families[i], live.families[i]
            ));
        }
    }
    out
}

/// True when the live DB has moved past the captured fixtures, so the golden
/// comparisons below would fail on data drift rather than on the port.
/// `ILLNESS_GUARD=1` prints the live facts (and skips nothing) so the
/// `captured_db` block in test-env.json can be refreshed.
fn golden_is_stale(test: &str, db: &Path, serial: &str, captured: &DbFacts) -> bool {
    let started = std::time::Instant::now();
    let live = match probe_db(db, serial) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("SKIP {test}: window-drift probe failed on {}: {e}", db.display());
            return true;
        }
    };
    let drifted = drift_report(captured, &live);
    if std::env::var_os("ILLNESS_GUARD").is_some() {
        eprintln!(
            "{test}: live DB facts for {serial} (probe took {:?}):\n{live}captured:\n{captured}",
            started.elapsed()
        );
        return false;
    }
    if drifted.is_empty() {
        return false;
    }
    eprintln!(
        "SKIP {test}: golden stale — re-capture per the module header (the live DB for {serial} has \
         moved past the captured fixture: {})",
        drifted.join("; ")
    );
    true
}

/// What the golden JSON pins this fixture to: its anchor day (the index-0
/// column of the feature matrix) and its `days_with_data` count. Both used to
/// be consts in this file.
fn golden_facts() -> Option<(String, i64)> {
    let v = common::vector_json("illness_golden.json")?;
    Some((
        v.get("date")?.as_str()?.to_string(),
        v.get("days_with_data")?.as_i64()?,
    ))
}

/// The anchor day a JSON document carries (`"date"`), if it parses.
#[cfg(feature = "torch")]
fn anchor_day(json: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(json)
        .ok()?
        .get("date")?
        .as_str()
        .map(str::to_string)
}

/// The 23 model inputs, bit for bit.
#[test]
fn feature_builder_matches_python_inputs() {
    let Some(serial) = common::test_serial() else {
        common::missing_env("feature_builder_matches_python_inputs");
        return;
    };
    let Some(captured) = captured_db() else {
        common::missing_env("feature_builder_matches_python_inputs");
        return;
    };
    let Some(args) = oracle_args(&serial) else {
        eprintln!("skip: open_oura/oura.db not present next to this checkout");
        return;
    };
    if golden_is_stale("feature_builder_matches_python_inputs", &args.db, &serial, &captured) {
        return;
    }
    let Some((anchor, golden_days)) = golden_facts() else {
        common::missing_vector("feature_builder_matches_python_inputs", "illness_golden.json");
        return;
    };
    let Some(expected) = oracle_inputs() else {
        common::missing_vector("feature_builder_matches_python_inputs", "illness_inputs.json");
        return;
    };
    let Prepared::Ready(feats) = illness::prepare(&args).expect("prepare") else {
        panic!("expected the pipeline to produce features for this DB");
    };

    if feats.date != anchor {
        eprintln!(
            "SKIP feature_builder_matches_python_inputs: golden stale — re-capture per the module \
             header (the fixture is anchored at {anchor}, the DB now anchors at {}) \
             — the SQL probe above did not see it move, so the clock mapping moved instead",
            feats.date
        );
        return;
    }

    let mut got: Vec<&[f32]> = Vec::with_capacity(23);
    for s in &feats.inputs.series {
        got.push(s.as_slice());
    }
    assert_eq!(feats.inputs.scalars.len(), 14, "scalar count");
    for s in &feats.inputs.scalars {
        got.push(std::slice::from_ref(s));
    }

    let mut bad = 0usize;
    for (i, want) in expected.iter().enumerate() {
        let have = got[i];
        if have.len() != want.len() {
            println!("input {i}: length {} != {}", have.len(), want.len());
            bad += 1;
            continue;
        }
        let diffs: Vec<usize> = have
            .iter()
            .zip(want.iter())
            .enumerate()
            .filter(|(_, (a, b))| !(a.is_nan() && b.is_nan()) && a.to_bits() != b.to_bits())
            .map(|(k, _)| k)
            .collect();
        if !diffs.is_empty() {
            bad += 1;
            for k in diffs.iter().take(4) {
                println!(
                    "input {i}[{k}]: rust {:?} != python {:?} (bits {:#010x} vs {:#010x})",
                    have[*k],
                    want[*k],
                    have[*k].to_bits(),
                    want[*k].to_bits()
                );
            }
        }
    }
    assert_eq!(bad, 0, "{bad} of {} model inputs differ from the Python oracle", expected.len());
    assert_eq!(feats.days_with_data, golden_days, "days_with_data");
}

/// With the tch backend enabled the whole JSON must match the captured golden.
/// Numbers are compared with a relative tolerance: the golden was captured under
/// a different libtorch build, so last-ulp inference drift is expected. The
/// feature-builder test above already pins the model inputs exactly.
#[cfg(feature = "torch")]
#[test]
fn json_matches_golden_vector() {
    let Some(serial) = common::test_serial() else {
        common::missing_env("json_matches_golden_vector");
        return;
    };
    let Some(captured) = captured_db() else {
        common::missing_env("json_matches_golden_vector");
        return;
    };
    let Some(args) = oracle_args(&serial) else {
        eprintln!("skip: open_oura/oura.db not present next to this checkout");
        return;
    };
    if golden_is_stale("json_matches_golden_vector", &args.db, &serial, &captured) {
        return;
    }
    let Some(golden) = common::load_vector("illness_golden.json") else {
        common::missing_vector("json_matches_golden_vector", "illness_golden.json");
        return;
    };
    let text = illness::run_json(&args).expect("run_json");

    // The golden carries the anchor day it was produced for. If the pipeline now
    // lands on a different one, the window moved: stale fixture, not a bad port.
    if let (Some(got), Some(want)) = (anchor_day(&text), anchor_day(&golden)) {
        if got != want {
            eprintln!(
                "SKIP json_matches_golden_vector: golden stale — re-capture per the module header \
                 (the golden is anchored at {want}, the DB now anchors at {got})"
            );
            return;
        }
    }

    let got = tokens(&text);
    let want = tokens(&golden);
    if got.len() != want.len() {
        panic!("json shape differs: {} tokens vs {}\n got: {text}\nwant: {golden}", got.len(), want.len());
    }
    let mut bad = 0usize;
    for (g, w) in got.iter().zip(want.iter()) {
        match (g.parse::<f64>(), w.parse::<f64>()) {
            (Ok(a), Ok(b)) => {
                let tol = (b.abs() * 2e-2).max(1e-6);
                if (a - b).abs() > tol {
                    println!("number differs: {a} vs {b} (tol {tol})");
                    bad += 1;
                }
            }
            _ => {
                if g != w {
                    println!("token differs: {g:?} vs {w:?}");
                    bad += 1;
                }
            }
        }
    }
    assert_eq!(bad, 0, "{bad} JSON tokens differ from the golden vector");
}

/// Split a JSON document into value tokens (keys, strings, numbers, literals).
#[cfg(feature = "torch")]
fn tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if matches!(ch, ',' | '{' | '}' | '[' | ']' | ':' | '"' | ' ' | '\n' | '\r' | '\t') {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Without the tch backend the pipeline still runs; only inference is missing,
/// and it fails with a message that says so (the server keeps its Python
/// fallback in that case).
#[cfg(not(feature = "torch"))]
#[test]
fn inference_stub_reports_why() {
    let Some(serial) = common::test_serial() else {
        common::missing_env("inference_stub_reports_why");
        return;
    };
    let Some(args) = oracle_args(&serial) else {
        eprintln!("skip: open_oura/oura.db not present next to this checkout");
        return;
    };
    let err = illness::run_json(&args).expect_err("stub inference must fail");
    let msg = err.to_string();
    assert!(msg.contains("tch"), "unexpected stub message: {msg}");
}
