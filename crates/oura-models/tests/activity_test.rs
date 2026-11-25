//! Tests for the activity port (tools/run_activity_model.py).
//!
//!   cargo test -p oura-models --test activity_test
//!
//! What is verified here, with and without the `torch` feature:
//!
//!   * pure logic (calendar, clock dating, rounding, JSON rendering, step-packet
//!     unpacking, the reject rule) against hand-computed or Python-oracle values;
//!   * the whole data pipeline against FNV-1a-64 digests of the exact float32
//!     matrices the Python oracle hands the model, captured on the same DB and
//!     serial as tests/vectors/activity_golden.json (the digests themselves are in
//!     tests/vectors/activity_oracle.json);
//!   * the emitted JSON itself, re-rendered from tests/vectors/activity_golden.json
//!     and compared byte-for-byte, plus the row -> session mapping.
//!
//! With `--features torch` the real AAD model and gait decoder run, so
//! `activity_output_vs_golden` compares the sessions the model actually emits.
//! Without it the same test re-renders the golden through the JSON writer, and
//! the digest test runs with `--no-stepmotion` — which is how the oracle digests
//! were recorded too (its step decoder is a recorder that returns no gait).
//!
//! NO PERSONAL DATA IS EMBEDDED HERE. The ring serial and the captured
//! day-window facts live in tests/vectors/test-env.json, the oracle digests in
//! tests/vectors/activity_oracle.json, the emitted JSON in
//! tests/vectors/activity_golden.json — all gitignored. Without that directory
//! every fixture test prints SKIP and passes; the pure-logic tests below still
//! run.
//!
//! activity_oracle.json was produced by
//!   a throwaway probe script, not in the repo
//! (it imports tools/run_activity_model.py, replaces the model and the step
//! decoder with recorders, digests their inputs, and writes
//! C:/Temp/activity_digest.json). Regenerate it after any change to that script,
//! then save it as tests/vectors/activity_oracle.json. Captured 2026-11-24 with
//! torch 2.13.0+cpu on ../open_oura/oura.db, the serial named by
//! tests/vectors/test-env.json, tz=-5. These digests are bound to that DB snapshot: the
//! ring keeps syncing, so new captures add clock anchors and shift the day
//! window, and the clock/signal assertions go red on data drift alone.
//!
//! THE RE-CAPTURE PROCEDURE (what to run when the fixture goes stale):
//!   1. `python tools/run_activity_model.py ../open_oura/oura.db
//!      --tz -5 --json --full --serial <the test-env.json serial> >
//!      crates/oura-models/tests/vectors/activity_golden.json`
//!   2. run probe_activity_digest.py above, then save C:/Temp/activity_digest.json
//!      over tests/vectors/activity_oracle.json and set the `ran` count at the end
//!      of `activity_pipeline_inputs_match_python_oracle` to the number of days the
//!      fresh oracle let reach the model (the probe prints the per-kind counts).
//!   3. refresh `captured_window` in tests/vectors/test-env.json. To read the live
//!      values, run
//!      `ACTIVITY_GUARD=1 cargo test -p oura-models --test activity_test \
//!          activity_pipeline_inputs -- --nocapture`
//!      — it prints the block ready to paste.
//!
//! THE WINDOW-DRIFT GUARD: both live-DB tests here
//! (`activity_pipeline_inputs_match_python_oracle` and `activity_output_vs_golden`)
//! first probe the live DB with plain SQL (read-only, no clock, no model, tens
//! of ms) for the facts the fixtures are bound to — the clock anchor/boot counts,
//! read straight out of the oracle fixture's own `clock` field, and the MET day-window
//! bounds, which only `ring_timestamp` can express, so those live in
//! test-env.json's `captured_window`. If any of it moved, the fixture is stale rather than the
//! port, so the test prints "golden stale — re-capture per the module header" and
//! skips. If the facts match, the digests are compared for real and a genuine
//! model regression still fails. The guard is a pre-check, not a substitute: it
//! can miss drift it does not look at, and then the assertions below catch it.

use std::path::{Path, PathBuf};

use oura_models::activity::{self, DayOutcome, Matrix, Settings};

mod common;

/// The `--tz` the golden vector was captured with.
const TZ: f64 = -5.0;
/// A stand-in serial for the argparse checks: the parser treats it as an
/// opaque string, so that test needs no personal data. The real one — the ring
/// every fixture here was captured for — is in tests/vectors/test-env.json.
const FAKE_SERIAL: &str = "RING-SERIAL-0";

/// Python-oracle fingerprints of the model inputs (see module docs), read from
/// the local fixture tests/vectors/activity_oracle.json.
fn oracle() -> Option<serde_json::Value> {
    serde_json::from_str(&common::load_vector("activity_oracle.json")?).ok()
}


fn repo_root() -> PathBuf {
    // crates/oura-models -> crates -> repo
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crate lives under <repo>/crates/")
        .to_path_buf()
}

/// Same defaults as tools/_common.py resolve_db, plus the sibling open_oura
/// checkout this repo's data actually lives in.
fn db_path() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("OURA_DB") {
        let p = PathBuf::from(v);
        return p.is_file().then_some(p);
    }
    let repo = repo_root();
    [
        repo.join("oura.db"),
        repo.parent()?.join("open_oura").join("oura.db"),
        repo.join("captures").join("ring5.db"),
    ]
    .into_iter()
    .find(|p| p.is_file())
}

/// The exact CLI the golden vector was captured with.
fn oracle_args(db: &Path, serial: &str) -> Vec<String> {
    vec![
        db.to_string_lossy().into_owned(),
        "--tz".into(),
        TZ.to_string(),
        "--json".into(),
        "--serial".into(),
        serial.into(),
    ]
}

/// The golden vector is the whole history (`--json --full`), not the default
/// recent-days window the oracle digests were captured on.
fn golden_args(db: &Path, serial: &str) -> Vec<String> {
    let mut a = oracle_args(db, serial);
    a.push("--full".into());
    a
}

// ---------------------------------------------------------------------------
// Window-drift guard (see the module docs): plain SQL, no clock, no model
// ---------------------------------------------------------------------------

/// The MET day window the fixture was captured on, in the only clock SQL can
/// see — the ring's own counter. `met_rows` catches a re-sync that adds minutes
/// inside a day already in the window; the ds bounds catch a day aging out or a
/// new day arriving. Refresh it with the re-capture procedure in the module docs
/// (`ACTIVITY_GUARD=1` prints the live values); it lives in
/// tests/vectors/test-env.json, next to the serial it was measured for.
fn captured_window() -> Option<WindowFacts> {
    let w = common::test_env()?.get("captured_window")?.clone();
    let [met_rows, met_first_ds, met_last_ds]: [i64; 3] =
        common::ints(&w, &["met_rows", "met_first_ds", "met_last_ds"])?
            .try_into()
            .ok()?;
    Some(WindowFacts { met_rows, met_first_ds, met_last_ds })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WindowFacts {
    met_rows: i64,
    met_first_ds: i64,
    met_last_ds: i64,
}

/// The clock counts, over exactly the rows `Clock::build` counts: time-sync /
/// RTC rows carrying an integer `unix_time`, and ring-start rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ClockFacts {
    anchors: i64,
    boots: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DbFacts {
    window: WindowFacts,
    clock: ClockFacts,
}

fn probe_db(db: &Path, serial: &str) -> rusqlite::Result<DbFacts> {
    let conn = rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    // tag 0x50 = MET, 0x42/0x85 = time-sync/RTC anchors, 0x41 = ring start;
    // the (serial, tag) index keeps every one of these a few ms.
    let (met_rows, met_first_ds, met_last_ds) = conn.query_row(
        "SELECT COUNT(*), MIN(ring_timestamp), MAX(ring_timestamp) FROM events \
         WHERE serial = ?1 AND tag = 80 AND decoded_json IS NOT NULL",
        [serial],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
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
    Ok(DbFacts {
        window: WindowFacts { met_rows, met_first_ds, met_last_ds },
        clock: ClockFacts { anchors, boots },
    })
}

/// True when the live DB has moved past the captured fixture, so the digests
/// below would fail on data drift rather than on the port. `clock_header` is the
/// fixture's own `clock` field ([epochs, anchors, boots]); the day window needs
/// its own facts because only `ring_timestamp` can express it, so those live in
/// test-env.json. Both activity fixtures are re-captured in the same step, so
/// they share `captured_window`. `ACTIVITY_GUARD=1` prints the live facts (and
/// skips nothing) so test-env.json can be refreshed.
fn golden_is_stale(
    test: &str,
    db: &Path,
    serial: &str,
    captured: &WindowFacts,
    clock_header: &serde_json::Value,
) -> bool {
    let want_clock = match clock_header.as_array() {
        Some(c) if c.len() == 3 => ClockFacts {
            anchors: c[1].as_i64().unwrap_or(-1),
            boots: c[2].as_i64().unwrap_or(-1),
        },
        _ => return false, // no clock header to compare against: run the test
    };
    let live = match probe_db(db, serial) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("SKIP {test}: window-drift probe failed on {}: {e}", db.display());
            return true;
        }
    };
    if std::env::var_os("ACTIVITY_GUARD").is_some() {
        eprintln!(
            "{test}: live DB facts for {serial} — paste into test-env.json:\n  \
             \"captured_window\": {{ \"met_rows\": {}, \"met_first_ds\": {}, \"met_last_ds\": {} }},\n  \
             fixture clock: [{}, {}, {}]",
            live.window.met_rows,
            live.window.met_first_ds,
            live.window.met_last_ds,
            clock_header[0].as_i64().unwrap_or(-1),
            live.clock.anchors,
            live.clock.boots,
        );
        if live.window != *captured || live.clock != want_clock {
            eprintln!(
                "{test}: captured window {captured:?} / clock {want_clock:?} — stale, re-capture"
            );
        }
        return false;
    }
    if live.window == *captured && live.clock == want_clock {
        return false;
    }
    eprintln!(
        "SKIP {test}: golden stale — re-capture per the module header (the live DB for {serial} has \
         moved past the captured fixture: window {live_window:?} vs captured {captured:?}; \
         clock anchors/boots {live_clock}/{live_boots} vs the fixture's {want_anchors}/{want_boots})",
        live_window = live.window,
        live_clock = live.clock.anchors,
        live_boots = live.clock.boots,
        want_anchors = want_clock.anchors,
        want_boots = want_clock.boots,
    );
    true
}

// ---------------------------------------------------------------------------
// Digest helpers — byte-for-byte the Python probe's FNV-1a-64
// ---------------------------------------------------------------------------

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h = (h ^ *b as u64).wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// [n, cols, digest] over a float64 series, as the probe's d_f64.
fn digest_f64(rows: &[(f64, Vec<f64>)]) -> Vec<serde_json::Value> {
    let cols = if rows.is_empty() { 0 } else { 1 + rows[0].1.len() };
    let mut buf = Vec::with_capacity(16 + rows.len() * 8 * (cols + 1));
    buf.extend((rows.len() as u64).to_le_bytes());
    buf.extend((cols as u64).to_le_bytes());
    for (t, values) in rows {
        buf.extend(t.to_le_bytes());
        for v in values {
            buf.extend(v.to_le_bytes());
        }
    }
    vec![
        serde_json::Value::from(rows.len() as u64),
        serde_json::Value::from(cols as u64),
        serde_json::Value::from(fnv1a64(&buf)),
    ]
}

/// [n, cols, digest] over float32 rows, as the probe's d_f32.
fn digest_f32_rows(rows: &[Vec<f32>]) -> Vec<serde_json::Value> {
    let cols = if rows.is_empty() { 0 } else { rows[0].len() };
    let mut buf = Vec::with_capacity(16 + rows.iter().map(|r| r.len() * 4).sum::<usize>());
    buf.extend((rows.len() as u64).to_le_bytes());
    buf.extend((cols as u64).to_le_bytes());
    for row in rows {
        for v in row {
            buf.extend(v.to_le_bytes());
        }
    }
    vec![
        serde_json::Value::from(rows.len() as u64),
        serde_json::Value::from(cols as u64),
        serde_json::Value::from(fnv1a64(&buf)),
    ]
}

/// Same, for a dense matrix the pipeline already built.
fn digest_matrix(m: &Matrix) -> Vec<serde_json::Value> {
    let rows: Vec<Vec<f32>> = (0..m.rows()).map(|i| m.row(i).to_vec()).collect();
    digest_f32_rows(&rows)
}

/// [n, digest] over int64 values, as the probe's d_i64.
fn digest_i64(values: &[i64]) -> Vec<serde_json::Value> {
    let mut buf = Vec::with_capacity(8 + values.len() * 8);
    buf.extend((values.len() as u64).to_le_bytes());
    for v in values {
        buf.extend(v.to_le_bytes());
    }
    vec![serde_json::Value::from(values.len() as u64), serde_json::Value::from(fnv1a64(&buf))]
}

/// A printed label must come from the behaviour table (or be the numeric code the
/// model sent when the table has no entry for it — see `sessions_from`).
fn label_in_table(label: &str) -> bool {
    (-1i64..=70_937).any(|c| activity::behavior(c) == Some(label))
}

// ---------------------------------------------------------------------------
// Pure logic, no DB, no torch
// ---------------------------------------------------------------------------

#[test]
fn calendar_matches_python() {
    // day_start_min for the first golden day, as the oracle computes it.
    let d = activity::parse_date("2026-10-09").expect("date");
    assert_eq!(d.days_from_civil() as f64 * 1440.0 - TZ * 60.0, 29_858_700.0);
    assert_eq!(d.weekday(), 4); // 2026-10-09 is a Friday; Python's weekday() is Mon=0
    assert_eq!(activity::Date::from_local_unix(29_858_700.0 * 60.0, TZ).iso(), "2026-10-09");
    // A minute just before local midnight belongs to the previous day.
    assert_eq!(activity::Date::from_local_unix(29_858_700.0 * 60.0 - 1.0, TZ).iso(), "2026-10-08");

    assert_eq!(activity::Date::from_days(0).iso(), "1970-01-01");
    assert_eq!(activity::Date::from_days(-1).iso(), "1969-12-31");
    assert_eq!(activity::parse_date("2024-02-29").unwrap().weekday(), 3);
    assert_eq!(
        activity::parse_date("2024-03-01").unwrap().days_from_civil()
            - activity::parse_date("2024-02-01").unwrap().days_from_civil(),
        29
    );
    // civil_from_days / days_from_civil round-trip across centuries.
    for z in (-40_000i64)..40_000 {
        assert_eq!(activity::Date::from_days(z).days_from_civil(), z, "day {z}");
    }
    assert_eq!(activity::parse_date("2026-13-01"), None);
    assert_eq!(activity::parse_date("2026-1-1"), None);
}

#[test]
fn rounding_matches_python() {
    // CPython's round() is correctly-rounded ties-to-even on the binary value.
    // Every `want` here is this venv's `round(x, n)`, not a guess: 0.5835 is
    // exactly a tie at 583.5, so ties-to-even gives 584, while 2.675 and 1.005
    // fall on the near side of their ties.
    for (x, digits, want) in [
        (2.5, 0, 2.0),
        (0.5835, 3, 0.584),
        (0.135, 2, 0.14),
        (1.005, 2, 1.0),
        (2.675, 2, 2.67),
        (-0.5, 0, -0.0),
        (0.0, 3, 0.0),
    ] {
        let got = activity::py_round(x, digits);
        assert_eq!(got, want, "round({x}, {digits})");
    }
    for (x, want) in [(0.5, 0), (1.5, 2), (2.5, 2), (-0.5, 0), (42.0, 42)] {
        assert_eq!(activity::py_round_int(x), want, "round({x})");
    }
    // json.dumps float repr: always a float, shortest round-trip, and CPython's
    // exponential switch at decpt outside (-4, 16]. Every string here is this
    // venv's repr() of the value on the left.
    assert_eq!(activity::fmt_float(0.7), "0.7");
    assert_eq!(activity::fmt_float(0.75), "0.75");
    assert_eq!(activity::fmt_float(1.0), "1.0");
    assert_eq!(activity::fmt_float(0.0), "0.0");
    assert_eq!(activity::fmt_float(-0.0), "-0.0");
    assert_eq!(activity::fmt_float(0.1 + 0.2), "0.30000000000000004");
    assert_eq!(activity::fmt_float(0.0001), "0.0001");
    assert_eq!(activity::fmt_float(1.0e-5), "1e-05");
    assert_eq!(activity::fmt_float(4.0e-7), "4e-07");
    assert_eq!(activity::fmt_float(1.0e15), "1000000000000000.0");
    assert_eq!(activity::fmt_float(1.0e16), "1e+16");
    assert_eq!(activity::fmt_float(1.0e23), "1e+23");
    assert_eq!(activity::fmt_float(123_456_789_012_345.6), "123456789012345.6");
    assert_eq!(activity::fmt_float(5.0e-324), "5e-324");
    assert_eq!(activity::fmt_float(f64::MAX), "1.7976931348623157e+308");
    assert_eq!(activity::fmt_float(f64::NAN), "NaN");
}

#[test]
fn behavior_table() {
    assert_eq!(activity::behavior(-1), Some("nothing"));
    assert_eq!(activity::behavior(0), Some("<empty>"));
    assert_eq!(activity::behavior(14), Some("walking"));
    assert_eq!(activity::behavior(52), Some("yardwork"));
    assert_eq!(activity::behavior(65535), Some("other"));
    assert_eq!(activity::behavior(70937), Some("meditation"));
    assert_eq!(activity::behavior(44), None);
    assert_eq!(activity::behavior(47), None);
}

#[test]
fn step_packet_unpack_matches_python() {
    // Expected values come from tools/activity_features.py unpack_real_step on the
    // same two packets (14 bytes each, carry byte = 0xFF).
    let p1: Vec<u8> = vec![1, 2, 3, 0x81, 5, 6, 7, 8, 9, 10, 11, 0xFF, 12, 13];
    let p2: Vec<u8> = vec![0, 1, 2, 3, 4, 0x7F, 6, 7, 8, 9, 10, 11, 12, 0b1111_1111];
    let want: Vec<i64> = vec![
        43, 11, 12, 3, 5, 7, 1, 5, 6, 7, 8, 19, 21, 23, 127, 12, 13, 0, 1, 4, 7, 9, 127, 6, 7, 8, 9,
    ];
    assert_eq!(activity::unpack_real_step(&p1, &p2).expect("unpack"), want);
    assert!(activity::unpack_real_step(&p1[..13], &p2).is_err());

    // decoder order -> AAD column order.
    let row: Vec<f32> = (0..11).map(|i| i as f32).collect();
    assert_eq!(
        activity::decoder_to_aad(&row).expect("permutation"),
        [6.0, 7.0, 9.0, 10.0, 8.0, 5.0, 4.0, 0.0, 3.0, 1.0, 2.0]
    );
    assert!(activity::decoder_to_aad(&row[..10]).is_err());
}

/// NaN-aware row comparison: the padding placeholder is NaN, and NaN != NaN.
fn assert_row_eq(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "row length {got:?} != {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g == w) || (g.is_nan() && w.is_nan()),
            "row[{i}]: {got:?} != {want:?}"
        );
    }
}

#[test]
fn tensor_padding_and_reject_rule() {
    // tensor(): a series with no sample inside the MET span gets a NaN row at lo.
    let padded = activity::tensor(&[], 2, Some((5.0, 10.0)));
    assert_eq!(padded.rows(), 1);
    assert_row_eq(padded.row(0), &[5.0, f32::NAN]);
    let merged = activity::tensor(&[vec![20.0, 1.0]], 2, Some((5.0, 10.0)));
    assert_eq!(merged.rows(), 2);
    assert_row_eq(merged.row(0), &[5.0, f32::NAN]);
    assert_row_eq(merged.row(1), &[20.0, 1.0]);
    // No window -> no placeholder (met and step_motion never get one).
    assert_eq!(activity::tensor(&[], 2, None).rows(), 0);

    let ramp = |n: f64| -> Vec<Vec<f64>> {
        (0..n as i64).map(|i| vec![i as f64, 1.0]).collect()
    };
    let one = |t: f64| -> Vec<Vec<f64>> { vec![vec![t, 1.0]] };
    // A full day: last_valid = the last worn minute, and the series starts there,
    // so nothing is dropped.
    assert!(!activity::model_would_reject(&ramp(60.0), &one(60.0), &one(60.0), &one(60.0), &one(60.0)));
    // Fewer than MIN_VALID_METS worn minutes: reject iff the series starts late.
    assert!(activity::model_would_reject(&ramp(60.0)[10..], &one(5.0), &one(5.0), &one(5.0), &one(5.0)));
    // A series that starts at minute 0 is never rejected on the short branch.
    assert!(!activity::model_would_reject(&ramp(60.0), &one(5.0), &one(5.0), &one(5.0), &one(5.0)));
    // Heart rate that stops early caps last_valid below the first MET minute.
    assert!(activity::model_would_reject(&ramp(60.0)[10..], &one(60.0), &one(60.0), &one(1.0), &one(60.0)));
    // An empty motion series puts last_motion at 0, so no minute qualifies: the
    // short branch decides, and only a late start rejects.
    assert!(!activity::model_would_reject(&ramp(60.0), &[], &one(60.0), &one(60.0), &one(60.0)));
    assert!(activity::model_would_reject(&ramp(60.0)[5..], &[], &one(60.0), &one(60.0), &one(60.0)));
}

#[test]
fn ring_clock_dates_and_flags_undated() {
    let anchor = |ds: i64, unix: i64| (ds, 0x42_i64, Some(format!("{{\"unix_time\": {unix}}}")), ds / 10 + 5);
    let plain = |ds: i64| (ds, 0x50_i64, None::<String>, ds / 10 + 5);

    // Consistent anchors: 100 s of counter, 100 s of wall clock.
    let clock = activity::Clock::build(
        [anchor(1000, 1_000_000), plain(1500), anchor(2000, 1_000_100)].into_iter(),
    );
    assert_eq!(clock.epochs.len(), 1);
    assert_eq!(clock.dated_unix(1500, Some(1_000_050)), Some(1_000_050.0));
    assert_eq!(clock.undated_reason(1500, Some(1_000_050)), None);
    // Interpolating inside a consistent anchor pair uses the nearest anchor
    // (ties go to the earlier one), and the prediction is within the 6 h future
    // slack of the capture time, so it is used as-is.
    assert_eq!(clock.unix_s(1500, Some(1_000_000)), Some(1_000_050.0));
    // A capture time far in the past kills both the anchor prediction and every
    // plausible anchor, leaving the boot's own fallback clamped to capture + 6 h.
    assert_eq!(clock.unix_s(1500, Some(900_000)), Some(155.0));

    // Stalled counter: the wall clock moved far more than the counter did. The
    // agreement tolerance is max(1800 s, 2% of the counter span), so a stall has
    // to disagree by more than that to be one.
    let clock = activity::Clock::build([anchor(1000, 1_000_000), anchor(2000, 1_005_000)].into_iter());
    assert_eq!(clock.undated_reason(1500, Some(1_500_000)), None);
    // Late anchor minus the counter gap, because that stays in the past.
    assert_eq!(clock.dated_unix(1500, Some(1_500_000)), Some(1_004_950.0));
    // A capture time that puts the late anchor in the future projects the early one.
    assert_eq!(clock.unix_s(1500, Some(900_000)), Some(1_000_050.0));
    // A sample between two boots inside the stall cannot be assigned a reboot.
    let clock = activity::Clock::build(
        [
            anchor(1000, 1_000_000),
            (1400, 0x41, None::<String>, 1_000_040),
            (1600, 0x41, None::<String>, 1_000_060),
            anchor(2000, 1_005_000),
        ]
        .into_iter(),
    );
    assert_eq!(clock.undated_reason(1500, Some(1_500_000)), Some("ambiguous_reboot_stall"));
    assert_eq!(clock.dated_unix(1500, Some(1_500_000)), None);

    // Accelerated counter: the counter ran far ahead of the wall clock.
    let clock = activity::Clock::build([anchor(1000, 1_000_000), anchor(100_000, 1_000_050)].into_iter());
    assert_eq!(clock.undated_reason(50_000, Some(1_500_000)), Some("accelerated_counter"));
    // Still extrapolates from the nearest anchor; callers drop it via is_dated.
    assert_eq!(clock.unix_s(50_000, Some(1_500_000)), Some(1_004_900.0));

    // No anchors at all, and a counter outside every boot window.
    let clock = activity::Clock::build([plain(1000), plain(1010)].into_iter());
    assert_eq!(clock.undated_reason(1005, Some(1_000_000)), Some("missing_anchor"));
    assert_eq!(clock.dated_unix(1005, Some(1_000_000)), None);
    assert_eq!(clock.undated_reason(90_000, Some(1_000_000)), Some("missing_anchor"));
    // Without a capture timestamp the fallback is the epoch's own guess: the
    // capture time of its highest-ds row, walked back by the counter gap.
    assert_eq!(clock.unix_s(1005, None), Some(106.0 - (1010 - 1005) as f64 / 10.0));

    // A reset (the counter drops by more than the 6 h slack) starts a new epoch,
    // and that boot is dated by its own anchor, not the older boot's.
    let clock = activity::Clock::build(
        [
            (1000_i64, 0x42_i64, Some("{\"unix_time\": 1000000}".to_string()), 100_000_i64),
            (500_000, 0x42, Some("{\"unix_time\": 1050000}".to_string()), 150_000),
            (100, 0x50, None::<String>, 200_000),
            (200, 0x42, Some("{\"unix_time\": 1000200}".to_string()), 200_010),
        ]
        .into_iter(),
    );
    assert_eq!(clock.epochs.len(), 2);
    assert_eq!(clock.undated_reason(150, Some(1_000_300)), None);
    assert_eq!(clock.dated_unix(150, Some(1_000_300)), Some(1_000_195.0));
}

#[test]
fn cli_matches_python_argparse() {
    let a = activity::parse_args(&[
        "oura.db".into(),
        "--tz".into(),
        "-5".into(),
        "--json".into(),
        "--serial".into(),
        FAKE_SERIAL.into(),
        "--min-duration".into(),
        "10".into(),
    ])
    .expect("args");
    assert_eq!(a.db.as_deref(), Some("oura.db"));
    assert_eq!(a.tz, -5.0);
    assert!(a.json);
    assert_eq!(a.serial.as_deref(), Some(FAKE_SERIAL));
    assert_eq!(a.min_duration, 10.0);
    assert_eq!(a.threshold, 0.5);

    // The legacy "bare float means --tz" handling, both spellings.
    let a = activity::parse_args(&["--json".into(), "-5".into()]).expect("args");
    assert_eq!(a.tz, -5.0);
    assert_eq!(a.db, None);
    // A positional that isn't a real path is retried as the legacy "--tz" float;
    // when it isn't a float either, argparse keeps it as the db.
    let a = activity::parse_args(&["missing.db".into(), "-8".into()]).expect("args");
    assert_eq!(a.tz, -8.0);
    assert_eq!(a.db.as_deref(), Some("missing.db"));
    // A real path stays a path even with a second numeric positional.
    let a = activity::parse_args(&["C:/nope/definitely-not-a-db.db".into(), "3".into()]).expect("args");
    assert_eq!(a.db.as_deref(), Some("C:/nope/definitely-not-a-db.db"));
    assert_eq!(a.tz, 3.0);
    assert!(activity::parse_args(&["--nope".into()]).is_err());

    assert_eq!(activity::numeric_key("automatic_activity_detection_3_1_15.pt"), vec![3, 1, 15]);
    // 3.1.13+ takes one global_settings row; older builds take three scalars.
    match activity::model_settings(&[3, 1, 15], 0.5, 10.0) {
        Settings::Global(row) => {
            assert_eq!(row.len(), 72);
            assert_eq!(row[3], 0.0);
            assert_eq!(row[12], 0.5);
            assert_eq!(row[26], 10.0);
            assert!(row[4].is_nan());
        }
        Settings::Legacy { .. } => panic!("3.1.15 must use the global_settings schema"),
    }
    match activity::model_settings(&[3, 1, 12], 0.4, 5.0) {
        Settings::Legacy { threshold, min_duration } => {
            assert_eq!((threshold, min_duration), (0.4, 5.0));
        }
        Settings::Global(_) => panic!("3.1.12 must use the positional schema"),
    }
}

/// The renderer, the key set/order and the row -> session mapping, against the
/// golden vector itself: re-rendering the golden's own sessions must reproduce
/// the file byte-for-byte.
#[test]
fn json_output_matches_golden_byte_for_byte() {
    let Some(text) = common::load_vector("activity_golden.json") else {
        common::missing_vector("json_output_matches_golden_byte_for_byte", "activity_golden.json");
        return;
    };
    let golden: serde_json::Value = serde_json::from_str(&text).expect("golden is valid JSON");
    assert_eq!(golden["model"].as_str().unwrap(), "3.1.15");
    assert_eq!(golden["pipeline"].as_str().unwrap(), "android-parity");

    let mut sessions = Vec::new();
    for s in golden["sessions"].as_array().unwrap() {
        let top3: Vec<(String, f64)> = s["top3"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p[0].as_str().unwrap().to_string(), p[1].as_f64().unwrap()))
            .collect();
        sessions.push(activity::Session {
            start: s["start"].as_str().unwrap().to_string(),
            end: s["end"].as_str().unwrap().to_string(),
            duration_min: s["duration_min"].as_i64().unwrap(),
            is_workout: s["is_workout"].as_f64().unwrap(),
            label: s["label"].as_str().unwrap().to_string(),
            label_confidence: s["label_confidence"].as_f64().unwrap(),
            top3,
        });
    }
    assert!(!sessions.is_empty());
    let rendered = activity::render_result(golden["model"].as_str().unwrap(), &sessions);
    // The golden was captured by the Python oracle on Windows, whose stdout
    // writes CRLF; the renderer emits LF, as the oracle does on POSIX. Compare
    // the bytes that carry meaning, and assert nothing else differs.
    let text_lf = text.replace("\r\n", "\n");
    assert_eq!(rendered, text_lf.trim_end(), "rendered JSON is not byte-identical to the golden");

    // The row -> session mapping: minutes from the model are local-day offsets,
    // so the printed start/end/duration must be self-consistent, and every label
    // must come from the behaviour table.
    for s in &sessions {
        let (date, start) = s.start.split_once(' ').expect("start has a date");
        let day = activity::parse_date(date).expect("start date parses");
        let minutes = |hhmm: &str| -> f64 {
            let (h, m) = hhmm.split_once(':').expect("hh:mm");
            h.parse::<f64>().unwrap() * 60.0 + m.parse::<f64>().unwrap()
        };
        let (start_min, end_min) = (minutes(start), minutes(&s.end));
        assert_eq!(activity::format_at(day, start_min, true), s.start);
        assert_eq!(activity::format_at(day, end_min, false), s.end);
        assert_eq!(activity::py_round_int(end_min - start_min), s.duration_min);
        assert!(label_in_table(&s.label), "unknown label {}", s.label);
        assert_eq!(s.top3[0].0, s.label);
        assert_eq!(s.top3[0].1, s.label_confidence);
    }
}

// ---------------------------------------------------------------------------
// The pipeline vs the Python oracle (needs the events DB, not torch)
// ---------------------------------------------------------------------------

#[test]
fn activity_pipeline_inputs_match_python_oracle() {
    let Some(db) = db_path() else {
        eprintln!("SKIP activity_pipeline_inputs_match_python_oracle: no events DB (set OURA_DB)");
        return;
    };
    let Some(serial) = common::test_serial() else {
        common::missing_env("activity_pipeline_inputs_match_python_oracle");
        return;
    };
    let Some(captured) = captured_window() else {
        common::missing_env("activity_pipeline_inputs_match_python_oracle");
        return;
    };
    let Some(o) = oracle() else {
        common::missing_vector("activity_pipeline_inputs_match_python_oracle", "activity_oracle.json");
        return;
    };
    if golden_is_stale("activity_pipeline_inputs_match_python_oracle", &db, &serial, &captured, &o["clock"]) {
        return;
    }

    let mut args = activity::Args::default();
    args.db = Some(db.to_string_lossy().into_owned());
    args.tz = TZ;
    args.serial = Some(serial.clone());
    // The oracle fixture's probe stubbed the step decoder (recorded its inputs,
    // empty gait out), so its Run/Rejected day decisions encode NO gait. Without the
    // torch feature our decoder is stubbed the same way, and packets are still
    // collected (compared below). With torch we must run with --no-stepmotion
    // to keep the decisions comparable; packet collection is feature-independent
    // and stays covered by the non-torch build.
    if cfg!(feature = "torch") {
        args.no_stepmotion = true;
    }
    let p = activity::prepare(&args, &db, &repo_root()).expect("activity pipeline");

    // Model and asset resolution.
    assert_eq!(p.model_version, o["model_version"].as_str().unwrap());
    assert_eq!(p.model.file_name().unwrap().to_string_lossy(), o["model_name"].as_str().unwrap());
    assert_eq!(
        p.step_model.as_ref().and_then(|m| m.file_name().map(|n| n.to_string_lossy().into_owned())),
        o["step_model_name"].as_str().map(|s| s.to_string()),
        "step decoder resolution"
    );
    // Demographics from profile.json next to the DB.
    let user: Vec<f64> = o["user_vector"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert_eq!(p.user_vector, user, "profile_vector");

    // Ring clock: boot epochs, time-sync anchors and ring_start markers.
    let anchors: usize = p.clock.epochs.iter().map(|e| e.anchors.len()).sum();
    let boots: usize = p.clock.epochs.iter().map(|e| e.boots.len()).sum();
    let want_clock: Vec<u64> = o["clock"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    assert_eq!(vec![p.clock.epochs.len() as u64, anchors as u64, boots as u64], want_clock, "clock counts");

    // The per-minute signal series, bit-for-bit.
    for name in ["met", "motion", "temperature", "heart_rate"] {
        let rows = match name {
            "met" => &p.signals.met,
            "motion" => &p.signals.motion,
            "temperature" => &p.signals.temperature,
            _ => &p.signals.heart_rate,
        };
        // Debug hook: ACTIVITY_DUMP=<path> writes the Rust met series so a
        // digest mismatch can be located row by row against the oracle.
        if name == "met" {
            if let Ok(path) = std::env::var("ACTIVITY_DUMP") {
                let mut text = String::new();
                for (t, values) in rows {
                    text.push_str(&format!("{:?},{:?}\n", t, values));
                }
                std::fs::write(path, text).expect("write dump");
            }
        }
        assert_eq!(digest_f64(rows), o["signals"][name].as_array().unwrap().clone(), "signals.{name}");
    }

    // Step-motion decoder inputs (the decoder itself is not linked, so the gait
    // series stays empty here — exactly as in the oracle capture). Under torch
    // the pipeline runs with --no-stepmotion, which skips collection; this
    // block then runs in the non-torch build only.
    if !cfg!(feature = "torch") {
    let si = p.step_input.as_ref().expect("step packets were collected");
    let want_si = o["step_input"].as_array().unwrap();
    assert_eq!(digest_i64(&si.timestamps), want_si[0].as_array().unwrap().clone(), "step timestamps");
    let raw_rows: Vec<Vec<f32>> = si.raw.iter().map(|r| r.iter().map(|v| *v as f32).collect()).collect();
    assert_eq!(digest_f32_rows(&raw_rows), want_si[1].as_array().unwrap().clone(), "step raw packets");
    assert_eq!(si.timestamps.first().copied(), want_si[2].as_i64(), "first step timestamp");
    assert_eq!(si.timestamps.last().copied(), want_si[3].as_i64(), "last step timestamp");
    let as_i64 = |v: &serde_json::Value| -> Vec<i64> {
        v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as i64).collect()
    };
    assert_eq!(si.raw.first(), Some(&as_i64(&want_si[4])), "first raw packet");
    assert_eq!(si.raw.last(), Some(&as_i64(&want_si[5])), "last raw packet");
    }

    // Every local day: same day list, same skip/run decision, same model inputs.
    let days = o["days"].as_array().unwrap();
    assert_eq!(p.days.len(), days.len(), "day count");
    let mut ran = 0;
    for (mine, want) in p.days.iter().zip(days.iter()) {
        let day = want[0].as_str().unwrap();
        assert_eq!(mine.day.iso(), day, "day boundary drift");
        assert_eq!(mine.day_start_min, want[1].as_f64().unwrap(), "{day}: day_start_min");
        assert_eq!(mine.day.weekday() as f64, want[2].as_f64().unwrap(), "{day}: weekday");
        let kind = match &mine.outcome {
            DayOutcome::NoMet => 0,
            DayOutcome::Rejected => 1,
            DayOutcome::Run(_) => 2,
        };
        assert_eq!(kind, want[3].as_i64().unwrap(), "{day}: run/reject decision");
        let req = match &mine.outcome {
            DayOutcome::Run(req) => req,
            _ => continue,
        };
        ran += 1;
        let c = &want[4];
        let ctx: Vec<f32> = c[0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        assert_eq!(req.context, [ctx[0], ctx[1], ctx[2], ctx[3]], "{day}: context tensor");
        assert_eq!(
            digest_f32_rows(&[req.user.clone()]),
            c[1].as_array().unwrap().clone(),
            "{day}: user tensor"
        );
        for (i, name) in ["met", "step_motion", "motion", "temperature", "heart_rate"].iter().enumerate() {
            let m = match *name {
                "met" => &req.met,
                "step_motion" => &req.step_motion,
                "motion" => &req.motion,
                "temperature" => &req.temperature,
                _ => &req.heart_rate,
            };
            assert_eq!(digest_matrix(m), c[2 + i].as_array().unwrap().clone(), "{day}: {name} tensor");
        }
        match &req.settings {
            Settings::Global(row) => {
                assert_eq!(
                    digest_f32_rows(&[row.clone()]),
                    c[7].as_array().unwrap().clone(),
                    "{day}: global_settings row"
                );
            }
            Settings::Legacy { .. } => panic!("{day}: oracle used a global_settings row"),
        }
        assert!(c[8].as_bool().unwrap() && c[9].as_bool().unwrap(), "{day}: the two None args");
    }
    assert_eq!(ran, 39, "days that reached the model");
}

/// The newest decrypted AAD model, resolved the way the oracle resolves it.
fn model_path() -> Option<PathBuf> {
    activity::find_model(&repo_root()).ok()
}

/// Without a linked TorchScript backend the pipeline must refuse loudly rather
/// than invent sessions. With one linked, this compares the full output.
#[test]
fn activity_output_vs_golden() {
    let Some(db) = db_path() else {
        eprintln!("SKIP activity_output_vs_golden: no events DB (set OURA_DB)");
        return;
    };
    let Some(serial) = common::test_serial() else {
        common::missing_env("activity_output_vs_golden");
        return;
    };
    let Some(captured) = captured_window() else {
        common::missing_env("activity_output_vs_golden");
        return;
    };
    let Some(o) = oracle() else {
        common::missing_vector("activity_output_vs_golden", "activity_oracle.json");
        return;
    };
    if golden_is_stale("activity_output_vs_golden", &db, &serial, &captured, &o["clock"]) {
        return;
    }
    let Some(model) = model_path() else {
        eprintln!("SKIP activity_output_vs_golden: no automatic_activity_detection_3_1_*.pt");
        return;
    };
    let args = activity::parse_args(&golden_args(&db, &serial)).expect("args");
    if activity::activity_backend(&model).is_none() {
        // No backend: the pipeline must stop at the model boundary, loudly.
        let e = activity::json_for(&args, &db, &repo_root()).expect_err("inference is not linked");
        assert!(e.to_string().contains("not linked in"), "unexpected error: {e}");
        eprintln!("SKIP inference half of activity_output_vs_golden: {e}");
        return;
    }
    // The golden was captured by redirecting the oracle's stdout on Windows, so it
    // carries CRLF line endings; the JSON itself is compared byte-for-byte.
    let want = common::load_vector("activity_golden.json")
        .expect("golden was loaded above")
        .replace("\r\n", "\n");
    let got = activity::json_for(&args, &db, &repo_root()).expect("activity run");
    assert_eq!(got.trim_end(), want.trim_end(), "activity JSON differs from the golden");
}
