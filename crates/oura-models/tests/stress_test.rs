//! Tests for the `stress` port (tools/run_stress_events.py).
//!
//! The astd segmenter is pure logic, so it gets direct unit tests against the
//! documented rules (±0.5/±0.4 zones, N=4 bins inside a 65-minute window,
//! same-type merge under 30 minutes, int32 day-relative timestamps, the int32
//! overflow the model raises on). `json_matches_python_oracle_live` replays the
//! whole pipeline against the oracle on the live DB; `json_matches_golden_vector`
//! compares it with tests/vectors/stress_golden.json and prints SKIP once the
//! live DB has moved past that snapshot. Both need the events DB and the project
//! venv (the DSA NN still runs in Python); without them they SKIP.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use oura_models::stress::{
    self, astd_event_detection, classify, fmt_local, iso_from_days, json_f64, py_round1,
    AstdError, RingClock, ALLOWED_MERGE_GAP_MS, BIN_WIDTH_MS, MAX_WINDOW_DELTA_MS,
    MIN_WINDOW_DELTA_MS, TYPE_RESTORED, TYPE_STRESSED,
};

mod common;

const BIN: i64 = BIN_WIDTH_MS; // 15 min, in ms

fn ts(n: usize, step: i64) -> Vec<f32> {
    (0..n).map(|i| (i as i64 * step) as f32).collect()
}

fn vals(n: usize, v: f32) -> Vec<f32> {
    vec![v; n]
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

// ---------------------------------------------------------------------------
// classify: the ±0.5 / ±0.4 zones
// ---------------------------------------------------------------------------

#[test]
fn classify_zone_boundaries() {
    // fully stressed at exactly -0.5, borderline only in (-0.5, -0.4]
    assert!(classify(-0.5).fully_stressed);
    assert!(!classify(-0.5).borderline_stressed);
    assert!(classify(-0.45).borderline_stressed);
    assert!(!classify(-0.45).fully_stressed);
    assert!(classify(-0.4).borderline_stressed);
    assert!(!classify(-0.39).borderline_stressed);
    assert!(!classify(-0.3).fully_stressed && !classify(-0.3).borderline_stressed);
    // mirrored on the restored side: fully at >= +0.5, borderline in [0.4, 0.5)
    assert!(classify(0.5).fully_restored);
    assert!(!classify(0.5).borderline_restored);
    assert!(classify(0.4).borderline_restored);
    assert!(classify(0.45).borderline_restored);
    assert!(!classify(0.55).borderline_restored && classify(0.55).fully_restored);
    // NaN is neither zone and not finite; 0 is neutral
    let nan = classify(f32::NAN);
    assert!(!nan.finite && !nan.fully_stressed && !nan.fully_restored);
    let zero = classify(0.0);
    assert!(zero.finite && !zero.fully_stressed && !zero.fully_restored);
}

// ---------------------------------------------------------------------------
// the N=4 / 65-minute window rule
// ---------------------------------------------------------------------------

#[test]
fn four_bins_make_one_event_shifted_by_one_bin() {
    let out = astd_event_detection(&vals(4, -0.6), &ts(4, BIN)).unwrap();
    assert_eq!(out.types, vec![TYPE_STRESSED]);
    assert_eq!(out.n_stressed, 1);
    assert_eq!(out.n_restored, 0);
    // reported start is one bin before the first stressed sample (astd:639)
    assert_eq!(out.starts_ms, vec![-BIN]);
    assert_eq!(out.ends_ms, vec![3 * BIN]);
    assert!(near(out.durations_min[0] as f64, 60.0));
    assert!(near(out.total_stressed_min as f64, 60.0));
}

#[test]
fn window_span_must_stay_inside_55_to_65_minutes() {
    // 4 bins 15 min apart -> span 60 min: accepted (see above).
    // Exactly the documented maximum: 4 bins 1000000 ms apart = 65 min.
    let out = astd_event_detection(&vals(4, -0.6), &ts(4, 1_000_000)).unwrap();
    assert_eq!(out.types.len(), 1);
    assert!(near(out.durations_min[0] as f64, 65.0));
    // one ms over the maximum -> no event at all
    assert!(astd_event_detection(&vals(4, -0.6), &ts(4, 1_000_001))
        .unwrap()
        .types
        .is_empty());
    // exactly the minimum (55 min) is accepted, one ms under is not
    assert_eq!(
        astd_event_detection(&vals(4, -0.6), &ts(4, 800_000))
            .unwrap()
            .types
            .len(),
        1
    );
    assert!(astd_event_detection(&vals(4, -0.6), &ts(4, 799_999))
        .unwrap()
        .types
        .is_empty());
    assert!(MIN_WINDOW_DELTA_MS == 3_300_000 && MAX_WINDOW_DELTA_MS == 3_900_000);
}

#[test]
fn three_bins_are_rejected_by_the_validator() {
    let e = astd_event_detection(&vals(3, -0.6), &ts(3, BIN)).unwrap_err();
    assert_eq!(e, AstdError::Validator(1));
    assert!(e.message().contains("insufficient data"));
}

// ---------------------------------------------------------------------------
// merging and boundary growth
// ---------------------------------------------------------------------------

#[test]
fn overlapping_windows_merge_into_one_event() {
    // 5 consecutive stressed bins produce two overlapping 4-bin windows; they
    // must collapse to a single event, not two.
    let out = astd_event_detection(&vals(5, -0.6), &ts(5, BIN)).unwrap();
    assert_eq!(out.types.len(), 1);
    assert_eq!(out.starts_ms, vec![-BIN]);
    assert_eq!(out.ends_ms, vec![4 * BIN]);
    assert!(near(out.durations_min[0] as f64, 75.0));
    assert_eq!(out.n_stressed, 1);
}

#[test]
fn clusters_more_than_30_minutes_apart_stay_separate() {
    // bins 0-3 stressed, 4-5 neutral, 6-9 stressed: gap 45 min > 30 min.
    let mut v = vals(10, -0.6);
    v[4] = 0.0;
    v[5] = 0.0;
    let out = astd_event_detection(&v, &ts(10, BIN)).unwrap();
    assert_eq!(out.types, vec![TYPE_STRESSED, TYPE_STRESSED]);
    assert_eq!(out.starts_ms, vec![-BIN, 5 * BIN]);
    assert_eq!(out.ends_ms, vec![3 * BIN, 9 * BIN]);
    assert_eq!(out.n_stressed, 2);
    assert!(near(out.total_stressed_min as f64, 120.0));
    assert!(ALLOWED_MERGE_GAP_MS == 1_800_000);
}

#[test]
fn clusters_within_30_minutes_merge() {
    // same shape, but only one neutral bin between them -> gap 30 min.
    let mut v = vals(9, -0.6);
    v[4] = 0.0;
    let out = astd_event_detection(&v, &ts(9, BIN)).unwrap();
    assert_eq!(out.types.len(), 1, "expected a single merged event");
    assert_eq!(out.starts_ms, vec![-BIN]);
    assert_eq!(out.ends_ms, vec![8 * BIN]);
    assert!(near(out.total_stressed_min as f64, 135.0));
}

#[test]
fn event_grows_across_a_bin_boundary_when_the_neighbour_qualifies() {
    // A NaN in the middle breaks the 4-bin window (only one NaN is tolerated in
    // the first window, and the second window starts on the NaN), but bin 4 is
    // fully stressed and exactly one bin after the event end -> astd:521 grows
    // the event to cover it.
    let mut v = vals(5, -0.6);
    v[1] = f32::NAN;
    let out = astd_event_detection(&v, &ts(5, BIN)).unwrap();
    assert_eq!(out.types.len(), 1);
    assert_eq!(out.ends_ms, vec![4 * BIN]);
    assert!(near(out.durations_min[0] as f64, 75.0));
}

// ---------------------------------------------------------------------------
// tolerance for borderline / missing bins
// ---------------------------------------------------------------------------

#[test]
fn one_borderline_bin_is_tolerated_two_are_not() {
    let mut one = vals(4, -0.6);
    one[1] = -0.45; // borderline, not fully
    assert_eq!(
        astd_event_detection(&one, &ts(4, BIN)).unwrap().types.len(),
        1,
        "a single borderline bin must not break the event"
    );

    let mut two = vals(4, -0.6);
    two[1] = -0.45;
    two[2] = -0.45;
    assert!(
        astd_event_detection(&two, &ts(4, BIN)).unwrap().types.is_empty(),
        "two non-fully bins must break the event"
    );

    // an out-of-zone bin is never tolerated, even one
    let mut neutral = vals(4, -0.6);
    neutral[2] = 0.0;
    assert!(astd_event_detection(&neutral, &ts(4, BIN))
        .unwrap()
        .types
        .is_empty());

    // both end bins must be fully in-zone
    let mut weak_end = vals(4, -0.6);
    weak_end[3] = -0.45;
    assert!(astd_event_detection(&weak_end, &ts(4, BIN))
        .unwrap()
        .types
        .is_empty());
}

#[test]
fn restored_events_are_typed_and_counted_separately() {
    let out = astd_event_detection(&vals(4, 0.6), &ts(4, BIN)).unwrap();
    assert_eq!(out.types, vec![TYPE_RESTORED]);
    assert_eq!(out.n_restored, 1);
    assert_eq!(out.n_stressed, 0);
    assert!(near(out.total_restored_min as f64, 60.0));
    assert!(near(out.total_stressed_min as f64, 0.0));
}

#[test]
fn a_quiet_day_returns_no_events_and_no_error() {
    let out = astd_event_detection(&vals(20, 0.0), &ts(20, BIN)).unwrap();
    assert_eq!(out, Default::default());
    assert_eq!(out.n_stressed, 0);
    assert!(out.types.is_empty());
}

// ---------------------------------------------------------------------------
// validator rejections (the oracle records these as {"day":…, "error":…})
// ---------------------------------------------------------------------------

#[test]
fn validator_rejects_bad_input_with_the_model_messages() {
    // duplicate timestamps are not strictly increasing
    let mut dup = ts(4, BIN);
    dup[2] = dup[1];
    let e = astd_event_detection(&vals(4, -0.6), &dup).unwrap_err();
    assert_eq!(e, AstdError::Validator(3));
    // the oracle records `str(e).splitlines()[-1]` of a graph raise, which
    // carries the exception class TorchScript raised it under
    assert_eq!(
        e.message(),
        "builtins.Exception: 3, Error, dsa_timestamps_ms is not strictly monotonically increasing."
    );

    // decreasing timestamps
    let back = vec![3_000_000.0, 2_000_000.0, 1_000_000.0, 0.0];
    assert_eq!(
        astd_event_detection(&vals(4, -0.6), &back).unwrap_err(),
        AstdError::Validator(3)
    );

    // a finite value outside [-1, +1]
    let mut wild = vals(4, -0.6);
    wild[1] = -1.5;
    let e = astd_event_detection(&wild, &ts(4, BIN)).unwrap_err();
    assert_eq!(e, AstdError::Validator(4));
    assert!(e.message().contains("outside [-1, +1]"));

    // NaN is not finite, so it is not a range violation
    let mut nan = vals(4, -0.6);
    nan[1] = f32::NAN;
    assert!(astd_event_detection(&nan, &ts(4, BIN)).is_ok());

    // length mismatch
    let e = astd_event_detection(&vals(5, -0.6), &ts(4, BIN)).unwrap_err();
    assert_eq!(e, AstdError::Validator(2));
    assert!(e.message().contains("length mismatch"));
}

// ---------------------------------------------------------------------------
// the C `int` cast the graph applies to day-relative timestamps
// ---------------------------------------------------------------------------

/// Four bins 15 minutes apart, the last one at `last`.
fn ts_ending_at(last: f64) -> Vec<f32> {
    let base = last - 3.0 * BIN as f64;
    (0..4).map(|i| (base + i as f64 * BIN as f64) as f32).collect()
}

#[test]
fn timestamps_inside_int32_are_accepted() {
    // 2147483520 is exactly representable in f32 and below i32::MAX.
    let out = astd_event_detection(&vals(4, -0.6), &ts_ending_at(2_147_483_520.0)).unwrap();
    assert_eq!(out.types.len(), 1);
    // the negative end of the range: a day starting exactly at i32::MIN
    let start = i32::MIN as f64;
    let ts: Vec<f32> = (0..4).map(|i| (start + i as f64 * BIN as f64) as f32).collect();
    assert_eq!(astd_event_detection(&vals(4, -0.6), &ts).unwrap().types.len(), 1);
}

#[test]
fn timestamps_outside_int32_abort_the_day() {
    // This is what the real DB hits: absolute ring milliseconds pass 2^31 after
    // ~25 days of uptime, so a day that mixes two rings' counters overflows.
    const OVER: f64 = 2_147_483_648.0; // 2^31, the first value that does not fit
    let e = astd_event_detection(&vals(4, -0.6), &ts_ending_at(OVER)).unwrap_err();
    assert_eq!(e, AstdError::IntOverflow);
    assert_eq!(
        e.message(),
        "RuntimeError: value cannot be converted to type int without overflow"
    );

    // 2^31-1 does not survive either: the tensor is float32, and 2147483647
    // rounds to 2147483648.0 there — the same thing the model's cast sees.
    assert_eq!(
        astd_event_detection(&vals(4, -0.6), &ts_ending_at(2_147_483_647.0)).unwrap_err(),
        AstdError::IntOverflow
    );
    // below i32::MIN as well
    assert_eq!(
        astd_event_detection(&vals(4, -0.6), &ts_ending_at(-29_504_048_900.0)).unwrap_err(),
        AstdError::IntOverflow
    );
}

#[test]
fn an_out_of_range_sample_breaks_a_day_that_has_events() {
    // The window itself is in range, but the model builds a timestamp->index map
    // over the whole day once any event exists (astd:494), so a later sample
    // past 2^31 still aborts it.
    let mut ts: Vec<f32> = ts_ending_at(3.0 * BIN as f64);
    ts.push(2_147_483_648.0);
    ts.push((2_147_483_648.0 + BIN as f64) as f32);
    let mut v = vals(4, -0.6);
    v.extend([0.0, 0.0]);
    assert_eq!(astd_event_detection(&v, &ts).unwrap_err(), AstdError::IntOverflow);
}

#[test]
fn the_int_cast_precedes_the_zone_check_but_not_the_finite_guard() {
    // astd:362,364 skip a window when an end bin is not finite; astd:367-368
    // then casts the window's timestamps to C `int` before any zone or span
    // check — so a day of finite-but-neutral samples still aborts on an
    // out-of-range timestamp, while a day of NaN samples never gets there.
    let ts = ts_ending_at(2_147_483_648.0);
    assert_eq!(
        astd_event_detection(&vals(4, 0.0), &ts).unwrap_err(),
        AstdError::IntOverflow
    );

    let nan = vec![f32::NAN; 4];
    let out = astd_event_detection(&nan, &ts).unwrap();
    assert_eq!(out, Default::default());
}

#[test]
fn the_validator_runs_before_the_int_cast() {
    // validate() is the first thing forward() calls, so a short day reports the
    // validator condition even with out-of-range timestamps.
    let ts: Vec<f32> = (0..3).map(|i| (2_147_483_648.0 + i as f64 * BIN as f64) as f32).collect();
    assert_eq!(
        astd_event_detection(&vals(3, -0.6), &ts).unwrap_err(),
        AstdError::Validator(1)
    );
}

// ---------------------------------------------------------------------------
// epoch_time.py mirror: ring deciseconds -> UTC
// ---------------------------------------------------------------------------

type Row = (i64, i64, Option<String>, i64);

fn anchor(ds: i64, unix: i64, cu: i64) -> Row {
    (ds, 0x42, Some(format!("{{\"unix_time\": {unix}}}")), cu)
}

#[test]
fn ring_clock_interpolates_between_time_anchors() {
    let clock = RingClock::from_rows(&[
        anchor(1_000_000, 1_700_000_000, 1_700_000_000),
        anchor(2_000_000, 1_700_000_100, 1_700_000_100),
    ]);
    // nearest anchor wins; the tie goes to the earlier one (epoch_time._nearest)
    // 500000 deciseconds past the anchor = +50000 s
    let u = clock.unix_s(1_500_000).unwrap();
    assert!(near(u, 1_700_050_000.0), "got {u}");
    assert!(near(clock.unix_s(1_000_000).unwrap(), 1_700_000_000.0));
    assert!(near(clock.unix_s(2_000_000).unwrap(), 1_700_000_100.0));
    // outside the anchor range the nearest anchor still extrapolates
    assert!(near(clock.unix_s(999_990).unwrap(), 1_699_999_999.0));
}

#[test]
fn ring_clock_falls_back_to_capture_time_without_anchors() {
    let clock = RingClock::from_rows(&[
        (1_000_000, 0x01, None, 1_700_000_000),
        (1_000_100, 0x01, None, 1_700_000_010),
    ]);
    assert!(near(clock.unix_s(1_000_100).unwrap(), 1_700_000_010.0));
    assert!(near(clock.unix_s(1_000_000).unwrap(), 1_700_000_000.0));
}

#[test]
fn ring_clock_splits_boots_and_dates_each_separately() {
    // A 1-day counter jump is far past the 6h reboot slack -> two epochs.
    let clock = RingClock::from_rows(&[
        anchor(1_000_000, 1_700_000_000, 1_700_000_000),
        anchor(1_000_010, 1_700_000_001, 1_700_000_001),
        anchor(5_000_000, 1_800_000_000, 1_800_000_000),
    ]);
    assert!(near(clock.unix_s(1_000_005).unwrap(), 1_700_000_000.5));
    assert!(near(clock.unix_s(5_000_000).unwrap(), 1_800_000_000.0));
}

#[test]
fn ring_clock_lost_time_between_reboots_is_undated() {
    // Anchors 10 s of counter apart but 100000 s of wall time -> stalled.
    let rows = [
        anchor(1_000_000, 1_700_000_000, 1_700_000_000),
        (1_000_050, 0x41, None, 1_700_050_000),
        (1_000_080, 0x41, None, 1_700_080_000),
        anchor(1_000_100, 1_700_100_000, 1_700_100_000),
    ];
    let clock = RingClock::from_rows(&rows);
    // between two reboots the lost time cannot be attributed -> None
    assert_eq!(clock.unix_s(1_000_060), None);
    // before the first reboot: extrapolate from the early anchor
    assert!(near(clock.unix_s(1_000_020).unwrap(), 1_700_000_002.0));
    // at/after the last reboot: extrapolate back from the late anchor
    assert!(near(clock.unix_s(1_000_090).unwrap(), 1_700_099_999.0));

    // with no boot marker in the stalled bracket the oracle takes the late side
    let clock = RingClock::from_rows(&rows[..1]
        .iter()
        .chain(rows[3..].iter())
        .cloned()
        .collect::<Vec<_>>());
    assert!(near(clock.unix_s(1_000_020).unwrap(), 1_700_099_992.0));
}

// ---------------------------------------------------------------------------
// calendar / formatting parity with the Python runner
// ---------------------------------------------------------------------------

#[test]
fn day_numbers_and_clock_strings_match_the_oracle() {
    assert_eq!(iso_from_days(0), "1970-01-01");
    assert_eq!(iso_from_days(20743), "2026-10-17");
    assert_eq!(iso_from_days(-1), "1969-12-31");
    assert_eq!(civil_days_check(20743), (2026, 10, 17));

    // the four events in tests/vectors/stress_golden.json, at --tz -5
    assert_eq!(fmt_local(Some(1_792_211_491.4), -5), "10-16 23:31");
    assert_eq!(fmt_local(Some(1_792_215_391.2), -5), "10-17 00:36");
    assert_eq!(fmt_local(Some(1_792_221_631.3), -5), "10-17 02:20");
    assert_eq!(fmt_local(Some(1_792_225_351.2), -5), "10-17 03:22");
    assert_eq!(fmt_local(Some(1_794_293_182.9), -5), "11-10 01:46");
    assert_eq!(fmt_local(Some(1_795_068_537.8), -5), "11-19 01:08");
    // sub-second input is dropped, exactly like datetime truncation
    assert_eq!(fmt_local(Some(1_792_211_491.999_999), -5), "10-16 23:31");
    assert_eq!(fmt_local(None, -5), "undated");
}

fn civil_days_check(days: i64) -> (i64, u32, u32) {
    oura_models::stress::civil_from_days(days)
}

// ---------------------------------------------------------------------------
// json.dumps / round parity
// ---------------------------------------------------------------------------

#[test]
fn floats_serialize_exactly_like_python_repr() {
    assert_eq!(json_f64(0.0), "0.0");
    assert_eq!(json_f64(127.0), "127.0");
    assert_eq!(json_f64(-0.0), "-0.0");
    assert_eq!(json_f64(57.6), "57.6");
    assert_eq!(json_f64(1_792_211_491.4), "1792211491.4");
    assert_eq!(json_f64(0.1 + 0.2), "0.30000000000000004");
    assert_eq!(json_f64(1e16), "1e+16");
    assert_eq!(json_f64(1.5e16), "1.5e+16");
    assert_eq!(json_f64(1e-5), "1e-05");
    assert_eq!(json_f64(1.2e-5), "1.2e-05");
    assert_eq!(json_f64(0.0001), "0.0001"); // Python keeps this one plain
    assert_eq!(json_f64(f64::NAN), "NaN");
    assert_eq!(json_f64(f64::INFINITY), "Infinity");
}

#[test]
fn rounding_matches_python_round_half_to_even() {
    assert!(near(py_round1(57.55), 57.5)); // 57.55 is 57.54999… in binary
    assert!(near(py_round1(64.95), 65.0)); // 64.95 is 64.95000…028 in binary
    assert!(near(py_round1(0.25), 0.2)); // exact tie -> to even
    assert!(near(py_round1(0.75), 0.8));
    assert!(near(py_round1(2.675), 2.7));
    assert!(near(py_round1(-0.05), -0.1));
    assert!(near(py_round1(127.0), 127.0));
}

// ---------------------------------------------------------------------------
// end-to-end: the whole pipeline, against the live oracle and the vector
// ---------------------------------------------------------------------------
//
// The committed vector was captured with the plain oracle invocation — no
// --serial (every ring in the DB), default --tz 1:
//
//   .venv/Scripts/python.exe tools/run_stress_events.py <db> --tz -5 --json \
//       --serial <the tests/vectors/test-env.json serial>
//
// That DB is live (the dashboard syncs into it), so a vector only describes the
// state it was captured at. `json_matches_python_oracle_live` re-runs the
// oracle instead and is the regression gate that survives a moving DB;
// `json_matches_golden_vector` compares against the committed vector and prints
// SKIP when the DB has moved past it.

fn repo_root() -> PathBuf {
    // crates/oura-models -> crates -> repo; overridable so the suite can be
    // driven against a checkout elsewhere (e.g. a scratch build of this crate).
    if let Ok(v) = std::env::var("OURA_REPO") {
        return PathBuf::from(v);
    }
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

fn golden_path() -> PathBuf {
    let local = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors")
        .join("stress_golden.json");
    if local.is_file() {
        return local;
    }
    // When this suite is built from a copy of the crate, the vectors live in the
    // checkout named by OURA_REPO.
    repo_root()
        .join("crates")
        .join("oura-models")
        .join("tests")
        .join("vectors")
        .join("stress_golden.json")
}

fn venv_python(repo: &Path) -> Option<PathBuf> {
    [".venv/Scripts/python.exe", ".venv/bin/python3", ".venv/bin/python"]
        .iter()
        .map(|p| repo.join(p))
        .find(|p| p.is_file())
}

/// The pipeline, run once per test binary against the live DB. The DSA series
/// still comes from tools/run_stress_model.py (the NN stays in Python), so this
/// needs the project venv; without a DB or a venv it prints SKIP and yields
/// None.
fn live_output() -> Option<&'static stress::StressOutput> {
    static CACHE: OnceLock<Option<stress::StressOutput>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let repo = repo_root();
            let (Some(db), Some(_py)) = (db_path(), venv_python(&repo)) else {
                eprintln!(
                    "SKIP stress end-to-end: no events DB (set OURA_DB) or no project venv \
                     at {}",
                    repo.display()
                );
                return None;
            };
            let Some(serial) = common::test_serial() else {
                common::missing_env("stress end-to-end");
                return None;
            };
            let args = [
                db.to_string_lossy().into_owned(),
                "--tz".into(),
                "-5".into(),
                "--json".into(),
                "--serial".into(),
                serial,
                "--repo".into(),
                repo.to_string_lossy().into_owned(),
            ];
            match stress::estimate(&args) {
                Ok(out) => Some(out),
                Err(e) => {
                    eprintln!("SKIP stress end-to-end: {e}");
                    None
                }
            }
        })
        .as_ref()
}

/// `(day, n_samples)` per emitted day entry; error days carry no n_samples in
/// either the oracle output or ours, so they show as -1.
fn fingerprint_days(results: &[stress::DayResult]) -> Vec<(i64, i64)> {
    results
        .iter()
        .map(|d| (d.day, if d.error.is_some() { -1 } else { d.n_samples as i64 }))
        .collect()
}

fn fingerprint_golden(golden: &serde_json::Value) -> Vec<(i64, i64)> {
    golden["days"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| {
            let day = d["day"].as_i64().unwrap();
            let n = if d.get("error").is_some() {
                -1
            } else {
                d["n_samples"].as_i64().unwrap_or(-1)
            };
            (day, n)
        })
        .collect()
}

fn first_difference<'a, T: std::fmt::Debug + PartialEq>(
    a: &'a [T],
    b: &'a [T],
) -> Option<(&'a T, &'a T)> {
    a.iter().zip(b).find(|(x, y)| x != y)
}

/// The oracle re-run right now, same DB, same flags: the gate that keeps
// meaning when the DB moves. Skips (loudly) if a sync lands mid-comparison.
#[test]
fn json_matches_python_oracle_live() {
    let Some(out) = live_output() else { return };
    // live_output() already said SKIP if this is missing.
    let Some(serial) = common::test_serial() else { return };
    let repo = repo_root();
    let Some(py) = venv_python(&repo) else { return }; // live_output already said SKIP
    let db = db_path().unwrap();
    let stat = |db: &Path| {
        std::fs::metadata(db)
            .ok()
            .map(|m| (m.len(), m.modified().ok()))
    };
    let stat_before = stat(&db);

    let script = repo.join("tools").join("run_stress_events.py");
    let ran = Command::new(&py)
        .arg(&script)
        .arg(&db)
        .arg("--tz")
        .arg("-5")
        .arg("--json")
        .arg("--serial")
        .arg(&serial)
        .output()
        .expect("run the Python oracle");
    assert!(
        ran.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&ran.stderr)
    );
    let stat_after = stat(&db);
    if stat_before.is_none() || stat_before != stat_after {
        eprintln!(
            "SKIP json_matches_python_oracle_live: the events DB changed between the two runs \
             ({:?} -> {:?} bytes) — re-run against a quiet DB",
            stat_before.map_or(0, |s| s.0),
            stat_after.map_or(0, |s| s.0)
        );
        return;
    }

    let want = String::from_utf8(ran.stdout).expect("utf8 stdout").trim().to_string();
    let got = out.json.trim();
    if got != want {
        let (a, b) = (fingerprint_days(&out.results), fingerprint_golden(&serde_json::from_str::<serde_json::Value>(&want).unwrap()));
        let diff = first_difference(&a, &b);
        panic!(
            "stress output differs from the Python oracle ({} days vs {})\n  first day-level difference: {diff:?}\n  rust  : {}…\n  oracle: {}…",
            a.len(),
            b.len(),
            &got[..got.len().min(200)],
            &want[..want.len().min(200)],
        );
    }
    // Shape, so an empty-vs-empty match cannot pass silently.
    let parsed: serde_json::Value = serde_json::from_str(&want).unwrap();
    assert!(parsed["days"].as_array().unwrap().len() > 40);
    assert_eq!(parsed["stress_baseline"], serde_json::json!({
        "night_hrv_baseline": out.baseline.night_hrv_baseline,
        "dhrv_baseline": out.baseline.dhrv_baseline,
    }));
}

/// The committed vector in tests/vectors/. The vector is a snapshot of a live
/// DB, so a day-count mismatch is reported as drift (SKIP), not as a failure;
/// when the DB still matches the capture, the output must be byte-identical.
#[test]
fn json_matches_golden_vector() {
    let Some(out) = live_output() else { return };
    let Ok(text) = std::fs::read_to_string(golden_path()) else {
        eprintln!("SKIP json_matches_golden_vector: {}", golden_path().display());
        return;
    };
    let want_text = text.trim();
    let golden: serde_json::Value =
        serde_json::from_str(want_text).expect("golden vector is valid JSON");

    let (got, want) = (fingerprint_days(&out.results), fingerprint_golden(&golden));
    if got != want {
        eprintln!(
            "SKIP json_matches_golden_vector: the events DB has moved since the vector was \
             captured ({} day entries in the vector, {} in the live DB; first difference {:?}) \
             — re-capture with: .venv/Scripts/python.exe tools/run_stress_events.py <db> --tz -5 --json \
             --serial <the tests/vectors/test-env.json serial>",
            want.len(),
            got.len(),
            first_difference(&got, &want),
        );
        return;
    }
    assert_eq!(
        out.json.trim(),
        want_text,
        "stress output differs from the golden vector"
    );
    // And structurally, so a key rename cannot hide behind formatting.
    let parsed: serde_json::Value = serde_json::from_str(out.json.trim()).unwrap();
    assert_eq!(parsed, golden);
    let keys: Vec<&str> = golden["days"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|d| d["events"].as_array().unwrap().iter())
        .flat_map(|e| e.as_object().unwrap().keys().map(|k| k.as_str()))
        .collect::<Vec<_>>();
    for k in ["type", "start_local", "end_local", "start_unix", "end_unix", "duration_min"] {
        assert!(keys.contains(&k), "golden event key {k} missing from the output");
    }
}
