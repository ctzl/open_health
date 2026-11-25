//! Proves the Rust incremental activity path (`incremental.rs`) returns exactly
//! what the cold path returns, and that its caches invalidate under the same
//! conditions `tools/event_cache.py` invalidates under.
//!
//! Two halves:
//!
//! 1. The six cases of `tools/test_event_cache.py`, mirrored one-for-one on a
//!    synthetic one-boot database (same schema, same fixture rows: one boot, two
//!    `0x42` anchors a day apart, three `0x50` signals). These need no model and
//!    no events DB, so they always run.
//! 2. End-to-end equivalence on the live events DB: cold, then incremental with
//!    no cache, then incremental again with nothing new, then incremental after
//!    simulated new rows. Every case must produce byte-identical JSON to a cold
//!    run on the same data, and the no-new-data case must do zero model calls
//!    (counted, not timed). Skips without the DB, the model asset, or a linked
//!    TorchScript backend.
//!
//! The mutation case works on a copy of the live DB under the temp dir: the
//! events table is append-only, so simulating a sync means INSERTing, and the
//! real DB must not be written. The read-only cases run against the live DB
//! itself, which is also what the benchmark example measures.

use anyhow::Result;
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};

use oura_models::activity;
use oura_models::incremental::{self, EventSource, RunPath, UnitCache, DS_MARGIN, MODEL_KEY};

mod common;

/// A stand-in serial for the fingerprint test: it only has to differ from other
/// stand-ins The live-DB cases take theirs from
/// tests/vectors/test-env.json, the same file the other fixtures are pinned to.
const FAKE_SERIAL: &str = "RING-SERIAL-0";
const TZ: f64 = -5.0;
const WHERE: &str = "decoded_json IS NOT NULL";
/// Ring counter units are 0.1 s, so a day is 864 000 of them (test_event_cache.py).
const DAY_DS: i64 = 864_000;

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

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "oura_inc_{}_{tag}_{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("temp dir");
        TempDir(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------------------
// 1. the event_cache.py contract, on a synthetic one-boot database
// ---------------------------------------------------------------------------

/// `test_event_cache.make_db` — the real events DDL.
fn make_db(path: &Path) -> Connection {
    let db = Connection::open(path).expect("open fixture db");
    db.execute_batch(
        "CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, serial TEXT NOT NULL,
            tag INTEGER NOT NULL, name TEXT NOT NULL, ring_timestamp INTEGER NOT NULL,
            body BLOB NOT NULL, decoded_json TEXT, captured_unix INTEGER NOT NULL,
            UNIQUE(serial, tag, ring_timestamp, body));",
    )
    .expect("create events");
    db
}

/// `test_event_cache.insert`.
fn insert(db: &Connection, tag: i64, ds: i64, captured: i64, decoded: Option<&str>) {
    db.execute(
        "INSERT INTO events(serial,tag,name,ring_timestamp,body,decoded_json,captured_unix) \
         VALUES (?,?,'t',?,?,?,?)",
        params!["S1", tag, ds, b"x".to_vec(), decoded, captured],
    )
    .expect("insert");
}

/// The `setUp` fixture of test_event_cache.py: one boot, two anchors a day
/// apart, signals around them.
fn seed_fixture(db: &Connection) {
    insert(db, 0x42, 1_000_000, 1_789_000_000, Some(r#"{"unix_time":1789000000}"#));
    insert(db, 0x42, 1_000_000 + DAY_DS, 1_789_086_400, Some(r#"{"unix_time":1789086400}"#));
    for ds in [900_000i64, 1_200_000, 1_800_000] {
        insert(db, 0x50, ds, 1_789_050_000, Some(r#"{"met":[1.0]}"#));
    }
}

fn source(db: &Path) -> EventSource {
    EventSource::new(db, None, MODEL_KEY, "test-1").expect("event source")
}

/// `test_no_new_rows_replays_cached_epochs`: with the watermark unmoved the clock
/// comes back out of the state file and no boot is re-tailed.
#[test]
fn no_new_rows_replays_cached_epochs() {
    let tmp = TempDir::new("noreplay");
    let path = tmp.path().join("t.db");
    let db = make_db(&path);
    seed_fixture(&db);
    db.close().ok();

    let mut src = source(&path);
    let (epochs1, tail1) = src.clock(WHERE).expect("cold clock");
    assert_eq!(epochs1.epochs.len(), 1, "one boot");
    assert_eq!(tail1.len(), 1, "a cold clock tails its only boot");
    src.commit(&epochs1, src.max_id(WHERE).expect("max id")).expect("commit");

    let mut src2 = source(&path);
    let (epochs2, tail2) = src2.clock(WHERE).expect("warm clock");
    assert_eq!(tail2, Vec::new(), "nothing new, nothing to re-date");
    // The round-trip form is what a later run actually reads, so compare that
    // (Python's test does the same with plain()).
    assert_eq!(src.state.epochs, src2.state.epochs, "cached epochs differ");
    assert_eq!(epochs2.epochs.len(), epochs1.epochs.len(), "epoch count differs");
    // And the replayed clock dates rows identically to the built one.
    for ds in (900_000i64..=1_800_000).step_by(100_000) {
        assert_eq!(
            epochs2.unix_s(ds, Some(1_789_050_000)),
            epochs1.unix_s(ds, Some(1_789_050_000)),
            "ds {ds} re-dated by the cache round-trip"
        );
    }
}

/// `test_new_anchor_tails_only_past_the_previous_last_anchor`.
#[test]
fn new_anchor_tails_only_past_the_previous_last_anchor() {
    let tmp = TempDir::new("anchor");
    let path = tmp.path().join("t.db");
    let db = make_db(&path);
    seed_fixture(&db);

    let mut src = source(&path);
    let (epochs, _) = src.clock(WHERE).expect("cold clock");
    src.commit(&epochs, src.max_id(WHERE).expect("max id")).expect("commit");

    insert(&db, 0x42, 1_900_000, 1_789_090_000, Some(r#"{"unix_time":1789090000}"#));
    db.close().ok();

    let mut src2 = source(&path);
    let (epochs2, tail) = src2.clock(WHERE).expect("warm clock");
    assert_eq!(tail.len(), 1, "exactly one boot changed");
    let window = tail[0];
    // Re-dating can only affect ds at/after the previous last anchor (plus the
    // inversion margin) — never the whole boot.
    assert_eq!(window.lo, 1_000_000 + DAY_DS - DS_MARGIN, "tail start");
    assert_eq!(window.hi, epochs2.epochs[window.epoch].max_ds.min(1_900_000 + DS_MARGIN), "tail end");
    assert!(window.lo > epochs2.epochs[window.epoch].min_ds, "not the epoch start");
}

/// `test_cold_run_seeding_prevents_whole_boot_tail`: `note_epochs` is what
/// `run_full` does before committing, and it is what stops the next run from
/// re-processing a whole boot.
#[test]
fn cold_run_seeding_prevents_whole_boot_tail() {
    let tmp = TempDir::new("seed");
    let path = tmp.path().join("t.db");
    let db = make_db(&path);
    seed_fixture(&db);

    let mut src = source(&path);
    let (epochs, _) = src.clock(WHERE).expect("cold clock");
    src.note_epochs(&epochs);
    src.commit(&epochs, src.max_id(WHERE).expect("max id")).expect("commit");

    insert(&db, 0x42, 1_900_000, 1_789_090_000, Some(r#"{"unix_time":1789090000}"#));
    db.close().ok();

    let mut src2 = source(&path);
    let (_, tail) = src2.clock(WHERE).expect("warm clock");
    assert_eq!(tail.len(), 1, "one boot re-tailed");
    assert!(tail[0].lo > 1_000_000, "not the epoch start: {:?}", tail[0]);
}

/// `test_rows_in_windows_dedupes_and_orders_like_a_full_load`: genuine duplicates
/// (distinct ids, identical payload) survive; rows covered by two overlapping
/// windows appear exactly once, in `(captured_unix, id)` order.
#[test]
fn rows_in_windows_dedupes_and_orders_like_a_full_load() {
    let tmp = TempDir::new("dedupe");
    let path = tmp.path().join("t.db");
    let db = make_db(&path);
    seed_fixture(&db);
    db.execute(
        "INSERT INTO events(serial,tag,name,ring_timestamp,body,decoded_json,captured_unix) \
         VALUES ('S1',80,'t',1200000,?,?,1789050000)",
        params![b"y".to_vec(), r#"{"met":[2.0]}"#],
    )
    .expect("duplicate row");
    db.close().ok();

    let src = source(&path);
    let reader = Connection::open(&path).expect("open reader");
    let mut stmt = reader
        .prepare(
            "SELECT ring_timestamp,tag,decoded_json,captured_unix FROM events \
             WHERE tag IN (80) AND decoded_json IS NOT NULL ORDER BY captured_unix,id",
        )
        .expect("prepare full load");
    let full = stmt
        .query_map([], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, Option<String>>(2)?, r.get::<_, i64>(3)?))
        })
        .expect("full load")
        .collect::<rusqlite::Result<Vec<_>>>()
        .expect("full load rows");
    assert_eq!(full.len(), 4, "three signals plus the duplicate");

    let got = src
        .rows_in_windows(&[(900_000, 1_500_000), (1_100_000, 1_800_000)], &[0x50], WHERE)
        .expect("windowed rows");
    assert_eq!(got, full, "windowed load differs from a full load");
}

/// `test_skeleton_watermark_survives_a_run_killed_before_commit`: the skeleton
/// cache has its own watermark, so a run that never committed does not leave the
/// next one appending the same rows twice.
#[test]
fn skeleton_watermark_survives_a_run_killed_before_commit() {
    let tmp = TempDir::new("skelwm");
    let path = tmp.path().join("t.db");
    let db = make_db(&path);
    seed_fixture(&db);
    db.close().ok();

    let mut src = source(&path);
    let (epochs1, _) = src.clock(WHERE).expect("cold clock"); // writes the skeleton, no commit
    let mut src2 = source(&path); // state still says watermark = 0
    let (epochs2, _) = src2.clock(WHERE).expect("second clock");
    assert_eq!(epochs2.epochs.len(), epochs1.epochs.len(), "rows were appended twice");
    assert_eq!(src2.state.epochs, src.state.epochs, "epoch JSON differs");
}

/// A changed WHERE clause reads a different skeleton file rather than corrupting
/// this one (`discard_skeleton_cache` in Python).
#[test]
fn where_clause_keys_the_skeleton_cache() {
    let tmp = TempDir::new("skelkey");
    let path = tmp.path().join("t.db");
    let db = make_db(&path);
    seed_fixture(&db);
    db.close().ok();

    let src = source(&path);
    assert_ne!(src.skel_path(WHERE), src.skel_path("tag IN (126,127) AND body IS NOT NULL"));
    assert_eq!(src.skel_path(WHERE), src.skel_path(WHERE), "stable key");
    src.discard_skeleton_cache();
}

/// A state-version change drops the cached clock instead of trusting it.
#[test]
fn state_version_change_drops_cached_clock() {
    let tmp = TempDir::new("statever");
    let path = tmp.path().join("t.db");
    let db = make_db(&path);
    seed_fixture(&db);
    db.close().ok();

    let mut src = source(&path);
    let (epochs, _) = src.clock(WHERE).expect("cold clock");
    src.commit(&epochs, src.max_id(WHERE).expect("max id")).expect("commit");
    assert!(src.has_state());

    let mut upgraded = EventSource::new(&path, None, MODEL_KEY, "test-2").expect("new version");
    assert!(!upgraded.has_state(), "a version bump must not reuse the cached clock");
    let (epochs2, tail) = upgraded.clock(WHERE).expect("rebuilt clock");
    assert_eq!(epochs2.epochs.len(), epochs.epochs.len(), "rebuild differs from the cached clock");
    assert_eq!(upgraded.state.epochs, src.state.epochs, "rebuilt epoch JSON differs");
    assert_eq!(tail.len(), 1, "and it tails its boot, like a cold run");
}

/// `test_unit_cache_invalidates_on_settings_fingerprint`.
#[test]
fn unit_cache_invalidates_on_settings_fingerprint() {
    let tmp = TempDir::new("units");
    let path = tmp.path().join("t.db");
    let db = make_db(&path);
    db.close().ok();

    let day = "2026-11-09";
    let session = activity::Session {
        start: "2026-11-09T07:00:00".into(),
        end: "2026-11-09T07:40:00".into(),
        duration_min: 40,
        is_workout: 1.0,
        label: "walk".into(),
        label_confidence: 0.91,
        top3: vec![("walk".into(), 0.91), ("run".into(), 0.05)],
    };

    let mut units = UnitCache::new(&path, MODEL_KEY, None, "fp-a");
    assert_eq!(units.path(), path.with_file_name(format!("{MODEL_KEY}.all.units.json")));
    units.set(day, vec![session.clone()]);
    units.save().expect("save");

    let same = UnitCache::new(&path, MODEL_KEY, None, "fp-a");
    assert_eq!(same.len(), 1, "same fingerprint must load");
    let got = same.get(day).expect("cached day");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].label, session.label);
    assert_eq!(got[0].duration_min, session.duration_min);
    assert_eq!(got[0].label_confidence, session.label_confidence);
    assert_eq!(got[0].top3, session.top3);

    let other = UnitCache::new(&path, MODEL_KEY, None, "fp-b");
    assert!(other.is_empty(), "a different fingerprint must not load");
    assert!(other.get(day).is_none());
}

/// The fingerprint must move when anything that changes the answer moves, and
/// only then.
#[test]
fn settings_fingerprint_tracks_the_inputs() {
    let base = activity::Args {
        db: None,
        tz: -5.0,
        threshold: 0.5,
        min_duration: 10.0,
        date: None,
        full: false,
        no_stepmotion: false,
        serial: Some(FAKE_SERIAL.into()),
        json: false,
        verbose: false,
        user_vector: vec![37.0, 1.0, 1.78, 75.0],
    };
    let user = [37.0, 1.0, 1.78, 75.0];
    let fp = |a: &activity::Args| {
        incremental::settings_fingerprint(a, "aad.pt", "3.1.15", &user, Some(Path::new("gait.pt")))
    };
    let want = fp(&base);
    assert_eq!(fp(&base), want, "stable for identical settings");

    let mut threshold = base.clone();
    threshold.threshold = 0.6;
    assert_ne!(fp(&threshold), want, "threshold must invalidate");

    let mut min_duration = base.clone();
    min_duration.min_duration = 15.0;
    assert_ne!(fp(&min_duration), want, "min duration must invalidate");

    let mut tz = base.clone();
    tz.tz = -6.0;
    assert_ne!(fp(&tz), want, "timezone must invalidate");

    let mut no_gait = base.clone();
    no_gait.no_stepmotion = true;
    assert_ne!(fp(&no_gait), want, "disabling gait must invalidate");

    let other_fp =
        incremental::settings_fingerprint(&base, "aad.pt", "3.1.16", &user, Some(Path::new("gait.pt")));
    assert_ne!(other_fp, want, "a model upgrade must invalidate");

    let other_user = incremental::settings_fingerprint(&base, "aad.pt", "3.1.15", &[38.0, 1.0, 1.78, 75.0], Some(Path::new("gait.pt")));
    assert_ne!(other_user, want, "a profile change must invalidate");
}

// ---------------------------------------------------------------------------
// 2. end-to-end equivalence on the live events DB
// ---------------------------------------------------------------------------

/// The dashboard shape: default args, no `--full`, one serial. `--full` is a
/// no-op in this port (the cold path always covers every day), so cold and
/// incremental runs are compared on identical arguments.
fn live_args(db: &Path, serial: &str) -> Vec<String> {
    vec![
        db.to_string_lossy().into_owned(),
        "--tz".into(),
        TZ.to_string(),
        "--serial".into(),
        serial.into(),
    ]
}

struct Run {
    json: String,
    report: incremental::RunReport,
    aad_calls: usize,
    gait_calls: usize,
}

fn run(db: &Path, serial: &str, extra: &[&str]) -> Result<Run> {
    let repo = repo_root();
    let mut argv = live_args(db, serial);
    argv.extend(extra.iter().map(|s| (*s).to_string()));
    let args = activity::parse_args(&argv)?;
    let model = activity::find_model(&repo)?;
    let step = if args.no_stepmotion {
        None
    } else {
        model.parent().and_then(|d| activity::find_step_decoder(d))
    };
    let backends = incremental::counted_backends(&model, step.as_deref())?;
    let report = incremental::run_model(&args, db, &repo, &backends)?;
    Ok(Run {
        json: activity::render_result(&report.model_version, &report.sessions),
        aad_calls: backends.counters.aad(),
        gait_calls: backends.counters.decoder(),
        report,
    })
}

/// Every sidecar this port owns for that DB, so a test can start from nothing.
fn clear_caches(db: &Path, serial: &str) {
    for suffix in ["inc.json", "units.json", "skel.json"] {
        let pattern = format!("{MODEL_KEY}.{serial}.");
        let Some(parent) = db.parent() else { continue };
        let Ok(entries) = std::fs::read_dir(parent) else { continue };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(&pattern) && name.ends_with(suffix) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

fn summary(r: &Run) -> String {
    format!(
        "{} sessions, {} days ({} recomputed, {} replayed), {} AAD calls, {} gait calls, {:.2} s",
        r.report.sessions.len(),
        r.report.days_total,
        r.report.days_recomputed,
        r.report.days_replayed,
        r.aad_calls,
        r.gait_calls,
        r.report.seconds
    )
}

/// Cold, then incremental with no cache, then incremental with nothing new, then
/// incremental after simulated new rows. Every case must match a cold run on the
/// same data, byte for byte.
#[test]
fn incremental_matches_cold_on_live_db() {
    let Some(db) = db_path() else {
        eprintln!("SKIP incremental_matches_cold_on_live_db: no events DB (set OURA_DB)");
        return;
    };
    let Some(serial) = common::test_serial() else {
        common::missing_env("incremental_matches_cold_on_live_db");
        return;
    };
    let serial = serial.as_str();
    let Ok(model) = activity::find_model(&repo_root()) else {
        eprintln!("SKIP incremental_matches_cold_on_live_db: no automatic_activity_detection_3_1_*.pt");
        return;
    };
    if activity::activity_backend(&model).is_none() {
        eprintln!("SKIP incremental_matches_cold_on_live_db: no TorchScript backend (build with --features torch)");
        return;
    }

    // --- case A: cold, with no caches at all ---------------------------------
    clear_caches(&db, serial);
    let cold = run(&db, serial, &[]).expect("cold run");
    assert_eq!(cold.report.path, RunPath::Cold, "case A must take the cold path");
    assert!(cold.report.sessions.len() > 0, "the cold path found no sessions");
    assert!(cold.aad_calls > 0, "the cold path must call the model");
    let baseline = cold.json.clone();
    eprintln!("case A cold          : {}", summary(&cold));

    // --- case B: incremental dispatch with an empty cache --------------------
    // With no epochs and no units, run_model must fall back to the cold path
    // (Python: `if not epochs or not units: run_full(...)`), and produce the
    // same answer while seeding the caches.
    clear_caches(&db, serial);
    let seeded = run(&db, serial, &[]).expect("cache-seeding run");
    assert_eq!(seeded.report.path, RunPath::Cold, "case B must fall back to cold");
    assert_eq!(seeded.json, baseline, "case B output differs from cold");
    assert_eq!(seeded.report.sessions.len(), cold.report.sessions.len());
    eprintln!("case B cold-seed     : {}", summary(&seeded));

    // --- case C: incremental again, nothing new ------------------------------
    let warm = run(&db, serial, &[]).expect("warm run");
    assert_eq!(warm.report.path, RunPath::Incremental, "case C must take the warm path");
    assert_eq!(warm.json, baseline, "case C output differs from cold");
    assert_eq!(warm.report.sessions.len(), cold.report.sessions.len(), "case C session count");
    assert_eq!(warm.report.days_recomputed, 0, "case C recomputed days");
    assert_eq!(warm.aad_calls, 0, "case C must not call the AAD model at all");
    assert_eq!(warm.gait_calls, 0, "case C must not call the gait decoder at all");
    assert!(warm.report.days_replayed > 0, "case C replayed nothing");
    assert_eq!(warm.report.new_signal_rows, 0, "case C saw new rows");
    eprintln!(
        "case C warm replay   : {}  (cold {:.2} s -> warm {:.2} s, {:.1}x)",
        summary(&warm),
        cold.report.seconds,
        warm.report.seconds,
        cold.report.seconds / warm.report.seconds.max(0.001)
    );
    assert!(
        warm.report.seconds < cold.report.seconds,
        "warm replay ({:.2} s) was not faster than cold ({:.2} s)",
        warm.report.seconds,
        cold.report.seconds
    );

    // --- case C2: --date forces one day to be recomputed (Python's force_day);
    // the answer must equal the cold path filtered to that same day.
    let day = cold.report.sessions.last().expect("a session to force").start[..10].to_string();
    let forced = run(&db, serial, &["--date", &day]).expect("forced-day run");
    assert_eq!(forced.report.path, RunPath::Incremental, "force_day runs the warm path");
    assert_eq!(forced.report.days_recomputed, 1, "force_day must recompute exactly that day");
    let filtered: Vec<activity::Session> =
        cold.report.sessions.iter().filter(|s| s.start.starts_with(&day)).cloned().collect();
    let want = activity::render_result(&cold.report.model_version, &filtered);
    assert_eq!(forced.json, want, "force_day output differs from the cold path for {day}");
    eprintln!("case C2 force_day {day}: {}", summary(&forced));

    // --- case D: simulate a sync, then incremental vs cold on a copy ---------
    // The events table is append-only, so a sync is an INSERT; the live DB must
    // not be written, hence the copy.
    let tmp = TempDir::new("copy");
    let copy = tmp.path().join("oura.db");
    // VACUUM INTO rather than copying the file: it is a consistent snapshot that
    // includes whatever is still in the WAL, and it leaves the source untouched.
    Connection::open(&db)
        .expect("open db for snapshot")
        .execute("VACUUM INTO ?1", params![copy.to_string_lossy().to_string()])
        .expect("snapshot the db");
    // The model reads profile.json from the directory holding the DB; without it
    // the user vector falls back to defaults and every feature shifts.
    if let Some(dir) = db.parent() {
        for sidecar in ["profile.json"] {
            let from = dir.join(sidecar);
            if from.is_file() {
                std::fs::copy(&from, tmp.path().join(sidecar)).expect("copy profile");
            }
        }
    }
    assert!(copy.is_file(), "no copy of {db:?}");

    let cold_copy = run(&copy, serial, &[]).expect("cold run on copy");
    assert_eq!(cold_copy.json, baseline, "the copy's cold run differs from the live DB");
    eprintln!("case D cold on copy  : {}", summary(&cold_copy));

    let inserted = simulate_sync(&copy, serial).expect("simulate new rows");
    assert!(inserted > 0, "simulating a sync inserted nothing");

    let warm2 = run(&copy, serial, &[]).expect("incremental after new rows");
    assert_eq!(warm2.report.path, RunPath::Incremental, "case D must take the warm path");
    assert!(warm2.report.days_recomputed > 0, "case D recomputed nothing after new rows");
    assert!(
        warm2.report.days_recomputed < warm2.report.days_total,
        "case D recomputed every day: {} of {}",
        warm2.report.days_recomputed,
        warm2.report.days_total
    );

    let cold_after = run(&copy, serial, &[]).expect("cold run after new rows");
    assert_eq!(warm2.json, cold_after.json, "case D output differs from cold after new rows");
    assert_eq!(
        warm2.report.sessions.len(),
        cold_after.report.sessions.len(),
        "case D session count differs"
    );
    eprintln!(
        "case D warm +{inserted} rows: {} (recomputed {} of {} days)",
        summary(&warm2),
        warm2.report.days_recomputed,
        warm2.report.days_total
    );
}

/// Append a day's worth of signal rows shifted 1 s later, the way a sync would:
/// new ids, new rows, same clock. `INSERT OR IGNORE` because the shifted ds can
/// collide with a real row's UNIQUE key.
fn simulate_sync(db: &Path, serial: &str) -> Result<usize> {
    let con = Connection::open(db)?;
    let rows: Vec<(String, i64, String, i64, Vec<u8>, Option<String>, i64)> = con
        .prepare(
            "SELECT serial, tag, name, ring_timestamp, body, decoded_json, captured_unix \
             FROM events WHERE serial = ?1 AND tag = 80 AND decoded_json IS NOT NULL \
             ORDER BY ring_timestamp DESC LIMIT 300",
        )?
        .query_map(params![serial], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get(4)?,
                r.get(5)?,
                r.get(6)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut inserted = 0;
    {
        let mut stmt = con.prepare(
            "INSERT OR IGNORE INTO events(serial,tag,name,ring_timestamp,body,decoded_json,captured_unix) \
             VALUES (?,?,?,?,?,?,?)",
        )?;
        for (serial, tag, name, ds, body, json, captured) in &rows {
            inserted += stmt.execute(params![serial, tag, name, ds + 10, body, json, captured])?;
        }
    }
    con.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)").ok();
    Ok(inserted)
}
