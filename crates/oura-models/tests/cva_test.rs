//! Golden-vector tests for the cva port (tools/run_cva_model.py).
//!
//!   cargo test -p oura-models                       # pipeline only, no libtorch
//!   cargo test -p oura-models --features torch      # + inference vs the golden
//!
//! They need the real events DB and the decrypted cva_1_*.pt; when either is
//! missing the test prints SKIP and passes, so CI without the private data does
//! not go red. Override the DB with OURA_DB=<path>.
//!
//! The committed vector was captured with the plain oracle invocation no
//! --serial (so every ring in the DB), demographics from profile.json:
//!
//!   .venv/Scripts/python.exe tools/run_cva_model.py <db> --json
//!
//! That DB is live (the dashboard syncs into it), so a vector only describes
//! the state it was captured at: the snapshot checks compare the counts they
//! were captured with first and print SKIP when the DB has moved on. The
//! differential test below re-runs the Python oracle instead, so it stays
//! meaningful against a moving DB keep that one as the regression gate.

use std::path::{Path, PathBuf};
use std::process::Command;

use oura_models::cva::{self, Segments};

mod common;

/// A serial no profile.json entry can match: the lookup must still fall back to
/// the "" default entry, and that needs no real ring.
const UNKNOWN_SERIAL: &str = "RING-SERIAL-NOT-IN-PROFILE";

/// Pipeline snapshot: FNV-1a-64 over the concatenated little-endian f32 bytes
/// of every segment, plus the counts it was taken at. Captured from the Python
/// oracle's build_segments(db, 0, None) on 2026-11-23 (torch 2.13.0+cpu).
/// Model-independent: it pins the decode, the >2s gap run split and the
/// 1500-sample chunking. The counts are the golden vector's own `segments` and
/// `measurements` fields; the hash is the one thing no vector carries, so it
/// lives in tests/vectors/test-env.json as `cva_fnv1a64` ("0x…"). Re-capture it
/// together with the golden vector.
fn oracle_fnv1a64() -> Option<u64> {
    let hex = common::test_env()?.get("cva_fnv1a64")?.as_str()?.to_string();
    u64::from_str_radix(hex.trim_start_matches("0x"), 16).ok()
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

fn golden_path() -> PathBuf {
    // Point at a freshly captured vector with OURA_CVA_GOLDEN=<path> — useful
    // when the committed one has drifted behind a live DB.
    if let Ok(v) = std::env::var("OURA_CVA_GOLDEN") {
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors")
        .join("cva_golden.json")
}

fn golden() -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(golden_path()).ok()?;
    serde_json::from_str(&text).ok()
}

/// The exact CLI the golden vector was captured with.
fn oracle_args(db: &Path) -> Vec<String> {
    vec![db.to_string_lossy().into_owned(), "--json".into()]
}

fn fnv1a64_segments(segs: &Segments) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for w in &segs.waves {
        for b in w.iter().flat_map(|s| s.to_le_bytes()) {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

fn drift_notice(what: &str, want: (usize, usize), segs: &Segments) -> String {
    format!(
        "SKIP {what}: the events DB has moved since it was captured \
         (snapshot: {want:?} segments/runs; live DB: {}/{})  re-capture per the module docs.",
        segs.waves.len(),
        segs.runs
    )
}

/// Structural invariants of the segment builder, independent of DB state.
fn check_shape(segs: &Segments) {
    assert!(segs.runs > 0, "no PPG measurements in this DB");
    assert!(!segs.waves.is_empty(), "no segments produced");
    assert!(segs.waves.iter().all(|w| w.len() == cva::SEG_LEN));
    assert_eq!(segs.ts.len(), segs.waves.len());
    assert!(segs.ts.windows(2).all(|w| w[0] <= w[1]), "segment timestamps not ordered");
}

/// Decode + run-splitting + 1500-sample chunking vs the frozen oracle snapshot.
/// Runs without libtorch.
#[test]
fn ppg_pipeline_matches_python_oracle() {
    let Some(db) = db_path() else {
        eprintln!("SKIP ppg_pipeline_matches_python_oracle: no events DB (set OURA_DB)");
        return;
    };
    let Some(g) = golden() else {
        eprintln!("SKIP ppg_pipeline_matches_python_oracle: {}", golden_path().display());
        return;
    };
    let (Some(want_segments), Some(want_runs)) = (
        g["segments"].as_u64().map(|v| v as usize),
        g["measurements"].as_u64().map(|v| v as usize),
    ) else {
        eprintln!(
            "SKIP ppg_pipeline_matches_python_oracle: {} carries no segments/measurements counts",
            golden_path().display()
        );
        return;
    };
    let Some(want_fp) = oracle_fnv1a64() else {
        common::missing_env("ppg_pipeline_matches_python_oracle");
        return;
    };
    let segs = cva::build_segments(&db, 0, None).expect("build_segments");
    check_shape(&segs);

    if segs.waves.len() != want_segments || segs.runs != want_runs {
        eprintln!(
            "{}",
            drift_notice("pipeline fingerprint", (want_segments, want_runs), &segs)
        );
        return;
    }
    let fingerprint = fnv1a64_segments(&segs);
    assert_eq!(
        format!("{fingerprint:#018x}"),
        format!("{want_fp:#018x}"),
        "decoded PPG bytes differ from the Python oracle"
    );
}

/// Same thing against the Python oracle re-run right now, so the gate survives
/// a DB that keeps growing. Needs the project venv (numpy); skipped without it.
#[test]
fn ppg_pipeline_matches_python_oracle_live() {
    let Some(db) = db_path() else {
        eprintln!("SKIP ppg_pipeline_matches_python_oracle_live: no events DB (set OURA_DB)");
        return;
    };
    let repo = repo_root();
    let py = [".venv/Scripts/python.exe", ".venv/bin/python"]
        .iter()
        .map(|p| repo.join(p))
        .find(|p| p.is_file());
    let Some(py) = py else {
        eprintln!("SKIP ppg_pipeline_matches_python_oracle_live: no project venv python");
        return;
    };
    let tools = repo.join("tools");
    // Same build_segments the oracle CLI calls; prints "runs segments fnv".
    let script = format!(
        "import sys; sys.path.insert(0, {tools:?})\n\
         import numpy as np\n\
         from run_cva_model import build_segments\n\
         segs, ts, n = build_segments({db:?}, 0, None)\n\
         h = 0xcbf29ce484222325\n\
         for s in segs:\n\
         \x20   for b in np.asarray(s, dtype=np.float32).tobytes():\n\
         \x20       h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF\n\
         print(n, len(segs), format(h, '016x'))\n"
    );
    let out = Command::new(&py)
        .args(["-c", &script])
        .output()
        .expect("run python oracle");
    if !out.status.success() {
        eprintln!(
            "SKIP ppg_pipeline_matches_python_oracle_live: python oracle failed: {}",
            String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or("")
        );
        return;
    }
    let py_out = String::from_utf8_lossy(&out.stdout);
    let mut it = py_out.split_whitespace();
    let (Some(py_runs), Some(py_segs), Some(py_fnv)) = (it.next(), it.next(), it.next()) else {
        panic!("unparsable oracle output: {py_out}");
    };
    let (py_runs, py_segs): (usize, usize) = (py_runs.parse().unwrap(), py_segs.parse().unwrap());

    let segs = cva::build_segments(&db, 0, None).expect("build_segments");
    check_shape(&segs);
    if (segs.runs, segs.waves.len()) != (py_runs, py_segs) {
        eprintln!(
            "SKIP ppg_pipeline_matches_python_oracle_live: the DB changed between the two runs \
             (python: {py_runs}/{py_segs}, rust: {}/{})  retry.",
            segs.runs,
            segs.waves.len()
        );
        return;
    }
    let fingerprint = fnv1a64_segments(&segs);
    assert_eq!(
        format!("{fingerprint:016x}"),
        py_fnv,
        "decoded PPG bytes differ from the Python oracle (runs {} segments {})",
        segs.runs,
        segs.waves.len()
    );
}

/// Demographics come from profile.json next to the DB, keyed by serial with ""
/// as the default entry same as tools/_common.py read_profile.
#[test]
fn profile_lookup_matches_common_read_profile() {
    let Some(db) = db_path() else {
        eprintln!("SKIP profile_lookup_matches_common_read_profile: no events DB");
        return;
    };
    if !db.with_file_name("profile.json").is_file() {
        eprintln!("SKIP profile_lookup: no profile.json next to the DB");
        return;
    }
    let def = cva::read_profile(&db, None);
    assert_eq!(def["sex"].as_str(), Some("M"), "default profile sex");
    assert_eq!(def["age"].as_f64(), Some(37.0), "default profile age");
    assert_eq!(def["height_m"].as_f64(), Some(1.78), "default profile height_m");
    assert_eq!(def["ring_size"].as_f64(), Some(10.0), "default profile ring_size");
    // An unknown serial still resolves through the "" default entry.
    let other = cva::read_profile(&db, Some(UNKNOWN_SERIAL));
    assert_eq!(other["age"].as_f64(), def["age"].as_f64());
}

/// Full output vs tests/vectors/cva_golden.json needs the torch feature.
#[cfg(feature = "torch")]
#[test]
fn cva_json_matches_golden() {
    let Some(db) = db_path() else {
        eprintln!("SKIP cva_json_matches_golden: no events DB (set OURA_DB)");
        return;
    };
    let Some(g) = golden() else {
        eprintln!("SKIP cva_json_matches_golden: {}", golden_path().display());
        return;
    };
    let segs = cva::build_segments(&db, 0, None).expect("build_segments");
    let want_counts = (
        g["segments"].as_u64().unwrap_or(0) as usize,
        g["measurements"].as_u64().unwrap_or(0) as usize,
    );
    if want_counts != (segs.waves.len(), segs.runs) {
        eprintln!("{}", drift_notice("golden comparison", want_counts, &segs));
        return;
    }
    let est = cva::estimate(&oracle_args(&db)).expect("cva pipeline");
    let got = est.json.to_json_line();
    let want = std::fs::read_to_string(golden_path()).expect("read golden vector");
    let want = want.trim();

    // Same keys, same rounding, same float repr compare the emitted line.
    if got != want {
        panic!(
            "cva output differs from golden\n got:  {got}\n want: {want}\n raw model: cva={:?} quality={:?} variance={:?} pwv={:?}",
            est.raw.daily_cva, est.raw.quality, est.raw.variance, est.raw.daily_pwv
        );
    }
    // And structurally, so a key rename cannot slip through a formatting fluke.
    let parsed: serde_json::Value = serde_json::from_str(&got).unwrap();
    let w: serde_json::Value = serde_json::from_str(want).unwrap();
    assert_eq!(parsed, w);
    let mut keys: Vec<&str> = w.as_object().unwrap().keys().map(|k| k.as_str()).collect();
    keys.sort();
    assert_eq!(
        keys,
        vec![
            "chronological_age",
            "measurements",
            "pwv_ms",
            "quality",
            "segments",
            "variance",
            "vascular_age"
        ]
    );
}

/// Without the torch feature the pipeline still runs and inference refuses
/// loudly instead of inventing numbers.
#[cfg(not(feature = "torch"))]
#[test]
fn cva_without_torch_refuses_inference() {
    let Some(db) = db_path() else {
        eprintln!("SKIP cva_without_torch_refuses_inference: no events DB (set OURA_DB)");
        return;
    };
    let err = cva::estimate(&oracle_args(&db)).expect_err("no torch backend in this build");
    let msg = err.to_string();
    assert!(msg.contains("no torch backend"), "unexpected error: {msg}");
}
