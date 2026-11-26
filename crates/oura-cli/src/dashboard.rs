//! `oura dashboard` — a local web health dashboard following
//! `notes/dashboard-v2-brainstorm.md`.
//!
//! Rust owns the DB + every non-model calculation (per-night HRV/RHR/skin-temp,
//! SpO2 % via Oura's calibration, device/data-health, baselines + deltas, the
//! digest) and the HTTP server. The torch models (sleep hypnogram, activity, CVA)
//! run through the Python runners, shelled out exactly like `oura sessions`. All
//! data stays on this machine.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use oura_summary::{
    bedtime_overrides_path, feature_modes_path, profile_path, write_bedtime_override, ModelInputs,
    ModelOutputs, ModelRunner,
};
// Re-exported so `dashboard::Demographics` (main.rs) and the profile/feature
// handlers keep working after the summary logic moved into `oura-summary`.
pub use oura_summary::{
    read_profile_for, write_feature_mode, write_profile, write_profile_for, Demographics,
};

const INDEX_HTML: &str = include_str!("../../../dashboard/web/index.html");
const STYLES_CSS: &str = include_str!("../../../dashboard/web/styles.css");
const APP_JS: &str = include_str!("../../../dashboard/web/app.js");
const DNA_HTML: &str = include_str!("../../../dashboard/web/dna.html");
const DNA_JS: &str = include_str!("../../../dashboard/web/dna.js");
const DNA_CSS: &str = include_str!("../../../dashboard/web/dna.css");
const BLOOD_HTML: &str = include_str!("../../../dashboard/web/blood.html");
const BLOOD_JS: &str = include_str!("../../../dashboard/web/blood.js");
const BLOOD_CSS: &str = include_str!("../../../dashboard/web/blood.css");
const PROFILE_JS: &str = include_str!("../../../dashboard/web/profile.js");
const LOGS_HTML: &str = include_str!("../../../dashboard/web/logs.html");
const LOGS_JS: &str = include_str!("../../../dashboard/web/logs.js");
const LOGS_CSS: &str = include_str!("../../../dashboard/web/logs.css");
const RING_HTML: &str = include_str!("../../../dashboard/web/ring.html");
const RING_JS: &str = include_str!("../../../dashboard/web/ring.js");
const RING_CSS: &str = include_str!("../../../dashboard/web/ring.css");

fn read_ring_key(path: Option<&Path>) -> Result<String> {
    let path = path.ok_or_else(|| anyhow!("dashboard was started without --key-file"))?;
    let key = std::fs::read_to_string(path)
        .with_context(|| format!("reading key file {}", path.display()))?;
    validate_ring_key(&key)
}

fn write_ring_key(path: Option<&Path>, key: &str) -> Result<String> {
    let path = path.ok_or_else(|| anyhow!("dashboard was started without --key-file"))?;
    let key = validate_ring_key(key)?;
    std::fs::write(path, format!("{key}\n"))
        .with_context(|| format!("writing key file {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(key)
}

// ---------------------------------------------------------------------------
// Activity log — a ring of what the dashboard did (syncs, torch model runs,
// live-HR sessions, feature toggles), surfaced on /logs. Persisted as JSONL
// next to the DB (same sidecar convention as profile.json) and replayed into
// the in-memory ring on startup, so history survives restarts. The file is
// bounded: once it passes LOG_FILE_MAX_BYTES it is rewritten from the ring.
// ---------------------------------------------------------------------------

const ACTIVITY_MAX: usize = 400;
const LOG_FILE_MAX_BYTES: u64 = 2_000_000;

fn activity_log() -> &'static Mutex<std::collections::VecDeque<Value>> {
    static LOG: OnceLock<Mutex<std::collections::VecDeque<Value>>> = OnceLock::new();
    LOG.get_or_init(Default::default)
}

/// JSONL sidecar path; set once by `serve()` via `init_activity_log`. Tests run without one.
fn activity_log_slot() -> &'static OnceLock<Option<PathBuf>> {
    static P: OnceLock<OnceLock<Option<PathBuf>>> = OnceLock::new();
    P.get_or_init(Default::default)
}

fn activity_log_path() -> Option<&'static PathBuf> {
    activity_log_slot().get_or_init(|| None).as_ref()
}

fn activity_log_file(db: &Path) -> PathBuf {
    db.parent()
        .unwrap_or(Path::new("."))
        .join("activity_log.jsonl")
}

/// Point the log at `db`'s sidecar and replay its tail into the ring.
fn init_activity_log(db: &Path) {
    let path = activity_log_file(db);
    if let Ok(text) = std::fs::read_to_string(&path) {
        let log = activity_log();
        let mut log = log.lock().unwrap_or_else(|e| e.into_inner());
        for line in text.lines() {
            if let Ok(v) = serde_json::from_str::<Value>(line) {
                if v["ts"].is_number() && v["kind"].is_string() {
                    log.push_back(v);
                }
            }
        }
        while log.len() > ACTIVITY_MAX {
            log.pop_front();
        }
    }
    let _ = activity_log_slot().set(Some(path));
}

/// Trim to `max` chars, marking the cut — log details must never blow up the page.
fn clip(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max).collect();
    format!("{cut}… (truncated)")
}

fn log_activity(kind: &str, status: &str, detail: &str) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    tracing::info!("[activity:{kind}] {status}: {}", detail.replace('\n', " | "));
    let entry = json!({ "ts": ts, "kind": kind, "status": status, "detail": detail });
    let mut log = activity_log().lock().unwrap_or_else(|e| e.into_inner());
    log.push_back(entry.clone());
    while log.len() > ACTIVITY_MAX {
        log.pop_front();
    }
    // Persist after the in-memory insert: the ring is the source of truth and
    // the file is rebuilt from it whenever it grows past the byte cap.
    if let Some(path) = activity_log_path() {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{}", serde_json::to_string(&entry).unwrap_or_default());
            let oversized = f
                .metadata()
                .map(|m| m.len() > LOG_FILE_MAX_BYTES)
                .unwrap_or(false);
            if oversized {
                drop(f);
                if let Ok(text) = std::fs::write(
                    path,
                    log.iter()
                        .map(|e| format!("{}\n", serde_json::to_string(e).unwrap_or_default()))
                        .collect::<String>()
                        .as_bytes(),
                ) {
                    let _ = text;
                }
            }
        }
    }
}

/// Newest-first entries, optionally filtered by kind, capped at `limit`.
fn list_activity(kind: &str, limit: usize) -> Vec<Value> {
    let log = activity_log().lock().unwrap_or_else(|e| e.into_inner());
    log.iter()
        .rev()
        .filter(|e| kind.is_empty() || e["kind"].as_str() == Some(kind))
        .take(limit)
        .cloned()
        .collect()
}

fn clear_activity() {
    activity_log()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
    if let Some(path) = activity_log_path() {
        let _ = std::fs::write(path, "");
    }
}

fn validate_ring_key(key: &str) -> Result<String> {
    let key = key.trim();
    if key.len() != 32 || !key.as_bytes().iter().all(u8::is_ascii_hexdigit) {
        return Err(anyhow!("auth key must be exactly 16 bytes of hex"));
    }
    Ok(key.to_ascii_lowercase())
}

// ── python model orchestration (shared with `oura sessions` via `pyrunner`) ────
fn repo_root() -> Option<PathBuf> {
    crate::pyrunner::repo_root(Path::new("tools/run_activity_model.py"))
}

fn python_bin(root: &Path) -> PathBuf {
    crate::pyrunner::venv_python(root)
}

/// Bound model subprocesses and drain stdout concurrently (large results must not
/// fill the pipe while the parent waits). A stalled runner is killed and reaped.
fn model_output(
    mut command: Command,
    input: Option<&[u8]>,
    timeout: std::time::Duration,
    label: &str,
) -> Option<Value> {
    use std::io::{Read, Write};
    command
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().ok()?;
    let stdout = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .take(32 * 1024 * 1024)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    // Model stderr (torch warnings, tracebacks) is the only clue when a run
    // silently returns None — capture it for the /logs page instead of dropping it.
    let stderr = child.stderr.take()?;
    let err_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stderr.take(64 * 1024).read_to_end(&mut bytes);
        bytes
    });
    if let Some(input) = input {
        if child
            .stdin
            .take()
            .and_then(|mut pipe| pipe.write_all(input).ok())
            .is_none()
        {
            let _ = child.kill();
            let _ = child.wait();
            let _ = reader.join();
            log_activity("model", "failed", &format!("{label}: could not write stdin"));
            return None;
        }
    }
    let started = std::time::Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if started.elapsed() < timeout => {
                std::thread::sleep(std::time::Duration::from_millis(20))
            }
            _ => {
                tracing::warn!("model runner timed out or could not be monitored");
                let _ = child.kill();
                let _ = child.wait();
                log_activity(
                    "model",
                    "timeout",
                    &format!(
                        "{label}: killed after {}s",
                        started.elapsed().as_secs_f64().round()
                    ),
                );
                break None;
            }
        }
    };
    let bytes = reader.join().ok()?.ok()?;
    let err = String::from_utf8_lossy(&err_reader.join().unwrap_or_default()).into_owned();
    let status = status?;
    let ms = started.elapsed().as_millis();
    if !status.success() {
        log_activity(
            "model",
            "failed",
            &format!(
                "{label}: exit {} in {ms} ms\n{}",
                status.code().unwrap_or(-1),
                clip(&err, 1500)
            ),
        );
        return None;
    }
    let parsed: Option<Value> = serde_json::from_slice(&bytes).ok();
    log_activity(
        "model",
        if parsed.is_some() { "ok" } else { "failed" },
        &format!(
            "{label}: {ms} ms{}",
            if parsed.is_some() {
                String::new()
            } else {
                format!(" — output was not valid JSON\n{}", clip(&err, 1500))
            }
        ),
    );
    parsed
}

fn run_py_json(root: &Path, py: &Path, script: &str, args: &[String]) -> Option<Value> {
    run_py_json_stdin_optional(root, py, script, args, None)
}

fn run_py_json_stdin(
    root: &Path,
    py: &Path,
    script: &str,
    args: &[String],
    input: &[u8],
) -> Option<Value> {
    run_py_json_stdin_optional(root, py, script, args, Some(input))
}

/// One model run in flight, shared by every caller asking for the identical
/// invocation. Model runs take minutes; a second build for the same ring must
/// not stack up a duplicate torch process — it joins this slot and reuses the
/// result instead.
#[derive(Default)]
struct ModelSlot {
    state: Mutex<(bool, Option<Value>)>, // (done, result)
    cv: std::sync::Condvar,
}

impl ModelSlot {
    fn publish(&self, result: Option<Value>) {
        let mut s = self.state.lock().unwrap();
        s.0 = true;
        s.1 = result;
        self.cv.notify_all();
    }
}

fn model_slots() -> &'static Mutex<std::collections::HashMap<String, Arc<ModelSlot>>> {
    static S: OnceLock<Mutex<std::collections::HashMap<String, Arc<ModelSlot>>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

/// Dedup key for a model invocation: script + ring serial + exact inputs. The
/// serial is what the runners filter events and key their caches by, so two
/// rings' runs never collide; different inputs (tz, demographics, sleep
/// ranges) hash apart and run independently. `None` (no guard) when the call
/// carries no `--serial`.
fn model_key(script: &str, args: &[String], input: Option<&[u8]>) -> Option<(String, String)> {
    let serial = args.windows(2).find(|w| w[0] == "--serial")?.get(1)?.clone();
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    script.hash(&mut h);
    args.hash(&mut h);
    input.hash(&mut h);
    Some((format!("{script}|{serial}|{:x}", h.finish()), serial))
}

/// Removes the slot from the registry and wakes any joiners even if the run
/// panics, so a later call can claim a fresh slot.
struct SlotGuard {
    key: String,
    slot: Arc<ModelSlot>,
    published: bool,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        if !self.published {
            self.slot.publish(None);
        }
        model_slots().lock().unwrap().remove(&self.key);
    }
}

/// Runs `run` under the in-flight guard for `key`: the first caller executes it
/// and publishes the result; concurrent callers with the same key wait on the
/// slot and reuse it instead of starting a duplicate model process. Bounded —
/// if the owner somehow never finishes, joiners degrade like a failed model
/// would. Extracted from `run_py_json_stdin_optional` so it can be tested
/// without spawning Python.
fn dedup_model_run<F>(script: &str, serial: &str, key: String, run: F) -> Option<Value>
where
    F: FnOnce() -> Option<Value>,
{
    let (slot, owner) = {
        let mut slots = model_slots().lock().unwrap();
        match slots.get(&key).cloned() {
            Some(existing) => (existing, false),
            None => {
                let slot = Arc::new(ModelSlot::default());
                slots.insert(key.clone(), slot.clone());
                (slot, true)
            }
        }
    };
    if !owner {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
        let mut state = slot.state.lock().unwrap();
        while !state.0 {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            let (g, _) = slot.cv.wait_timeout(state, remaining).unwrap();
            state = g;
        }
        let done = state.0;
        let result = state.1.clone();
        drop(state);
        log_activity(
            "model",
            if done { "joined" } else { "join-timeout" },
            &format!(
                "{script}: {} (ring {serial})",
                if done { "reused in-flight run" } else { "in-flight run never finished" }
            ),
        );
        return result;
    }
    let mut guard = SlotGuard { key, slot, published: false };
    let result = run();
    guard.slot.publish(result.clone());
    guard.published = true;
    result
}

fn run_py_json_stdin_optional(
    root: &Path,
    py: &Path,
    script: &str,
    args: &[String],
    input: Option<&[u8]>,
) -> Option<Value> {
    let run = || {
        let mut command = Command::new(py);
        command.current_dir(root).arg(root.join(script)).args(args);
        model_output(command, input, std::time::Duration::from_secs(180), script)
    };
    match model_key(script, args, input) {
        Some((key, serial)) => dedup_model_run(script, &serial, key, run),
        None => run(),
    }
}

/// The argv each model runner is invoked with — shared by [`PythonRunner`] and
/// [`RustRunner`] so the two backends can never drift apart in what they ask
/// for (the in-process entry points take the exact same argv as the scripts).
struct ModelArgv {
    sleep: Vec<String>,
    cva: Vec<String>,
    activity: Vec<String>,
    illness: Vec<String>,
    stress: Vec<String>,
}

fn model_argv(
    db: &Path,
    tz: f64,
    demo: &Demographics,
    serial: &str,
) -> ModelArgv {
    let sleep = vec![
        db.display().to_string(),
        tz.to_string(),
        "--json".into(),
        "--batch".into(),
        "--serial".into(),
        serial.to_string(),
    ];
    let cva = vec![
        db.display().to_string(),
        "--json".into(),
        "--sex".into(),
        demo.sex.to_string(),
        "--age".into(),
        demo.age.to_string(),
        "--height".into(),
        demo.height_m.to_string(),
        "--weight".into(),
        demo.weight_kg.to_string(),
        "--ring".into(),
        demo.ring_size.to_string(),
        "--serial".into(),
        serial.to_string(),
    ];
    let activity = vec![
        db.display().to_string(),
        "--tz".into(),
        tz.to_string(),
        "--json".into(),
        "--serial".into(),
        serial.to_string(),
    ];
    let illness = vec![db.display().to_string(), tz.to_string(), "--json".into(), "--serial".into(), serial.to_string()];
    let stress = vec![db.display().to_string(), "--tz".into(), tz.to_string(), "--json".into(), "--serial".into(), serial.to_string()];
    ModelArgv { sleep, cva, activity, illness, stress }
}

/// The signal-estimator top-up both runners share: a missing or partial sleep
/// batch is filled from the Rust per-night estimator (see
/// [`sleep_batch_needs_fallback`]).
fn finish_sleep(
    sleep_batch: Option<Value>,
    db: &Path,
    sleep_ranges: &[[i64; 3]],
    serial: &str,
) -> Option<Value> {
    let mut sleep_batch = sleep_batch;
    if sleep_batch_needs_fallback(sleep_batch.as_ref(), sleep_ranges.len())
        && !sleep_ranges.is_empty()
    {
        let signals = oura_summary::stage_nights_from_signals(db, sleep_ranges, Some(serial));
        sleep_batch = merge_sleep_fallback(sleep_batch, signals);
    }
    sleep_batch
}

/// The web dashboard's [`ModelRunner`]: shells out to the Python torch runners,
/// exactly as before, and falls back to Rust signal-based sleep staging when the
/// Python environment or private `.pt` model file is unavailable on a user's PC.
struct PythonRunner;
/// The event rows the sleep plumbing needs, in `run_sleep_model.load_rows`'s
/// exact order and filter — the cache fingerprint is only reproducible when the
/// row set matches Python's. Also returns the raw (ds, tag, json, cu) tuples
/// `build_epochs` consumes for the clock. `None` = DB unreadable.
fn load_nssa_rows(
    db: &Path,
    serial: &str,
) -> Option<(Vec<nssa_plumbing::rows::EventRow>, Vec<(i64, u8, Option<String>, i64)>)> {
    use nssa_plumbing::rows::EventRow;
    let conn = rusqlite::Connection::open_with_flags(
        db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()?;
    let mut stmt = conn
        .prepare(
            "SELECT ring_timestamp, tag, COALESCE(decoded_json, '{}'), captured_unix, body \
             FROM events \
             WHERE (decoded_json IS NOT NULL OR (tag = 65 AND LENGTH(body) >= 14)) \
               AND serial = ?1 ORDER BY captured_unix, id",
        )
        .ok()?;
    let mut rows = Vec::new();
    let mut clock_rows = Vec::new();
    let mut q = stmt
        .query_map(rusqlite::params![serial], |r| {
            let ds: i64 = r.get(0)?;
            let tag: i64 = r.get(1)?;
            let js: String = r.get(2)?;
            let cu: i64 = r.get(3)?;
            let body: Option<Vec<u8>> = r.get(4)?;
            Ok((ds, tag as u8, js, cu, body))
        })
        .ok()?;
    while let Some((ds, tag, js, cu, body)) = q.next().and_then(|v| v.ok()) {
        clock_rows.push((ds, tag, Some(js.clone()), cu));
        let mut row = EventRow {
            ds,
            tag,
            cu,
            ..Default::default()
        };
        let payload: Value = serde_json::from_str(&js).unwrap_or_default();
        match tag {
            nssa_plumbing::rows::TAG_IBI => {
                let ibi: Vec<i64> = payload
                    .get("ibi_ms")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
                    .unwrap_or_default();
                if ibi.len() == 6 {
                    row.ibi_ms = Some([ibi[0], ibi[1], ibi[2], ibi[3], ibi[4], ibi[5]]);
                }
                row.amplitude = payload
                    .get("amplitude")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
                    .or(Some(Vec::new()));
            }
            nssa_plumbing::rows::TAG_BEDTIME => {
                row.bedtime_start_ds = payload.get("bedtime_start_ds").and_then(|v| v.as_i64());
                row.bedtime_end_ds = payload.get("bedtime_end_ds").and_then(|v| v.as_i64());
            }
            nssa_plumbing::rows::TAG_ACM | nssa_plumbing::rows::TAG_TEMP => row.body = body,
            _ => {}
        }
        rows.push(row);
    }
    Some((rows, clock_rows))
}

/// T2c: answer sleep windows from `nssa_cache.json` in-process; only windows the
/// cache cannot answer are delegated to the Python model (which fills the cache
/// for the next recompute). The cold torch spawn — seconds per rebuild — is gone
/// from warm recomputes. `None` = the batch failed outright, exactly as
/// `run_py_json_stdin` would report it.
fn sleep_batch_cached(
    root: &Path,
    py: &Path,
    db: &Path,
    tz: f64,
    demo: &Demographics,
    sleep_ranges: &[[i64; 3]],
    serial: &str,
    argv: &ModelArgv,
) -> Option<Value> {
    use nssa_plumbing::cache::{self, Served};
    use nssa_plumbing::clock::{build_epochs, Clock};
    use nssa_plumbing::grid::demographics;
    let run_python = |ranges: &[&[i64; 3]]| -> Option<Value> {
        let stdin = serde_json::to_vec(&ranges.iter().map(|r| *r).collect::<Vec<_>>()).ok()?;
        run_py_json_stdin(root, py, "tools/run_sleep_model.py", &argv.sleep, &stdin)
    };
    if sleep_ranges.is_empty() {
        return Some(Value::Array(Vec::new()));
    }
    let (rows, clock_rows) = match load_nssa_rows(db, serial) {
        Some(v) => v,
        None => return run_python(&sleep_ranges.iter().collect::<Vec<_>>()),
    };
    let ncache = match cache::Cache::load(db, Some(serial)) {
        Some(c) => c,
        None => return run_python(&sleep_ranges.iter().collect::<Vec<_>>()),
    };
    let clock_refs: Vec<(i64, u8, Option<&str>, i64)> = clock_rows
        .iter()
        .map(|(ds, tag, js, cu)| (*ds, *tag, js.as_deref(), *cu))
        .collect();
    let clock = Clock::new(build_epochs(&clock_refs));
    let demo3 = demographics(demo.age, demo.height_m, demo.weight_kg, &demo.sex.to_string());
    let mut slots: Vec<Value> = Vec::with_capacity(sleep_ranges.len());
    let mut missed: Vec<(usize, [i64; 3])> = Vec::new();
    for &[s, e, cu] in sleep_ranges {
        match cache::serve_window(&clock, &rows, &ncache, s, e, Some(cu), tz, &demo3) {
            Served::Ok(v) => slots.push(v),
            // Python's batch prints null for a window error too — same slot.
            Served::Err(_) => slots.push(Value::Null),
            Served::Missed => {
                slots.push(Value::Null);
                missed.push((slots.len() - 1, [s, e, cu]));
            }
        }
    }
    if !missed.is_empty() {
        let refs: Vec<&[i64; 3]> = missed.iter().map(|(_, r)| r).collect();
        let out = run_python(&refs)?;
        let arr = out.as_array()?.clone();
        if arr.len() != missed.len() {
            return None;
        }
        for ((idx, _), v) in missed.iter().zip(arr) {
            slots[*idx] = v;
        }
    }
    Some(Value::Array(slots))
}

impl ModelRunner for PythonRunner {
    fn run(&self, input: ModelInputs) -> ModelOutputs {
        let ModelInputs {
            db,
            tz,
            demo,
            sleep_ranges,
            serial,
        } = input;
        let root = repo_root();
        let py = root.as_deref().map(python_bin);
        let argv = model_argv(db, tz, demo, serial);

        let (sleep_batch, cva, activity, illness, stress) = match (root.as_deref(), py.as_deref()) {
            (Some(r), Some(p)) => std::thread::scope(|s| {
                let sh = s.spawn(|| {
                    // T2c: serve from nssa_cache.json in-process; Python only
                    // runs for windows the cache cannot answer.
                    sleep_batch_cached(r, p, db, tz, demo, sleep_ranges, serial, &argv)
                });
                let ch = s.spawn(|| run_py_json(r, p, "tools/run_cva_model.py", &argv.cva));
                let ah = s.spawn(|| run_py_json(r, p, "tools/run_activity_model.py", &argv.activity));
                let ih = s.spawn(|| run_py_json(r, p, "tools/run_illness_model.py", &argv.illness));
                let sh2 = s.spawn(|| {
                    run_py_json(r, p, "tools/run_stress_events.py", &argv.stress)
                });
                (
                    sh.join().ok().flatten(),
                    ch.join().ok().flatten(),
                    ah.join().ok().flatten(),
                    ih.join().ok().flatten(),
                    sh2.join().ok().flatten(),
                )
            }),
            _ => (None, None, None, None, None),
        };
        let sleep_batch = finish_sleep(sleep_batch, db, sleep_ranges, serial);
        ModelOutputs {
            sleep_batch,
            cva,
            activity,
            illness,
            stress,
        }
    }
}

/// The web dashboard's in-process [`ModelRunner`] (built with `--features
/// torch`): cva / activity / illness / stress run through the `oura-models`
/// crate — same argv contract as the Python scripts, golden-tested against them
/// — with no interpreter start-up and no process boundary. Sleep staging stays
/// on Python (its model is not ported). Any model that fails in Rust falls back
/// to that model's Python script, so a broken Rust path can never take a panel
/// down; `OURA_DASHBOARD_PYTHON=1` forces the whole runner back to Python even
/// in a torch build (A/B benching).
#[cfg(feature = "torch")]
struct RustRunner;

#[cfg(feature = "torch")]
impl ModelRunner for RustRunner {
    fn run(&self, input: ModelInputs) -> ModelOutputs {
        let ModelInputs {
            db,
            tz,
            demo,
            sleep_ranges,
            serial,
        } = input;
        let root = repo_root();
        let py = root.as_deref().map(python_bin);
        let argv = model_argv(db, tz, demo, serial);
        // oura-models resolves its .pt files from its own manifest, but
        // activity's repo_root() reads OURA_REPO (then cwd) — pin it to the
        // checkout the dashboard is running from.
        if let (Some(r), None) = (root.as_deref(), std::env::var_os("OURA_REPO")) {
            std::env::set_var("OURA_REPO", r);
        }

        let (sleep_batch, cva, activity, illness, stress) = match (root.as_deref(), py.as_deref()) {
            (Some(r), Some(p)) => std::thread::scope(|s| {
                let sh = s.spawn(|| {
                    // T2c: serve from nssa_cache.json in-process; Python only
                    // runs for windows the cache cannot answer.
                    sleep_batch_cached(r, p, db, tz, demo, sleep_ranges, serial, &argv)
                });
                let ch = s.spawn(|| {
                    run_rust_model("cva", "tools/run_cva_model.py", r, p, &argv.cva, || {
                        oura_models::cva::json_for_args(&argv.cva)
                    })
                });
                let ah = s.spawn(|| {
                    run_rust_model("activity", "tools/run_activity_model.py", r, p, &argv.activity, || {
                        // Incremental dispatch: replays cached days, recomputes
                        // only what new rows dirtied; falls back to the cold
                        // path internally when no valid cache exists.
                        oura_models::incremental::json_for_args(&argv.activity)
                    })
                });
                let ih = s.spawn(|| {
                    run_rust_model("illness", "tools/run_illness_model.py", r, p, &argv.illness, || {
                        oura_models::illness::json_for_args(&argv.illness)
                    })
                });
                let sh2 = s.spawn(|| {
                    run_rust_model("stress", "tools/run_stress_events.py", r, p, &argv.stress, || {
                        oura_models::stress::json_for_args(&argv.stress)
                    })
                });
                (
                    sh.join().ok().flatten(),
                    ch.join().ok().flatten(),
                    ah.join().ok().flatten(),
                    ih.join().ok().flatten(),
                    sh2.join().ok().flatten(),
                )
            }),
            _ => (None, None, None, None, None),
        };
        let sleep_batch = finish_sleep(sleep_batch, db, sleep_ranges, serial);
        ModelOutputs {
            sleep_batch,
            cva,
            activity,
            illness,
            stress,
        }
    }
}

/// One model through the in-process path, with its Python script as fallback.
/// Deduped like the Python path — the slot key hashes the backend tag too, so a
/// Rust run and a Python run of the same model never share a slot.
#[cfg(feature = "torch")]
fn run_rust_model(
    label: &str,
    py_script: &str,
    root: &Path,
    py: &Path,
    args: &[String],
    rust: impl FnOnce() -> Result<String>,
) -> Option<Value> {
    let run_py = || run_py_json(root, py, py_script, args);
    let Some((key, serial)) = model_key(&format!("rust/{label}"), args, None) else {
        return run_py();
    };
    dedup_model_run(&format!("rust/{label}"), &serial, key, || {
        let started = std::time::Instant::now();
        // A panic inside a model escapes thread::scope and takes down the whole
        // server (seen once with concurrent torch init). Treat it like an Err:
        // log it and fall back to the Python runner.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(rust))
            .map_err(|_| anyhow!("{label} (rust) panicked"))
            .and_then(|inner| inner)
            .and_then(|json| {
                serde_json::from_str::<Value>(&json)
                    .map_err(|e| anyhow!("{label}: output was not valid JSON — {e}"))
            });
        let ms = started.elapsed().as_millis();
        match outcome {
            Ok(v) => {
                log_activity("model", "ok", &format!("{label} (rust): {ms} ms"));
                Some(v)
            }
            Err(e) => {
                log_activity(
                    "model",
                    "failed",
                    &format!("{label} (rust): {ms} ms — {e}; falling back to {py_script}"),
                );
                run_py()
            }
        }
    })
}

/// Whether the Python sleep batch must be topped up from the signal estimator.
///
/// `run_sleep_model.py --batch` returns one entry per sleep range, in order: a staged
/// dict, or `null` for a window it couldn't stage (no torch/models, or NSSA inputs
/// missing), still exiting 0. A top-up is needed when the batch is missing, the wrong
/// length (the per-range contract was broken), or carries any null night. Kept beside
/// [`merge_sleep_fallback`] so the "when to fall back" and the "what to replace" stay
/// in agreement — this predicate is the line whose all-or-nothing form was the bug.
fn sleep_batch_needs_fallback(batch: Option<&Value>, range_count: usize) -> bool {
    match batch.and_then(|v| v.as_array()) {
        None => true,
        Some(a) => a.len() != range_count || a.iter().any(Value::is_null),
    }
}

/// Fill the Python sleep batch's missing per-night entries with the signal-derived
/// fallback. `signals` is authoritative on night count and order (exactly one entry
/// per sleep range), so the merged result has one entry per range: the Python staging
/// where present, otherwise the signal estimate.
///
/// A Python `null` (window the model couldn't stage) is replaced by its signal
/// estimate — otherwise the night keeps its physiology but shows no sleep at all.
/// Nights Python staged are preserved. A Python entry with no signal counterpart
/// (length mismatch — not expected from the current producer) is dropped, since
/// without a range it cannot be mapped onto a night anyway; a missing Python entry is
/// simply the signal estimate. If signals are unavailable, the Python batch is
/// returned untouched.
fn merge_sleep_fallback(python: Option<Value>, signals: Option<Value>) -> Option<Value> {
    let Some(Value::Array(sig)) = signals else {
        return python;
    };
    let py = match python {
        Some(Value::Array(py)) => py,
        _ => Vec::new(),
    };
    let merged = sig
        .into_iter()
        .enumerate()
        .map(|(i, s)| match py.get(i) {
            Some(p) if !p.is_null() => p.clone(),
            _ => s,
        })
        .collect();
    Some(Value::Array(merged))
}

/// `build_summary` for the web dashboard. Built with `--features torch` it runs
/// cva/activity/illness/stress in-process (RustRunner, per-model Python
/// fallback); otherwise it shells out to Python as before.
fn build_summary(db: &Path, tz: f64, serial: Option<&str>) -> Result<Value> {
    #[cfg(feature = "torch")]
    let runner: &dyn ModelRunner = if std::env::var_os("OURA_DASHBOARD_PYTHON").is_some() {
        &PythonRunner
    } else {
        &RustRunner
    };
    #[cfg(not(feature = "torch"))]
    let runner: &dyn ModelRunner = &PythonRunner;
    oura_summary::build_summary(db, tz, serial, runner)
}

// ── summary cache ─────────────────────────────────────────────────────────────
// build_summary spawns torch subprocesses (~seconds); without a cache every page
// load re-pays that. We memoise the last result and reuse it until the inputs
// change — the DB (a sync appends events) or profile.json (an edit changes the CVA
// inputs / weight-based kcal). Both are cheap mtime stats, so a sync or profile
// edit transparently invalidates the cache with no explicit wiring.
struct SummaryCache {
    db: PathBuf,
    tz: f64,
    serial: String, // the summary is per-ring; each ring caches separately
    token: CacheToken, // (db change-counter, profile.json, feature_modes.json, bedtime_overrides.json) mtimes
    value: Arc<Value>,
}

type CacheToken = (
    i64,
    Option<SystemTime>,
    Option<SystemTime>,
    Option<SystemTime>,
    i64,
    u32,
);

/// Bump when `build_summary`'s output shape changes — new fields, new semantics,
/// removed keys. The DB and sidecar inputs can be byte-identical across a binary
/// update, so without this a rebuilt binary keeps serving the previous run's
/// sidecar as "fresh" and the new fields never appear (breathing/perf added
/// under v2).
const SUMMARY_SHAPE_VERSION: u32 = 2;

/// SQLite change signal for the DB, for derived-data cache keys.
///
/// `PRAGMA data_version` alone is NOT sufficient in WAL mode: a fresh read-only
/// connection reads the change counter from the DB file header, which only
/// advances when the WAL is checkpointed — so a sync's commits can stay
/// invisible to the probe for an arbitrarily long time (observed: HR cache
/// served stale minutes after a sync). MAX(rowid) of the tables a sync writes
/// is visible to every WAL reader the instant the commit lands; combine both
/// (data_version still catches updates/deletes that leave rowids alone).
/// The file mtime is NOT a safe proxy either — any read-write open checkpoints
/// the WAL and bumps mtime without changing data, which made the cache thrash
/// (and re-ran every torch model) after each sync, live-HR session, or even the
/// summary build's own migration probe.
fn db_data_version(db: &Path) -> i64 {
    rusqlite::Connection::open_with_flags(
        db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .and_then(|c| {
        let dv: i64 = c.query_row("PRAGMA data_version", [], |r| r.get(0))?;
        let (ev, rd, dev): (Option<i64>, Option<i64>, Option<i64>) = c.query_row(
            "SELECT (SELECT MAX(rowid) FROM events), \
                    (SELECT MAX(rowid) FROM readings), \
                    (SELECT MAX(rowid) FROM device)",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        Ok(dv
            ^ ev.unwrap_or(0).wrapping_mul(0x9e37_79b9_7f4a_7c15u64.cast_signed())
            ^ rd.unwrap_or(0).wrapping_mul(0x632b_e658_f119_47a3)
            ^ dev.unwrap_or(0).wrapping_mul(0x8950_4f4d_59ce_4fdbu64.cast_signed()))
    })
    .unwrap_or(0)
}

/// Change-counter of the DB plus mtimes of every sidecar input the summary
/// depends on plus the store decoder version — any change rebuilds it. Covers a
/// sync (oura.db), a profile edit (profile.json), a feature toggle
/// (feature_modes.json), a manual bedtime adjustment (bedtime_overrides.json),
/// and historical decode migrations.
fn summary_token(db: &Path) -> CacheToken {
    (
        db_data_version(db),
        mtime(&profile_path(db)),
        mtime(&feature_modes_path(db)),
        mtime(&bedtime_overrides_path(db)),
        oura_store::storage::DECODER_VERSION,
        SUMMARY_SHAPE_VERSION,
    )
}

fn summary_cache() -> &'static Mutex<Vec<SummaryCache>> {
    static C: OnceLock<Mutex<Vec<SummaryCache>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(Vec::new()))
}

fn mtime(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// Sidecar holding the last built summary, next to the DB (same convention as
/// profile.json). Lets a restarted server serve the previous run's summary on
/// first paint instead of paying for the torch models before anything renders;
/// the embedded token decides whether it is fresh or merely stale-but-shaped,
/// exactly like the in-memory entry it seeds.
fn summary_cache_path(db: &Path, serial: &str) -> PathBuf {
    db.parent()
        .unwrap_or(Path::new("."))
        .join(format!("summary_cache.{serial}.json"))
}

fn token_to_json(t: &CacheToken) -> Value {
    let mt = |s: &Option<SystemTime>| {
        s.map(|s| {
            let d = s.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
            vec![d.as_secs() as i64, d.subsec_nanos() as i64]
        })
    };
    json!([t.0, mt(&t.1), mt(&t.2), mt(&t.3), t.4, t.5])
}

fn token_from_json(v: &Value) -> Option<CacheToken> {
    let a = v.as_array()?;
    if a.len() != 6 {
        return None;
    }
    let mt = |v: &Value| -> Option<Option<SystemTime>> {
        if v.is_null() {
            return Some(None);
        }
        let p = v.as_array()?;
        Some(Some(
            SystemTime::UNIX_EPOCH
                + std::time::Duration::new(p[0].as_u64()?, p[1].as_u64()? as u32),
        ))
    };
    Some((
        a[0].as_i64()?,
        mt(&a[1])?,
        mt(&a[2])?,
        mt(&a[3])?,
        a[4].as_i64()?,
        a[5].as_u64()? as u32,
    ))
}

fn load_summary_disk(db: &Path, tz: f64, serial: &str) -> Option<(CacheToken, Arc<Value>)> {
    let env: Value =
        serde_json::from_slice(&std::fs::read(summary_cache_path(db, serial)).ok()?).ok()?;
    // the token is about the data, but the payload itself is rendered for a tz;
    // a summary built under a different offset is not reusable
    if env["tz"].as_f64()? != tz {
        return None;
    }
    Some((token_from_json(&env["token"])?, Arc::new(env["value"].clone())))
}

/// Resolve `?serial=` against the registry: a unique prefix is accepted, and with
/// exactly one registered ring an omitted serial resolves to it (single-ring DBs
/// keep working unchanged).
fn resolve_summary_serial(db: &Path, want: Option<&str>) -> Result<String> {
    let store = oura_store::storage::Store::open_read_only(db).context("opening DB")?;
    let rings = store.list_rings().unwrap_or_default();
    let Some(want) = want else {
        return oura_summary::resolve_summary_serial(&store, None);
    };
    let up = want.to_ascii_uppercase();
    let matches: Vec<&String> = rings
        .iter()
        .filter(|r| r.0.to_ascii_uppercase() == up || r.0.to_ascii_uppercase().starts_with(&up))
        .map(|r| &r.0)
        .collect();
    let exact = rings.iter().find(|r| r.0.to_ascii_uppercase() == up).map(|r| &r.0);
    match exact.or(matches.first().map(|v| &**v)) {
        Some(s) if exact.is_some() || matches.len() == 1 => Ok((*s).clone()),
        _ if matches.is_empty() => Err(anyhow!("no registered ring matches {want}")),
        _ => Err(anyhow!("{want} is ambiguous ({} rings match)", matches.len())),
    }
}

/// Cached `build_summary`: recompute only when oura.db, profile.json, or
/// feature_modes.json changes. Keyed per ring, so switching rings in the UI
/// doesn't evict the other ring's build (a handful of rings at most).
/// Keys with a background rebuild already in flight, so a polling client can't
/// stack up duplicate model runs. Key is "db-path|serial" (tz is a float and
/// can't hash; in practice one server serves one tz).
fn recompute_in_flight() -> &'static Mutex<std::collections::HashSet<String>> {
    static S: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

/// Dbs with a BLE sync currently writing. Events land continuously during a
/// sync, so a rebuild spawned mid-sync is invalidated by its own token recheck
/// at completion (observed: a full model batch thrown away per sync).
/// `cached_summary` defers spawning while a sync is in flight; `warm_summaries`
/// triggers the one rebuild that matters at sync end, against final data.
fn sync_writing() -> &'static Mutex<std::collections::HashSet<PathBuf>> {
    static S: OnceLock<Mutex<std::collections::HashSet<PathBuf>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

fn sync_in_flight(db: &Path) -> bool {
    sync_writing()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .contains(db)
}

fn store_summary(
    db: &Path,
    tz: f64,
    serial: &str,
    token: CacheToken,
    value: &Arc<Value>,
) {
    let newer_exists = {
        let mut guard = summary_cache().lock().unwrap();
        // don't clobber an entry a concurrent build already stored for this exact
        // state. token.0 is an XOR hash over the change counters — its magnitude
        // carries no ordering (a sync can make it *drop*, which would make any
        // older entry look "newer" and block storing forever). Callers only store
        // under the current token, so an entry equal to this token is up to date.
        let newer = guard
            .iter()
            .any(|c| c.db == db && c.tz == tz && c.serial == serial && c.token == token);
        if !newer {
            guard.retain(|c| !(c.db == db && c.tz == tz && c.serial == serial));
            guard.push(SummaryCache {
                db: db.to_path_buf(),
                tz,
                serial: serial.to_string(),
                token: token.clone(),
                value: value.clone(),
            });
        }
        newer
    };
    if newer_exists {
        return;
    }
    // Persist for the next process start (tmp+rename so a crash mid-write can't
    // leave a half-written sidecar). ~1 MB per ring, once per rebuild.
    let env = json!({ "tz": tz, "token": token_to_json(&token), "value": &**value });
    let path = summary_cache_path(db, serial);
    let tmp = path.with_extension("json.tmp");
    if let Ok(bytes) = serde_json::to_vec(&env) {
        if std::fs::write(&tmp, &bytes).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}

/// Cached `build_summary`. Returns `(value, recomputing)`: on a cache miss with a
/// previous value on hand, the stale summary is served immediately and the rebuild
/// happens on a background thread — the dashboard shows cached data while the
/// torch models run instead of blocking the page for ~a minute. Only the very
/// first request (nothing cached yet) builds synchronously.
fn cached_summary(db: &Path, tz: f64, serial: Option<&str>) -> Result<(Arc<Value>, bool)> {
    let serial = resolve_summary_serial(db, serial)?;
    let token = summary_token(db);
    let key = format!("{}|{}", db.display(), serial);
    {
        let mut guard = summary_cache().lock().unwrap();
        // Process restart: seed from the previous run's sidecar before deciding.
        // A token match serves it as fresh below; a mismatch falls into the
        // stale path, which serves it immediately and rebuilds in the background
        // — the same stale-while-revalidate the in-memory cache already does.
        if !guard.iter().any(|c| c.db == db && c.tz == tz && c.serial == serial) {
            if let Some((disk_token, value)) = load_summary_disk(db, tz, &serial) {
                guard.push(SummaryCache {
                    db: db.to_path_buf(),
                    tz,
                    serial: serial.clone(),
                    token: disk_token,
                    value,
                });
            }
        }
        if let Some(c) = guard
            .iter()
            .find(|c| c.db == db && c.tz == tz && c.serial == serial && c.token == token)
        {
            return Ok((c.value.clone(), false));
        }
        // token mismatch but a previous build exists: serve it, rebuild in background
        if let Some(stale) = guard
            .iter()
            .find(|c| c.db == db && c.tz == tz && c.serial == serial)
        {
            let value = stale.value.clone();
            drop(guard);
            let spawned = {
                let mut inflight = recompute_in_flight().lock().unwrap();
                if sync_in_flight(db) {
                    // A sync is still draining events into the db: a rebuild now
                    // would race the stream and its token recheck would discard
                    // the result. Stay stale+recomputing; warm_summaries fires
                    // the rebuild against final data when the sync ends.
                    false
                } else {
                    inflight.insert(key.clone()) // false if a rebuild is already running
                }
            };
            if spawned {
                // RAII: clear the in-flight flag even if the build panics, so a
                // later request can retry instead of serving stale forever.
                struct InFlightGuard(String);
                impl Drop for InFlightGuard {
                    fn drop(&mut self) {
                        recompute_in_flight().lock().unwrap().remove(&self.0);
                    }
                }
                let (db2, tz2, serial2) = (db.to_path_buf(), tz, serial.clone());
                std::thread::spawn(move || {
                    let _guard = InFlightGuard(key);
                    let started = std::time::Instant::now();
                    // If inputs changed mid-build (a sync landed, a profile edit),
                    // rebuild against the new state immediately instead of waiting
                    // for the next client request to notice the mismatch — that
                    // wait is what left the dashboard on stale data for minutes.
                    // Bounded so a pathological writer can't spin the models
                    // forever. Never retry over a still-running sync: the
                    // post-sync warm_summaries kick is the one against final data.
                    let mut attempt = 0;
                    loop {
                        attempt += 1;
                        let token_now = summary_token(&db2);
                        match build_summary(&db2, tz2, Some(&serial2)) {
                            Ok(v) => {
                                let value = Arc::new(v);
                                if summary_token(&db2) == token_now {
                                    store_summary(&db2, tz2, &serial2, token_now, &value);
                                    log_activity(
                                        "summary",
                                        "ok",
                                        &format!(
                                            "background recompute finished in {} ms",
                                            started.elapsed().as_millis()
                                        ),
                                    );
                                    break;
                                }
                                if sync_in_flight(&db2) {
                                    log_activity(
                                        "summary",
                                        "retry",
                                        "inputs still changing (sync in progress); \
                                         superseding this build, post-sync rebuild follows",
                                    );
                                    break;
                                }
                                if attempt >= 3 {
                                    // Out of retries: cache this build anyway (it is
                                    // still newer than what the client has); the token
                                    // mismatch makes the next request rebuild again.
                                    store_summary(&db2, tz2, &serial2, token_now, &value);
                                    log_activity(
                                        "summary",
                                        "ok",
                                        &format!(
                                            "background recompute finished in {} ms \
                                             (retry cap, inputs still moving)",
                                            started.elapsed().as_millis()
                                        ),
                                    );
                                    break;
                                }
                                log_activity(
                                    "summary",
                                    "retry",
                                    "inputs changed mid-build; rebuilding against new state",
                                );
                            }
                            Err(e) => {
                                log_activity(
                                    "summary",
                                    "error",
                                    &format!("background recompute failed: {e}"),
                                );
                                break;
                            }
                        }
                    }
                });
            }
            return Ok((value, true));
        }
    }
    // cold start: no previous value to fall back to, build synchronously
    let value = Arc::new(build_summary(db, tz, Some(&serial))?);
    // re-stat the inputs: if any changed while we were building (e.g. a sync landed),
    // the result may predate that change — don't cache it, so the next request
    // rebuilds against the new state instead of serving stale data.
    let token_after = summary_token(db);
    if token_after == token {
        store_summary(db, tz, &serial, token, &value);
    }
    Ok((value, false))
}

// ── hourly-HR cache ───────────────────────────────────────────────────────────
// hr_bins decodes every HR event in the DB (~2.4 s over a full history) and the
// day view refetches the whole series on every day switch. Memoise per
// (db, serial, tz, bin size, days, window) and invalidate on the same DB
// data_version + decoder version the summary cache uses, so a sync refreshes it
// transparently. The windowed day path (from/to set) is an index seek and still
// memoises — switching back to a day you just looked at should be instant.
struct HrCache {
    db: PathBuf,
    serial: String, // "" = all rings (no ?serial=)
    tz: f64,
    minutes: u32,
    days: u32,
    from: i64, // resolved-unix window; i64::MIN = unbounded (full path)
    to: i64,
    version: (i64, i64),
    body: Arc<Vec<u8>>, // serialized once; the warm path writes these bytes verbatim
}

fn hr_cache() -> &'static Mutex<Vec<HrCache>> {
    static C: OnceLock<Mutex<Vec<HrCache>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(Vec::new()))
}

fn cached_hr_bins(
    db: &Path,
    tz: f64,
    days: u32,
    minutes: u32,
    serial: Option<&str>,
) -> Result<Arc<Vec<u8>>> {
    cached_hr_window(db, tz, days, minutes, serial, i64::MIN, i64::MAX)
}

fn cached_hr_window(
    db: &Path,
    tz: f64,
    days: u32,
    minutes: u32,
    serial: Option<&str>,
    from: i64,
    to: i64,
) -> Result<Arc<Vec<u8>>> {
    let version = (db_data_version(db), oura_store::storage::DECODER_VERSION);
    let key_serial = serial.unwrap_or_default();
    let same = |c: &HrCache| {
        c.db == db
            && c.serial == key_serial
            && c.tz == tz
            && c.minutes == minutes
            && c.days == days
            && c.from == from
            && c.to == to
    };
    {
        let guard = hr_cache().lock().unwrap();
        if let Some(c) = guard.iter().find(|c| same(c) && c.version == version) {
            return Ok(c.body.clone());
        }
    }
    let value = if from == i64::MIN {
        oura_summary::hourly_hr::hr_bins(db, tz, days, minutes, serial)?
    } else {
        oura_summary::hourly_hr::hr_bins_window(db, tz, from, to, minutes, serial)?
    };
    let body = Arc::new(serde_json::to_vec(&value)?);
    let mut guard = hr_cache().lock().unwrap();
    // don't clobber a fresher entry a concurrent request already stored
    if !guard.iter().any(|c| same(c) && c.version > version) {
        guard.retain(|c| !same(c));
        guard.push(HrCache {
            db: db.to_path_buf(),
            serial: key_serial.to_string(),
            tz,
            minutes,
            days,
            from,
            to,
            version,
            body: body.clone(),
        });
    }
    Ok(body)
}

pub async fn serve(
    port: u16,
    lan: bool,
    db: PathBuf,
    tz: f64,
    name: String,
    key_file: Option<PathBuf>,
    seed: Demographics,
) -> Result<()> {
    // seed profile.json from the CLI demographics on first run; afterwards the
    // file is the editable source of truth.
    if !profile_path(&db).exists() {
        let _ = write_profile(&db, &seed.to_json());
    }
    // replay the persisted activity log (activity_log.jsonl next to the DB)
    init_activity_log(&db);
    // Profiles (docs/profiles.md): legacy blood rows (serial='') belong to the
    // primary ring — reassign once at startup so the first profile shows them.
    // Then move imported PDFs into per-profile document dirs
    // (<blood_dir>/<serial>/) and create those dirs for every ring, so it is
    // obvious where new lab PDFs go.
    {
        let db = db.clone();
        let res = tokio::task::spawn_blocking(move || {
            let rings = oura_store::storage::Store::open_read_only(&db)
                .and_then(|s| s.list_rings())
                .ok()?;
            let primary = rings.iter().min_by_key(|r| r.3).map(|r| r.0.clone())?;
            let backfill = crate::blood::backfill_serial(&primary);
            let migrated = crate::blood::migrate_profile_dirs();
            crate::blood::ensure_profile_dirs(
                &rings.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
            );
            Some((backfill, migrated))
        })
        .await
        .unwrap_or(None);
        if let Some((backfill, migrated)) = res {
            let n = backfill["reassigned"].as_i64().unwrap_or(0);
            if n > 0 {
                println!("blood: reassigned {n} legacy import(s) to the primary ring");
            }
            let m = migrated["moved"].as_i64().unwrap_or(0);
            if m > 0 {
                println!("blood: moved {m} imported PDF(s) into per-profile document dirs");
            }
        }
    }
    let bind_ip = if lan { "0.0.0.0" } else { "127.0.0.1" };
    let listener = TcpListener::bind((bind_ip, port))
        .await
        .context("binding port")?;
    if lan {
        println!("open_oura dashboard running — open http://127.0.0.1:{port} locally, or http://<this machine's LAN IP>:{port} from other devices");
    } else {
        println!("open_oura dashboard running — open http://127.0.0.1:{port} in your browser");
    }
    // The start entry carries the pid and the binary's link time so a log read
    // can tell which build served the entries below it — the failure mode this
    // exists for is a restart that silently kept the OLD binary serving (the
    // old exe holds the port, the new build never starts, and the log otherwise
    // shows old behavior with no marker saying why).
    log_activity(
        "app",
        "start",
        &format!(
            "dashboard starting (pid {}, built {}, port {port}, lan {lan}, tz {tz}, db {})",
            std::process::id(),
            exe_built_utc(),
            db.display(),
        ),
    );
    // Ctrl+C gets an end entry before the process goes. A hard kill
    // (taskkill /F, Stop-Process) can log nothing — a start with no matching
    // end is how those read in the log.
    tokio::spawn(async {
        if tokio::signal::ctrl_c().await.is_ok() {
            log_activity(
                "app",
                "end",
                &format!("dashboard stopping (pid {}, Ctrl+C)", std::process::id()),
            );
            std::process::exit(0);
        }
    });
    let res: Result<()> = async {
        loop {
            let (sock, _) = listener.accept().await?;
            let (db, name, key_file) = (db.clone(), name.clone(), key_file.clone());
            tokio::spawn(async move {
                let _ = handle(sock, port, lan, db, tz, name, key_file).await;
            });
        }
    }
    .await;
    log_activity(
        "app",
        "end",
        &format!(
            "dashboard stopped (pid {}, {})",
            std::process::id(),
            match &res {
                Ok(()) => "listener closed".to_string(),
                Err(e) => format!("error: {e}"),
            }
        ),
    );
    res
}

/// When the running binary was linked, as UTC text ("2026-11-22T18:04:12Z").
fn exe_built_utc() -> String {
    let secs = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // civil_from_days (Howard Hinnant's algorithm) — no date crate in this crate
    let days = (secs / 86_400) as i64;
    let (h, m, s) = ((secs % 86_400) / 3600, (secs % 3600) / 60, secs % 60);
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y0 = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y0 + 1 } else { y0 };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Serve a web asset: prefer the on-disk file under `dashboard/web/` (so the UI
/// can be edited and refreshed without recompiling) and fall back to the copy
/// embedded at build time.
fn asset(name: &str, embedded: &'static str) -> Vec<u8> {
    if let Some(root) = repo_root() {
        if let Ok(bytes) = std::fs::read(root.join("dashboard/web").join(name)) {
            return bytes;
        }
    }
    embedded.as_bytes().to_vec()
}

/// Look up a `key` in a `a=b&c=d` query string, percent-decoding the value.
fn query_param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == key).then(|| percent_decode(v))
    })
}

/// Minimal application/x-www-form-urlencoded decode: `%XX` escapes and `+` → space.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = |c: u8| (c as char).to_digit(16);
                match (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                    (Some(h), Some(l)) => {
                        out.push((h * 16 + l) as u8);
                        i += 3;
                    }
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn header<'a>(req: &'a str, name: &str) -> Option<&'a str> {
    req.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        k.trim().eq_ignore_ascii_case(name).then(|| v.trim())
    })
}

async fn write_resp(sock: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) -> Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    sock.write_all(head.as_bytes()).await?;
    sock.write_all(body).await?;
    Ok(())
}

async fn json_resp(sock: &mut TcpStream, v: &Value) -> Result<()> {
    write_resp(
        sock,
        "200 OK",
        "application/json",
        serde_json::to_vec(v)?.as_slice(),
    )
    .await
}

async fn handle(
    mut sock: TcpStream,
    port: u16,
    lan: bool,
    db: PathBuf,
    tz: f64,
    name: String,
    key_file: Option<PathBuf>,
) -> Result<()> {
    let mut buf = [0u8; 8192];
    let mut n = 0usize;
    // A single read can return a partial request — TCP segments arrive as they
    // please, and a request line split across segments truncates the target:
    // the query string (the ring serial) silently vanishes and /api/summary
    // then claims no serial was passed. Read until the header block and any
    // Content-Length body have fully arrived, the buffer is full, or EOF.
    loop {
        if let Some(sep) = buf[..n].windows(4).position(|w| w == [13, 10, 13, 10]) {
            let head = String::from_utf8_lossy(&buf[..sep]).into_owned();
            let want = header(&head, "content-length")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            if n - (sep + 4) >= want {
                break;
            }
        }
        if n == buf.len() {
            break;
        }
        match sock.read(&mut buf[n..]).await? {
            0 => break,
            k => n += k,
        }
    }
    let req = String::from_utf8_lossy(&buf[..n]);
    let method = req.split_whitespace().next().unwrap_or("GET");
    let target = req.split_whitespace().nth(1).unwrap_or("/");
    // route on the path only — a query string (e.g. cache-buster) must not 404 the API
    let path = target.split('?').next().unwrap_or(target);
    let query = target.split_once('?').map(|(_, q)| q).unwrap_or("");
    let body = req.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("");

    // Loopback-only (DNS-rebind guard). With --lan the user opted into serving
    // the LAN, so any host is accepted as long as it targets our port (a
    // rebinding attack still cannot reach a nonstandard port blindly).
    let host_ok = if lan {
        header(&req, "host").is_some_and(|h| h.ends_with(&format!(":{port}")))
    } else {
        header(&req, "host")
            .is_some_and(|h| h == format!("127.0.0.1:{port}") || h == format!("localhost:{port}"))
    };
    if !host_ok {
        return write_resp(&mut sock, "403 Forbidden", "text/plain", b"forbidden").await;
    }
    // vendored SVG icons (Phosphor), served from disk
    if path.starts_with("/icons/") && path.ends_with(".svg") && !path.contains("..") {
        if let Some(root) = repo_root() {
            if let Ok(bytes) = std::fs::read(
                root.join("dashboard/web")
                    .join(path.trim_start_matches('/')),
            ) {
                return write_resp(&mut sock, "200 OK", "image/svg+xml", &bytes).await;
            }
        }
        return write_resp(&mut sock, "404 Not Found", "text/plain", b"not found").await;
    }

    // CSRF guard for mutating/action endpoints: a same-origin fetch can set this
    // header; an <img>/<form> or cross-origin fetch cannot (CORS preflight we never approve).
    // GET /api/ring-key is guarded too — it discloses the ring auth key, so only the
    // dashboard's own same-origin page (which sets the header) may read it.
    let csrf_ok = header(&req, "x-oura-dash").is_some();
    let forbid = matches!(
        (method, path),
        ("POST", "/api/profile")
            | ("POST", "/api/research-marker")
            | ("POST", "/api/bedtime")
            | ("POST", "/api/rings")
            | ("POST", "/api/sync")
            | ("POST", "/api/live-hr")
            | ("POST", "/api/feature")
            | ("POST", "/api/ring-key")
            | ("GET", "/api/ring-key")
            | ("POST", "/api/dna/fetch")
            | ("POST", "/api/blood/import-dir")
            | ("POST", "/api/blood/import")
            | ("POST", "/api/blood/assign")
            | ("POST", "/api/dna/assign")
            | ("POST", "/api/logs/clear")
    ) && !csrf_ok;
    if forbid {
        return write_resp(&mut sock, "403 Forbidden", "text/plain", b"forbidden").await;
    }

    // A dashboard can own only one BLE operation at a time.
    static BLE_GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    let _ble_guard = if method == "POST"
        && matches!(path, "/api/sync" | "/api/live-hr" | "/api/feature")
    {
        match BLE_GATE.try_lock() {
            Ok(guard) => Some(guard),
            Err(_) => {
                let busy = json!({"ok": false, "error": "Ring is busy; wait for the current operation to finish.", "message": "Ring is busy; wait for the current operation to finish."});
                if path == "/api/live-hr" {
                    // The live client consumes newline-delimited records.
                    return write_resp(&mut sock, "200 OK", "application/x-ndjson", format!("{busy}\n").as_bytes()).await;
                }
                return json_resp(&mut sock, &busy).await;
            }
        }
    } else {
        None
    };
    let effective_tz = if tz == 0.0 {
        query_param(query, "tz")
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|h| h.is_finite() && (-24.0..=24.0).contains(h))
            .unwrap_or(tz)
    } else {
        tz
    };
    match (method, path) {
        (_, "/") | (_, "/index.html") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/html; charset=utf-8",
                &asset("index.html", INDEX_HTML),
            )
            .await
        }
        (_, "/styles.css") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/css; charset=utf-8",
                &asset("styles.css", STYLES_CSS),
            )
            .await
        }
        (_, "/app.js") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/javascript; charset=utf-8",
                &asset("app.js", APP_JS),
            )
            .await
        }
        (_, "/dna") | (_, "/dna.html") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/html; charset=utf-8",
                &asset("dna.html", DNA_HTML),
            )
            .await
        }
        (_, "/dna.js") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/javascript; charset=utf-8",
                &asset("dna.js", DNA_JS),
            )
            .await
        }
        (_, "/dna.css") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/css; charset=utf-8",
                &asset("dna.css", DNA_CSS),
            )
            .await
        }
        (_, "/blood") | (_, "/blood.html") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/html; charset=utf-8",
                &asset("blood.html", BLOOD_HTML),
            )
            .await
        }
        (_, "/blood.js") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/javascript; charset=utf-8",
                &asset("blood.js", BLOOD_JS),
            )
            .await
        }
        (_, "/profile.js") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/javascript; charset=utf-8",
                &asset("profile.js", PROFILE_JS),
            )
            .await
        }
        (_, "/blood.css") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/css; charset=utf-8",
                &asset("blood.css", BLOOD_CSS),
            )
            .await
        }
        (_, "/logs") | (_, "/logs.html") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/html; charset=utf-8",
                &asset("logs.html", LOGS_HTML),
            )
            .await
        }
        (_, "/logs.js") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/javascript; charset=utf-8",
                &asset("logs.js", LOGS_JS),
            )
            .await
        }
        (_, "/logs.css") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/css; charset=utf-8",
                &asset("logs.css", LOGS_CSS),
            )
            .await
        }
        (_, "/ring") | (_, "/ring.html") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/html; charset=utf-8",
                &asset("ring.html", RING_HTML),
            )
            .await
        }
        (_, "/ring.js") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/javascript; charset=utf-8",
                &asset("ring.js", RING_JS),
            )
            .await
        }
        (_, "/ring.css") => {
            write_resp(
                &mut sock,
                "200 OK",
                "text/css; charset=utf-8",
                &asset("ring.css", RING_CSS),
            )
            .await
        }
        ("GET", "/api/logs") => {
            // The /logs page's feed: activity ring (syncs, model runs, live-HR,
            // feature toggles; persisted to activity_log.jsonl), newest first.
            // `kind` filters; `limit` caps.
            let kind = query_param(query, "kind").unwrap_or_default();
            let limit = query_param(query, "limit")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(200)
                .clamp(1, ACTIVITY_MAX);
            json_resp(&mut sock, &json!({ "entries": list_activity(&kind, limit) })).await
        }
        ("POST", "/api/logs/clear") => {
            clear_activity();
            json_resp(&mut sock, &json!({ "ok": true })).await
        }
        ("GET", "/api/blood/report") => {
            let serial = query_param(query, "serial").unwrap_or_default();
            let v = tokio::task::spawn_blocking(move || {
                crate::blood::report((!serial.is_empty()).then_some(serial.as_str()))
            })
            .await
            .map_err(|e| anyhow!(e))?;
            json_resp(&mut sock, &v).await
        }
        ("POST", "/api/blood/import-dir") => {
            let req =
                serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
            let serial = req["serial"].as_str().unwrap_or("").to_string();
            let v = tokio::task::spawn_blocking(move || crate::blood::import_dir(&serial))
                .await
                .map_err(|e| anyhow!(e))?;
            json_resp(&mut sock, &v).await
        }
        ("POST", "/api/blood/import") => {
            let req =
                serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
            let path = req["path"].as_str().unwrap_or("").to_string();
            let serial = req["serial"].as_str().unwrap_or("").to_string();
            let v = tokio::task::spawn_blocking(move || crate::blood::import_path(&path, &serial))
                .await
                .map_err(|e| anyhow!(e))?;
            json_resp(&mut sock, &v).await
        }
        ("POST", "/api/blood/assign") => {
            let req =
                serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
            let sha = req["sha"].as_str().unwrap_or("").to_string();
            let serial = req["serial"].as_str().unwrap_or("").to_string();
            let v = tokio::task::spawn_blocking(move || crate::blood::assign(&sha, &serial))
                .await
                .map_err(|e| anyhow!(e))?;
            json_resp(&mut sock, &v).await
        }
        ("GET", "/api/dna/files") => {
            let v = tokio::task::spawn_blocking(crate::dna::list_files)
                .await
                .map_err(|e| anyhow!(e))?;
            // Profiles (docs/profiles.md): hide genome files assigned to another
            // profile; unassigned files stay visible to everyone.
            let serial = query_param(query, "serial").unwrap_or_default();
            let v = if v.is_object() && !serial.is_empty() {
                let db = db.clone();
                tokio::task::spawn_blocking(move || {
                    let assign = oura_store::storage::Store::open_read_only(&db)
                        .and_then(|s| s.list_dna_assignments())
                        .unwrap_or_default();
                    let mut v = v;
                    if let Some(list) = v["genomes"].as_array_mut() {
                        list.retain(|g| match assign.get(g["name"].as_str().unwrap_or("")) {
                            Some(owner) => *owner == serial,
                            None => true,
                        });
                        for g in list.iter_mut() {
                            let name = g["name"].as_str().unwrap_or("").to_string();
                            if let Some(obj) = g.as_object_mut() {
                                obj.insert(
                                    "serial".into(),
                                    Value::String(
                                        assign.get(name.as_str()).cloned().unwrap_or_default(),
                                    ),
                                );
                            }
                        }
                    }
                    v
                })
                .await
                .map_err(|e| anyhow!(e))?
            } else {
                v
            };
            json_resp(&mut sock, &v).await
        }
        ("POST", "/api/dna/assign") => {
            let req =
                serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
            let filename = req["filename"].as_str().unwrap_or("").to_string();
            let serial = req["serial"].as_str().unwrap_or("").to_string();
            if filename.is_empty() {
                return json_resp(&mut sock, &json!({"ok": false, "message": "filename required"}))
                    .await;
            }
            let db2 = db.clone();
            let v = tokio::task::spawn_blocking(move || {
                match oura_store::storage::Store::open(&db2)
                    .and_then(|s| s.set_dna_assignment(&filename, &serial))
                {
                    Ok(()) => json!({"ok": true}),
                    Err(e) => json!({"ok": false, "message": e.to_string()}),
                }
            })
            .await
            .map_err(|e| anyhow!(e))?;
            json_resp(&mut sock, &v).await
        }
        ("GET", "/api/dna/report") => {
            // parsing a genome is CPU/IO-heavy → off the async executor; cached per
            // (genome, selected scores).
            let file = query_param(query, "file").unwrap_or_default();
            let scores = query_param(query, "scores").unwrap_or_default();
            let v = tokio::task::spawn_blocking(move || crate::dna::report(&file, &scores))
                .await
                .map_err(|e| anyhow!(e))?;
            json_resp(&mut sock, &v).await
        }
        ("POST", "/api/dna/fetch") => {
            // downloads a PUBLIC PGS Catalog score definition from EBI (no genome
            // data leaves). CSRF-gated like the other action endpoints.
            let req =
                serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
            let id = req["pgs_id"].as_str().unwrap_or("").to_string();
            let v = tokio::task::spawn_blocking(move || crate::dna::fetch(&id))
                .await
                .map_err(|e| anyhow!(e))?;
            json_resp(&mut sock, &v).await
        }
        ("GET", "/api/dna/lookup") => {
            let file = query_param(query, "file").unwrap_or_default();
            let q = query_param(query, "q").unwrap_or_default();
            let v = tokio::task::spawn_blocking(move || crate::dna::lookup(&file, &q))
                .await
                .map_err(|e| anyhow!(e))?;
            json_resp(&mut sock, &v).await
        }
        ("GET", "/api/rings") => {
            // The switcher's data: every registered ring with its display name.
            let v = match oura_store::storage::Store::open_read_only(&db)
                .and_then(|s| Ok((s.list_rings()?, s.ring_macs()?)))
            {
                Ok((rings, macs)) => Value::Array(
                    rings
                        .iter()
                        .map(|(serial, name, note, _added, _seen, last_sync, _cursor, count)| {
                            json!({
                                "serial": serial, "name": name, "note": note,
                                "mac": macs.get(serial),
                                "last_sync": last_sync, "events": count,
                            })
                        })
                        .collect(),
                ),
                Err(e) => json!({ "error": e.to_string() }),
            };
            json_resp(&mut sock, &v).await
        }
        ("GET", "/api/ring-diagnostics") => {
            // The /ring page: health view for every registered ring (or one via
            // ?serial=). Computed from the DB per request; cheap enough (last
            // 30 days of batch anchors).
            let serial = query_param(query, "serial").unwrap_or_default();
            let v = tokio::task::spawn_blocking(move || ring_diagnostics(&db, &serial))
                .await
                .unwrap_or_else(|e| json!({ "error": e.to_string() }));
            json_resp(&mut sock, &v).await
        }
        ("POST", "/api/rings") => {
            // Rename (and optionally annotate) a registered ring. The write bumps
            // oura.db's mtime, which invalidates the summary cache on its own.
            let req = serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
            let serial = req["serial"].as_str().unwrap_or("");
            let name = req["name"].as_str().unwrap_or("").trim();
            // note absent → leave it alone; note present (even "") → set/clear it.
            let note = req["note"].as_str().map(|s| s.trim());
            if serial.is_empty() || name.is_empty() {
                return json_resp(&mut sock, &json!({ "ok": false, "error": "serial and name are required" })).await;
            }
            match oura_store::storage::Store::open(&db).and_then(|s| s.rename_ring(serial, name, note)) {
                Ok(true) => json_resp(&mut sock, &json!({ "ok": true })).await,
                Ok(false) => json_resp(&mut sock, &json!({ "ok": false, "error": format!("no ring registered with serial {serial}") })).await,
                Err(e) => json_resp(&mut sock, &json!({ "ok": false, "error": e.to_string() })).await,
            }
        }
        ("GET", "/api/summary") => {
            // building the summary shells out to torch models → off the async
            // executor; cached so only the first load (or post-sync/edit) pays for it.
            // After that a stale value is served instantly while a background
            // thread rebuilds — the page never blocks on the models.
            let serial = query_param(query, "serial");
            let built = tokio::task::spawn_blocking(move || {
                cached_summary(&db, effective_tz, serial.as_deref())
            })
            .await
            .map_err(|e| anyhow!(e))?;
            match built {
                Ok((mut v, recomputing)) => {
                    if recomputing {
                        // Arc::make_mut clones once (the cache keeps its copy); the
                        // flag tells the page fresh numbers are on their way.
                        if let Some(obj) = Arc::make_mut(&mut v).as_object_mut() {
                            obj.insert("recomputing".into(), json!(true));
                        }
                    }
                    json_resp(&mut sock, &v).await
                }
                Err(e) => json_resp(&mut sock, &json!({ "error": e.to_string() })).await,
            }
        }
        ("GET", "/api/hourly-hr") => {
            // HR bars per `minutes` slot (default hourly, the iOS HeartRate screen's
            // data); `days` = 0 → all of them, the day page picks out the day it shows
            let days = query_param(query, "days")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let minutes = query_param(query, "minutes")
                .and_then(|v| v.parse().ok())
                .unwrap_or(60);
            let serial = query_param(query, "serial");
            let from = query_param(query, "from").and_then(|v| v.parse::<i64>().ok());
            let to = query_param(query, "to").and_then(|v| v.parse::<i64>().ok());
            let body = tokio::task::spawn_blocking(move || {
                let serial = serial
                    .as_deref()
                    .map(|s| resolve_summary_serial(&db, Some(s)))
                    .transpose()?;
                let serial = serial.as_deref();
                // With an explicit window, read only that window's rows instead
                // of binning the whole history; both paths share the memo cache.
                if let (Some(f), Some(t)) = (from, to) {
                    if f < t {
                        return cached_hr_window(&db, effective_tz, days, minutes, serial, f, t);
                    }
                }
                cached_hr_bins(&db, effective_tz, days, minutes, serial)
            })
            .await
            .map_err(|e| anyhow!(e))?;
            match body {
                Ok(bytes) => write_resp(&mut sock, "200 OK", "application/json", &bytes).await,
                Err(e) => json_resp(&mut sock, &json!({ "error": e.to_string() })).await,
            }
        }
        ("GET", "/api/research-export") => {
            // ZIP of raw per-beat / per-window CSVs for [start, end) — the same
            // `research_export` the iOS app and `oura research-export` use.
            let num = |k: &str| query_param(query, k).and_then(|v| v.parse::<f64>().ok());
            let (Some(start), Some(end)) = (num("start"), num("end")) else {
                return write_resp(&mut sock, "400 Bad Request", "text/plain", b"start and end (unix seconds) are required").await;
            };
            let streams: Vec<_> = query_param(query, "streams")
                .unwrap_or_default()
                .split(',')
                .filter_map(oura_summary::research_export::Stream::from_id)
                .collect();
            let tz = num("tz").unwrap_or(effective_tz);
            let serial = query_param(query, "serial");
            let built = tokio::task::spawn_blocking(move || {
                use oura_summary::research_export as rx;
                let serial = serial
                    .as_deref()
                    .map(|s| resolve_summary_serial(&db, Some(s)))
                    .transpose()?;
                let export = rx::build(&db, &rx::Request { start_unix: start, end_unix: end, streams, tz_offset_hours: tz, serial })?;
                let name = rx::bundle_name(start, tz);
                Ok::<_, anyhow::Error>((rx::to_zip(&export, &name)?, name))
            })
            .await
            .map_err(|e| anyhow!(e))?;
            match built {
                Ok((zip, name)) => {
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/zip\r\nContent-Disposition: attachment; filename=\"{name}.zip\"\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
                        zip.len()
                    );
                    sock.write_all(head.as_bytes()).await?;
                    sock.write_all(&zip).await?;
                    Ok(())
                }
                Err(e) => write_resp(&mut sock, "400 Bad Request", "text/plain", e.to_string().as_bytes()).await,
            }
        }
        ("POST", "/api/research-marker") => {
            let body_json = serde_json::from_str::<Value>(body).ok();
            let label = body_json
                .as_ref()
                .and_then(|v| v["label"].as_str().map(str::to_owned))
                .unwrap_or_default();
            let serial = body_json
                .as_ref()
                .and_then(|v| v["serial"].as_str().map(str::to_owned));
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as f64 / 1000.0)
                .unwrap_or(0.0);
            let serial = serial
                .as_deref()
                .map(|s| resolve_summary_serial(&db, Some(s)))
                .transpose()
                .map(|s| s.unwrap_or_default());
            match serial.and_then(|s| oura_summary::research_export::add_marker(&db, now, &label, Some(&s))) {
                Ok(()) => json_resp(&mut sock, &json!({ "ok": true, "unix": now })).await,
                Err(e) => json_resp(&mut sock, &json!({ "error": e.to_string() })).await,
            }
        }
        ("GET", "/api/profile") => {
            let serial = query_param(query, "serial").unwrap_or_default();
            json_resp(&mut sock, &read_profile_for(&db, &serial).to_json()).await
        }
        ("POST", "/api/profile") => {
            match serde_json::from_str::<Value>(body.trim_end_matches('\0')) {
                Ok(v) => {
                    let serial = v["serial"].as_str().unwrap_or("").to_string();
                    match write_profile_for(&db, &serial, &v) {
                        Ok(d) => json_resp(&mut sock, &d.to_json()).await,
                        Err(e) => json_resp(&mut sock, &json!({ "error": e.to_string() })).await,
                    }
                }
                Err(e) => json_resp(&mut sock, &json!({ "error": e.to_string() })).await,
            }
        }
        ("POST", "/api/bedtime") => {
            let req =
                serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
            let raw_start_ds = req["raw_start_ds"].as_i64();
            let captured_unix = req["captured_unix"].as_i64();
            let reset = req["reset"].as_bool().unwrap_or(false);
            let (start_ds, end_ds) = if reset {
                (None, None)
            } else {
                match (req["start_ds"].as_i64(), req["end_ds"].as_i64()) {
                    (Some(s), Some(e)) => (Some(s), Some(e)),
                    _ => {
                        return json_resp(
                            &mut sock,
                            &json!({ "ok": false, "error": "start_ds and end_ds are required" }),
                        )
                        .await;
                    }
                }
            };
            let Some(raw_start_ds) = raw_start_ds else {
                return json_resp(
                    &mut sock,
                    &json!({ "ok": false, "error": "raw_start_ds is required" }),
                )
                .await;
            };
            match write_bedtime_override(&db, raw_start_ds, captured_unix, start_ds, end_ds) {
                Ok(v) => json_resp(&mut sock, &v).await,
                Err(e) => {
                    json_resp(&mut sock, &json!({ "ok": false, "error": e.to_string() })).await
                }
            }
        }
        ("GET", "/api/ring-key") => match read_ring_key(key_file.as_deref()) {
            Ok(key) => json_resp(&mut sock, &json!({ "ok": true, "key": key })).await,
            Err(e) => json_resp(&mut sock, &json!({ "ok": false, "message": e.to_string() })).await,
        },
        ("POST", "/api/ring-key") => {
            let req =
                serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
            match write_ring_key(key_file.as_deref(), req["key"].as_str().unwrap_or("")) {
                Ok(_) => json_resp(&mut sock, &json!({ "ok": true })).await,
                Err(e) => {
                    json_resp(&mut sock, &json!({ "ok": false, "message": e.to_string() })).await
                }
            }
        }
        ("POST", "/api/sync") => {
            // Optional {"serial": "..."} targets one specific ring (see resolve_ring_target).
            let req = serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
            let target = req["serial"].as_str().unwrap_or("").trim().to_string();
            let (serial, key_file) = resolve_ring_target(&db, &target, key_file.as_deref());
            let db2 = db.clone();
            let res = tokio::task::spawn_blocking(move || {
                run_sync(&db, &name, &serial, key_file.as_deref())
            })
                .await
                .map_err(|e| anyhow!(e))?;
            if res.is_ok() {
                warm_summaries(&db2);
            }
            let v = match res {
                Ok(msg) => json!({ "ok": true, "message": msg }),
                Err(e) => json!({ "ok": false, "message": e.to_string() }),
            };
            json_resp(&mut sock, &v).await
        }
        ("POST", "/api/live-hr") => {
            // Same per-ring targeting as /api/sync (name filter + stored key).
            let req = serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
            let target = req["serial"].as_str().unwrap_or("").trim().to_string();
            let (serial, key_file) = resolve_ring_target(&db, &target, key_file.as_deref());
            live_hr(&mut sock, &db, &name, &serial, key_file.as_deref()).await
        }
        ("POST", "/api/feature") => {
            let req =
                serde_json::from_str::<Value>(body.trim_end_matches('\0')).unwrap_or(Value::Null);
            let feature = req["feature"].as_str().unwrap_or("").to_string();
            let mode = req["mode"].as_str().unwrap_or("").to_string();
            let target = req["serial"].as_str().unwrap_or("").trim().to_string();
            let (serial, key_file) = resolve_ring_target(&db, &target, key_file.as_deref());
            let res = tokio::task::spawn_blocking(move || {
                run_feature(&db, &name, &serial, key_file.as_deref(), &feature, &mode)
            })
            .await
            .map_err(|e| anyhow!(e))?;
            let v = match res {
                Ok(msg) => json!({ "ok": true, "message": msg }),
                Err(e) => json!({ "ok": false, "message": e.to_string() }),
            };
            json_resp(&mut sock, &v).await
        }
        _ => write_resp(&mut sock, "404 Not Found", "text/plain", b"not found").await,
    }
}

/// BLE targeting for a dashboard action. With a serial, the CLI gets
/// `--expect-serial` (advertisements carry no serial or stable address, so the
/// CLI probes candidates and verifies the serial over GATT) and the ring's
/// stored auth key (rings.auth_key_hex) beats the global --key-file; without
/// one, the CLI defaults stand. The stored key is written to a sibling file
/// because `oura` takes a path.
fn resolve_ring_target(
    db: &Path,
    serial: &str,
    default_key: Option<&Path>,
) -> (String, Option<PathBuf>) {
    if serial.is_empty() {
        return (String::new(), default_key.map(PathBuf::from));
    }
    let key = match oura_store::storage::Store::open_read_only(db).and_then(|s| s.ring_key(serial)) {
        Ok(Some(hex)) => {
            let p = db.with_file_name(format!("ring-{serial}.key"));
            if std::fs::write(&p, format!("{hex}\n")).is_ok() {
                Some(p)
            } else {
                default_key.map(PathBuf::from)
            }
        }
        _ => default_key.map(PathBuf::from),
    };
    (serial.to_string(), key)
}

/// BLE targeting flags for a CLI child: prefer `--expect-serial` (verified over
/// GATT) when a serial is known, else the legacy `--name` filter.
fn target_args(c: &mut Command, name: &str, serial: &str) {
    if serial.is_empty() {
        c.arg("--name").arg(name);
    } else {
        c.arg("--expect-serial").arg(serial);
    }
}

/// After a successful sync, proactively spawn summary rebuilds for every cached
/// (tz, serial). The recompute is otherwise request-driven: it only starts when
/// some page happens to ask for a summary after the token changed — so a sync
/// whose tab was closed or backgrounded mid-run leaves the models idle for
/// minutes. Reuses cached_summary's token check + in-flight guard, so this is
/// a no-op when nothing changed or a rebuild is already running.
fn warm_summaries(db: &Path) {
    let targets: Vec<(f64, String)> = summary_cache()
        .lock()
        .unwrap()
        .iter()
        .map(|c| (c.tz, c.serial.clone()))
        .collect();
    for (tz, serial) in targets {
        let _ = cached_summary(db, tz, Some(&serial));
    }
}

/// Human-readable sync target for the activity log: the GATT serial plus the
/// display name from the ring switcher (rings table). The old bare
/// `name=Oura` label was the CLI's default BLE name filter, not a real ring —
/// registered rings carry user-chosen names, so the log must show which
/// physical ring a sync targets.
fn describe_ring(db: &Path, name_filter: &str, serial: &str) -> String {
    let rings = oura_store::storage::Store::open_read_only(db)
        .and_then(|s| s.list_rings())
        .unwrap_or_default();
    if !serial.is_empty() {
        return match rings.iter().find(|r| r.0 == serial) {
            Some(r) => format!("serial={serial} name=\"{}\"", r.1),
            None => format!("serial={serial} (not in rings table)"),
        };
    }
    // No serial: the CLI child picks by advertised BLE name (`--name` filter).
    // Say so, and list what is actually registered so the ambiguity is visible.
    let listed: Vec<String> = rings
        .iter()
        .map(|r| format!("{}=\"{}\"", r.0, r.1))
        .collect();
    format!(
        "name-filter=\"{name_filter}\" (no serial; registered: {})",
        if listed.is_empty() { "none".to_string() } else { listed.join(", ") }
    )
}

/// Per-ring health data for the `/ring` diagnostics page: identity and key
/// provenance, sync state, a 14-day wearing-coverage heatmap, recording gaps,
/// per-stream last-seen times, and recent activity-log entries that mention
/// the ring. All derived from the DB at request time.
///
/// Coverage is grounded in the sensing streams themselves: the ring produces
/// IBI / heart-rate / motion samples continuously while powered and worn, so
/// an hour with zero such samples is a real not-recording period. (A lagging
/// tick counter is NOT evidence of a gap: buffered replays and clock re-syncs
/// break any batch-boundary method — proven on a 24/7-worn ring whose every
/// "counter stall" contained thousands of buffered samples.)
///
/// Ring ticks (`ring_timestamp`, deciseconds) are mapped to wall clock through
/// `time_sync` events, which carry the phone's unix time next to the ring's
/// counter — exact piecewise-linear anchors. Each gap is then characterised:
/// a `ring_start` boot at resume means the ring powered off (dead battery);
/// a battery reading that rose across the gap means it was on the charger;
/// otherwise the ring simply saw nothing (off-finger in a drawer).
/// Event names carry no index in the events table, but their numeric `tag`
/// does (UNIQUE(serial, tag, ring_timestamp, body)). Resolve the mapping once
/// per process so every diagnostics query can use the index.
fn event_tags(conn: &rusqlite::Connection) -> &std::collections::HashMap<String, i64> {
    static TAGS: std::sync::OnceLock<std::collections::HashMap<String, i64>> =
        std::sync::OnceLock::new();
    TAGS.get_or_init(|| {
        let mut m = std::collections::HashMap::new();
        if let Ok(mut s) = conn.prepare("SELECT name, tag FROM events GROUP BY name, tag") {
            if let Ok(q) = s.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))) {
                for (n, t) in q.filter_map(|v| v.ok()) {
                    m.entry(n).or_insert(t);
                }
            }
        }
        m
    })
}

/// The scan is minute-scale, so serve the full-list result from a short-lived
/// cache; the page polls and several viewers shouldn't each pay for it.
fn ring_diagnostics(db: &Path, only_serial: &str) -> Value {
    if !only_serial.is_empty() {
        return ring_diagnostics_inner(db, only_serial);
    }
    static CACHE: std::sync::OnceLock<std::sync::Mutex<(std::time::Instant, Value)>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new((std::time::Instant::now() - std::time::Duration::from_secs(9999), Value::Null)));
    if let Ok(g) = cache.lock() {
        if g.0.elapsed().as_secs() < 300 {
            return g.1.clone();
        }
    }
    let v = ring_diagnostics_inner(db, "");
    if let Ok(mut g) = cache.lock() {
        *g = (std::time::Instant::now(), v.clone());
    }
    v
}

fn ring_diagnostics_inner(db: &Path, only_serial: &str) -> Value {
    const COVERAGE_DAYS: i64 = 14;
    const GAP_MIN_S: i64 = 3600; // hour-granularity: gaps of >= 1 empty hour
    const MAX_GAPS: usize = 40;
    let conn = match rusqlite::Connection::open_with_flags(
        db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) {
        Ok(c) => c,
        Err(e) => return json!({ "error": e.to_string() }),
    };
    let mut stmt = match conn.prepare(
        "SELECT r.serial, r.name, r.key_source, r.note, r.added_unix, r.last_seen_unix,
                d.firmware, d.api_version, d.mac, s.last_sync_unix
         FROM rings r LEFT JOIN device d ON d.serial = r.serial
         LEFT JOIN sync_state s ON s.serial = r.serial
         WHERE (?1 = '' OR r.serial = ?1)
         ORDER BY r.added_unix",
    ) {
        Ok(s) => s,
        Err(e) => return json!({ "error": e.to_string() }),
    };
    let rings: Vec<(String, String, String, String, i64, i64, String, String, String, i64)> = stmt
        .query_map(rusqlite::params![only_serial], |r| {
            let opt = |i: usize| -> String { r.get::<_, Option<String>>(i).ok().flatten().unwrap_or_default() };
            Ok((
                r.get(0)?,
                r.get(1)?,
                opt(2),
                opt(3),
                r.get(4)?,
                r.get(5)?,
                opt(6),
                opt(7),
                opt(8),
                r.get::<_, Option<i64>>(9)?.unwrap_or(0),
            ))
        })
        .ok()
        .map(|q| q.filter_map(|v| v.ok()).collect())
        .unwrap_or_default();
    let tags = event_tags(&conn);
    let mut out: Vec<Value> = Vec::new();
    for (serial, name, key_source, note, added_s, last_seen_s, firmware, api_version, mac, last_sync_s) in rings {
        let (tag_sync, tag_start, tag_dbg) = (
            tags.get("time_sync").copied(),
            tags.get("ring_start").copied(),
            tags.get("debug_data").copied(),
        );
        let tags_live: Vec<i64> = ["ibi_and_amplitude_event", "ehr_trace_event", "motion_event"]
            .iter()
            .filter_map(|n| tags.get(*n).copied())
            .collect();
        if tags_live.is_empty() {
            continue;
        }
        // Clock anchors: time_sync events pair the ring's tick counter with the
        // phone's wall clock. Piecewise-linear map between them; cleaned so both
        // sequences advance (clock re-syncs make the raw list non-monotonic).
        let Some(tag_sync) = tag_sync else { continue };
        let mut stmt = match conn.prepare(
            "SELECT ring_timestamp, decoded_json FROM events
             WHERE serial = ?1 AND tag = ?2 AND decoded_json IS NOT NULL
             ORDER BY ring_timestamp",
        ) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut anchors: Vec<(i64, i64)> = Vec::new();
        let rows = stmt.query_map(rusqlite::params![&serial, tag_sync], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?))
        });
        if let Ok(q) = rows {
            for (rt, dj) in q.filter_map(|v| v.ok()) {
                let Some(dj) = dj else { continue };
                let Ok(parsed) = serde_json::from_str::<Value>(&dj) else { continue };
                let Some(u) = parsed.get("unix_time").and_then(Value::as_i64) else {
                    continue;
                };
                match anchors.last() {
                    // keep ticks strictly increasing; tolerate small wall-clock
                    // nudges but drop large backward re-syncs
                    Some(&(lrt, lu)) if rt > lrt && u >= lu - 3600 => anchors.push((rt, u)),
                    None => anchors.push((rt, u)),
                    _ => {}
                }
            }
        }
        if anchors.len() < 2 {
            continue; // no clock map for this ring: nothing defensible to report
        }
        let to_wall = |rt: i64| -> i64 {
            let i = anchors.partition_point(|&(r, _)| r <= rt).saturating_sub(1);
            let (r0, u0) = anchors[i];
            let (r1, u1) = anchors[(i + 1).min(anchors.len() - 1)];
            if r1 <= r0 {
                u0
            } else {
                u0 + (rt - r0) * (u1 - u0) / (r1 - r0)
            }
        };
        let to_tick = |u: i64| -> i64 {
            // inverse of to_wall over the anchor list
            let i = anchors.iter().rposition(|&(_, w)| w <= u).unwrap_or(0);
            let (r0, u0) = anchors[i];
            let (r1, u1) = anchors[(i + 1).min(anchors.len() - 1)];
            if u1 <= u0 {
                r0
            } else {
                r0 + (u - u0) * (r1 - r0) / (u1 - u0)
            }
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let cov_start = ((now - COVERAGE_DAYS * 86400) / 3600) * 3600;
        // Coverage: hours containing at least one sensing sample (IBI, EHR,
        // motion). The ring emits these continuously while powered and worn,
        // so an empty hour is a real not-recording period — buffered replays
        // carry original tick times, so late delivery cannot fake a gap here.
        let scan_from = to_tick(cov_start) - 2_000_000; // margin for tick/wall drift
        // Bucket per tick-hour in SQL (ring_timestamp is deciseconds): the map
        // to wall clock is near-identity, so per-bucket mapping is plenty and
        // the query returns ~340 rows instead of ~1M.
        let mut sql = String::from(
            "SELECT ring_timestamp / 36000, MIN(ring_timestamp), MAX(ring_timestamp) FROM events\n             WHERE serial = ?1 AND ring_timestamp > ?2 AND (",
        );
        let mut bind: Vec<&dyn rusqlite::types::ToSql> = vec![&serial, &scan_from];
        for (i, t) in tags_live.iter().enumerate() {
            if i > 0 {
                sql.push_str(" OR");
            }
            sql.push_str(&format!(" tag = ?{}", i + 3));
            bind.push(t);
        }
        sql.push_str(") GROUP BY 1");
        let mut stmt = match conn.prepare(&sql) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut live_hours: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        // Spanning a bucket across its wall-clock range is only sound while the
        // anchor segment maps ticks ~1:1 to seconds. When a ring powers off its
        // tick clock freezes until the next time_sync, and that segment's ratio
        // explodes — a single bucket would then paint the whole powered-off
        // window "live". Detect abnormal segments and map such buckets to a
        // single hour instead.
        let seg_ok = |t: i64| -> bool {
            let i = anchors.partition_point(|&(r, _)| r <= t).saturating_sub(1);
            let (r0, u0) = anchors[i];
            let (r1, u1) = anchors[(i + 1).min(anchors.len() - 1)];
            r1 <= r0 || (u1 - u0).max(0) * 10 <= 2 * (r1 - r0) + 600
        };
        {
            let rows = stmt.query_map(bind.as_slice(), |r| Ok((r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)));
            if let Ok(q) = rows {
                for (t0, t1) in q.filter_map(|v| v.ok()) {
                    // Span only real sample ticks (not bucket edges): the gap
                    // edges are exactly where fictitious edges would lie.
                    let (w0, w1) = if seg_ok(t0) && seg_ok(t1) {
                        (to_wall(t0), to_wall(t1))
                    } else {
                        let m = to_wall((t0 + t1) / 2);
                        (m, m)
                    };
                    if w1 >= cov_start {
                        for hh in (w0.max(cov_start) / 3600)..=(w1 / 3600) {
                            live_hours.insert(hh);
                        }
                    }
                }
            }
        }
        drop(stmt);
        // Recording gaps: runs of >= 1 empty hour on the hour grid, kept to the
        // most recent MAX_GAPS; each is then characterised (boot / battery).
        let boots: Vec<i64> = {
            let mut acc: Vec<i64> = Vec::new();
            if let (Some(tag_start), Ok(mut stmt)) = (
                tag_start,
                conn.prepare("SELECT ring_timestamp FROM events WHERE serial = ?1 AND tag = ?2"),
            ) {
                if let Ok(q) = stmt.query_map(rusqlite::params![&serial, tag_start], |r| r.get::<_, i64>(0)) {
                    acc = q.filter_map(|v| v.ok()).map(&to_wall).collect();
                }
            }
            acc
        };
        let battery_at = |rt: i64, before: bool| -> Option<(i64, i64, i64)> {
            let tag_dbg = tag_dbg?;
            let (op, dir) = if before { ("<=", "DESC") } else { (">=", "ASC") };
            let sql = format!(
                "SELECT ring_timestamp, decoded_json FROM events
                 WHERE serial = ?1 AND tag = ?2 AND ring_timestamp {op} ?3
                 ORDER BY ring_timestamp {dir} LIMIT 300"
            );
            let mut stmt = match conn.prepare(&sql) {
                Ok(s) => s,
                Err(_) => return None,
            };
            let rows = stmt.query_map(rusqlite::params![&serial, tag_dbg, rt], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?))
            });
            // nearest row that actually carries a battery reading
            for (rt0, dj) in rows.ok()?.filter_map(|v| v.ok()) {
                let Some(dj) = dj else { continue };
                let Ok(parsed) = serde_json::from_str::<Value>(&dj) else { continue };
                let Some(pct) = parsed.get("battery_pct").and_then(Value::as_i64) else {
                    continue;
                };
                let mv = parsed.get("voltage_mv").and_then(Value::as_i64).unwrap_or(-1);
                return Some((to_wall(rt0), pct, mv));
            }
            None
        };
        let mut gaps: Vec<Value> = Vec::new();
        // Past the ring's last sync the state is UNKNOWABLE, not off-finger:
        // the ring may be recording perfectly with everything still buffered
        // on-device. Evidence ends at the last sync (the newest delivered
        // event is the fallback when sync_state lags), so the hour-grid scan
        // never claims a cause beyond that point.
        let mut max_delivered = 0i64;
        if let Ok(mut st) = conn.prepare("SELECT MAX(captured_unix) FROM events WHERE serial = ?1") {
            if let Ok(v) = st.query_row(rusqlite::params![&serial], |r| r.get::<_, Option<i64>>(0)) {
                max_delivered = v.unwrap_or(0);
            }
        }
        let observed_until = last_sync_s.max(max_delivered);
        let mut h = cov_start / 3600;
        let h_end = now / 3600;
        while h <= h_end {
            if !live_hours.contains(&h) {
                let gs = h * 3600;
                while h + 1 <= h_end && !live_hours.contains(&(h + 1)) {
                    h += 1;
                }
                let ge = ((h + 1) * 3600 - 1).min(observed_until.max(gs));
                if gs < observed_until && ge - gs >= GAP_MIN_S {
                    let boot = boots
                        .iter()
                        .any(|&b| b >= gs - 300 && b <= ge + 900);
                    let bb = battery_at(to_tick(gs), true);
                    let ba = battery_at(to_tick(ge), false);
                    let kind = if boot {
                        "power-off"
                    } else if bb
                        .as_ref()
                        .zip(ba.as_ref())
                        .is_some_and(|(b, a)| a.1 > b.1 + 5)
                    {
                        "charging"
                    } else {
                        "idle"
                    };
                    let batt = |b: &Option<(i64, i64, i64)>| match b {
                        Some((s, p, v)) => json!({ "s": s, "pct": p, "mv": v }),
                        None => Value::Null,
                    };
                    gaps.push(json!({
                        "from_s": gs, "to_s": ge, "dur_s": ge - gs,
                        "kind": kind, "boot": boot,
                        "batt_before": batt(&bb), "batt_after": batt(&ba),
                    }));
                }
            }
            h += 1;
        }
        // The tail past the observation cutoff is an explicit unknown — never
        // guessed at as off-finger time.
        let unknown_from = observed_until
            .max(live_hours.iter().next_back().map_or(0, |lh| (lh + 1) * 3600));
        if unknown_from > cov_start && now - unknown_from >= GAP_MIN_S {
            gaps.push(json!({
                "from_s": unknown_from, "to_s": now, "dur_s": now - unknown_from,
                "kind": "unknown", "boot": false,
                "batt_before": Value::Null, "batt_after": Value::Null,
            }));
        }
        if gaps.len() > MAX_GAPS {
            gaps.drain(..gaps.len() - MAX_GAPS);
        }
        // Battery curve: the fuel gauge emits debug_data battery_level_changed
        // events whenever pct or voltage moves. They are buffered and burst at
        // sync (captured_unix bunches), so their real time comes from the ring
        // clock through the same anchors as the coverage map. Thinning keeps
        // every pct change and >=1 point per 10 min of flat stretch; the last
        // reading before the window stays as the line's left anchor.
        let mut battery: Vec<Value> = Vec::new();
        if let Some(tag_dbg) = tag_dbg {
            if let Ok(mut stmt) = conn.prepare(
                "SELECT ring_timestamp, decoded_json FROM events
                 WHERE serial = ?1 AND tag = ?2 AND ring_timestamp > ?3
                   AND decoded_json LIKE '%battery_pct%'
                 ORDER BY ring_timestamp",
            ) {
                let rows = stmt.query_map(rusqlite::params![&serial, tag_dbg, scan_from], |r| {
                    Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?))
                });
                let mut prev: Option<(i64, i64)> = None; // (t, pct)
                if let Ok(q) = rows {
                    for (rt, dj) in q.filter_map(|v| v.ok()) {
                        let Some(dj) = dj else { continue };
                        let Ok(parsed) = serde_json::from_str::<Value>(&dj) else { continue };
                        let Some(pct) = parsed.get("battery_pct").and_then(Value::as_i64) else {
                            continue;
                        };
                        let mv = parsed.get("voltage_mv").and_then(Value::as_i64).unwrap_or(-1);
                        let t = to_wall(rt);
                        if let Some((lt, lp)) = prev {
                            if t - lt < 60 || (pct == lp && t - lt < 600) {
                                continue;
                            }
                        }
                        if t < cov_start - 7200 {
                            prev = Some((t, pct));
                            continue;
                        }
                        battery.push(json!({ "t": t, "pct": pct, "mv": mv }));
                        prev = Some((t, pct));
                    }
                }
            }
        }
        let coverage: Vec<Value> = live_hours.iter().map(|h| json!({ "h": h })).collect();
        // Per-stream last-seen, oldest first: a stream that died is a clue.
        let mut stmt = match conn.prepare(
            "SELECT tag, MAX(captured_unix), COUNT(*) FROM events WHERE serial = ?1
             GROUP BY tag ORDER BY MAX(captured_unix) ASC LIMIT 15",
        ) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let streams: Vec<Value> = stmt
            .query_map(rusqlite::params![&serial], |r| {
                let tag = r.get::<_, i64>(0)? as u8;
                Ok(json!({
                    "name": oura_protocol::events::event_name(tag),
                    "last_s": r.get::<_, i64>(1)?,
                    "count": r.get::<_, i64>(2)?,
                }))
            })
            .ok()
            .map(|q| q.filter_map(|v| v.ok()).collect())
            .unwrap_or_default();
        // Activity-log entries that mention this ring (sync attempts, failures).
        let log: Vec<Value> = list_activity("", 400)
            .into_iter()
            .filter(|e| e["detail"].as_str().is_some_and(|d| d.contains(&serial)))
            .take(12)
            .collect();
        out.push(json!({
            "serial": serial,
            "name": name,
            "key_source": key_source,
            "note": note,
            "added_s": added_s,
            "last_seen_s": last_seen_s,
            "firmware": firmware,
            "api_version": api_version,
            "mac": mac,
            "last_sync_s": last_sync_s,
            "observed_until_s": observed_until,
            "coverage": coverage,
            "gaps": gaps,
            "battery": battery,
            "streams": streams,
            "log": log,
        }));
    }
    json!({ "rings": out })
}

/// Drain the ring by invoking our own binary's `sync` subcommand (reuses all the
/// BLE + cursor logic). Returns the last stdout line on success.
///
/// BLE scanning on macOS is flaky: the ring advertises only periodically (and not at
/// all for a few seconds after a disconnect), so a single short scan often misses a
/// ring that is right there. We use a longer scan window and retry the transient
/// "no matching ring" miss a couple of times before giving up.
fn run_sync(db: &Path, name: &str, serial: &str, key_file: Option<&Path>) -> Result<String> {
    // Mark the db as mid-write for the whole transfer (all retries): summary
    // rebuilds defer until it ends — one rebuild against final data (warm_summaries)
    // instead of racing the event stream. RAII so a panic/unlock still clears it.
    struct SyncGuard(PathBuf);
    impl Drop for SyncGuard {
        fn drop(&mut self) {
            sync_writing()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&self.0);
        }
    }
    let _sync = SyncGuard(db.to_path_buf());
    sync_writing()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(db.to_path_buf());
    let exe = std::env::current_exe().context("locating oura binary")?;
    let target = describe_ring(db, name, serial);
    log_activity("sync", "start", &format!("sync started ({target})"));
    let started = std::time::Instant::now();
    let mut last = String::from("sync failed");
    for attempt in 0..3 {
        let mut c = Command::new(&exe);
        c.arg("--db").arg(db);
        target_args(&mut c, name, serial);
        c.arg("--scan-timeout")
            .arg("40"); // global flag; wider than the 25 s default
        if let Some(k) = key_file {
            c.arg("--key-file").arg(k);
        }
        c.arg("sync");
        let out = c.output().context("running `oura sync`")?;
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let msg = stdout.lines().last().unwrap_or("synced").trim().to_string();
            log_activity(
                "sync",
                "ok",
                &format!(
                    "sync finished in {}s ({target})\n{}",
                    started.elapsed().as_secs_f64().round(),
                    clip(&stdout, 4000)
                ),
            );
            return Ok(msg);
        }
        let stderr = String::from_utf8_lossy(&out.stderr);
        log_activity(
            "sync",
            "retry",
            &format!(
                "attempt {} failed after {}s ({target})\n{}",
                attempt + 1,
                started.elapsed().as_secs_f64().round(),
                clip(&stderr, 4000)
            ),
        );
        last = stderr
            .lines()
            .last()
            .unwrap_or("sync failed")
            .trim()
            .to_string();
        // only a scan miss is worth retrying; a real error (auth, etc.) is not
        let transient = {
            let l = last.to_lowercase();
            l.contains("no matching")
                || l.contains("not found")
                || l.contains("no device")
                || l.contains("timed out")
                || l.contains("timeout")
        };
        if !transient || attempt == 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    log_activity(
        "sync",
        "failed",
        &format!(
            "sync failed after {}s ({target}): {last}",
            started.elapsed().as_secs_f64().round()
        ),
    );
    Err(anyhow!("{last}"))
}

/// How long one dashboard live heart-rate session streams (the page can stop it early).
const LIVE_HR_SECONDS: u64 = 60;

/// Stream live heart rate as newline-delimited JSON while our own `live-hr`
/// subcommand runs (BLE stays in the CLI, as for sync). Lines: `{"status":"live"}`
/// once connected, `{"bpm":..,"ibi_ms":..}` per beat, then `{"done":true}` or
/// `{"error":".."}`; the body ends with the connection. If the page stops early the
/// child is killed, and the ring leaves live mode on its own within ~20 s.
async fn live_hr(
    sock: &mut TcpStream,
    db: &Path,
    name: &str,
    serial: &str,
    key_file: Option<&Path>,
) -> Result<()> {
    sock.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
    )
    .await?;
    let (mut rd, mut wr) = sock.split();
    let Some(key_file) = key_file else {
        let msg = "live heart rate needs the ring's auth key: start the dashboard with --key-file";
        return send_line(&mut wr, &json!({ "error": msg })).await;
    };
    let exe = std::env::current_exe().context("locating oura binary")?;
    let mut cmd = tokio::process::Command::new(&exe);
    cmd.arg("--db").arg(db);
    if serial.is_empty() {
        cmd.arg("--name").arg(name);
    } else {
        cmd.arg("--expect-serial").arg(serial);
    }
    let mut child = cmd
        .arg("--scan-timeout")
        .arg("120") // a worn ring advertises only in sparse bursts
        .arg("--key-file")
        .arg(key_file)
        .arg("live-hr")
        .arg("--seconds")
        .arg(LIVE_HR_SECONDS.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("running `oura live-hr`")?;
    let stderr = child.stderr.take().context("child stderr")?;
    let stderr_task = tokio::spawn(read_stderr_tail(stderr));
    log_activity(
        "live-hr",
        "start",
        &format!(
            "live heart-rate session started ({} s window)",
            LIVE_HR_SECONDS
        ),
    );
    let mut lines = tokio::io::BufReader::new(child.stdout.take().context("child stdout")?).lines();
    let mut probe = [0u8; 1];
    let session = tokio::time::timeout(std::time::Duration::from_secs(210), async {
        loop {
            tokio::select! {
                line = lines.next_line() => {
                    let Some(line) = line? else { break };
                    let msg = if line.starts_with("Streaming live heart rate") {
                        json!({ "status": "live" })
                    } else if let Some((bpm, ibi_ms)) = parse_beat_line(&line) {
                        json!({ "bpm": bpm, "ibi_ms": ibi_ms })
                    } else { continue };
                    send_line(&mut wr, &msg).await?;
                }
                n = rd.read(&mut probe) => if matches!(n, Ok(0) | Err(_)) {
                    return Err(anyhow!("client disconnected"));
                },
            }
        }
        Ok::<_, anyhow::Error>(child.wait().await?.success())
    })
    .await;
    let ok = match session {
        Ok(Ok(ok)) => ok,
        _ => {
            // Retain the BLE gate until the process is killed and reaped.
            let _ = child.kill().await;
            let _ = child.wait().await;
            stderr_task.abort();
            log_activity("live-hr", "stopped", "live heart-rate session stopped early (client disconnected or timeout)");
            return send_line(
                &mut wr,
                &json!({"error": "Live heart rate stopped or timed out."}),
            )
            .await;
        }
    };
    let stderr = stderr_task.await.unwrap_or_default();
    let last = if ok {
        log_activity("live-hr", "ok", "live heart-rate session finished");
        json!({ "done": true })
    } else {
        let msg = stderr
            .lines()
            .rfind(|l| !l.trim().is_empty())
            .unwrap_or("live heart rate failed");
        log_activity("live-hr", "failed", &format!("live heart-rate session failed: {}", clip(&stderr, 2000)));
        json!({ "error": msg.trim() })
    };
    send_line(&mut wr, &last).await
}

async fn read_stderr_tail(mut reader: impl tokio::io::AsyncRead + Unpin) -> String {
    let mut tail = Vec::new();
    let mut buf = [0u8; 4096];
    while let Ok(n) = reader.read(&mut buf).await {
        if n == 0 {
            break;
        }
        tail.extend_from_slice(&buf[..n]);
        if tail.len() > 8192 {
            tail.drain(..tail.len() - 8192);
        }
    }
    String::from_utf8_lossy(&tail).into_owned()
}

async fn send_line(wr: &mut (impl AsyncWriteExt + Unpin), v: &Value) -> Result<()> {
    wr.write_all(format!("{v}\n").as_bytes()).await?;
    Ok(())
}

/// Parse one `oura live-hr` beat line, e.g. `  81 bpm (IBI 733 ms)`.
fn parse_beat_line(line: &str) -> Option<(u16, u16)> {
    let (bpm, ibi) = line.trim().strip_suffix(" ms)")?.split_once(" bpm (IBI ")?;
    Some((bpm.parse().ok()?, ibi.parse().ok()?))
}

/// Toggle an on-ring feature via our `feature-mode` subcommand (BLE, auth-gated).
/// `feature` is whitelisted; `mode` is off|automatic. Writing a feature mode REQUIRES
/// the auth key, so we fail early (with a clear message) if the dashboard was launched
/// without `--key-file`. Scan misses retry; the ring's "already in that mode" rejection
/// (result `0x20`) is reported as a no-op success rather than an error.
fn run_feature(
    db: &Path,
    name: &str,
    serial: &str,
    key_file: Option<&Path>,
    feature: &str,
    mode: &str,
) -> Result<String> {
    const FEATS: &[&str] = &[
        "daytime_hr",
        "spo2",
        "exercise_hr",
        "real_steps",
        "cva_ppg",
        "ambient",
        "resting_hr",
    ];
    if !FEATS.contains(&feature) {
        return Err(anyhow!("unknown feature"));
    }
    if mode != "off" && mode != "automatic" {
        return Err(anyhow!("invalid mode"));
    }
    let key_file = key_file.ok_or_else(|| {
        anyhow!("can't change ring features: the dashboard was started without --key-file (auth key needed to write feature modes)")
    })?;
    let exe = std::env::current_exe().context("locating oura binary")?;
    let on = if mode == "off" { "off" } else { "on" };
    log_activity("feature", "start", &format!("feature {feature} -> {mode}"));
    let mut last = String::from("toggle failed");
    for attempt in 0..3 {
        let mut c = Command::new(&exe);
        c.arg("--db").arg(db);
        target_args(&mut c, name, serial);
        c.arg("--scan-timeout")
            .arg("40")
            .arg("--key-file")
            .arg(key_file)
            .arg("feature-mode")
            .arg(feature)
            .arg("--mode")
            .arg(mode);
        let out = c.output().context("running `oura feature-mode`")?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        // 0 = off, 1 = automatic — keep feature_modes.json in step with the toggle so
        // the dashboard shows the new state on the next reload, not only after a sync.
        let mode_int = if mode == "off" { 0 } else { 1 };
        if stdout.contains("SUCCESS") {
            write_feature_mode(db, feature, mode_int);
            log_activity("feature", "ok", &format!("{feature} turned {on}"));
            return Ok(format!("{feature} turned {on}"));
        }
        if stdout.contains("result 0x20") {
            // ring refuses to set a mode it's already in → it's already in that state
            write_feature_mode(db, feature, mode_int);
            log_activity("feature", "ok", &format!("{feature} was already {on} (no change)"));
            return Ok(format!("{feature} is already {on} (no change needed)"));
        }
        last = stdout
            .lines()
            .chain(stderr.lines())
            .rfind(|l| !l.trim().is_empty())
            .unwrap_or("toggle failed")
            .trim()
            .to_string();
        let transient = {
            let l = last.to_lowercase();
            l.contains("no matching")
                || l.contains("not found")
                || l.contains("no device")
                || l.contains("timed out")
                || l.contains("timeout")
        };
        if !transient || attempt == 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    log_activity("feature", "failed", &format!("feature {feature} -> {mode} failed: {last}"));
    Err(anyhow!("{last}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_key_requires_serial_and_separates_rings() {
        let args = |serial: &str| {
            vec!["db".to_string(), "--json".into(), "--serial".into(), serial.into()]
        };
        // no --serial: unguarded (returns None)
        assert!(model_key("s.py", &["db".into(), "--json".into()], None).is_none());
        let a = model_key("s.py", &args("AAA"), None).unwrap().0;
        let b = model_key("s.py", &args("BBB"), None).unwrap().0;
        assert_ne!(a, b, "different rings must not share a slot");
        // same invocation -> same key; different inputs -> different key
        assert_eq!(a, model_key("s.py", &args("AAA"), None).unwrap().0);
        assert_ne!(a, model_key("s.py", &args("AAA"), Some(b"win1")).unwrap().0);
        assert_ne!(
            a,
            model_key("other.py", &args("AAA"), None).unwrap().0,
            "different scripts must not share a slot"
        );
    }

    #[test]
    fn concurrent_identical_runs_execute_once_and_share_the_result() {
        let key = format!("dedup-test-{}", std::process::id());
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let tx = Arc::new(tx);

        let spawn = |rx_done: bool| {
            let runs = runs.clone();
            let key = key.clone();
            let tx = tx.clone();
            std::thread::spawn(move || {
                dedup_model_run("t.py", "RING", key, || {
                    runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    tx.send(()).ok(); // owner is inside the run
                    if rx_done {
                        // hold the slot until the joiner has arrived
                        std::thread::sleep(std::time::Duration::from_millis(150));
                    }
                    Some(json!({"ok": true}))
                })
            })
        };
        let owner = spawn(true);
        rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap(); // owner running
        let joiner = spawn(false);
        let a = owner.join().unwrap();
        let b = joiner.join().unwrap();
        assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 1, "model ran twice");
        assert_eq!(a, b);
        assert_eq!(a.unwrap()["ok"], true);

        // the slot is released afterwards: a later call runs again
        let c = dedup_model_run("t.py", "RING", key, || Some(json!({"ok": "second"})));
        assert_eq!(c.unwrap()["ok"], "second");
    }

    #[test]
    fn fallback_fills_only_null_nights_and_keeps_python_stages() {
        // Python staged nights 0 and 2 but emitted null for night 1 (NSSA could not
        // stage it, or no torch). The fallback must fill exactly night 1 from signals
        // and leave the Python results untouched.
        let python = json!([{"source": "nssa", "stages": [1, 2]}, null, {"source": "nssa"}]);
        let signals = json!([{"source": "signals"}, {"source": "signals", "stages": [2, 3]}, {"source": "signals"}]);
        let out = merge_sleep_fallback(Some(python), Some(signals)).unwrap();
        assert_eq!(
            out,
            json!([
                {"source": "nssa", "stages": [1, 2]},
                {"source": "signals", "stages": [2, 3]},
                {"source": "nssa"},
            ])
        );
    }

    #[test]
    fn fallback_takes_whole_signal_array_when_python_absent_or_empty() {
        let signals = json!([{"source": "signals"}]);
        // Python crashed (no torch) → None.
        assert_eq!(
            merge_sleep_fallback(None, Some(signals.clone())),
            Some(signals.clone())
        );
        // Python returned an empty batch.
        assert_eq!(
            merge_sleep_fallback(Some(json!([])), Some(signals.clone())),
            Some(signals)
        );
    }

    #[test]
    fn fallback_length_follows_signals_one_per_range() {
        // Signals are authoritative on night count/order (one per range). A Python
        // batch shorter than signals must not drop the uncovered nights, and a Python
        // batch longer must not append orphans that map to no range.
        let signals = json!([{"n": 0}, {"n": 1}]);
        // Python shorter (only night 0 staged): night 1 comes from signals.
        assert_eq!(
            merge_sleep_fallback(Some(json!([{"py": 0}])), Some(signals.clone())),
            Some(json!([{"py": 0}, {"n": 1}]))
        );
        // Python longer: the orphan third entry is dropped.
        assert_eq!(
            merge_sleep_fallback(
                Some(json!([null, {"py": 1}, {"orphan": true}])),
                Some(signals)
            ),
            Some(json!([{"n": 0}, {"py": 1}]))
        );
    }

    #[test]
    fn fallback_is_a_noop_when_signals_unavailable() {
        // No events to stage from: keep whatever Python returned rather than blanking.
        let python = json!([null]);
        assert_eq!(
            merge_sleep_fallback(Some(python.clone()), None),
            Some(python)
        );
    }

    #[test]
    fn needs_fallback_fires_on_null_missing_or_wrong_length_only() {
        // The gate that was the bug: a fully-staged batch of the right length needs
        // nothing; a null night, a missing batch, or a per-range count mismatch do.
        assert!(!sleep_batch_needs_fallback(Some(&json!([{"s": 1}, {"s": 2}])), 2));
        assert!(sleep_batch_needs_fallback(Some(&json!([{"s": 1}, null])), 2));
        assert!(sleep_batch_needs_fallback(None, 2));
        assert!(sleep_batch_needs_fallback(Some(&json!([{"s": 1}])), 2)); // too short
        assert!(sleep_batch_needs_fallback(Some(&json!([])), 1)); // empty vs one range
        assert!(!sleep_batch_needs_fallback(Some(&json!([])), 0)); // no ranges, no work
    }


    #[tokio::test]
    async fn live_stderr_is_drained_without_unbounded_retention() {
        let (mut writer, reader) = tokio::io::duplex(1024);
        let task = tokio::spawn(async move {
            writer.write_all(&vec![b'x'; 200000]).await.unwrap();
            writer.write_all(b"\nlast error\n").await.unwrap();
        });
        let tail = tokio::time::timeout(std::time::Duration::from_secs(2), read_stderr_tail(reader)).await.unwrap();
        task.await.unwrap();
        assert_eq!(tail.len(), 8192);
        assert!(tail.ends_with("last error\n"));
    }

    #[test]
    fn stalled_runner_is_killed_and_reaped() {
        let mut command = Command::new("python3");
        command.args(["-c", "import time; time.sleep(30)"]);
        assert!(model_output(command, None, std::time::Duration::from_millis(60), "test").is_none());
    }
    #[test]
    fn large_runner_output_does_not_deadlock_and_input_reaches_eof() {
        let mut command = Command::new("python3");
        command.args([
            "-c",
            "import json,sys; print(json.dumps({'input':sys.stdin.read(),'data':'x'*200000}))",
        ]);
        let value =
            model_output(command, Some(b"window"), std::time::Duration::from_secs(5), "test").unwrap();
        assert_eq!(value["input"], "window");
        assert_eq!(value["data"].as_str().unwrap().len(), 200000);
    }
    #[test]
    fn activity_ring_caps_filters_and_clears() {
        // Other tests log concurrently, so assert only on this test's own kind.
        for i in 0..(ACTIVITY_MAX + 25) {
            log_activity("sync", "ok", &format!("ringtest-{i}"));
        }
        let all = list_activity("", ACTIVITY_MAX);
        assert_eq!(all.len(), ACTIVITY_MAX); // ring capped
        let syncs = list_activity("sync", 1000);
        assert!(!syncs.is_empty());
        assert!(syncs.iter().all(|e| e["kind"] == "sync"));
        assert_eq!(syncs[0]["detail"], format!("ringtest-{}", ACTIVITY_MAX + 24)); // newest first
        clear_activity();
        assert!(list_activity("", 10).is_empty());
    }

    #[test]
    fn parses_live_hr_beat_lines() {
        assert_eq!(parse_beat_line("  81 bpm (IBI 733 ms)"), Some((81, 733)));
        assert_eq!(
            parse_beat_line("No beats captured. Make sure the ring is worn."),
            None
        );
    }

    // T2c debug probe (run with --ignored --nocapture): recompute the cache
    // fingerprint for one real bedtime and dump the text for diffing against
    // the Python probe. Personal inputs come from the environment so nothing
    // personal lives in this file; the probe skips when they are absent:
    //   PROBE_DB, PROBE_SERIAL, PROBE_AGE, PROBE_HEIGHT, PROBE_WEIGHT,
    //   PROBE_SEX, PROBE_START, PROBE_END, PROBE_REF, PROBE_OUT
    #[test]
    #[ignore]
    fn probe_nssa_fingerprint() {
        use nssa_plumbing::cache;
        use nssa_plumbing::clock::{build_epochs, Clock};
        use nssa_plumbing::grid::demographics;
        use nssa_plumbing::inputs::collect_nssa_inputs;
        let env = |k: &str| std::env::var(k).ok();
        let (Some(db), Some(serial)) = (env("PROBE_DB"), env("PROBE_SERIAL")) else {
            eprintln!("skip: set PROBE_DB + PROBE_SERIAL to run this probe");
            return;
        };
        let (age, height, weight, sex) = match (
            env("PROBE_AGE").and_then(|v| v.parse().ok()),
            env("PROBE_HEIGHT").and_then(|v| v.parse().ok()),
            env("PROBE_WEIGHT").and_then(|v| v.parse().ok()),
            env("PROBE_SEX"),
        ) {
            (Some(a), Some(h), Some(w), Some(s)) => (a, h, w, s),
            _ => {
                eprintln!("skip: PROBE_AGE/HEIGHT/WEIGHT/SEX missing");
                return;
            }
        };
        let win = ["PROBE_START", "PROBE_END", "PROBE_REF"].map(|k| {
            env(k).and_then(|v| v.parse::<i64>().ok())
        });
        let [Some(start), Some(end), Some(reference)] = win else {
            eprintln!("skip: PROBE_START/END/REF missing");
            return;
        };
        let db = std::path::Path::new(&db);
        let (rows, clock_rows) = load_nssa_rows(db, &serial).expect("rows");
        let clock_refs: Vec<(i64, u8, Option<&str>, i64)> = clock_rows
            .iter()
            .map(|(ds, tag, js, cu)| (*ds, *tag, js.as_deref(), *cu))
            .collect();
        let clock = Clock::new(build_epochs(&clock_refs));
        let demo3 = demographics(age, height, weight, &sex);
        let inputs = collect_nssa_inputs(&rows, start, end, reference, &clock);
        let text = cache::fingerprint_text(Some(&serial), &demo3, &inputs);
        if let Some(out) = env("PROBE_OUT") {
            std::fs::write(out, &text).unwrap();
        }
        let key = cache::fingerprint(Some(&serial), &demo3, &inputs);
        let nc = cache::Cache::load(db, Some(&serial)).expect("cache");
        println!("rs key: {key} in cache: {}", nc.has_night(&key));
        println!(
            "rs inputs: acm {} ibi {} temp {}",
            inputs.acm.len(),
            inputs.ibi.len(),
            inputs.temp.len()
        );
    }
}
