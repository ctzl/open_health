//! Port of `tools/run_stress_events.py` — local stress/rest event detection.
//! Golden vector: tests/vectors/stress_golden.json. See docs/python-to-rust-plan.md T1b.
//!
//! Chain (same as the Python oracle):
//!   1. `stress_daytime_sensing_1_1_0` (the DSA series) still runs in Python:
//!      we spawn `tools/run_stress_model.py --json` exactly like
//!      `run_stress_events.py:stress_series()` does. That model's data pipeline
//!      + NN stay behind the subprocess for now; `--dsa-json <file>` feeds a
//!      pre-captured series instead, which keeps this path torch-free.
//!   2. `astd_event_detection_0_1_1` is hand-ported below in pure Rust
//!      (`astd_event_detection`). Its rules were recovered from the model's
//!      frozen TorchScript graph (the original astd_event_detection.py line
//!      numbers are kept as comments), so no tch/libtorch is needed here.
//!   3. Ring-decisecond dating mirrors `tools/epoch_time.py`
//!      (`build_epochs`/`make_unix_s`, the captured_unix-less path the oracle
//!      uses). `crates/oura-summary/src/ring_time.rs` holds a Rust equivalent,
//!      but this crate may not depend on it and that variant requires a
//!      captured_unix and differs in tie-breaking, so the oracle's exact
//!      semantics live here.
//!
//! Float discipline: the oracle hands the model float32 tensors, so every value
//! and timestamp comparison in the segmenter runs in `f32`, and reported
//! durations are `f32` results widened to `f64` before Python-style rounding —
//! mirroring `torch.tensor(..., dtype=torch.float32)` / `aten::to(_, 6)`.

use anyhow::{bail, Context, Result};
use rusqlite::Connection;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const USAGE: &str = "usage: oura-models stress [DB] [--tz N] [--serial S] [--json] \
                    [--python PATH] [--dsa-json FILE] [--repo PATH]";

#[derive(Default)]
struct Args {
    db: Option<String>,
    tz: i64,
    serial: Option<String>,
    json: bool,
    python: Option<String>,
    dsa_json: Option<String>,
    repo: Option<String>,
}

fn parse_args(args: &[String]) -> Result<Args> {
    let mut a = Args {
        tz: 1, // same default as the Python runner
        ..Default::default()
    };
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].clone();
        let mut take = || -> Result<String> {
            i += 1;
            args.get(i)
                .cloned()
                .with_context(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--json" => a.json = true,
            "--tz" => {
                let v = take()?;
                a.tz = v
                    .parse::<i64>()
                    .with_context(|| format!("--tz expects an hour offset, got {v}"))?;
            }
            "--serial" => a.serial = Some(take()?),
            "--python" => a.python = Some(take()?),
            "--dsa-json" => a.dsa_json = Some(take()?),
            "--repo" => a.repo = Some(take()?),
            "-h" | "--help" => bail!("{USAGE}"),
            s if s.starts_with('-') => bail!("unknown option {s}\n{USAGE}"),
            s => {
                if a.db.is_some() {
                    bail!("unexpected argument {s}\n{USAGE}");
                }
                a.db = Some(s.to_string());
            }
        }
        i += 1;
    }
    Ok(a)
}

/// `tools/_common.py:resolve_db` — explicit arg wins, else the first existing
/// default, else `repo/oura.db`; error when the resolved path is missing.
fn resolve_db(arg: Option<&str>, repo: &Path) -> Result<PathBuf> {
    let db = match arg {
        Some(a) => PathBuf::from(a),
        None => {
            let cwd = std::env::current_dir().unwrap_or_else(|_| repo.to_path_buf());
            [
                cwd.join("oura.db"),
                repo.join("oura.db"),
                repo.join("captures").join("ring5.db"),
            ]
            .into_iter()
            .find(|c| c.exists())
            .unwrap_or_else(|| repo.join("oura.db"))
        }
    };
    if !db.exists() {
        bail!(
            "error: database not found: {} (run `oura sync` first)",
            db.display()
        );
    }
    Ok(db)
}

/// Everything the CLI would print, computed: the `--json` payload plus the
/// counts the human-readable report needs. This is the library entry point the
/// tests drive; `run` below is the CLI front end over it.
pub struct StressOutput {
    pub json: String,
    pub baseline: Baseline,
    pub series_len: usize,
    pub n_days: usize,
    pub results: Vec<DayResult>,
}

pub fn estimate(args: &[String]) -> Result<StressOutput> {
    let args = parse_args(args)?;
    let repo = match &args.repo {
        Some(r) => PathBuf::from(r),
        None => std::env::current_dir().context("resolve repo root (pass --repo)")?,
    };
    let (series, baseline) = match &args.dsa_json {
        // Pre-captured run_stress_model.py output: no Python, no torch.
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read DSA series {}", Path::new(path).display()))?;
            dsa_series(
                &serde_json::from_str::<Value>(&text)
                    .with_context(|| format!("parse DSA series {}", Path::new(path).display()))?,
            )
        }
        None => {
            let db = resolve_db(args.db.as_deref(), &repo)?;
            dsa_series(&run_stress_model(
                &repo,
                &db,
                args.serial.as_deref(),
                args.python.as_deref(),
            )?)
        }
    };
    if series.is_empty() {
        bail!("no daytime-stress samples — enable `daytime_hr` and sync?");
    }

    // Date the ring timestamps (epoch_time.build_epochs + make_unix_s).
    let db = resolve_db(args.db.as_deref(), &repo)?;
    let clock = RingClock::from_db(&db, args.serial.as_deref())?;

    let mut days: BTreeMap<i64, Vec<(i64, f64)>> = BTreeMap::new();
    for (ts_ms, v) in &series {
        let ds = ts_ms.div_euclid(100);
        let Some(u) = clock.unix_s(ds) else { continue };
        let day = trunc_to_i64(u).div_euclid(86_400);
        days.entry(day).or_default().push((*ts_ms, *v));
    }

    let n_days = days.len();
    let mut results: Vec<DayResult> = Vec::new();
    for (day, pts) in days {
        if pts.len() < MIN_N_BINS {
            continue; // the model's own validator minimum
        }
        let base = pts[0].0;
        // torch.tensor([...], dtype=torch.float32): day-relative ms + scaled values.
        let ts: Vec<f32> = pts.iter().map(|(t, _)| (*t - base) as f32).collect();
        let vals: Vec<f32> = pts.iter().map(|(_, v)| *v as f32).collect();
        match astd_event_detection(&vals, &ts) {
            Err(e) => {
                let msg = e.message();
                // The shipped model indexes an empty tensor when a day has zero
                // events; that is a model-side bug, not bad input (see the
                // oracle docstring) — treat it as "no events that day".
                if msg.contains("out of bounds") {
                    results.push(DayResult::bug(day, pts.len()));
                } else {
                    results.push(DayResult::error(day, msg));
                }
            }
            Ok(out) => {
                let mut events = Vec::new();
                for i in 0..out.types.len() {
                    let start_ms = base + out.starts_ms[i];
                    let end_ms = base + out.ends_ms[i];
                    let u0 = clock.unix_s(start_ms.div_euclid(100));
                    let u1 = clock.unix_s(end_ms.div_euclid(100));
                    events.push(Event {
                        stressed: out.types[i] == TYPE_STRESSED,
                        start_local: fmt_local(u0, args.tz),
                        end_local: fmt_local(u1, args.tz),
                        start_unix: u0,
                        end_unix: u1,
                        duration_min: py_round1(out.durations_min[i] as f64),
                    });
                }
                results.push(DayResult {
                    day,
                    n_samples: pts.len(),
                    n_stressed: Some(out.n_stressed),
                    n_restored: Some(out.n_restored),
                    total_stressed_min: Some(py_round1(out.total_stressed_min as f64)),
                    total_restored_min: Some(py_round1(out.total_restored_min as f64)),
                    error: None,
                    events,
                });
            }
        }
    }

    Ok(StressOutput {
        json: render_json(&baseline, &results),
        baseline,
        series_len: series.len(),
        n_days,
        results,
    })
}

/// CLI front end over `estimate`: `--json` prints the payload, otherwise the
/// per-day report the oracle prints.
pub fn run(args: &[String]) -> Result<()> {
    let want_json = parse_args(args)?.json;
    if want_json {
        println!("{}", json_for_args(args)?);
        return Ok(());
    }
    let out = estimate(args)?;

    println!("Stress/rest events (astd_event_detection_0_1_1 on local DSA series)");
    println!(
        "  inputs: {} daytime-stress samples over {} day(s)",
        out.series_len, out.n_days
    );
    for res in &out.results {
        let date = iso_from_days(res.day);
        if let Some(err) = &res.error {
            println!("  {date}: model error: {err}");
            continue;
        }
        // mirrors `res.get('n_stressed', 0)` / `res.get('total_stressed_min', 0)`:
        // a day the model bug swallowed prints bare integer zeros.
        let ns = res
            .n_stressed
            .map(|v| v.to_string())
            .unwrap_or_else(|| "0".to_string());
        let nr = res
            .n_restored
            .map(|v| v.to_string())
            .unwrap_or_else(|| "0".to_string());
        let tot_s = res
            .total_stressed_min
            .map(json_f64)
            .unwrap_or_else(|| "0".to_string());
        let tot_r = res
            .total_restored_min
            .map(json_f64)
            .unwrap_or_else(|| "0".to_string());
        println!(
            "  {date}: {} samples | stressed {ns} ({tot_s} min) | restored {nr} ({tot_r} min)",
            res.n_samples,
        );
        for e in &res.events {
            println!(
                "      [{:<8}] {} -> {}  {} min",
                if e.stressed { "stressed" } else { "restored" },
                e.start_local,
                e.end_local,
                json_f64(e.duration_min)
            );
        }
    }
    Ok(())
}

/// The JSON `run` would print, as a string — the in-process entry point for the
/// dashboard's RustRunner (same argv contract as the CLI, golden-tested). The
/// DSA series it consumes still comes from `tools/run_stress_model.py` (see
/// module header), exactly as in the golden runs.
pub fn json_for_args(argv: &[String]) -> Result<String> {
    Ok(estimate(argv)?.json)
}

// ---------------------------------------------------------------------------
// DSA series (stress_daytime_sensing stays in Python for now)
// ---------------------------------------------------------------------------

/// Spawn `tools/run_stress_model.py --json` the way the oracle's
/// `stress_series()` does, and return its parsed output.
fn run_stress_model(
    repo: &Path,
    db: &Path,
    serial: Option<&str>,
    python: Option<&str>,
) -> Result<Value> {
    let script = repo.join("tools").join("run_stress_model.py");
    if !script.exists() {
        bail!(
            "error: model script not found: {} (pass --repo)",
            script.display()
        );
    }
    let python = find_python(repo, python)?;
    let mut cmd = Command::new(&python);
    cmd.arg(&script).arg(db).arg("--json");
    if let Some(s) = serial {
        cmd.arg("--serial").arg(s);
    }
    let out = cmd
        .output()
        .map_err(|e| anyhow::anyhow!("run {} with {python}: {e}", script.display()))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let msg = String::from_utf8_lossy(&out.stdout);
        bail!("{}", if err.trim().is_empty() { msg } else { err }.trim());
    }
    serde_json::from_slice(&out.stdout).context("parse run_stress_model.py JSON output")
}

/// `.venv` next to the repo, then `OURA_PYTHON`, then the bare interpreter.
fn find_python(repo: &Path, override_: Option<&str>) -> Result<String> {
    if let Some(p) = override_ {
        return Ok(p.to_string());
    }
    for cand in [
        repo.join(".venv").join("Scripts").join("python.exe"),
        repo.join(".venv").join("bin").join("python3"),
        repo.join(".venv").join("bin").join("python"),
    ] {
        if cand.exists() {
            return Ok(cand.to_string_lossy().into_owned());
        }
    }
    if let Ok(p) = std::env::var("OURA_PYTHON") {
        if !p.is_empty() {
            return Ok(p);
        }
    }
    Ok("python".to_string())
}

/// `(ring_ms, scaled_value)` sorted ascending, last write wins per timestamp
/// (the model's validator needs strictly increasing timestamps), plus the two
/// baselines the oracle echoes back as `stress_baseline`.
fn dsa_series(data: &Value) -> (Vec<(i64, f64)>, Baseline) {
    let mut dedup: BTreeMap<i64, f64> = BTreeMap::new();
    if let Some(samples) = data["samples"].as_array() {
        for s in samples {
            let ts = s["timestamp_ds"].as_i64().map(|ds| ds * 100);
            let v = s["intensity_scaled"].as_f64();
            if let (Some(ts), Some(v)) = (ts, v) {
                dedup.insert(ts, v);
            }
        }
    }
    let baseline = Baseline {
        night_hrv_baseline: data["night_hrv_baseline"].as_f64().unwrap_or(0.0),
        dhrv_baseline: data["dhrv_baseline"].as_f64().unwrap_or(0.0),
    };
    (dedup.into_iter().collect(), baseline)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Baseline {
    pub night_hrv_baseline: f64,
    pub dhrv_baseline: f64,
}

// ---------------------------------------------------------------------------
// astd_event_detection_0_1_1, hand-ported
//
// Recovered from the model's frozen TorchScript graph; `astd:<line>` comments
// are the astd_event_detection.py line numbers baked into that graph.
// Constants read out of the archive: bin_width_ms 900000, min_n_bins 4,
// min/max_window_delta_ms 3.3e6/3.9e6, allowed_merge_gap_ms 1.8e6,
// extreme/relaxed thresholds 0.5/0.4.
// ---------------------------------------------------------------------------

pub const BIN_WIDTH_MS: i64 = 900_000;
pub const MIN_N_BINS: usize = 4;
pub const MIN_WINDOW_DELTA_MS: i64 = 3_300_000;
pub const MAX_WINDOW_DELTA_MS: i64 = 3_900_000;
pub const ALLOWED_MERGE_GAP_MS: i64 = 1_800_000;
pub const EXTREME_THRESHOLD: f32 = 0.5;
pub const RELAXED_THRESHOLD: f32 = 0.4;

pub const TYPE_STRESSED: i64 = -1;
pub const TYPE_RESTORED: i64 = 1;

/// Per-sample classification (`preprocessor.classify`, astd:214-218).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Classes {
    pub finite: bool,
    pub fully_stressed: bool,
    pub borderline_stressed: bool,
    pub fully_restored: bool,
    pub borderline_restored: bool,
}

pub fn classify(v: f32) -> Classes {
    Classes {
        finite: !v.is_nan(),
        fully_stressed: v <= -EXTREME_THRESHOLD,
        borderline_stressed: v > -EXTREME_THRESHOLD && v <= -RELAXED_THRESHOLD,
        fully_restored: v >= EXTREME_THRESHOLD,
        borderline_restored: v >= RELAXED_THRESHOLD && v < EXTREME_THRESHOLD,
    }
}

/// The eight-tuple `forward` returns. `starts_ms`/`ends_ms` are day-relative
/// and already shifted the way the model shifts them (`start - bin_width_ms`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AstdOutput {
    pub n_stressed: i64,
    pub n_restored: i64,
    pub total_stressed_min: f32,
    pub total_restored_min: f32,
    pub types: Vec<i64>,
    pub starts_ms: Vec<i64>,
    pub ends_ms: Vec<i64>,
    pub durations_min: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AstdError {
    /// Validator rejection (codes 1-4 of `error_code_to_message`).
    Validator(u8),
    /// astd:537 — opposite-type events ended up on top of each other.
    OppositeTypeOverlap,
    /// The graph pushes timestamps through a C `int` (`int(torch.item(ts))` at
    /// astd:367-368 for window ends, astd:494 for the ts->index map), so a
    /// day-relative timestamp outside int32 aborts the whole day. Absolute ring
    /// ms pass 2^31 after ~25 days, which is why the oracle runs one day at a
    /// time — but a day whose samples mix two rings (a ring swap) still does.
    IntOverflow,
}

impl AstdError {
    /// The exact string the oracle records: `str(exception).splitlines()[-1]`.
    /// Graph-raised errors go through `ops.prim.RaiseException(msg,
    /// "builtins.Exception")`, so that last traceback line carries the class
    /// prefix; the int32 cast failure is a real `RuntimeError` instead.
    pub fn message(&self) -> String {
        match self {
            // validator.error_code_to_message, inlined in the graph
            AstdError::Validator(1) => raised("1, Condition, insufficient data: n_bins < 4."),
            AstdError::Validator(2) => raised(
                "2, Error, length mismatch: dsa_values and dsa_timestamps_ms have different lengths.",
            ),
            AstdError::Validator(3) => raised(
                "3, Error, dsa_timestamps_ms is not strictly monotonically increasing.",
            ),
            AstdError::Validator(4) => {
                raised("4, Error, finite dsa_values outside [-1, +1] detected.")
            }
            AstdError::Validator(c) => raised(&format!("{c}, unknown error code.")),
            AstdError::OppositeTypeOverlap => raised("opposite-type event overlap on time axis"),
            AstdError::IntOverflow => {
                "RuntimeError: value cannot be converted to type int without overflow".into()
            }
        }
    }
}

fn raised(msg: &str) -> String {
    format!("builtins.Exception: {msg}")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Ev {
    id: i64,
    start: i64,
    end: i64,
}

/// Whether `int(torch.item(t))` in the graph survives: the value is float32,
/// the cast target is a C `int`, so the comparison happens on the widened
/// float (2147483647.0 as float32 *is* 2147483648.0 and does overflow).
fn ts_fits_c_int(t: f32) -> bool {
    let d = t as f64;
    d <= i32::MAX as f64 && d >= i32::MIN as f64
}

/// `validator.validate` (astd:137-152).
fn validate(values: &[f32], ts: &[f32]) -> Result<(), AstdError> {
    if values.len() < MIN_N_BINS {
        return Err(AstdError::Validator(1)); // astd:137
    }
    if ts.len() != values.len() {
        return Err(AstdError::Validator(2)); // astd:140
    }
    // astd:145-146: the model diffs `ts[1:] - ts[:-1]` and rejects when any
    // step is not strictly positive.
    if ts.len() > 1 && ts.windows(2).any(|w| w[1] - w[0] <= 0.0) {
        return Err(AstdError::Validator(3)); // astd:145-146
    }
    // astd:149-152: "finite" there means `~isnan`, so +/-inf are checked too
    // (an inf input really does come back as code 4 from the shipped model).
    if values.iter().any(|v| !v.is_nan() && (*v < -1.0 || *v > 1.0)) {
        return Err(AstdError::Validator(4));
    }
    Ok(())
}

/// `processor.segment` (astd:352-541) plus the `forward` arithmetic
/// (astd:639-649). `values`/`ts_ms` are the float32 tensors the oracle builds.
pub fn astd_event_detection(values: &[f32], ts_ms: &[f32]) -> Result<AstdOutput, AstdError> {
    validate(values, ts_ms)?;
    let cls: Vec<Classes> = values.iter().map(|v| classify(*v)).collect();
    // segment reads timestamps back as ints (aten::item -> aten::Int): truncated
    // toward zero from their float32 representation.
    let ts: Vec<i64> = ts_ms.iter().map(|t| t.trunc() as i64).collect();
    let n = ts.len();

    let mut found: Vec<Ev> = Vec::new();
    if n >= MIN_N_BINS {
        // astd:359-383 — every window of N=4 consecutive bins, stressed first
        // then restored (the graph emits two identical loops; only masks differ).
        let flags: [(Vec<bool>, Vec<bool>, i64); 2] = [
            (
                cls.iter().map(|c| c.fully_stressed).collect(),
                cls.iter().map(|c| c.borderline_stressed).collect(),
                TYPE_STRESSED,
            ),
            (
                cls.iter().map(|c| c.fully_restored).collect(),
                cls.iter().map(|c| c.borderline_restored).collect(),
                TYPE_RESTORED,
            ),
        ];
        for (fully, borderline, id) in &flags {
            for j in 0..=(n - MIN_N_BINS) {
                let j_last = j + MIN_N_BINS - 1;
                if !cls[j].finite || !cls[j_last].finite {
                    continue; // astd:362,364
                }
                // astd:367-368: `first_ts = int(torch.item(ts[i]))` and the same
                // for the window end — a C `int` cast, before any span check.
                if !ts_fits_c_int(ts_ms[j]) || !ts_fits_c_int(ts_ms[j_last]) {
                    return Err(AstdError::IntOverflow);
                }
                let (first_ts, last_ts) = (ts[j], ts[j_last]);
                let covered = last_ts - first_ts + BIN_WIDTH_MS; // astd:369
                if covered < MIN_WINDOW_DELTA_MS || covered > MAX_WINDOW_DELTA_MS {
                    continue; // astd:370,372
                }
                let win = j..=j_last;
                let n_finite = win.clone().filter(|i| cls[*i].finite).count();
                let n_fully = win.clone().filter(|i| fully[*i]).count();
                let n_borderline = win.clone().filter(|i| borderline[*i]).count();
                let n_nan = MIN_N_BINS - n_finite; // astd:312
                let n_other = n_finite - n_fully - n_borderline; // astd:313
                // astd:331-335: both end bins must be fully in-zone, no bin may
                // be out-of-zone, and at most one bin may be nan/borderline.
                let ok = fully[j]
                    && fully[j_last]
                    && n_other == 0
                    && (n_nan + n_borderline) <= 1
                    && n_fully >= 1; // astd:316
                if !ok {
                    continue;
                }
                found.push(Ev {
                    id: *id,
                    start: first_ts,
                    end: last_ts,
                });
            }
        }
    }

    // astd:463-464: nothing found -> the model returns empty tensors.
    if found.is_empty() {
        return Ok(AstdOutput::default());
    }

    // astd:471-483: insertion sort by start (stable, so equal starts keep the
    // stressed-before-restored order the two scan loops produced).
    insertion_sort_by_start(&mut found);

    // astd:404-417 (first of two merge passes).
    let mut merged = merge_same_type(&found);

    // astd:509-521: grow an event across a bin boundary when the neighbouring
    // sample is timestamped exactly one bin away and is fully in the same zone.
    // astd:494: the ts->index map keys are C `int`s too, so once any window
    // exists every timestamp in the day has to fit, even ones no window used.
    if ts_ms.iter().any(|t| !ts_fits_c_int(*t)) {
        return Err(AstdError::IntOverflow);
    }
    let mut ts_to_idx: HashMap<i64, usize> = HashMap::new();
    for (i, t) in ts.iter().enumerate() {
        ts_to_idx.insert(*t, i); // a later duplicate wins, as in Python
    }
    for ev in merged.iter_mut() {
        let zone: Vec<bool> = if ev.id == TYPE_STRESSED {
            cls.iter().map(|c| c.fully_stressed).collect()
        } else {
            cls.iter().map(|c| c.fully_restored).collect()
        };
        let prev_ts = ev.start - BIN_WIDTH_MS;
        if let Some(&i) = ts_to_idx.get(&prev_ts) {
            if zone[i] {
                ev.start = prev_ts; // astd:517
            }
        }
        let next_ts = ev.end + BIN_WIDTH_MS;
        if let Some(&i) = ts_to_idx.get(&next_ts) {
            if zone[i] {
                ev.end = next_ts; // astd:521
            }
        }
    }

    // astd:404-417 again: boundary growth can make neighbours mergeable.
    let merged = merge_same_type(&merged);

    // astd:534-537: opposite-type events must not overlap on the time axis.
    for k in 1..merged.len() {
        if merged[k].id != merged[k - 1].id && merged[k].start <= merged[k - 1].end {
            return Err(AstdError::OppositeTypeOverlap);
        }
    }

    // forward, astd:639-649.
    let mut out = AstdOutput::default();
    for ev in &merged {
        let start_ms = ev.start - BIN_WIDTH_MS; // astd:639
        let dur_ms = ev.end - start_ms; // astd:641 (aten::to(_, 6) = float32)
        let dur = (dur_ms as f32) / 60_000.0f32;
        out.types.push(ev.id);
        out.starts_ms.push(start_ms);
        out.ends_ms.push(ev.end);
        out.durations_min.push(dur);
        if ev.id == TYPE_STRESSED {
            out.n_stressed += 1;
            out.total_stressed_min += dur; // f32 accumulation, as aten::sum does
        } else {
            out.n_restored += 1;
            out.total_restored_min += dur;
        }
    }
    Ok(out)
}

fn insertion_sort_by_start(v: &mut [Ev]) {
    for i in 1..v.len() {
        let cur = v[i];
        let mut jj = i as i64 - 1;
        while jj >= 0 && v[jj as usize].start > cur.start {
            v[jj as usize + 1] = v[jj as usize];
            jj -= 1;
        }
        v[(jj + 1) as usize] = cur;
    }
}

/// astd:404-417: same-type events whose gap is <= `allowed_merge_gap_ms`
/// collapse into the earlier one, extending its end.
fn merge_same_type(sorted: &[Ev]) -> Vec<Ev> {
    let mut m: Vec<Ev> = Vec::new();
    for cur in sorted {
        if let Some(last) = m.last_mut() {
            if last.id == cur.id && cur.start - last.end <= ALLOWED_MERGE_GAP_MS {
                if cur.end > last.end {
                    last.end = cur.end; // astd:413
                }
                continue;
            }
        }
        m.push(*cur);
    }
    m
}

// ---------------------------------------------------------------------------
// Ring deciseconds -> UTC (tools/epoch_time.py, captured_unix-less path)
// ---------------------------------------------------------------------------

const EPOCH_RESET_SLACK_DS: i64 = 6 * 3600 * 10;
const ANCHOR_AGREEMENT_S: f64 = 30.0 * 60.0;
const ANCHOR_AGREEMENT_FRACTION: f64 = 0.02;

#[derive(Clone, Debug)]
struct Epoch {
    min_ds: i64,
    max_ds: i64,
    fallback_unix: i64,
    anchors: Vec<(i64, i64, usize)>, // (ring ds, unix, insertion index), sorted
    boots: Vec<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bracket {
    Consistent,
    Stalled { before: (i64, i64), after: (i64, i64) },
    Erratic,
}

/// Boot epochs + time anchors, built exactly like `epoch_time.build_epochs` over
/// `(ring_timestamp, tag, decoded_json, captured_unix)` rows in
/// `(captured_unix, id)` order.
pub struct RingClock {
    epochs: Vec<Epoch>,
}

impl RingClock {
    pub fn from_rows(rows: &[(i64, i64, Option<String>, i64)]) -> RingClock {
        let mut epochs: Vec<Epoch> = Vec::new();
        let mut anchor_seq = 0usize;
        for (ds, tag, json, cu) in rows {
            match epochs.last_mut() {
                // Rows stay in (captured_unix, id) order: thousands of rows share
                // one capture second, and sorting them by ds would erase reboot
                // jumps.
                Some(e) if *ds >= e.max_ds - EPOCH_RESET_SLACK_DS => {
                    if *ds >= e.max_ds {
                        e.max_ds = *ds;
                        e.fallback_unix = *cu;
                    }
                    e.min_ds = e.min_ds.min(*ds);
                }
                _ => epochs.push(Epoch {
                    min_ds: *ds,
                    max_ds: *ds,
                    fallback_unix: *cu,
                    anchors: Vec::new(),
                    boots: Vec::new(),
                }),
            }
            if *tag == 0x41 {
                epochs.last_mut().unwrap().boots.push(*ds);
            }
            if *tag == 0x42 || *tag == 0x85 {
                let Some(js) = json.as_deref() else { continue };
                let Ok(value) = serde_json::from_str::<Value>(js) else { continue };
                let Some(unix) = value["unix_time"].as_i64() else { continue };
                epochs.last_mut().unwrap().anchors.push((*ds, unix, anchor_seq));
                anchor_seq += 1;
            }
        }
        for e in epochs.iter_mut() {
            e.anchors.sort(); // (ds, unix, insertion) — Python's tuple sort
        }
        RingClock { epochs }
    }

    pub fn from_db(db: &Path, serial: Option<&str>) -> Result<RingClock> {
        let con = Connection::open_with_flags(
            db,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("open {}", db.display()))?;
        let sql = format!(
            "SELECT ring_timestamp, tag, decoded_json, captured_unix FROM events \
             WHERE captured_unix IS NOT NULL{} ORDER BY captured_unix, id",
            if serial.is_some() {
                " AND serial = ?1"
            } else {
                ""
            }
        );
        let mut stmt = con.prepare(&sql).context("read events for epoch dating")?;
        let args: Vec<rusqlite::types::Value> = serial
            .map(|s| vec![rusqlite::types::Value::Text(s.to_string())])
            .unwrap_or_default();
        let mut out: Vec<(i64, i64, Option<String>, i64)> = Vec::new();
        {
            let mut rows = stmt
                .query_map(rusqlite::params_from_iter(args.iter()), |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                })
                .context("query events")?;
            while let Some(row) = rows.next() {
                out.push(row.context("read event row")?);
            }
        }
        Ok(RingClock::from_rows(&out))
    }

    /// `epoch_time._pick_epoch` with `captured_unix = None`: the boot whose ds
    /// span is narrowest wins, first index on ties.
    fn pick_epoch(&self, ds: i64) -> usize {
        let candidates: Vec<usize> = (0..self.epochs.len())
            .filter(|&i| {
                let e = &self.epochs[i];
                e.min_ds - EPOCH_RESET_SLACK_DS <= ds && ds <= e.max_ds + EPOCH_RESET_SLACK_DS
            })
            .collect();
        if candidates.is_empty() {
            return self.epochs.len() - 1;
        }
        let mut best = candidates[0];
        for &i in &candidates[1..] {
            if (self.epochs[i].max_ds - self.epochs[i].min_ds)
                < (self.epochs[best].max_ds - self.epochs[best].min_ds)
            {
                best = i;
            }
        }
        best
    }

    /// `epoch_time._nearest`: ties resolve to the earliest insertion index.
    fn nearest(anchors: &[(i64, i64, usize)], ds: i64) -> (i64, i64) {
        let idx = anchors.partition_point(|a| a.0 < ds);
        if idx == 0 {
            return (anchors[0].0, anchors[0].1);
        }
        if idx >= anchors.len() {
            let a = anchors[anchors.len() - 1];
            return (a.0, a.1);
        }
        let (before, after) = (anchors[idx - 1], anchors[idx]);
        if (ds - before.0, before.2) <= (after.0 - ds, after.2) {
            (before.0, before.1)
        } else {
            (after.0, after.1)
        }
    }

    fn bracket(anchors: &[(i64, i64, usize)], ds: i64) -> Bracket {
        let idx = anchors.partition_point(|a| a.0 < ds);
        if idx == 0 || idx >= anchors.len() {
            return Bracket::Consistent;
        }
        let (prev, nxt) = (anchors[idx - 1], anchors[idx]);
        if nxt.0 == ds {
            return Bracket::Consistent;
        }
        let wall_s = (nxt.1 - prev.1) as f64;
        let counter_s = (nxt.0 - prev.0) as f64 / 10.0;
        let tolerance = ANCHOR_AGREEMENT_S.max(counter_s * ANCHOR_AGREEMENT_FRACTION);
        if wall_s < counter_s - tolerance {
            Bracket::Erratic
        } else if wall_s > counter_s + tolerance {
            Bracket::Stalled {
                before: (prev.0, prev.1),
                after: (nxt.0, nxt.1),
            }
        } else {
            Bracket::Consistent
        }
    }

    /// `epoch_time.make_unix_s(epochs)(ds)` — `None` only for the
    /// ambiguous-reboot-stall case, exactly like the oracle.
    pub fn unix_s(&self, ds: i64) -> Option<f64> {
        if self.epochs.is_empty() {
            return None;
        }
        let e = &self.epochs[self.pick_epoch(ds)];
        if !e.anchors.is_empty() {
            let (anchor_ds, anchor_unix) = Self::nearest(&e.anchors, ds);
            let predicted = anchor_unix as f64 + (ds - anchor_ds) as f64 / 10.0;
            return match Self::bracket(&e.anchors, ds) {
                // The oracle keeps the erratic prediction as-is (it never calls
                // is_dated), so the day grouping sees it too.
                Bracket::Erratic | Bracket::Consistent => Some(predicted),
                Bracket::Stalled { before, after } => {
                    let late = after.1 as f64 - (after.0 - ds) as f64 / 10.0;
                    let early = before.1 as f64 + (ds - before.0) as f64 / 10.0;
                    let boots: Vec<i64> = e
                        .boots
                        .iter()
                        .copied()
                        .filter(|b| *b > before.0 && *b <= after.0)
                        .collect();
                    if !boots.is_empty() {
                        let first = *boots.iter().min().unwrap();
                        let last = *boots.iter().max().unwrap();
                        if first <= ds && ds < last {
                            return None; // lost time cannot be assigned to one reboot
                        }
                        return Some(if ds >= last { late } else { early });
                    }
                    Some(late) // captured_unix is None in the oracle
                }
            };
        }
        Some(e.fallback_unix as f64 - (e.max_ds - ds) as f64 / 10.0)
    }
}

// ---------------------------------------------------------------------------
// Output rendering — byte-compatible with `json.dumps(..., indent=None)`
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub stressed: bool,
    pub start_local: String,
    pub end_local: String,
    pub start_unix: Option<f64>,
    pub end_unix: Option<f64>,
    pub duration_min: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DayResult {
    pub day: i64,
    pub n_samples: usize,
    pub n_stressed: Option<i64>,
    pub n_restored: Option<i64>,
    pub total_stressed_min: Option<f64>,
    pub total_restored_min: Option<f64>,
    pub error: Option<String>,
    pub events: Vec<Event>,
}

impl DayResult {
    fn error(day: i64, msg: String) -> DayResult {
        DayResult {
            day,
            n_samples: 0,
            n_stressed: None,
            n_restored: None,
            total_stressed_min: None,
            total_restored_min: None,
            error: Some(msg),
            events: Vec::new(),
        }
    }

    /// A day the model-side zero-events bug swallowed: the oracle emits only
    /// day/n_samples/events for those.
    fn bug(day: i64, n_samples: usize) -> DayResult {
        DayResult {
            day,
            n_samples,
            n_stressed: None,
            n_restored: None,
            total_stressed_min: None,
            total_restored_min: None,
            error: None,
            events: Vec::new(),
        }
    }
}

fn render_json(baseline: &Baseline, days: &[DayResult]) -> String {
    let mut s = String::new();
    s.push_str("{\"stress_baseline\": {\"night_hrv_baseline\": ");
    s.push_str(&json_f64(baseline.night_hrv_baseline));
    s.push_str(", \"dhrv_baseline\": ");
    s.push_str(&json_f64(baseline.dhrv_baseline));
    s.push_str("}, \"days\": [");
    for (i, d) in days.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str("{\"day\": ");
        s.push_str(&d.day.to_string());
        if let Some(err) = &d.error {
            s.push_str(", \"error\": ");
            s.push_str(&json_str(err));
            s.push('}');
            continue;
        }
        s.push_str(", \"n_samples\": ");
        s.push_str(&d.n_samples.to_string());
        if let (Some(ns), Some(nr), Some(ts), Some(tr)) = (
            d.n_stressed,
            d.n_restored,
            d.total_stressed_min,
            d.total_restored_min,
        ) {
            s.push_str(&format!(
                ", \"n_stressed\": {ns}, \"n_restored\": {nr}, \
                 \"total_stressed_min\": {}, \"total_restored_min\": {}",
                json_f64(ts),
                json_f64(tr)
            ));
        }
        s.push_str(", \"events\": [");
        for (j, e) in d.events.iter().enumerate() {
            if j > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!(
                "{{\"type\": \"{}\", \"start_local\": {}, \"end_local\": {}, \
                 \"start_unix\": {}, \"end_unix\": {}, \"duration_min\": {}}}",
                if e.stressed { "stressed" } else { "restored" },
                json_str(&e.start_local),
                json_str(&e.end_local),
                opt_f64(e.start_unix),
                opt_f64(e.end_unix),
                json_f64(e.duration_min),
            ));
        }
        s.push_str("]}");
    }
    s.push_str("]}");
    s
}

fn opt_f64(v: Option<f64>) -> String {
    match v {
        Some(v) => json_f64(v),
        None => "null".to_string(),
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `repr(float)` as `json.dumps` writes it: shortest round-trip decimal, with
/// Python's exponent switch outside [1e-4, 1e16).
pub fn json_f64(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string(); // json.dumps emits the bare token
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let raw = format!("{v}"); // Rust: shortest round-trip, like Python's repr
    let (sign, digits) = match raw.strip_prefix('-') {
        Some(rest) => ("-", rest.to_string()),
        None => ("", raw),
    };
    let int_part = digits.split('.').next().unwrap_or("0");
    let int_digits = int_part.trim_start_matches('0');
    if int_digits.len() >= 17 {
        // Python switches to exponent form at 1e16 and above.
        let (all, exp) = strip_point(&digits);
        return format!("{sign}{}e+{exp:02}", mantissa(&all));
    }
    if let Some(frac) = digits.split_once('.').map(|(_, f)| f) {
        let lead = frac.len() - frac.trim_start_matches('0').len();
        if !frac.is_empty() && int_digits.is_empty() && lead >= 4 {
            let all = frac.trim_start_matches('0');
            return format!("{sign}{}e-{lead_plus_one:02}", mantissa(&all), lead_plus_one = lead + 1);
        }
    }
    if !digits.contains('.') {
        return format!("{sign}{digits}.0");
    }
    format!("{sign}{digits}")
}

fn strip_point(digits: &str) -> (String, i64) {
    let (int, frac) = digits.split_once('.').unwrap_or((digits, ""));
    let all = format!("{int}{frac}");
    (all, int.trim_start_matches('0').len() as i64 - 1)
}

fn mantissa(all: &str) -> String {
    let d = all.trim_end_matches('0');
    if d.contains('.') || d.len() <= 1 {
        return d.to_string();
    }
    format!("{}.{}", &d[..1], &d[1..])
}

/// Python's `round(x, 1)`: correctly rounded, ties to even, on the exact value.
pub fn py_round1(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.1}").parse::<f64>().unwrap_or(x)
}

fn trunc_to_i64(v: f64) -> i64 {
    if !v.is_finite() {
        return 0;
    }
    v.trunc() as i64
}

// ---------------------------------------------------------------------------
// Calendar helpers (no chrono in this crate)
// ---------------------------------------------------------------------------

/// Howard Hinnant's civil_from_days: days since 1970-01-01 -> (y, m, d).
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d)
}

pub fn iso_from_days(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// `_fmt(unix, tz)`: `datetime(1970,1,1) + timedelta(seconds=unix + tz*3600)`
/// then `strftime("%m-%d %H:%M")`. `timedelta` turns the float into whole
/// microseconds (round-half-even) and the datetime drops the sub-second part.
pub fn fmt_local(unix: Option<f64>, tz_hours: i64) -> String {
    let Some(unix) = unix else {
        return "undated".to_string();
    };
    let seconds = unix + (tz_hours as f64) * 3600.0;
    let micros = (seconds * 1_000_000.0).round_ties_even() as i128;
    let day_us: i128 = 86_400 * 1_000_000;
    let days = micros.div_euclid(day_us);
    let rem = micros.rem_euclid(day_us) / 1_000_000;
    let (_y, m, d) = civil_from_days(days as i64);
    format!("{m:02}-{d:02} {:02}:{:02}", rem / 3600, (rem % 3600) / 60)
}
