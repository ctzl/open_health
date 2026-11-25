//! Port of `tools/run_activity_model.py` — Oura's automatic_activity_detection
//! (AAD) model with Android-parity inputs. Golden vector:
//! `tests/vectors/activity_golden.json`. See docs/python-to-rust-plan.md, Task 1.
//!
//! What this module covers: model/asset resolution, the DB read, ring-clock
//! dating (mirror of `tools/epoch_time.py`), per-minute signal bucketing
//! (`build_signals`), real-step packet unpacking (`activity_features.
//! unpack_real_step` + `decoder_to_aad`), the per-day feature builder
//! (`run_day`'s tensor assembly), `model_would_reject`, and the exact output
//! JSON. It implements the cold path (`run_full`), which is the reference the
//! Python incremental path is required to match.
//!
//! Inference sits behind [`ActivityInference`] / [`StepMotionDecoder`]. Under the
//! `torch` feature both TorchScript calls are live: the AAD CModule's
//! `forward_is` and the `steps_motion_decoder` gait decoder, through the vendored
//! tch/LibTorch stack (see the `tch_status` note at the end of this file).
//! Without that feature the default [`UnimplementedInference`] fails with
//! [`InferenceUnavailable`]; the plumbing still runs, and `--no-stepmotion` keeps
//! the stub comparable with a decoder-stubbed oracle.
//!
//! The incremental path lives in [`crate::incremental`]: the port of
//! `event_cache.EventSource` / `UnitCache` plus `run_incremental`, over Rust-owned
//! `activity.rs.<serial>.{inc,units,skel}.json` sidecars (Python's
//! `.inc.json` / `.units.json` / `.skel.pkl` are never read or written from
//! here). `--full` is accepted but always taken, so the cold path above stays the
//! reference the incremental path has to match — a match that
//! `tests/activity_incremental_test.rs` measures rather than assumes.

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OpenFlags};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Constants mirrored from the Python script
// ---------------------------------------------------------------------------

pub const NON_WEAR_MET_THRESHOLD: f64 = 0.2;
pub const MIN_VALID_METS: usize = 10;
pub const ACCEPTABLE_LAST_HR_MISSING_MINUTES: f64 = 5.0;
pub const STATE_VERSION: &str = "activity-inc-2";
pub const WHERE: &str = "decoded_json IS NOT NULL";
pub const SIGNAL_TAGS: [i64; 4] = [0x50, 0x47, 0x46, 0x80];
pub const STEP_TAGS: [i64; 2] = [126, 127];

/// AAD behaviour codes -> names (BEHAVIOR in the Python script).
pub fn behavior(code: i64) -> Option<&'static str> {
    Some(match code {
        -1 => "nothing",
        0 => "<empty>",
        1 => "badminton",
        2 => "boxing",
        3 => "crossCountrySkiing",
        4 => "crossTraining",
        5 => "cycling",
        6 => "dance",
        7 => "elliptical",
        8 => "strengthTraining",
        9 => "hockey",
        10 => "pilates",
        11 => "rowing",
        12 => "running",
        13 => "swimming",
        14 => "walking",
        15 => "yoga",
        16 => "golf",
        17 => "tennis",
        18 => "climbing",
        19 => "downhillSkiing",
        20 => "snowboarding",
        21 => "hiking",
        22 => "horsebackRiding",
        23 => "volleyball",
        24 => "basketball",
        25 => "americanFootball",
        26 => "soccer",
        27 => "baseball",
        28 => "coreExercise",
        29 => "cricket",
        30 => "HIIT",
        31 => "diving",
        32 => "fitnessClass",
        33 => "floorball",
        34 => "gymnastics",
        35 => "handball",
        36 => "houseWork",
        37 => "iceSkating",
        38 => "jumpingRope",
        39 => "martialArts",
        40 => "flexibility",
        41 => "mountainBiking",
        42 => "nordicWalking",
        48 => "stairExercise",
        49 => "stretching",
        50 => "surfing",
        51 => "waterFitness",
        52 => "yardwork",
        53 => "padel",
        69 => "skateboarding",
        65535 => "other",
        65536 => "nap",
        65537 => "sleep",
        65538 => "pause",
        70937 => "meditation",
        71201 => "eating",
        71227 => "relax",
        71239 => "transport",
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Numeric / calendar helpers (Python-compatible semantics)
// ---------------------------------------------------------------------------

/// Python's `round()` is round-half-to-even; Rust's `round()` is half-away.
pub fn round_ties_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// Round `x * 10^digits` to an integer exactly the way CPython does: the f64 is
/// an exact binary rational m·2^e, so the scaled value is a rational with a
/// power-of-two denominator, and ties go to even. Returning the integer (rather
/// than a float) lets both `py_round` and `py_round_int` share one exact path —
/// Rust's `format!("{:.3}", 0.5835)` gives "0.584" where CPython gives 0.583,
/// because its Grisu fast path rounds the shortest digits, not the exact value.
fn round_scaled_exact(x: f64, digits: usize) -> Option<i128> {
    if !x.is_finite() || digits > 17 {
        return None;
    }
    const POW10: [i128; 18] = {
        let mut a = [0i128; 18];
        a[0] = 1;
        let mut i = 1;
        while i < 18 {
            a[i] = a[i - 1] * 10;
            i += 1;
        }
        a
    };
    let bits = x.to_bits();
    let neg = bits >> 63 != 0;
    let frac = (bits & 0x000F_FFFF_FFFF_FFFF) as i128;
    let exp = ((bits >> 52) & 0x7FF) as i32;
    let (m, e) = match exp {
        0 if frac == 0 => return Some(0), // +/-0
        0 => (frac, -1074),               // subnormal
        _ => (frac | 1 << 52, exp - 1075),
    };
    let n = m.checked_mul(POW10[digits])?; // |m| < 2^53, 10^17 < 2^57
    if e >= 0 {
        // Already an integer: scaling by 10^digits and shifting is exact, but the
        // shift can overflow i128 for huge inputs — those need no rounding.
        return n.checked_shl(e as u32).map(|q| if neg { -q } else { q }).or(Some(i128::MAX));
    }
    let k = (-e) as u32;
    let q = if k >= 111 {
        // n < 2^53 * 10^17 < 2^110 <= 2^k, so 2*rem < den and this rounds to zero.
        0
    } else {
        let den = 1i128 << k;
        let mut q = n / den;
        let rem = n - q * den; // truncating division, so 0 <= rem < den
        if rem * 2 > den || (rem * 2 == den && q.rem_euclid(2) != 0) {
            q += 1;
        }
        q
    };
    Some(if neg { -q } else { q })
}

/// Python's `round(x, n)`: correctly rounded to n decimals, ties-to-even.
pub fn py_round(x: f64, digits: usize) -> f64 {
    match round_scaled_exact(x, digits) {
        Some(q) => q as f64 / 10f64.powi(digits as i32),
        None => x,
    }
}

/// Python's `round(x)` -> int (ties-to-even).
pub fn py_round_int(x: f64) -> i64 {
    round_scaled_exact(x, 0)
        .and_then(|q| i64::try_from(q).ok())
        .unwrap_or(0)
}

/// A proleptic-Gregorian calendar day (mirrors `datetime.date`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Date {
    pub y: i32,
    pub m: u32,
    pub d: u32,
}

impl Date {
    /// days_from_civil (Howard Hinnant); equals `calendar.timegm(d)/86400`.
    pub fn days_from_civil(self) -> i64 {
        let y = self.y as i64 - i64::from(self.m <= 2);
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = y - era * 400;
        let mp = i64::from(self.m) + if self.m > 2 { -3 } else { 9 };
        let doy = (153 * mp + 2) / 5 + i64::from(self.d) - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146097 + doe - 719468
    }

    /// civil_from_days.
    pub fn from_days(z: i64) -> Date {
        let z = z + 719468;
        let era = if z >= 0 { z } else { z - 146096 } / 146097;
        let doe = z - era * 146097;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        Date { y: (y + i64::from(m <= 2)) as i32, m: m as u32, d: d as u32 }
    }

    /// `datetime.date.weekday()`: Monday == 0.
    pub fn weekday(self) -> u32 {
        (self.days_from_civil() + 3).rem_euclid(7) as u32
    }

    pub fn shift_days(self, n: i64) -> Date {
        Date::from_days(self.days_from_civil() + n)
    }

    pub fn iso(self) -> String {
        format!("{:04}-{:02}-{:02}", self.y, self.m, self.d)
    }

    /// `datetime.utcfromtimestamp(unix + tz*3600).date()`
    pub fn from_local_unix(unix: f64, tz_hours: f64) -> Date {
        Date::from_days(((unix + tz_hours * 3600.0) / 86400.0).floor() as i64)
    }
}

/// `datetime.date.fromisoformat`: exactly `YYYY-MM-DD`, and a real calendar day.
pub fn parse_date(s: &str) -> Option<Date> {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' || !b.iter().enumerate().all(|(i, c)| c.is_ascii_digit() || i == 4 || i == 7) {
        return None;
    }
    let y: i32 = s[0..4].parse().ok()?;
    let m: u32 = s[5..7].parse().ok()?;
    let d: u32 = s[8..10].parse().ok()?;
    if !(1..=12).contains(&m) {
        return None;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let dim = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m as usize - 1];
    if d < 1 || d > dim {
        return None;
    }
    Some(Date { y, m, d })
}

/// `re.findall(r"\d+", name)` as a version sort key.
pub fn numeric_key(name: &str) -> Vec<i64> {
    let mut out = Vec::new();
    let bytes = name.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if let Ok(v) = name[start..i].parse::<i64>() {
                out.push(v);
            }
        } else {
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CLI (parse_args)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Args {
    pub db: Option<String>,
    pub tz: f64,
    pub threshold: f64,
    pub min_duration: f64,
    pub date: Option<String>,
    pub full: bool,
    pub no_stepmotion: bool,
    pub serial: Option<String>,
    pub json: bool,
    pub verbose: bool,
    pub user_vector: Vec<f64>,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            db: None,
            tz: 1.0,
            threshold: 0.5,
            min_duration: 10.0,
            date: None,
            full: false,
            no_stepmotion: false,
            serial: None,
            json: false,
            verbose: false,
            user_vector: Vec::new(),
        }
    }
}

fn as_float(s: &str) -> Option<f64> {
    s.parse::<f64>().ok()
}

/// Mirror of `parse_args`, including the legacy "bare float means --tz" handling.
pub fn parse_args(argv: &[String]) -> Result<Args> {
    let mut a = Args::default();
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].clone();
        let value = |i: &mut usize| -> Result<String> {
            *i += 1;
            argv.get(*i).cloned().with_context(|| format!("missing value for {arg}"))
        };
        match argv[i].as_str() {
            "--tz" => a.tz = as_float(&value(&mut i)?).context("--tz expects a float")?,
            "--threshold" => a.threshold = as_float(&value(&mut i)?).context("--threshold expects a float")?,
            "--min-duration" => a.min_duration = as_float(&value(&mut i)?).context("--min-duration expects a float")?,
            "--date" => a.date = Some(value(&mut i)?),
            "--serial" => a.serial = Some(value(&mut i)?),
            "--full" => a.full = true,
            "--no-stepmotion" => a.no_stepmotion = true,
            "--json" => a.json = true,
            "--verbose" => a.verbose = true,
            s if s.starts_with("--") => bail!("unrecognized option: {s}"),
            s => positional.push(s.to_string()),
        }
        i += 1;
    }
    // `parse_known_args` plus the extra-positional / numeric-db fallbacks.
    if positional.len() > 1 {
        if let Some(x) = as_float(&positional[1]) {
            a.tz = x;
        }
        positional.truncate(1);
    }
    a.db = positional.into_iter().next();
    if let Some(db) = a.db.clone() {
        if !Path::new(&db).exists() {
            if let Some(x) = as_float(&db) {
                a.tz = x;
                a.db = None;
            }
        }
    }
    if std::env::var("STEPMOTION").map(|v| v == "0").unwrap_or(false) {
        a.no_stepmotion = true;
    }
    Ok(a)
}

/// `resolve_db` from tools/_common.py.
pub fn resolve_db(arg: Option<&str>, repo: &Path) -> Result<PathBuf> {
    let db = match arg {
        Some(a) => PathBuf::from(a),
        None => {
            let cwd = std::env::current_dir().unwrap_or_else(|_| repo.to_path_buf());
            [cwd.join("oura.db"), repo.join("oura.db"), repo.join("captures").join("ring5.db")]
                .into_iter()
                .find(|p| p.exists())
                .unwrap_or_else(|| repo.join("oura.db"))
        }
    };
    if !db.exists() {
        bail!("error: database not found: {} (run `oura sync` first)", db.display());
    }
    Ok(db)
}

// ---------------------------------------------------------------------------
// Model resolution (find_model / find_step_decoder)
// ---------------------------------------------------------------------------

fn newest_glob(dir: &Path, prefix: &str, suffix: &str) -> Option<PathBuf> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(prefix) && name.ends_with(suffix) {
                cands.push(entry.path());
            }
        }
    }
    if cands.is_empty() {
        return None;
    }
    cands.sort_by(|a, b| {
        numeric_key(&a.file_name().unwrap().to_string_lossy()).cmp(&numeric_key(&b.file_name().unwrap().to_string_lossy()))
    });
    cands.pop()
}

/// Newest decrypted AAD model: notes/models, then the sibling open_oura checkout.
pub fn find_model(repo: &Path) -> Result<PathBuf> {
    let dirs = [
        repo.join("notes").join("models"),
        repo.parent().unwrap_or(repo).join("open_oura").join("notes").join("models"),
    ];
    for d in &dirs {
        if let Some(p) = newest_glob(d, "automatic_activity_detection_3_1_", ".pt") {
            return Ok(p);
        }
    }
    bail!("error: no decrypted automatic_activity_detection_3_1_*.pt in notes/models — decrypt the assets with model_key.hex first")
}

/// Newest steps_motion_decoder in the same directory (also unpinned).
pub fn find_step_decoder(dir: &Path) -> Option<PathBuf> {
    newest_glob(dir, "steps_motion_decoder_", ".pt")
}

// ---------------------------------------------------------------------------
// Ring clock: mirror of tools/epoch_time.py (build_epochs / make_unix_s / is_dated)
// ---------------------------------------------------------------------------

pub const EPOCH_RESET_SLACK_DS: i64 = 6 * 3600 * 10;
pub const FUTURE_SLACK_S: f64 = 6.0 * 3600.0;
pub const ANCHOR_AGREEMENT_S: f64 = 30.0 * 60.0;
pub const ANCHOR_AGREEMENT_FRACTION: f64 = 0.02;

#[derive(Clone, Debug, Default)]
pub struct Epoch {
    pub min_ds: i64,
    pub max_ds: i64,
    /// captured_unix of the row that set `max_ds` (Python's `e[2]`).
    pub fallback_unix: i64,
    pub capture_min: i64,
    pub capture_max: i64,
    /// (ds, unix) time-sync/RTC anchors, in insertion order.
    pub anchors: Vec<(i64, i64)>,
    /// ring_start ds values.
    pub boots: Vec<i64>,
}

#[derive(Debug)]
pub struct Clock {
    pub epochs: Vec<Epoch>,
    /// Per epoch: anchors sorted by ds carrying their original insertion index,
    /// plus the parallel ds list (`_prepared`).
    prepared: Vec<(Vec<(i64, i64, usize)>, Vec<i64>)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bracket {
    Consistent,
    Stalled,
    Erratic,
}

impl Clock {
    /// `build_epochs` over rows in (captured_unix, id) order.
    pub fn build(rows: impl Iterator<Item = (i64, i64, Option<String>, i64)>) -> Clock {
        let mut epochs: Vec<Epoch> = Vec::new();
        for (ds, tag, js, cu) in rows {
            let start_new = epochs.last().map(|e| ds < e.max_ds - EPOCH_RESET_SLACK_DS).unwrap_or(true);
            if start_new {
                epochs.push(Epoch {
                    min_ds: ds,
                    max_ds: ds,
                    fallback_unix: cu,
                    capture_min: cu,
                    capture_max: cu,
                    anchors: Vec::new(),
                    boots: Vec::new(),
                });
            } else {
                let e = epochs.last_mut().unwrap();
                if ds >= e.max_ds {
                    e.max_ds = ds;
                    e.fallback_unix = cu;
                }
                e.min_ds = e.min_ds.min(ds);
                e.capture_min = e.capture_min.min(cu);
                e.capture_max = e.capture_max.max(cu);
            }
            let e = epochs.last_mut().unwrap();
            if tag == 0x41 {
                e.boots.push(ds);
            }
            if tag == 0x42 || tag == 0x85 {
                if let Some(js) = js {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&js) {
                        if let Some(u) = v.get("unix_time").and_then(|u| u.as_i64()) {
                            e.anchors.push((ds, u));
                        }
                    }
                }
            }
        }
        let prepared = epochs
            .iter()
            .map(|e| {
                let mut indexed: Vec<(i64, i64, usize)> =
                    e.anchors.iter().enumerate().map(|(i, (ds, unix))| (*ds, *unix, i)).collect();
                indexed.sort();
                let dsl = indexed.iter().map(|a| a.0).collect();
                (indexed, dsl)
            })
            .collect();
        Clock { epochs, prepared }
    }

    /// Rebuild a `Clock` from an epoch list that came out of the incremental
    /// cache instead of a scan (see `incremental::EventSource::clock`). The
    /// derived `prepared` index is recomputed here exactly as `build` does, so a
    /// cached clock dates rows identically to a freshly built one.
    pub fn from_epochs(epochs: Vec<Epoch>) -> Clock {
        let prepared = epochs
            .iter()
            .map(|e| {
                let mut indexed: Vec<(i64, i64, usize)> =
                    e.anchors.iter().enumerate().map(|(i, (ds, unix))| (*ds, *unix, i)).collect();
                indexed.sort();
                let dsl = indexed.iter().map(|a| a.0).collect();
                (indexed, dsl)
            })
            .collect();
        Clock { epochs, prepared }
    }

    fn bisect_left(dsl: &[i64], ds: i64) -> usize {
        dsl.partition_point(|x| *x < ds)
    }

    /// `bracket()` — how the anchors around `ds` relate.
    fn bracket(&self, ei: usize, ds: i64) -> (Bracket, Option<((i64, i64), (i64, i64))>) {
        let (anchors, dsl) = &self.prepared[ei];
        let idx = Self::bisect_left(dsl, ds);
        if idx == 0 || idx >= anchors.len() {
            return (Bracket::Consistent, None);
        }
        let prev = (anchors[idx - 1].0, anchors[idx - 1].1);
        let nxt = (anchors[idx].0, anchors[idx].1);
        if nxt.0 == ds {
            return (Bracket::Consistent, None);
        }
        let wall_s = (nxt.1 - prev.1) as f64;
        let counter_s = (nxt.0 - prev.0) as f64 / 10.0;
        let tolerance = ANCHOR_AGREEMENT_S.max(counter_s * ANCHOR_AGREEMENT_FRACTION);
        if wall_s < counter_s - tolerance {
            (Bracket::Erratic, None)
        } else if wall_s > counter_s + tolerance {
            (Bracket::Stalled, Some((prev, nxt)))
        } else {
            (Bracket::Consistent, None)
        }
    }

    /// `_nearest` — closest anchor; ties resolve to the earliest insertion index.
    fn nearest(&self, ei: usize, ds: i64) -> (i64, i64) {
        let (anchors, dsl) = &self.prepared[ei];
        let idx = Self::bisect_left(dsl, ds);
        if idx <= 0 {
            return (anchors[0].0, anchors[0].1);
        }
        if idx >= anchors.len() {
            let last = &anchors[anchors.len() - 1];
            return (last.0, last.1);
        }
        let before = &anchors[idx - 1];
        let after = &anchors[idx];
        if (ds - before.0, before.2) <= (after.0 - ds, after.2) {
            (before.0, before.1)
        } else {
            (after.0, after.1)
        }
    }

    fn capture_distance(e: &Epoch, captured_unix: i64) -> i64 {
        if captured_unix < e.capture_min {
            e.capture_min - captured_unix
        } else if captured_unix > e.capture_max {
            captured_unix - e.capture_max
        } else {
            0
        }
    }

    /// `_pick_epoch` — the boot that owns `ds`, by capture-window proximity.
    fn pick_epoch(&self, ds: i64, captured_unix: Option<i64>) -> usize {
        let candidates: Vec<usize> = (0..self.epochs.len())
            .filter(|&i| {
                let e = &self.epochs[i];
                e.min_ds - EPOCH_RESET_SLACK_DS <= ds && ds <= e.max_ds + EPOCH_RESET_SLACK_DS
            })
            .collect();
        if candidates.is_empty() {
            return self.epochs.len().saturating_sub(1);
        }
        let mut best = candidates[0];
        match captured_unix {
            Some(cu) => {
                let mut bd = Self::capture_distance(&self.epochs[best], cu);
                for &i in &candidates[1..] {
                    let d = Self::capture_distance(&self.epochs[i], cu);
                    if d < bd {
                        bd = d;
                        best = i;
                    }
                }
            }
            None => {
                let mut bs = self.epochs[best].max_ds - self.epochs[best].min_ds;
                for &i in &candidates[1..] {
                    let s = self.epochs[i].max_ds - self.epochs[i].min_ds;
                    if s < bs {
                        bs = s;
                        best = i;
                    }
                }
            }
        }
        best
    }

    /// The `plausible` cross-boot fallback in `make_unix_s`.
    fn plausible(&self, ds: i64, captured_unix: i64) -> Option<f64> {
        let mut best: Option<f64> = None;
        for e in &self.epochs {
            if !(e.min_ds - EPOCH_RESET_SLACK_DS <= ds && ds <= e.max_ds + EPOCH_RESET_SLACK_DS) {
                continue;
            }
            for &(a_ds, unix) in &e.anchors {
                let v = unix as f64 + (ds - a_ds) as f64 / 10.0;
                if v <= captured_unix as f64 + FUTURE_SLACK_S {
                    best = Some(best.map_or(v, |b| b.max(v)));
                }
            }
        }
        best
    }

    /// `make_unix_s`'s `unix_s`: the wall clock for a ring counter value.
    pub fn unix_s(&self, ds: i64, captured_unix: Option<i64>) -> Option<f64> {
        let ei = self.pick_epoch(ds, captured_unix);
        let e = &self.epochs[ei];
        if !e.anchors.is_empty() {
            let (anchor_ds, anchor_unix) = self.nearest(ei, ds);
            let predicted = anchor_unix as f64 + (ds - anchor_ds) as f64 / 10.0;
            match self.bracket(ei, ds) {
                (Bracket::Stalled, Some((before, after))) => {
                    let late = after.1 as f64 - (after.0 - ds) as f64 / 10.0;
                    let early = before.1 as f64 + (ds - before.0) as f64 / 10.0;
                    let boots: Vec<i64> = e.boots.iter().copied().filter(|b| before.0 < *b && *b <= after.0).collect();
                    if !boots.is_empty() {
                        let bmin = *boots.iter().min().unwrap();
                        let bmax = *boots.iter().max().unwrap();
                        if bmin <= ds && ds < bmax {
                            return None; // lost time cannot be assigned to one of several reboots
                        }
                        return Some(if ds >= bmax { late } else { early });
                    }
                    return Some(match captured_unix {
                        None => late,
                        Some(cu) if late <= cu as f64 + FUTURE_SLACK_S => late,
                        _ => early,
                    });
                }
                (Bracket::Consistent, _)
                    if match captured_unix {
                        None => true,
                        Some(cu) => predicted <= cu as f64 + FUTURE_SLACK_S,
                    } =>
                {
                    return Some(predicted)
                }
                (Bracket::Erratic, _) => return Some(predicted), // undated; callers check is_dated
                (Bracket::Consistent, _) => {
                    if let Some(cu) = captured_unix {
                        if let Some(v) = self.plausible(ds, cu) {
                            return Some(v);
                        }
                    }
                }
                (Bracket::Stalled, None) => {}
            }
        }
        let fallback = e.fallback_unix as f64 - (e.max_ds - ds) as f64 / 10.0;
        Some(match captured_unix {
            Some(cu) => fallback.min(cu as f64 + FUTURE_SLACK_S),
            None => fallback,
        })
    }

    /// `undated_reason`: None when dated, else the reason string.
    pub fn undated_reason(&self, ds: i64, captured_unix: Option<i64>) -> Option<&'static str> {
        let candidates: Vec<usize> = (0..self.epochs.len())
            .filter(|&i| {
                let e = &self.epochs[i];
                e.min_ds - EPOCH_RESET_SLACK_DS <= ds && ds <= e.max_ds + EPOCH_RESET_SLACK_DS
            })
            .collect();
        if candidates.is_empty() {
            return Some("missing_anchor");
        }
        let ei = self.pick_epoch(ds, captured_unix);
        let e = &self.epochs[ei];
        if e.anchors.is_empty() {
            return Some("missing_anchor");
        }
        match self.bracket(ei, ds) {
            (Bracket::Erratic, _) => Some("accelerated_counter"),
            (Bracket::Stalled, Some((before, after))) => {
                let boots: Vec<i64> = e.boots.iter().copied().filter(|b| before.0 < *b && *b <= after.0).collect();
                if !boots.is_empty() {
                    let bmin = *boots.iter().min().unwrap();
                    let bmax = *boots.iter().max().unwrap();
                    if bmin <= ds && ds < bmax {
                        return Some("ambiguous_reboot_stall");
                    }
                }
                None
            }
            (Bracket::Stalled, None) => None,
            (Bracket::Consistent, _) => {
                let (anchor_ds, anchor_unix) = self.nearest(ei, ds);
                let predicted = anchor_unix as f64 + (ds - anchor_ds) as f64 / 10.0;
                let cu = captured_unix.unwrap_or(i64::MIN);
                if predicted <= cu as f64 + FUTURE_SLACK_S {
                    return None;
                }
                if self.plausible(ds, cu).is_some() {
                    return None;
                }
                Some("accelerated_counter")
            }
        }
    }

    /// `is_dated`.
    pub fn is_dated(&self, ds: i64, captured_unix: Option<i64>) -> bool {
        self.undated_reason(ds, captured_unix).is_none()
    }

    /// `is_dated` + `unix_s` in one pass (both resolve the same boot).
    pub fn dated_unix(&self, ds: i64, captured_unix: Option<i64>) -> Option<f64> {
        if self.undated_reason(ds, captured_unix).is_some() {
            return None;
        }
        self.unix_s(ds, captured_unix)
    }
}

// ---------------------------------------------------------------------------
// profile.json (tools/_common.py read_profile)
// ---------------------------------------------------------------------------

/// Demographics from `profile.json` next to the DB. The `""` key is the default;
/// a legacy flat object is treated as that default.
pub fn read_profile(db_path: &Path, serial: Option<&str>) -> serde_json::Map<String, serde_json::Value> {
    let path = db_path.parent().unwrap_or(Path::new(".")).join("profile.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return serde_json::Map::new(),
    };
    let value: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return serde_json::Map::new(),
    };
    let obj = match value {
        serde_json::Value::Object(o) => o,
        _ => return serde_json::Map::new(),
    };
    if obj.contains_key("age") || obj.contains_key("sex") {
        return obj; // legacy flat object = the only profile
    }
    let key = serial.unwrap_or("");
    let entry = obj.get(key).or_else(|| obj.get("")).cloned().unwrap_or(serde_json::Value::Null);
    match entry {
        serde_json::Value::Object(o) => o,
        _ => serde_json::Map::new(),
    }
}

/// `[age, sex(M=1), height_m, weight_kg]` with the script's defaults.
pub fn profile_vector(db_path: &Path, serial: Option<&str>) -> Vec<f64> {
    let p = read_profile(db_path, serial);
    // `profile.get(k) or default`: null, 0 and "" all fall back.
    let pick = |k: &str, default: f64| -> f64 {
        let v = p.get(k);
        let n = match v {
            Some(serde_json::Value::Number(n)) => n.as_f64(),
            Some(serde_json::Value::String(s)) => s.parse().ok(),
            _ => None,
        };
        match n {
            Some(n) if n != 0.0 => n,
            _ => default,
        }
    };
    let sex = p
        .get("sex")
        .and_then(|v| v.as_str())
        .map(|s| s.to_uppercase())
        .unwrap_or_else(|| "M".to_string());
    vec![
        pick("age", 30.0),
        if sex == "M" { 1.0 } else { 0.0 },
        pick("height_m", 1.78),
        pick("weight_kg", 75.0),
    ]
}

// ---------------------------------------------------------------------------
// Step packets (tools/activity_features.py)
// ---------------------------------------------------------------------------

/// The ring's two 14-byte real-step packets -> the decoder's 27 ints.
pub fn unpack_real_step(p1: &[u8], p2: &[u8]) -> Result<Vec<i64>> {
    if p1.len() != 14 || p2.len() != 14 {
        bail!("real-step feature packets must both be 14 bytes");
    }
    let carry = p2[13] as i64;
    let g = |b: &[u8], i: usize| -> i64 { b[i] as i64 };
    Ok(vec![
        (g(p2, 10) << 2) | (carry & 0x3),
        g(p2, 11),
        g(p2, 12),
        (g(p1, 0) << 1) | (g(p1, 3) >> 7),
        (g(p1, 1) << 1) | ((carry >> 7) & 1),
        (g(p1, 2) << 1) | ((carry >> 6) & 1),
        g(p1, 3) & 0x7F,
        g(p1, 4),
        g(p1, 5),
        g(p1, 6),
        g(p1, 7),
        (g(p1, 8) << 1) | (g(p1, 11) >> 7),
        (g(p1, 9) << 1) | ((carry >> 5) & 1),
        (g(p1, 10) << 1) | ((carry >> 4) & 1),
        g(p1, 11) & 0x7F,
        g(p1, 12),
        g(p1, 13),
        g(p2, 0),
        g(p2, 1),
        (g(p2, 2) << 1) | (g(p2, 5) >> 7),
        (g(p2, 3) << 1) | ((carry >> 3) & 1),
        (g(p2, 4) << 1) | ((carry >> 2) & 1),
        g(p2, 5) & 0x7F,
        g(p2, 6),
        g(p2, 7),
        g(p2, 8),
        g(p2, 9),
    ])
}

/// decoder order (sum, y, z, total, strideFreq, strideAmp, firstFreq, firstAmp,
/// gaitAmp, highFrac, midFrac) -> AAD column order.
pub fn decoder_to_aad(row: &[f32]) -> Result<[f32; 11]> {
    if row.len() != 11 {
        bail!("step-motion decoder row must have 11 values");
    }
    Ok([row[6], row[7], row[9], row[10], row[8], row[5], row[4], row[0], row[3], row[1], row[2]])
}

// ---------------------------------------------------------------------------
// Inference behind a trait
// ---------------------------------------------------------------------------

/// A dense float32 matrix in row-major order — exactly what `tensor()` builds.
#[derive(Clone, Debug, Default)]
pub struct Matrix {
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn rows(&self) -> usize {
        if self.cols == 0 {
            0
        } else {
            self.data.len() / self.cols
        }
    }
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
    pub fn from_rows(cols: usize, rows: impl IntoIterator<Item = Vec<f64>>) -> Matrix {
        let mut data = Vec::new();
        for r in rows {
            debug_assert!(r.len() == cols || r.is_empty());
            data.extend(r.iter().map(|v| *v as f32));
        }
        Matrix { cols, data }
    }
}

/// Trailing settings argument(s), adapted to the loaded model schema.
#[derive(Clone, Debug)]
pub enum Settings {
    /// AAD >= 3.1.13: one optional `global_settings` row read by column
    /// (3=allow_non_wear, 12=is_workout_proba_threshold,
    /// 26=min_workout_duration_minutes); NaN keeps the model's baked-in default.
    Global(Vec<f32>),
    /// AAD <= 3.1.12: three scalar tensors positionally.
    Legacy { threshold: f32, min_duration: f32 },
}

/// Everything `model(context, user, met, step_rows, motion, temperature,
/// heart_rate, None, None, *settings)` receives for one local day.
#[derive(Clone, Debug)]
pub struct DayRequest {
    pub day: Date,
    pub day_start_min: f64,
    pub context: [f32; 4],
    pub user: Vec<f32>,
    pub met: Matrix,
    pub step_motion: Matrix,
    pub motion: Matrix,
    pub temperature: Matrix,
    pub heart_rate: Matrix,
    pub settings: Settings,
}

/// One row of the model's `workouts` output:
/// `[start_min, end_min, is_workout, label0, conf0, label1, conf1, label2, conf2]`.
pub type WorkoutRow = [f64; 9];

pub trait ActivityInference {
    fn run_day(&self, req: &DayRequest) -> Result<Vec<WorkoutRow>>;
}

pub trait StepMotionDecoder {
    /// `decoder(int64 timestamps, float32 [n,27] raw)` -> (out_ts ms, [m,11] rows).
    fn decode(&self, timestamps: &[i64], raw: &[Vec<i64>]) -> Result<(Vec<i64>, Vec<[f32; 11]>)>;
}

/// Raised when no TorchScript backend is linked into this build. `run` treats it
/// as fatal instead of swallowing it the way Python swallows a rejected day.
#[derive(Debug)]
pub struct InferenceUnavailable(pub &'static str);

impl std::fmt::Display for InferenceUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for InferenceUnavailable {}

/// Placeholder until a TorchScript backend is linked in (see `tch_status`).
pub struct UnimplementedInference;

impl ActivityInference for UnimplementedInference {
    fn run_day(&self, _req: &DayRequest) -> Result<Vec<WorkoutRow>> {
        Err(InferenceUnavailable(tch_status()).into())
    }
}

impl StepMotionDecoder for UnimplementedInference {
    fn decode(&self, _timestamps: &[i64], _raw: &[Vec<i64>]) -> Result<(Vec<i64>, Vec<[f32; 11]>)> {
        Err(InferenceUnavailable(tch_status()).into())
    }
}

/// Why no backend is linked in a `--no-default-features` build. The torch path
/// needs the venv's libtorch: activate `.venv` so `.cargo/config.toml` supplies
/// LIBTORCH_USE_PYTORCH=1 and the C++20 flag, then `--features torch`.
/// Verified 2026-11-24 with tch =0.26.0 + torch 2.13.0+cpu + the vendored
/// torch-sys patch (its `module` typedef collides with C++20's keyword).
pub fn tch_status() -> &'static str {
    "activity inference is not linked in: this build has no `torch` feature, so the pipeline stops at the model boundary instead of inventing sessions"
}

// ---------------------------------------------------------------------------
// Signal bucketing (build_signals)
// ---------------------------------------------------------------------------

/// One per-minute sample: bucket timestamp + the model's column values.
/// Kept in DB capture order, as the oracle's build_signals leaves it.
pub type Sample = (f64, Vec<f64>);

#[derive(Clone, Debug, Default)]
pub struct Signals {
    pub met: Vec<Sample>,
    pub motion: Vec<Sample>,
    pub temperature: Vec<Sample>,
    pub heart_rate: Vec<Sample>,
}

fn json_f64(v: Option<&serde_json::Value>) -> Option<f64> {
    match v {
        Some(serde_json::Value::Number(n)) => n.as_f64(),
        _ => None,
    }
}

/// Mirror of `build_signals`: date every row through the clock and bucket it into
/// the per-minute series Android's TimeseriesDb tables would have held.
pub fn build_signals(rows: impl Iterator<Item = (i64, i64, Option<String>, i64)>, clock: &Clock) -> Signals {
    let scale: f64 = std::env::var("ACM_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let mut out = Signals::default();
    for (ds, tag, payload, captured) in rows {
        let payload = match payload {
            Some(p) => p,
            None => continue,
        };
        if !SIGNAL_TAGS.contains(&tag) {
            continue; // parsed by Python, but the value is unused for other tags
        }
        let value: serde_json::Value = match serde_json::from_str(&payload) {
            Ok(v) => v,
            Err(_) => continue,
        };
        // Undated data (an untrustworthy boot clock) belongs to no calendar day.
        let unix = match clock.dated_unix(ds, Some(captured)) {
            Some(u) => u,
            None => continue,
        };
        // Ring summaries are minute buckets; epoch reconstruction can leave them a
        // few seconds off the boundary, so snap to the bucket.
        let minute = round_ties_even(unix / 60.0);
        match tag {
            0x50 => {
                if let Some(met) = value.get("met").and_then(|m| m.as_array()) {
                    for (i, v) in met.iter().enumerate() {
                        out.met.push((minute + i as f64, vec![json_f64(Some(v)).unwrap_or(0.0)]));
                    }
                }
            }
            0x47 => {
                let get = |k: &str| json_f64(value.get(k)).unwrap_or(0.0);
                out.motion.push((
                    minute,
                    vec![
                        get("orientation"),
                        get("motion_seconds"),
                        get("avg_x") * scale,
                        get("avg_y") * scale,
                        get("avg_z") * scale,
                        f64::NAN,
                        get("low_intensity"),
                        get("high_intensity"),
                    ],
                ));
            }
            0x46 => {
                if let Some(v) = value.get("temps_c").and_then(|t| t.as_array()).and_then(|a| a.first()).and_then(|v| json_f64(Some(v))) {
                    out.temperature.push((minute, vec![v]));
                }
            }
            0x80 => {
                if let Some(bpm) = value.get("hr_bpm").and_then(|h| h.as_array()) {
                    if !bpm.is_empty() {
                        let sum: f64 = bpm.iter().filter_map(|v| json_f64(Some(v))).sum();
                        out.heart_rate.push((minute, vec![sum / bpm.len() as f64]));
                    }
                }
            }
            _ => {}
        }
    }
    // No sort here: build_signals keeps the DB's capture order, exactly like the
    // oracle. Every consumer sorts at the point of use (run_day's `select`,
    // decode_step_packets), because a stable sort there decides which duplicate
    // minute wins.
    out
}

// ---------------------------------------------------------------------------
// Step-motion series (decode_step_packets)
// ---------------------------------------------------------------------------

/// The gait decoder's AAD input series: (minute, 11 columns).
pub type StepSeries = Vec<(f64, [f32; 11])>;

pub struct StepPackets {
    pub timestamps: Vec<i64>,
    pub raw: Vec<Vec<i64>>,
}

/// Collect and order the decoder inputs from the raw 0x7E/0x7F packets.
pub fn collect_step_packets(packets: &[(i64, i64, Vec<u8>, i64)], clock: &Clock) -> Result<StepPackets> {
    let mut seconds: HashMap<i64, Vec<u8>> = HashMap::new();
    for (ds, tag, body, captured) in packets {
        if *tag == 0x7F && body.len() == 14 {
            if let Some(u) = clock.dated_unix(*ds, Some(*captured)) {
                seconds.insert(round_ties_even(u * 10.0) as i64, body.clone());
            }
        }
    }
    let mut raw: Vec<Vec<i64>> = Vec::new();
    let mut timestamps: Vec<i64> = Vec::new();
    for (ds, tag, body, captured) in packets {
        if *tag != 0x7E || body.len() != 14 {
            continue;
        }
        let u = match clock.dated_unix(*ds, Some(*captured)) {
            Some(u) => u,
            None => continue,
        };
        let wall_ds = round_ties_even(u * 10.0) as i64;
        let second = match seconds.get(&(wall_ds + 1)) {
            Some(s) => s,
            None => continue,
        };
        raw.push(unpack_real_step(body, second)?);
        timestamps.push(wall_ds * 100);
    }
    // The decoder windows consecutive packets: feed them in time order (the DB is
    // in capture order, which interleaves boots after a re-anchor), as iOS does.
    let mut order: Vec<usize> = (0..timestamps.len()).collect();
    order.sort_by_key(|&i| timestamps[i]);
    timestamps = order.iter().map(|&i| timestamps[i]).collect();
    raw = order.iter().map(|&i| raw[i].clone()).collect();
    Ok(StepPackets { timestamps, raw })
}

pub fn decode_step_packets(packets: &StepPackets, decoder: &dyn StepMotionDecoder) -> Result<StepSeries> {
    if packets.timestamps.is_empty() {
        return Ok(Vec::new());
    }
    // Keep all three 10-second sub-windows, as Android's TimeseriesDbStepMotion does.
    let (out_ts, out_data) = decoder.decode(&packets.timestamps, &packets.raw)?;
    // Python: `sorted((ts / 60_000.0, decoder_to_aad(values)) ...)` — the decoder
    // emits its own column order, the AAD feature order is a permutation of it.
    let mut out: StepSeries = Vec::with_capacity(out_ts.len());
    for (ts, values) in out_ts.iter().zip(out_data.iter()) {
        out.push((*ts as f64 / 60_000.0, decoder_to_aad(values)?));
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(out)
}

// ---------------------------------------------------------------------------
// Per-day feature assembly (run_day)
// ---------------------------------------------------------------------------

/// `model_would_reject` — mirror of the AAD preprocessor's `get_last_valid_time`.
pub fn model_would_reject(met: &[Vec<f64>], motion: &[Vec<f64>], temperature: &[Vec<f64>], heart_rate: &[Vec<f64>], step_rows: &[Vec<f64>]) -> bool {
    let last = |rows: &[Vec<f64>]| -> f64 { rows.last().map(|r| r[0]).unwrap_or(0.0) };
    let last_motion = last(motion).ceil();
    let worn: Vec<f64> = met.iter().filter(|r| r[1] > NON_WEAR_MET_THRESHOLD && r[0] <= last_motion).map(|r| r[0]).collect();
    if worn.len() < MIN_VALID_METS {
        return met[0][0] > 0.0;
    }
    let last_valid = worn[worn.len() - 1]
        .min(last_motion)
        .min(last(heart_rate).ceil() + ACCEPTABLE_LAST_HR_MISSING_MINUTES)
        .min(last(step_rows).ceil())
        .min(last(temperature).ceil());
    met[0][0] > last_valid
}

/// `tensor()`: rows as float32; `window=(lo, hi)` inserts a NaN placeholder for a
/// series with no sample inside the MET span (the model would index an empty one).
pub fn tensor(rows: &[Vec<f64>], columns: usize, window: Option<(f64, f64)>) -> Matrix {
    let mut owned: Vec<Vec<f64>> = rows.to_vec();
    if let Some((lo, hi)) = window {
        if !rows.iter().any(|r| lo <= r[0] && r[0] <= hi) {
            let mut row = vec![lo];
            row.extend(std::iter::repeat(f64::NAN).take(columns - 1));
            owned.push(row);
            owned.sort_by(|a, b| a[0].total_cmp(&b[0]));
        }
    }
    Matrix::from_rows(columns, owned)
}

fn range_of<T>(rows: &[(f64, T)], lo: f64, hi: f64) -> (usize, usize) {
    let start = rows.partition_point(|r| r.0 < lo);
    let end = rows.partition_point(|r| r.0 < hi);
    (start, end.max(start))
}

/// What happened for one local day: no data, a degenerate day, or a model call.
pub enum DayOutcome {
    NoMet,
    Rejected,
    Run(DayRequest),
}

pub struct PreparedDay {
    pub day: Date,
    pub day_start_min: f64,
    pub outcome: DayOutcome,
}

pub struct Pipeline {
    pub model: PathBuf,
    pub model_version: String,
    pub step_model: Option<PathBuf>,
    pub user_vector: Vec<f64>,
    /// The ring clock, kept for diagnostics/tests (epoch + anchor counts).
    pub clock: Clock,
    /// Decoder inputs actually collected (None when step motion is disabled).
    pub step_input: Option<StepPackets>,
    /// The per-minute series the model calls are built from.
    pub signals: Signals,
    pub days: Vec<PreparedDay>,
}

/// Trailing settings for the loaded schema. Python reads
/// `model.forward.schema.arguments`; tch exposes no schema introspection, so this
/// keys off the asset version — 3.1.13+ ships `global_settings` (the script's own
/// docstring), which is the branch the golden model (3.1.15) takes.
pub fn model_settings(version: &[i64], threshold: f64, min_duration: f64) -> Settings {
    if version.len() >= 3 && (version[0], version[1], version[2]) >= (3, 1, 13) {
        let mut row = vec![f32::NAN; 72];
        row[3] = 0.0; // allow_non_wear=0 -> non-wear segments filtered, as before
        row[12] = threshold as f32;
        row[26] = min_duration as f32;
        Settings::Global(row)
    } else {
        Settings::Legacy { threshold: threshold as f32, min_duration: min_duration as f32 }
    }
}

/// One row of the decoded-events query: (ring_timestamp, tag, decoded_json, captured_unix).
fn decoded_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<(i64, i64, Option<String>, i64)> {
    Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
}

/// One row of the raw-packet query (tags 126/127); `body` is a BLOB.
fn packet_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<(i64, i64, Vec<u8>, i64)> {
    Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
}

/// The cold path's data plumbing: every day's model request.
pub fn prepare(args: &Args, db_path: &Path, repo: &Path) -> Result<Pipeline> {
    let model = find_model(repo)?;
    if !model.exists() {
        bail!("error: model not found: {}", model.display());
    }
    let model_name = model.file_name().unwrap().to_string_lossy().to_string();
    let version = numeric_key(&model_name);
    let model_version = version.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(".");
    let step_model = model.parent().and_then(|d| find_step_decoder(d));

    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("opening {}", db_path.display()))?;
    let serial = args.serial.clone();
    let clause = if serial.is_some() { " AND serial = ?" } else { "" };
    let sql = format!("SELECT ring_timestamp,tag,decoded_json,captured_unix FROM events WHERE {WHERE}{clause} ORDER BY captured_unix,id");

    // Pass 1: boot epochs from the anchor rows (same row order as Python).
    let clock = {
        let mut stmt = conn.prepare(&sql)?;
        let rows = match &serial {
            Some(s) => stmt.query_map(params![s], decoded_row)?,
            None => stmt.query_map([], decoded_row)?,
        };
        Clock::build(rows.filter_map(|r| r.ok()))
    };
    if clock.epochs.is_empty() {
        bail!("error: no decoded events in {}", db_path.display());
    }

    // Pass 2: the per-minute signal series.
    let signals = {
        let mut stmt = conn.prepare(&sql)?;
        let rows = match &serial {
            Some(s) => stmt.query_map(params![s], decoded_row)?,
            None => stmt.query_map([], decoded_row)?,
        };
        build_signals(rows.filter_map(|r| r.ok()), &clock)
    };
    if signals.met.is_empty() {
        bail!("no MET events in DB — cannot run activity model");
    }

    // Step-motion series.
    let mut steps: StepSeries = Vec::new();
    let mut step_input: Option<StepPackets> = None;
    if !args.no_stepmotion {
        match &step_model {
            None => eprintln!(
                "warning: step decoder not found ({}); running without gait features, expect fewer and differently labelled sessions",
                model.parent().unwrap_or(Path::new(".")).join("steps_motion_decoder_2_0_0.pt").display()
            ),
            Some(path) => {
                let sql = format!(
                    "SELECT ring_timestamp, tag, body, captured_unix FROM events WHERE tag IN (126,127) AND body IS NOT NULL{clause} ORDER BY captured_unix,id"
                );
                let packets: Vec<(i64, i64, Vec<u8>, i64)> = {
                    let mut stmt = conn.prepare(&sql)?;
                    let rows = match &serial {
                        Some(s) => stmt.query_map(params![s], packet_row)?,
                        None => stmt.query_map([], packet_row)?,
                    };
                    rows.filter_map(|r| r.ok()).collect()
                };
                let collected = collect_step_packets(&packets, &clock)?;
                step_input = Some(StepPackets {
                    timestamps: collected.timestamps.clone(),
                    raw: collected.raw.clone(),
                });
                match step_decoder_backend(path) {
                    Some(decoder) => steps = decode_step_packets(&collected, decoder.as_ref())?,
                    None => eprintln!(
                        "warning: step decoder inference is not linked in ({}); running without gait features",
                        path.display()
                    ),
                }
            }
        }
    }

    // The official app evaluates one local calendar day at a time.
    let mut dates: Vec<Date> = signals.met.iter().map(|(t, _)| Date::from_local_unix(t * 60.0, args.tz)).collect();
    dates.sort();
    dates.dedup();

    let user_vector = profile_vector(db_path, args.serial.as_deref());
    let settings = model_settings(&version, args.threshold, args.min_duration);

    let days = dates
        .into_iter()
        .map(|day| {
            let day_start_min = day.days_from_civil() as f64 * 1440.0 - args.tz * 60.0;
            PreparedDay {
                day,
                day_start_min,
                outcome: build_day_request(day, day_start_min, &user_vector, &signals, &steps, &settings),
            }
        })
        .collect();

    Ok(Pipeline { model, model_version, step_model, user_vector, clock, step_input, signals, days })
}

/// `run_day` up to the model call.
pub fn build_day_request(
    day: Date,
    day_start_min: f64,
    user_vector: &[f64],
    signals: &Signals,
    steps: &StepSeries,
    settings: &Settings,
) -> DayOutcome {
    let day_end_min = day_start_min + 24.0 * 60.0;
    let select = |rows: &[Sample]| -> Vec<Vec<f64>> {
        // Python's `sorted((... if day_start_min <= t < day_end_min), key=row[0])`:
        // filter first, then a stable sort. The series arrive in capture order, so
        // a binary-search window here would be wrong.
        let mut out: Vec<Vec<f64>> = rows
            .iter()
            .filter(|(t, _)| *t >= day_start_min && *t < day_end_min)
            .map(|(t, v)| {
                let mut row = vec![t - day_start_min];
                row.extend(v.iter().copied());
                row
            })
            .collect();
        out.sort_by(|a, b| a[0].total_cmp(&b[0]));
        out
    };

    // The ring may resend a MET minute; Android's Realm series has one value per timestamp.
    let mut by_minute: HashMap<i64, Vec<f64>> = HashMap::new();
    for row in select(&signals.met) {
        by_minute.insert(round_ties_even(row[0]) as i64, row);
    }
    let mut met: Vec<Vec<f64>> = by_minute.into_values().collect();
    met.sort_by(|a, b| a[0].total_cmp(&b[0]));
    if met.is_empty() {
        return DayOutcome::NoMet;
    }

    let motion = select(&signals.motion);
    let temperature = select(&signals.temperature);
    let heart_rate = select(&signals.heart_rate);
    let (a, b) = range_of(steps, day_start_min, day_end_min);
    let day_steps: Vec<Vec<f64>> = steps[a..b]
        .iter()
        .map(|(t, v)| {
            let mut row = vec![t - day_start_min];
            row.extend(v.iter().map(|x| *x as f64));
            row
        })
        .collect();

    if model_would_reject(&met, &motion, &temperature, &heart_rate, &day_steps) {
        return DayOutcome::Rejected;
    }
    let (lo, hi) = (met[0][0], met[met.len() - 1][0]);
    let pad = |t: f64| {
        let mut row = vec![t];
        row.extend(std::iter::repeat(f64::NAN).take(11));
        row
    };
    let mut step_rows: Vec<Vec<f64>> = Vec::with_capacity(day_steps.len() + 2);
    step_rows.push(pad(lo));
    step_rows.extend(day_steps);
    step_rows.push(pad(hi));

    let mut user: Vec<f32> = user_vector.iter().map(|v| *v as f32).collect();
    user.extend(std::iter::repeat(f32::NAN).take(10));

    DayOutcome::Run(DayRequest {
        day,
        day_start_min,
        context: [day.y as f32, day.m as f32, day.d as f32, day.weekday() as f32],
        user,
        met: tensor(&met, 2, None),
        step_motion: Matrix::from_rows(12, step_rows),
        motion: tensor(&motion, 9, Some((lo, hi))),
        temperature: tensor(&temperature, 2, Some((lo, hi))),
        heart_rate: tensor(&heart_rate, 2, Some((lo, hi))),
        settings: settings.clone(),
    })
}

// ---------------------------------------------------------------------------
// The tch backend: the same two TorchScript graphs the Python oracle loads with
// torch.jit.load(...).map_location("cpu").eval().
// ---------------------------------------------------------------------------

/// Load a TorchScript graph the way `torch.jit.load(p, map_location="cpu").eval()`
/// does: on CPU, in inference mode, with grad disabled for the whole call.
#[cfg(feature = "torch")]
fn load_graph(path: &Path, what: &str) -> Result<tch::CModule> {
    let mut m = tch::CModule::load(path).with_context(|| format!("torch.jit.load({})", path.display()))?;
    m.set_eval();
    let _ = what;
    Ok(m)
}

#[cfg(feature = "torch")]
fn tensor_rows(t: &tch::Tensor, kind: tch::Kind) -> Result<Vec<f64>> {
    let flat = t.reshape(&[-1]).to_kind(kind).to_device(tch::Device::Cpu);
    Vec::<f64>::try_from(&flat).context("read model output as f64")
}

/// The AAD model itself (`automatic_activity_detection_3_1_*.pt`).
#[cfg(feature = "torch")]
pub struct TchActivity {
    inner: tch::CModule,
}

#[cfg(feature = "torch")]
impl TchActivity {
    pub fn load(path: &Path) -> Result<TchActivity> {
        Ok(TchActivity { inner: load_graph(path, "activity model")? })
    }
}

#[cfg(feature = "torch")]
impl ActivityInference for TchActivity {
    fn run_day(&self, req: &DayRequest) -> Result<Vec<WorkoutRow>> {
        use tch::{Device, IValue, Kind, Tensor};
        let _nograd = tch::no_grad_guard();
        // `tensor(rows, cols)` in the oracle: a [rows, cols] float32 grid.
        let grid = |m: &Matrix| {
            Tensor::from_slice(&m.data)
                .reshape([m.rows() as i64, m.cols as i64])
                .to_kind(Kind::Float)
                .to_device(Device::Cpu)
        };
        let mut ivs: Vec<IValue> = Vec::with_capacity(10);
        // context/user are 1-D (`torch.tensor([...], dtype=torch.float32)`), not grids.
        ivs.push(IValue::Tensor(Tensor::from_slice(&req.context).to_kind(Kind::Float)));
        ivs.push(IValue::Tensor(Tensor::from_slice(&req.user).to_kind(Kind::Float)));
        for m in [&req.met, &req.step_motion, &req.motion, &req.temperature, &req.heart_rate] {
            ivs.push(IValue::Tensor(grid(m)));
        }
        ivs.push(IValue::None); // the two optional args Python passes as None
        ivs.push(IValue::None);
        match &req.settings {
            Settings::Global(row) => ivs.push(IValue::Tensor(
                Tensor::from_slice(row)
                    .reshape(&[1, row.len() as i64])
                    .to_kind(Kind::Float)
                    .to_device(Device::Cpu),
            )),
            Settings::Legacy { threshold, min_duration } => {
                // <=3.1.12 takes three `float` args positionally; at the JIT
                // boundary a 0-dim torch.tensor(x) is an IValue double.
                ivs.push(IValue::Double(f64::from(*threshold)));
                ivs.push(IValue::Double(f64::from(*min_duration)));
                ivs.push(IValue::Double(0.0));
            }
        }
        let out = self
            .inner
            .forward_is(&ivs)
            .context("activity model forward(context, user, met, step_motion, motion, temperature, heart_rate, None, None, settings)")?;
        let items: Vec<IValue> = match out {
            IValue::Tuple(v) => v,
            IValue::TensorList(v) => v.into_iter().map(IValue::Tensor).collect(),
            other => bail!("unexpected activity model return type: {other:?}"),
        };
        // `workouts, _, _ = model(...)`: the first element is [n, 9].
        let flat = match items.first() {
            Some(IValue::Tensor(t)) => tensor_rows(t, Kind::Double)?,
            _ => bail!("activity model returned no workouts tensor"),
        };
        if flat.len() % 9 != 0 {
            bail!("activity model returned {} values, not a multiple of 9", flat.len());
        }
        Ok(flat
            .chunks(9)
            .map(|c| {
                let mut row = [0.0; 9];
                row.copy_from_slice(c);
                row
            })
            .collect())
    }
}

/// The step-motion decoder (`steps_motion_decoder_2_0_0.pt`).
#[cfg(feature = "torch")]
pub struct TchStepDecoder {
    inner: tch::CModule,
}

#[cfg(feature = "torch")]
impl TchStepDecoder {
    pub fn load(path: &Path) -> Result<TchStepDecoder> {
        Ok(TchStepDecoder { inner: load_graph(path, "step decoder")? })
    }
}

#[cfg(feature = "torch")]
impl StepMotionDecoder for TchStepDecoder {
    fn decode(&self, timestamps: &[i64], raw: &[Vec<i64>]) -> Result<(Vec<i64>, Vec<[f32; 11]>)> {
        use tch::{Device, IValue, Kind, Tensor};
        if raw.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let cols = raw[0].len();
        if raw.iter().any(|r| r.len() != cols) {
            bail!("step packets must all unpack to the same width");
        }
        let _nograd = tch::no_grad_guard();
        // decoder(torch.tensor(timestamps, dtype=torch.int64),
        //        torch.tensor(raw, dtype=torch.float32))
        let ts = Tensor::from_slice(timestamps).to_device(Device::Cpu);
        let flat: Vec<f32> = raw.iter().flatten().map(|v| *v as f32).collect();
        let grid = Tensor::from_slice(&flat)
            .reshape(&[raw.len() as i64, cols as i64])
            .to_kind(Kind::Float)
            .to_device(Device::Cpu);
        let out = self
            .inner
            .forward_is(&[IValue::Tensor(ts), IValue::Tensor(grid)])
            .context("step decoder forward(timestamps, raw)")?;
        let items: Vec<IValue> = match out {
            IValue::Tuple(v) => v,
            other => bail!("unexpected step decoder return type: {other:?}"),
        };
        if items.len() < 2 {
            bail!("step decoder returned {} outputs, expected 2", items.len());
        }
        let out_ts = match &items[0] {
            IValue::Tensor(t) => Vec::<i64>::try_from(&t.reshape(&[-1]).to_kind(Kind::Int64))
                .context("read step decoder timestamps")?,
            _ => bail!("step decoder timestamps are not a tensor"),
        };
        let flat = match &items[1] {
            IValue::Tensor(t) => tensor_rows(t, Kind::Double)?,
            _ => bail!("step decoder rows are not a tensor"),
        };
        if flat.len() % 11 != 0 || flat.len() / 11 != out_ts.len() {
            bail!(
                "step decoder returned {} values for {} windows",
                flat.len(),
                out_ts.len()
            );
        }
        let rows = flat
            .chunks(11)
            .map(|c| {
                let mut row = [0.0f32; 11];
                for (slot, v) in row.iter_mut().zip(c) {
                    *slot = *v as f32;
                }
                row
            })
            .collect();
        Ok((out_ts, rows))
    }
}

/// The step-motion backend for a resolved decoder file. `None` = no backend is
/// linked into this build (the tch/torch blocker).
pub fn step_decoder_backend(path: &Path) -> Option<Box<dyn StepMotionDecoder>> {
    #[cfg(feature = "torch")]
    {
        TchStepDecoder::load(path).ok().map(|d| Box::new(d) as Box<dyn StepMotionDecoder>)
    }
    #[cfg(not(feature = "torch"))]
    {
        let _ = path;
        None
    }
}

/// The activity-model backend for a resolved model file. `None` = no backend is
/// linked into this build, so `run`/`json_for` stop at the model boundary
/// instead of inventing sessions.
pub fn activity_backend(path: &Path) -> Option<Box<dyn ActivityInference>> {
    #[cfg(feature = "torch")]
    {
        TchActivity::load(path).ok().map(|m| Box::new(m) as Box<dyn ActivityInference>)
    }
    #[cfg(not(feature = "torch"))]
    {
        let _ = path;
        None
    }
}

// ---------------------------------------------------------------------------
// Output (the exact JSON the Python script prints)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Session {
    pub start: String,
    pub end: String,
    pub duration_min: i64,
    pub is_workout: f64,
    pub label: String,
    pub label_confidence: f64,
    pub top3: Vec<(String, f64)>,
}

/// `local_midnight + timedelta(minutes=m)` rendered like `strftime`.
pub fn format_at(day: Date, minutes: f64, with_date: bool) -> String {
    let secs = minutes * 60.0;
    let day_offset = (secs / 86400.0).floor() as i64;
    let rem = secs - day_offset as f64 * 86400.0;
    let h = (rem / 3600.0).floor() as i64;
    let m = ((rem - h as f64 * 3600.0) / 60.0).floor() as i64;
    if with_date {
        let d = day.shift_days(day_offset);
        format!("{:04}-{:02}-{:02} {:02}:{:02}", d.y, d.m, d.d, h, m)
    } else {
        format!("{h:02}:{m:02}")
    }
}

/// `workouts.tolist()` rows -> the session dicts the script builds.
pub fn sessions_from(day: Date, workouts: &[WorkoutRow]) -> Vec<Session> {
    workouts
        .iter()
        .map(|row| {
            let top: Vec<(String, f64)> = (0..3)
                .map(|i| {
                    let code = row[3 + 2 * i] as i64;
                    let name = behavior(code).map(|s| s.to_string()).unwrap_or_else(|| code.to_string());
                    (name, py_round(row[4 + 2 * i], 3))
                })
                .collect();
            Session {
                start: format_at(day, row[0], true),
                end: format_at(day, row[1], false),
                duration_min: py_round_int(row[1] - row[0]),
                is_workout: py_round(row[2], 3),
                label: top[0].0.clone(),
                label_confidence: top[0].1,
                top3: top,
            }
        })
        .collect()
}

/// Python's `json.dumps` float rendering (shortest round-trip, always a float).
pub fn fmt_float(v: f64) -> String {
    if !v.is_finite() {
        return v.to_string(); // json.dumps emits the bare NaN/Infinity tokens
    }
    // CPython's float repr: shortest round-trip digits, and exponential notation
    // only when the decimal point lands outside (-4, 16] — so 4e-07 and 1e+16
    // are exponential, 0.0001 and 1e15 are not. Rust's Display never switches,
    // so the digits come from `{:e}` (same shortest algorithm) and are then
    // laid out by Python's rule.
    let raw = format!("{:e}", v);
    let (neg, rest) = match raw.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, raw.as_str()),
    };
    let (mantissa, exponent) = rest.split_once('e').expect("LowerExp always emits an exponent");
    let exp: i32 = exponent.parse().expect("LowerExp exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let digits = digits.trim_start_matches('0').to_string();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if digits.is_empty() {
        out.push_str("0.0"); // -0.0 keeps its sign, as repr(-0.0) does
        return out;
    }
    let decpt = exp + 1;
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push_str(&format!("e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs()));
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..(-decpt) {
            out.push('0');
        }
        out.push_str(&digits);
    } else if decpt as usize >= digits.len() {
        out.push_str(&digits);
        for _ in 0..(decpt as usize - digits.len()) {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    out
}

fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Byte-for-byte what `json.dumps(result, indent=2)` prints, key order included.
pub fn render_result(model_version: &str, sessions: &[Session]) -> String {
    let mut s = String::new();
    s.push_str("{\n  \"model\": \"");
    s.push_str(&esc(model_version));
    s.push_str("\",\n  \"pipeline\": \"android-parity\",\n  \"sessions\": [");
    for (i, session) in sessions.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("\n    {\n");
        s.push_str(&format!("      \"start\": \"{}\",\n", esc(&session.start)));
        s.push_str(&format!("      \"end\": \"{}\",\n", esc(&session.end)));
        s.push_str(&format!("      \"duration_min\": {},\n", session.duration_min));
        s.push_str(&format!("      \"is_workout\": {},\n", fmt_float(session.is_workout)));
        s.push_str(&format!("      \"label\": \"{}\",\n", esc(&session.label)));
        s.push_str(&format!("      \"label_confidence\": {},\n", fmt_float(session.label_confidence)));
        s.push_str("      \"top3\": [");
        for (j, (name, prob)) in session.top3.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            s.push_str(&format!("\n        [\n          \"{}\",\n          {}\n        ]", esc(name), fmt_float(*prob)));
        }
        s.push_str("\n      ]\n    }");
    }
    if !sessions.is_empty() {
        s.push('\n');
    }
    s.push_str("  ]\n}");
    s
}

/// The human-readable table (the script's non-`--json` output).
fn print_table(sessions: &[Session], model_version: &str, threshold: f64) {
    println!("Activity sessions — Oura AAD v{model_version}, Android-parity inputs\n");
    if sessions.is_empty() {
        println!("  No activity segments detected.");
        return;
    }
    println!("  {:<10} {:<13} {:>4}  {:>7}  activity (model confidence)", "date", "time", "dur", "workout");
    for session in sessions {
        let (date, start) = session.start.split_once(' ').unwrap_or((session.start.as_str(), ""));
        let alternatives = session
            .top3
            .iter()
            .skip(1)
            .map(|(name, prob)| format!("{name} {prob:.2}"))
            .collect::<Vec<_>>()
            .join("   ");
        let mark = if session.is_workout >= threshold { '✓' } else { ' ' };
        println!(
            "  {:<10} {}-{:8} {:>3}m  {:.2} {}  {} {:.2}   ·   {}",
            date,
            start,
            session.end,
            session.duration_min,
            session.is_workout,
            mark,
            session.label,
            session.label_confidence,
            alternatives
        );
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// `workouts` for every day -> the session list, `--date` filter applied.
fn collect_sessions(args: &Args, pipeline: &Pipeline) -> Result<Vec<Session>> {
    let inference = activity_backend(&pipeline.model).unwrap_or_else(|| Box::new(UnimplementedInference));
    let mut sessions: Vec<Session> = Vec::new();
    for day in &pipeline.days {
        if let DayOutcome::Run(req) = &day.outcome {
            match inference.run_day(req) {
                Ok(rows) => sessions.extend(sessions_from(day.day, &rows)),
                Err(e) => {
                    if e.downcast_ref::<InferenceUnavailable>().is_some() {
                        return Err(e); // a missing backend is not a rejected day
                    }
                    if args.verbose {
                        eprintln!("{}: model rejected incomplete day ({e})", day.day.iso());
                    }
                }
            }
        }
    }
    if let Some(date) = &args.date {
        sessions.retain(|s| s.start.starts_with(date));
    }
    Ok(sessions)
}

/// The `--json` payload for these arguments, model call included. Split out of
/// `run` so a test can diff it against the golden vector.
pub fn json_for(args: &Args, db_path: &Path, repo: &Path) -> Result<String> {
    let pipeline = prepare(args, db_path, repo)?;
    let sessions = collect_sessions(args, &pipeline)?;
    Ok(render_result(&pipeline.model_version, &sessions))
}

pub fn run(argv: &[String]) -> Result<()> {
    let args = parse_args(argv)?;
    if args.json {
        println!("{}", json_for_args(argv)?);
        return Ok(());
    }
    let mut args = args;
    let repo = repo_root();
    let db_path = resolve_db(args.db.as_deref(), &repo)?;
    args.user_vector = profile_vector(&db_path, args.serial.as_deref());

    // The cold path (`run_full`) is the reference the Python incremental path must
    // match; `--full` is therefore accepted but always taken. The incremental unit
    // cache is not ported (see the module header).
    let pipeline = prepare(&args, &db_path, &repo)?;
    let sessions = collect_sessions(&args, &pipeline)?;
    print_table(&sessions, &pipeline.model_version, args.threshold);
    Ok(())
}

/// The JSON `run` would print, as a string — the in-process entry point for the
/// dashboard's RustRunner (same argv contract as the CLI, golden-tested).
pub fn json_for_args(argv: &[String]) -> Result<String> {
    let mut args = parse_args(argv)?;
    let repo = repo_root();
    let db_path = resolve_db(args.db.as_deref(), &repo)?;
    args.user_vector = profile_vector(&db_path, args.serial.as_deref());
    let pipeline = prepare(&args, &db_path, &repo)?;
    let sessions = collect_sessions(&args, &pipeline)?;
    Ok(render_result(&pipeline.model_version, &sessions))
}

/// The repo root, i.e. the parent of `tools/` (Python: `Path(__file__).parent.parent`).
pub fn repo_root() -> PathBuf {
    match std::env::var("OURA_REPO") {
        Ok(p) => PathBuf::from(p),
        Err(_) => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    }
}
