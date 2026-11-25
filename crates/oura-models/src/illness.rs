//! Rust port of `tools/run_illness_model.py` (Oura "Symptom Radar").
//!
//! Oracle command (golden vector `tests/vectors/illness_golden.json`):
//!   `.venv/Scripts/python.exe tools/run_illness_model.py <db> -5 --json --serial <serial>`
//!
//! Status: the whole data + feature pipeline (DB scan, epoch dating, bedtime
//! windows, nightly biometrics, respiratory rate, 30-day series, long-term
//! baselines, demographics) is ported and checked against the Python oracle by
//! `tests/illness_test.rs`. The TorchScript graph itself sits behind
//! [`IllnessModel`]: the tch backend is feature-gated (`--features torch`) and
//! does not build in this environment yet, so [`NoInference`] is wired in and
//! `run` returns an error the server can fall back on.
//!
//! Numeric policy (the ibi-correct lesson): every value numpy computes in
//! float32 is computed here in f32 at the same point — the 30-day columns are
//! built in f64 and cast to f32 exactly where numpy does `dtype=np.float32` /
//! `.astype(np.float32)`, and the long-term baselines average those f32-rounded
//! values in f64, exactly like the oracle's `nz()`.

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OpenFlags};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// module constants (mirror of the Python module)
// ---------------------------------------------------------------------------

const N_DAYS: usize = 30;
const DAY: f64 = 86400.0; // seconds per local calendar day index
const IBI_TAGS: [i64; 3] = [0x44, 0x60, 0x71];
const TEMP_TAGS: [i64; 3] = [0x46, 0x69, 0x75];
const HRV_TAG: i64 = 0x5D;
const BEDTIME_TAG: i64 = 0x76;
const RING_START_TAG: i64 = 0x41;
const TIME_SYNC_TAGS: [i64; 2] = [0x42, 0x85];

/// decision -> (risk level, traffic light) — see illness-detection.md §4
const DECISION: [(i64, &str, &str); 3] = [
    (0, "BelowThreshold", "NO_SIGNS"),
    (1, "Low", "MINOR_SIGNS"),
    (2, "Moderate", "MAJOR_SIGNS"),
];
/// output tuple index -> symptom-card label, in the Python dict's insertion order
const SHOWN_BIOMARKERS: [(usize, &str); 4] = [
    (2, "AverageBreath"),
    (4, "LowestHeartRate"),
    (5, "AverageHrv"),
    (6, "TemperatureDeviation"),
];

// ---------------------------------------------------------------------------
// small numeric helpers
// ---------------------------------------------------------------------------

/// numpy `np.median`: sort, average of the two middle values for even counts.
fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// numpy `np.mean` over a non-empty slice (pairwise summation, like numpy).
fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    fn rec(a: &[f64]) -> f64 {
        if a.len() <= 8 {
            let mut s = 0.0;
            for x in a {
                s += x;
            }
            return s;
        }
        let m = a.len() / 2;
        rec(&a[..m]) + rec(&a[m..])
    }
    rec(v) / (v.len() as f64)
}

/// numpy `np.std` (population, ddof=0).
fn std_dev(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let m = mean(v);
    let var = mean(&v.iter().map(|x| (x - m) * (x - m)).collect::<Vec<f64>>());
    var.sqrt()
}

/// `nz()`: the finite values, widened to f64.
fn finite(v: &[f32]) -> Vec<f64> {
    v.iter().filter(|x| x.is_finite()).map(|x| *x as f64).collect()
}

fn finite64(v: &[f64]) -> Vec<f64> {
    v.iter().filter(|x| x.is_finite()).copied().collect()
}

/// `int((unix + tz * 3600) // DAY)` — the local calendar day index every map in
/// the oracle is keyed by (wake day, activity day).
fn day_index(unix_local: f64) -> i64 {
    (unix_local / DAY).floor() as i64
}

/// Howard Hinnant's `civil_from_days`, verbatim from the Python `civil()`.
fn civil(days: i64) -> (i64, i64, i64) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (y + if m <= 2 { 1 } else { 0 }, m, d)
}

/// `date(y,m,d).weekday()` with Mon=0; day 0 (1970-01-01) was a Thursday.
fn weekday_from_days(days: i64) -> i64 {
    (days + 3).rem_euclid(7)
}

// ---------------------------------------------------------------------------
// epoch dating — port of tools/epoch_time.py (build_epochs / make_unix_s / is_dated)
// ---------------------------------------------------------------------------

const EPOCH_RESET_SLACK_DS: i64 = 6 * 3600 * 10;
const FUTURE_SLACK_S: f64 = 6.0 * 3600.0;
const ANCHOR_AGREEMENT_S: f64 = 30.0 * 60.0;
const ANCHOR_AGREEMENT_FRACTION: f64 = 0.02;

#[derive(Clone)]
struct Epoch {
    min_ds: i64,
    max_ds: i64,
    fallback_unix: i64,
    capture_min: i64,
    capture_max: i64,
    anchors: Vec<(i64, i64)>, // (ds, unix_time) from time-sync / rtc events
    boots: Vec<i64>,          // ring_start ds
}

/// One boot's anchors pre-sorted by ds, with the parallel ds list for bisect.
struct Anch {
    indexed: Vec<(i64, i64, usize)>, // (ds, unix, insertion index)
    dsl: Vec<i64>,
}

pub struct Clock {
    epochs: Vec<Epoch>,
    anch: Vec<Anch>,
}

enum Bracket {
    Consistent,
    Stalled((i64, i64), (i64, i64)),
    Erratic,
}

impl Clock {
    fn new() -> Clock {
        Clock { epochs: vec![], anch: vec![] }
    }

    fn blank(ds: i64, cu: i64) -> Epoch {
        Epoch {
            min_ds: ds,
            max_ds: ds,
            fallback_unix: cu,
            capture_min: cu,
            capture_max: cu,
            anchors: vec![],
            boots: vec![],
        }
    }

    /// `build_epochs()` fed in `(captured_unix, id)` order.
    fn add_row(&mut self, ds: i64, tag: i64, dj: &str, cu: i64) {
        if let Some(e) = self.epochs.last_mut() {
            if ds >= e.max_ds - EPOCH_RESET_SLACK_DS {
                if ds >= e.max_ds {
                    e.max_ds = ds;
                    e.fallback_unix = cu;
                }
                e.min_ds = e.min_ds.min(ds);
                e.capture_min = e.capture_min.min(cu);
                e.capture_max = e.capture_max.max(cu);
            } else {
                self.epochs.push(Clock::blank(ds, cu));
            }
        } else {
            self.epochs.push(Clock::blank(ds, cu));
        }
        let last = self.epochs.last_mut().unwrap();
        if tag == RING_START_TAG {
            last.boots.push(ds);
        }
        if TIME_SYNC_TAGS.contains(&tag) && !dj.is_empty() {
            if let Ok(v) = serde_json::from_str::<Value>(dj) {
                if let Some(obj) = v.as_object() {
                    if let Some(u) = json_int(obj.get("unix_time")) {
                        last.anchors.push((ds, u));
                    }
                }
            }
        }
    }

    /// Diagnostics: how many boot epochs the sync events produced.
    pub(crate) fn epoch_count(&self) -> usize {
        self.epochs.len()
    }

    /// `_prepared()`
    fn finish(&mut self) {
        self.anch.clear();
        for e in &self.epochs {
            let mut indexed: Vec<(i64, i64, usize)> = e
                .anchors
                .iter()
                .enumerate()
                .map(|(i, (ds, unix))| (*ds, *unix, i))
                .collect();
            indexed.sort();
            let dsl = indexed.iter().map(|a| a.0).collect();
            self.anch.push(Anch { indexed, dsl });
        }
    }

    /// `bisect.bisect_left`
    fn bisect_left(dsl: &[i64], ds: i64) -> usize {
        let mut lo = 0usize;
        let mut hi = dsl.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if dsl[mid] < ds {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn capture_distance(e: &Epoch, cu: i64) -> i64 {
        if cu < e.capture_min {
            e.capture_min - cu
        } else if cu > e.capture_max {
            cu - e.capture_max
        } else {
            0
        }
    }

    /// `_pick_epoch()`
    fn pick_epoch(&self, ds: i64, cu: Option<i64>) -> usize {
        let candidates: Vec<usize> = (0..self.epochs.len())
            .filter(|&i| {
                let e = &self.epochs[i];
                e.min_ds - EPOCH_RESET_SLACK_DS <= ds && ds <= e.max_ds + EPOCH_RESET_SLACK_DS
            })
            .collect();
        if candidates.is_empty() {
            return self.epochs.len() - 1;
        }
        match cu {
            Some(cu) => *candidates
                .iter()
                .min_by_key(|i| Self::capture_distance(&self.epochs[**i], cu))
                .unwrap(),
            None => *candidates
                .iter()
                .min_by_key(|i| {
                    let e = &self.epochs[**i];
                    e.max_ds - e.min_ds
                })
                .unwrap(),
        }
    }

    /// `_nearest()` — ties resolve to the earliest insertion index.
    fn nearest(&self, ei: usize, ds: i64) -> (i64, i64) {
        let a = &self.anch[ei];
        let idx = Self::bisect_left(&a.dsl, ds);
        if idx == 0 {
            return (a.indexed[0].0, a.indexed[0].1);
        }
        if idx >= a.indexed.len() {
            let x = a.indexed.last().unwrap();
            return (x.0, x.1);
        }
        let (bd, bu, bi) = a.indexed[idx - 1];
        let (ad, au, ai) = a.indexed[idx];
        if (ds - bd, bi) <= (ad - ds, ai) {
            (bd, bu)
        } else {
            (ad, au)
        }
    }

    /// `_bracket()`
    fn bracket(&self, ei: usize, ds: i64) -> Bracket {
        let a = &self.anch[ei];
        let idx = Self::bisect_left(&a.dsl, ds);
        if idx == 0 || idx >= a.indexed.len() {
            return Bracket::Consistent;
        }
        let prev = a.indexed[idx - 1];
        let nxt = a.indexed[idx];
        if nxt.0 == ds {
            return Bracket::Consistent;
        }
        let wall_s = nxt.1 - prev.1;
        let counter_s = (nxt.0 - prev.0) as f64 / 10.0;
        let tolerance = ANCHOR_AGREEMENT_S.max(counter_s * ANCHOR_AGREEMENT_FRACTION);
        if (wall_s as f64) < counter_s - tolerance {
            Bracket::Erratic
        } else if (wall_s as f64) > counter_s + tolerance {
            Bracket::Stalled((prev.0, prev.1), (nxt.0, nxt.1))
        } else {
            Bracket::Consistent
        }
    }

    /// `make_unix_s()(ds, captured_unix)`; None = the time cannot be assigned.
    fn unix_s(&self, ds: i64, cu: i64) -> Option<f64> {
        let ei = self.pick_epoch(ds, Some(cu));
        let e = &self.epochs[ei];
        if !e.anchors.is_empty() {
            let (anchor_ds, anchor_unix) = self.nearest(ei, ds);
            let predicted = anchor_unix as f64 + (ds - anchor_ds) as f64 / 10.0;
            match self.bracket(ei, ds) {
                Bracket::Stalled(before, after) => {
                    let late = after.1 as f64 - (after.0 - ds) as f64 / 10.0;
                    let early = before.1 as f64 + (ds - before.0) as f64 / 10.0;
                    let boots: Vec<i64> = e
                        .boots
                        .iter()
                        .copied()
                        .filter(|b| before.0 < *b && *b <= after.0)
                        .collect();
                    if !boots.is_empty() {
                        let lo = *boots.iter().min().unwrap();
                        let hi = *boots.iter().max().unwrap();
                        if lo <= ds && ds < hi {
                            return None; // lost time cannot be assigned to one reboot
                        }
                        return Some(if ds >= hi { late } else { early });
                    }
                    return Some(if late <= cu as f64 + FUTURE_SLACK_S { late } else { early });
                }
                Bracket::Consistent => {
                    if predicted <= cu as f64 + FUTURE_SLACK_S {
                        return Some(predicted);
                    }
                }
                Bracket::Erratic => return Some(predicted), // undated; is_dated() checks
            }
            // Borrow another boot's clock only when this ds continues its counter.
            let mut best: Option<f64> = None;
            for epoch in &self.epochs {
                if !(epoch.min_ds - EPOCH_RESET_SLACK_DS <= ds
                    && ds <= epoch.max_ds + EPOCH_RESET_SLACK_DS)
                {
                    continue;
                }
                for (a_ds, unix) in &epoch.anchors {
                    let p = *unix as f64 + (ds - a_ds) as f64 / 10.0;
                    if p <= cu as f64 + FUTURE_SLACK_S {
                        best = Some(best.map_or(p, |b: f64| b.max(p)));
                    }
                }
            }
            if let Some(p) = best {
                return Some(p);
            }
        }
        let fallback = e.fallback_unix as f64 - (e.max_ds - ds) as f64 / 10.0;
        Some(fallback.min(cu as f64 + FUTURE_SLACK_S))
    }

    /// `undated_reason()`
    fn undated_reason(&self, ds: i64, cu: i64) -> Option<&'static str> {
        let any = self
            .epochs
            .iter()
            .any(|e| e.min_ds - EPOCH_RESET_SLACK_DS <= ds && ds <= e.max_ds + EPOCH_RESET_SLACK_DS);
        if !any {
            return Some("missing_anchor");
        }
        let ei = self.pick_epoch(ds, Some(cu));
        let e = &self.epochs[ei];
        if e.anchors.is_empty() {
            return Some("missing_anchor");
        }
        match self.bracket(ei, ds) {
            Bracket::Erratic => Some("accelerated_counter"),
            Bracket::Stalled(before, after) => {
                let boots: Vec<i64> = e
                    .boots
                    .iter()
                    .copied()
                    .filter(|b| before.0 < *b && *b <= after.0)
                    .collect();
                if !boots.is_empty() {
                    let lo = *boots.iter().min().unwrap();
                    let hi = *boots.iter().max().unwrap();
                    if lo <= ds && ds < hi {
                        return Some("ambiguous_reboot_stall");
                    }
                }
                None
            }
            Bracket::Consistent => {
                let (anchor_ds, anchor_unix) = self.nearest(ei, ds);
                if anchor_unix as f64 + (ds - anchor_ds) as f64 / 10.0 <= cu as f64 + FUTURE_SLACK_S
                {
                    return None;
                }
                for epoch in &self.epochs {
                    if !(epoch.min_ds - EPOCH_RESET_SLACK_DS <= ds
                        && ds <= epoch.max_ds + EPOCH_RESET_SLACK_DS)
                    {
                        continue;
                    }
                    for (a_ds, unix) in &epoch.anchors {
                        if *unix as f64 + (ds - a_ds) as f64 / 10.0 <= cu as f64 + FUTURE_SLACK_S {
                            return None;
                        }
                    }
                }
                Some("accelerated_counter")
            }
        }
    }

    fn is_dated(&self, ds: i64, cu: i64) -> bool {
        self.undated_reason(ds, cu).is_none()
    }
}

/// JSON value -> Python `int(x)` (truncate floats, parse digit strings).
fn json_int(v: Option<&Value>) -> Option<i64> {
    match v? {
        Value::Number(n) => n.as_f64().map(|x| x.trunc() as i64),
        Value::String(s) => s
            .trim()
            .parse::<i64>()
            .ok()
            .or_else(|| s.trim().parse::<f64>().ok().map(|x| x.trunc() as i64)),
        _ => None,
    }
}

/// JSON value -> Python `float(x)`; None for null/absent.
fn json_f64(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// DB scan
// ---------------------------------------------------------------------------

const SELECT_ROWS: &str =
    "SELECT ring_timestamp, tag, decoded_json, captured_unix FROM events \
     WHERE decoded_json IS NOT NULL ORDER BY captured_unix, id";
const SELECT_ROWS_SERIAL: &str =
    "SELECT ring_timestamp, tag, decoded_json, captured_unix FROM events \
     WHERE decoded_json IS NOT NULL AND serial = ? ORDER BY captured_unix, id";

fn open_db(path: &Path) -> Result<Connection> {
    Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("open {}", path.display()))
}

fn for_each_row<F: FnMut(i64, i64, &str, i64)>(
    conn: &Connection,
    serial: Option<&str>,
    mut f: F,
) -> Result<()> {
    let sql = if serial.is_some() { SELECT_ROWS_SERIAL } else { SELECT_ROWS };
    let mut stmt = conn.prepare(sql)?;
    let mut rows = match serial {
        Some(s) => stmt.query_map(params![s], row_to_tuple)?,
        None => stmt.query_map([], row_to_tuple)?,
    };
    while let Some(row) = rows.next() {
        let (ds, tag, dj, cu) = row?;
        f(ds, tag, &dj, cu);
    }
    Ok(())
}

fn row_to_tuple(r: &rusqlite::Row<'_>) -> rusqlite::Result<(i64, i64, String, i64)> {
    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
}

// ---------------------------------------------------------------------------
// nightly biometrics + activity
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Night {
    avg_breath: f64,
    avg_hr: f64,
    lowest_hr: f64,
    avg_hrv: f64,
    skin_temp: f64,
    dur: f64,
}

#[derive(Default)]
struct Raw {
    windows: Vec<(f64, f64)>,
    ibi_t: Vec<f64>,
    ibi_v: Vec<f64>,
    hrv_t: Vec<f64>,
    hrv_rmssd: Vec<f64>,
    hrv_hr: Vec<f64>,
    temp_t: Vec<f64>,
    temp_v: Vec<f64>,
    sed: BTreeMap<i64, i64>,
    rest: BTreeMap<i64, i64>,
}

/// `bed_windows` + `nightly_biometrics` collection + `activity_by_day` in one
/// streaming pass over the dated rows (same row order as the oracle's list).
fn collect_raw(conn: &Connection, clock: &Clock, tz: f64, serial: Option<&str>) -> Result<Raw> {
    let mut raw = Raw::default();
    for_each_row(conn, serial, |ds, tag, dj, cu| {
        if !clock.is_dated(ds, cu) {
            return; // undated data belongs to no night or day
        }
        let parsed: Value = match serde_json::from_str(dj) {
            Ok(v) => v,
            Err(_) => return,
        };
        let d = match parsed.as_object() {
            Some(o) => o,
            None => return,
        };
        let base = match clock.unix_s(ds, cu) {
            Some(b) => b,
            None => return,
        };
        if tag == BEDTIME_TAG {
            let s = json_f64(d.get("bedtime_start_ds"));
            let e = json_f64(d.get("bedtime_end_ds"));
            if let (Some(s), Some(e)) = (s, e) {
                if e > s {
                    // bed_windows(): the start/end ds are dated on their own, NOT as an
                    // offset of the event row's ds — they can land in another boot epoch.
                    let (sd, ed) = (s.trunc() as i64, e.trunc() as i64);
                    if let (Some(ws), Some(we)) = (clock.unix_s(sd, cu), clock.unix_s(ed, cu)) {
                        raw.windows.push((ws, we));
                    }
                }
            }
            return;
        }
        if IBI_TAGS.contains(&tag) {
            let mut acc = 0.0;
            if let Some(list) = d.get("ibi_ms").and_then(|v| v.as_array()) {
                for item in list {
                    // Python: `for v in (d.get("ibi_ms") or []): if v:`
                    let v = match json_f64(Some(item)) {
                        Some(v) if v != 0.0 => v,
                        _ => continue,
                    };
                    acc += v / 1000.0;
                    raw.ibi_t.push(base + acc);
                    raw.ibi_v.push(v);
                }
            }
            return;
        }
        if tag == HRV_TAG {
            let step = json_f64(d.get("interval_min")).unwrap_or(5.0) * 60.0;
            let rm: Vec<Option<f64>> = d
                .get("rmssd_ms")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(|x| json_f64(Some(x))).collect())
                .unwrap_or_default();
            let hb: Vec<Option<f64>> = d
                .get("hr_bpm")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(|x| json_f64(Some(x))).collect())
                .unwrap_or_default();
            let n = rm.len();
            for (i, r) in rm.iter().enumerate() {
                // `base` is the packet's LAST average; earlier ones walk back by `step`.
                let r = match r {
                    Some(r) if *r > 0.0 => *r,
                    _ => continue,
                };
                raw.hrv_t.push(base - (n as i64 - 1 - i as i64) as f64 * step);
                raw.hrv_rmssd.push(r);
                raw.hrv_hr.push(hb.get(i).copied().flatten().unwrap_or(f64::NAN));
            }
            return;
        }
        if TEMP_TAGS.contains(&tag) {
            if let Some(list) = d.get("temps_c").and_then(|v| v.as_array()) {
                for item in list {
                    if let Some(tv) = json_f64(Some(item)) {
                        if tv != 0.0 && tv > 20.0 {
                            // skin temp; drop obvious off-finger lows
                            raw.temp_t.push(base);
                            raw.temp_v.push(tv);
                        }
                    }
                }
            }
            return;
        }
        // activity_by_day: the oracle only parses rows whose JSON text contains
        // the literal `"met"` — keep that shortcut.
        if dj.contains("\"met\"") {
            if let Some(list) = d.get("met").and_then(|v| v.as_array()) {
                for (i, item) in list.iter().enumerate() {
                    let mv = match json_f64(Some(item)) {
                        Some(mv) => mv,
                        None => continue,
                    };
                    let day = day_index(base + i as f64 * 60.0 + tz * 3600.0);
                    if mv < 1.05 {
                        *raw.rest.entry(day).or_insert(0) += 60;
                    } else if mv < 2.0 {
                        *raw.sed.entry(day).or_insert(0) += 60;
                    }
                }
            }
        }
    })?;
    Ok(raw)
}

/// A time-sorted index over one sample array, so a `[start, end]` window can be
/// resolved without an O(n) scan per night.
struct TimeIndex {
    order: Vec<usize>,
    sorted: Vec<f64>,
}

impl TimeIndex {
    fn new(times: &[f64]) -> TimeIndex {
        let mut order: Vec<usize> = (0..times.len()).collect();
        order.sort_by(|&a, &b| times[a].partial_cmp(&times[b]).unwrap_or(std::cmp::Ordering::Equal));
        let sorted = order.iter().map(|&i| times[i]).collect();
        TimeIndex { order, sorted }
    }

    /// Indices whose time lies in `[start, end]`, returned in original array
    /// order — that is numpy's boolean-mask order, which `np.interp` depends on.
    fn masked(&self, start: f64, end: f64) -> Vec<usize> {
        let lo = partition_point(&self.sorted, |x| x < start);
        let hi = partition_point(&self.sorted, |x| x <= end);
        let mut idx = self.order[lo..hi].to_vec();
        idx.sort_unstable();
        idx
    }
}

fn partition_point(v: &[f64], pred: impl Fn(f64) -> bool) -> usize {
    let mut lo = 0usize;
    let mut hi = v.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if pred(v[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// `nightly_biometrics()`: per wake-day biometrics, longest sleep per day wins.
fn nightly_biometrics(raw: &Raw, tz: f64) -> BTreeMap<i64, Night> {
    let hrv = TimeIndex::new(&raw.hrv_t);
    let ibi = TimeIndex::new(&raw.ibi_t);
    let temp = TimeIndex::new(&raw.temp_t);
    let mut per_day: BTreeMap<i64, Night> = BTreeMap::new();
    for &(start, end) in &raw.windows {
        let dur = end - start;
        if dur < 3600.0 {
            continue; // naps under an hour do not count
        }
        let wake_day = day_index(end + tz * 3600.0);
        let hi = hrv.masked(start, end);
        if hi.len() < 3 {
            continue;
        }
        let rmssd = median(&hi.iter().map(|&i| raw.hrv_rmssd[i]).collect::<Vec<f64>>());
        let mut hr_win: Vec<f64> = hi.iter().map(|&i| raw.hrv_hr[i]).collect();
        hr_win.retain(|x| x.is_finite());
        if hr_win.len() < 3 {
            continue;
        }
        let avg_hr = median(&hr_win);
        let lowest = hr_win.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let ii = ibi.masked(start, end);
        let breath = if ii.len() > 200 {
            let ts: Vec<f64> = ii.iter().map(|&i| raw.ibi_t[i]).collect();
            let vs: Vec<f64> = ii.iter().map(|&i| raw.ibi_v[i]).collect();
            respiratory_rate(&ts, &vs).0
        } else {
            f64::NAN
        };
        let ti = temp.masked(start, end);
        let skin = if ti.is_empty() {
            f64::NAN
        } else {
            median(&ti.iter().map(|&i| raw.temp_v[i]).collect::<Vec<f64>>())
        };
        let night = Night {
            avg_breath: breath,
            avg_hr,
            lowest_hr: lowest,
            avg_hrv: rmssd,
            skin_temp: skin,
            dur,
        };
        let replace = match per_day.get(&wake_day) {
            Some(prev) => dur > prev.dur,
            None => true,
        };
        if replace {
            per_day.insert(wake_day, night);
        }
    }
    per_day
}

// ---------------------------------------------------------------------------
// respiratory rate — port of tools/respiratory_rate.py. The venv has no scipy,
// so the windowed-sinc FIR fallback is the live path in the oracle.
// ---------------------------------------------------------------------------

const FS: f64 = 4.0;
const RESP_LO: f64 = 0.15;
const RESP_HI: f64 = 0.40;
const WIN_S: f64 = 60.0;
const STEP_S: f64 = 30.0;
const IBI_LO: f64 = 300.0;
const IBI_HI: f64 = 2000.0;

/// `np.hanning(n)`
fn hanning(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos())
        .collect()
}

/// `np.sinc(x)` = sin(pi x)/(pi x)
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let p = std::f64::consts::PI * x;
    p.sin() / p
}

/// `np.interp` semantics, including numpy's binary search over `xp`.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let num = xp.len();
    if num == 1 {
        return fp[0];
    }
    if x < xp[0] {
        return fp[0];
    }
    if x >= xp[num - 1] {
        return fp[num - 1];
    }
    let mut ilo = 0usize;
    let mut ihi = num - 1;
    while ihi - ilo > 1 {
        let mid = (ihi + ilo) / 2;
        if x < xp[mid] {
            ihi = mid;
        } else {
            ilo = mid;
        }
    }
    let slope = (fp[ihi] - fp[ilo]) / (xp[ihi] - xp[ilo]);
    fp[ilo] + slope * (x - xp[ilo])
}

/// `_tachogram()`: ectopic filter, then linear resample onto the FS grid.
fn tachogram(ts: &[f64], ibi: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut keep: Vec<(f64, f64)> = ibi
        .iter()
        .zip(ts.iter())
        .filter(|(b, _)| **b >= IBI_LO && **b <= IBI_HI)
        .map(|(b, t)| (*t, *b))
        .collect();
    if keep.len() < 30 {
        return None;
    }
    let vals: Vec<f64> = keep.iter().map(|(_, b)| *b).collect();
    let med = median(&vals);
    keep.retain(|(_, b)| (b - med).abs() <= 0.30 * med);
    if keep.len() < 30 || keep[keep.len() - 1].0 - keep[0].0 < WIN_S {
        return None;
    }
    let (t0, t1) = (keep[0].0, keep[keep.len() - 1].0);
    let xs: Vec<f64> = keep.iter().map(|(t, _)| *t).collect();
    let ys: Vec<f64> = keep.iter().map(|(_, b)| *b).collect();
    // np.arange(t0, t1, 1/FS): n = ceil((stop-start)/step), then numpy drops the last
    // index when start + step*n lands on or past stop (exact spans).
    let step = 1.0 / FS;
    let mut n = ((t1 - t0) / step).ceil() as usize;
    if n > 0 && t0 + step * n as f64 >= t1 {
        n -= 1;
    }
    let grid: Vec<f64> = (0..n).map(|i| t0 + i as f64 * step).collect();
    let res: Vec<f64> = grid.iter().map(|&x| interp(x, &xs, &ys)).collect();
    Some((grid, res))
}

/// `_bandpass()` — the windowed-sinc FIR fallback (scipy is not installed).
fn bandpass(x: &[f64]) -> Vec<f64> {
    let m = mean(x);
    let x: Vec<f64> = x.iter().map(|v| v - m).collect();
    let n = 129usize;
    let w = hanning(n);
    let mut kern = vec![0.0; n];
    for i in 0..n {
        let t = i as f64 - (n as f64 - 1.0) / 2.0;
        let lo = 2.0 * RESP_LO / FS * sinc(2.0 * RESP_LO / FS * t);
        let hi = 2.0 * RESP_HI / FS * sinc(2.0 * RESP_HI / FS * t);
        kern[i] = (hi - lo) * w[i];
    }
    let km = mean(&kern);
    for k in kern.iter_mut() {
        *k -= km;
    }
    // np.convolve(x, kern, mode="same") with an odd kernel: the centred window,
    // offset (129-1)/2 = 64. out[i] = sum_j x[j]*kern[i+off-j] over the j where both
    // indices are in range, accumulated in increasing j (numpy's loop order).
    let off = (n - 1) / 2;
    let mut out = vec![0.0; x.len()];
    for i in 0..x.len() {
        let mut acc = 0.0;
        let lo = i.saturating_sub(off);
        let hi = (i + off).min(x.len() - 1);
        for j in lo..=hi {
            acc += x[j] * kern[i + off - j];
        }
        out[i] = acc;
    }
    out
}

/// Naive complex DFT (for the small non-power-of-two sub-transforms).
fn dft(v: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = v.len();
    let mut out = vec![(0.0, 0.0); n];
    for k in 0..n {
        let (mut sr, mut si) = (0.0, 0.0);
        for (i, (re, im)) in v.iter().enumerate() {
            let ang = -2.0 * std::f64::consts::PI * ((i * k) % n) as f64 / n as f64;
            let (c, s) = (ang.cos(), ang.sin());
            sr += re * c - im * s;
            si += re * s + im * c;
        }
        out[k] = (sr, si);
    }
    out
}

/// Iterative radix-2 FFT (n must be a power of two).
fn radix2(v: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = v.len();
    let mut a = v.to_vec();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let wl = (ang.cos(), ang.sin());
        let mut i = 0usize;
        while i < n {
            let mut w = (1.0, 0.0);
            for k in 0..len / 2 {
                let u = a[i + k];
                let x = a[i + k + len / 2];
                let t = (x.0 * w.0 - x.1 * w.1, x.0 * w.1 + x.1 * w.0);
                a[i + k] = (u.0 + t.0, u.1 + t.1);
                a[i + k + len / 2] = (u.0 - t.0, u.1 - t.1);
                w = (w.0 * wl.0 - w.1 * wl.1, w.0 * wl.1 + w.1 * wl.0);
            }
            i += len;
        }
        len <<= 1;
    }
    a
}

/// Complex FFT for composite n = n1*n2 (four-step algorithm), n1 being the
/// largest power of two dividing n. The pipeline only hits n = 240 = 16*15.
fn fft(v: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = v.len();
    if n <= 1 {
        return v.to_vec();
    }
    let n1 = 1usize << (n.trailing_zeros() as usize);
    let n2 = n / n1;
    if n2 == 1 {
        return radix2(v);
    }
    // A[k1][n2i] = sum_{n1i} x[n1i*n2 + n2i] * W_{n1}^{n1i*k1}
    let mut a = vec![vec![(0.0, 0.0); n2]; n1];
    for n2i in 0..n2 {
        let col: Vec<(f64, f64)> = (0..n1).map(|n1i| v[n1i * n2 + n2i]).collect();
        let col = radix2(&col);
        for k1 in 0..n1 {
            a[k1][n2i] = col[k1];
        }
    }
    // B[k1][k2] = sum_{n2i} a[k1][n2i] * W_n^{n2i*k1} * W_{n2}^{n2i*k2}
    let mut out = vec![(0.0, 0.0); n];
    for k1 in 0..n1 {
        let col: Vec<(f64, f64)> = (0..n2)
            .map(|n2i| {
                let ang = -2.0 * std::f64::consts::PI * ((n2i * k1) % n) as f64 / n as f64;
                let (c, s) = (ang.cos(), ang.sin());
                let (re, im) = a[k1][n2i];
                (re * c - im * s, re * s + im * c)
            })
            .collect();
        let b = dft(&col);
        for k2 in 0..n2 {
            out[k1 + n1 * k2] = b[k2];
        }
    }
    out
}

/// `np.abs(np.fft.rfft(x)) ** 2` — numpy takes the modulus (hypot) then squares.
fn rfft_power(x: &[f64]) -> Vec<f64> {
    let c: Vec<(f64, f64)> = x.iter().map(|&v| (v, 0.0)).collect();
    let s = fft(&c);
    s[..x.len() / 2 + 1]
        .iter()
        .map(|(re, im)| re.hypot(*im).powi(2))
        .collect()
}

/// `_dominant_freq()`
fn dominant_freq(seg: &[f64]) -> Option<f64> {
    let n = seg.len();
    let m = mean(seg);
    let w = hanning(n);
    let seg: Vec<f64> = seg.iter().zip(w.iter()).map(|(v, w)| (v - m) * w).collect();
    let power = rfft_power(&seg);
    // np.fft.rfftfreq(n, 1/FS) = k*FS/n; keep the respiratory band
    let band: Vec<(usize, f64)> = (0..power.len())
        .map(|k| (k, k as f64 * FS / n as f64))
        .filter(|(_, f)| *f >= RESP_LO && *f <= RESP_HI)
        .collect();
    if band.is_empty() {
        return None;
    }
    let bp: Vec<f64> = band.iter().map(|(k, _)| power[*k]).collect();
    let mx = bp.iter().fold(f64::MIN, |a, &b| a.max(b));
    if mx < 4.0 * median(&bp) {
        return None; // no real peak
    }
    let mut best = 0usize;
    for i in 0..bp.len() {
        if bp[i] > bp[best] {
            best = i; // np.argmax: first maximum
        }
    }
    Some(band[best].1)
}

/// `respiratory_rate()` -> (avg br/min, variability, n_windows)
fn respiratory_rate(ts: &[f64], ibi: &[f64]) -> (f64, f64, usize) {
    let x = match tachogram(ts, ibi) {
        Some((_grid, x)) => x,
        None => return (f64::NAN, f64::NAN, 0),
    };
    let x = bandpass(&x);
    let win = (WIN_S * FS) as usize;
    let step = (STEP_S * FS) as usize;
    let mut rr: Vec<f64> = vec![];
    let mut i = 0usize;
    while i + win <= x.len() {
        if let Some(f) = dominant_freq(&x[i..i + win]) {
            rr.push(f * 60.0);
        }
        i += step;
    }
    if rr.len() < 3 {
        return (f64::NAN, f64::NAN, rr.len());
    }
    (median(&rr), std_dev(&rr), rr.len())
}

// ---------------------------------------------------------------------------
// 30-day series + long-term baselines
// ---------------------------------------------------------------------------

struct Series {
    breath: Vec<f32>,
    avg_hr: Vec<f32>,
    low_hr: Vec<f32>,
    hrv: Vec<f32>,
    temp_dev: Vec<f32>,
    sed: Vec<f32>,
    rest: Vec<f32>,
    skin: Vec<f64>,
    base_temp: f64,
}

fn night_val(per_day: &BTreeMap<i64, Night>, day: i64, f: impl Fn(&Night) -> f64) -> f64 {
    per_day.get(&day).map(f).unwrap_or(f64::NAN)
}

/// `col(fn)` — one [30,1] float32 column, index 0 = anchor.
fn col_of(days: &[i64], f: impl Fn(i64) -> f64) -> Vec<f32> {
    days.iter().map(|&d| f(d) as f32).collect()
}

/// `series()` — index 0 = anchor (today), 29 = oldest; missing days are NaN.
fn build_series(
    per_day: &BTreeMap<i64, Night>,
    sed: &BTreeMap<i64, i64>,
    rest: &BTreeMap<i64, i64>,
    anchor: i64,
) -> Series {
    let days: Vec<i64> = (0..N_DAYS).map(|i| anchor - i as i64).collect();
    let breath = col_of(&days, |d| night_val(per_day, d, |n| n.avg_breath));
    let avg_hr = col_of(&days, |d| night_val(per_day, d, |n| n.avg_hr));
    let low_hr = col_of(&days, |d| night_val(per_day, d, |n| n.lowest_hr));
    let hrv = col_of(&days, |d| night_val(per_day, d, |n| n.avg_hrv));
    // skin stays float64 in the oracle; only temp_dev is cast to float32
    let skin: Vec<f64> = days.iter().map(|&d| night_val(per_day, d, |n| n.skin_temp)).collect();
    let sf = finite64(&skin);
    let base_temp = if !sf.is_empty() { median(&sf) } else { f64::NAN };
    let temp_dev: Vec<f32> = skin.iter().map(|v| (v - base_temp) as f32).collect();
    // Oura pairs each night with the full day *before* it, not the wake day's
    // partial morning (activityMap.get(day.minusDays(1))).
    let sed_c = col_of(&days, |d| sed.get(&(d - 1)).map(|v| *v as f64).unwrap_or(f64::NAN));
    let rest_c = col_of(&days, |d| rest.get(&(d - 1)).map(|v| *v as f64).unwrap_or(f64::NAN));
    Series { breath, avg_hr, low_hr, hrv, temp_dev, sed: sed_c, rest: rest_c, skin, base_temp }
}

// ---------------------------------------------------------------------------
// model I/O
// ---------------------------------------------------------------------------

/// The 23 tensors `forward()` takes: 9 series of shape [30,1], then 14 scalars
/// of shape [1,1] (cycle phase/reason, predicted period/ovulation, age, BMI,
/// sex, day_of_week, and the 8 long-term baselines).
pub struct ModelInputs {
    pub series: [Vec<f32>; 9],
    pub scalars: Vec<f32>,
}

/// The parts of the 18-tensor output tuple the script reads: index 0 score,
/// index 1 decision, indices 2..8 the seven `[is_out, value, lower, upper]`
/// biomarker vectors.
pub struct ModelOutput {
    pub score: f64,
    pub decision: f64,
    pub biomarkers: Vec<Vec<f64>>,
}

/// Inference backend. Normalization, calibration and thresholds all live inside
/// the TorchScript graph — we only feed raw physiological values.
pub trait IllnessModel {
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput>;
}

/// The backend when the crate is built without `--features torch`: the whole data
/// and feature pipeline still runs, only inference is missing, so `run` fails and
/// the server keeps its PythonRunner fallback.
pub struct NoInference;

impl IllnessModel for NoInference {
    fn forward(&self, _inputs: &ModelInputs) -> Result<ModelOutput> {
        bail!(
            "illness inference unavailable: this binary was built without the tch backend \
             (--features torch); the 23 model inputs were built fine"
        )
    }
}

/// Resolve the model file the way the script does: `REPO/notes/models`, then the
/// sibling `../open_oura/notes/models`. REPO is the crate's workspace root (the
/// script's parent-of-parent), with the current directory as a fallback.
#[cfg(feature = "torch")]
fn model_path() -> Option<PathBuf> {
    let name = "illness_detection_0_5_1.pt";
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo = manifest.parent().and_then(|p| p.parent()).unwrap_or(Path::new("."));
    let mut dirs = vec![repo.join("notes/models")];
    if let Some(parent) = repo.parent() {
        dirs.push(parent.join("open_oura/notes/models"));
    }
    dirs.push(PathBuf::from("notes/models"));
    dirs.into_iter().map(|d| d.join(name)).find(|p| p.exists())
}

#[cfg(feature = "torch")]
pub struct TchIllnessModel {
    inner: tch::CModule,
}

#[cfg(feature = "torch")]
impl TchIllnessModel {
    pub fn load(path: &Path) -> Result<TchIllnessModel> {
        let mut m = tch::CModule::load(path).with_context(|| format!("load {}", path.display()))?;
        m.set_eval();
        Ok(TchIllnessModel { inner: m })
    }
}

#[cfg(feature = "torch")]
impl IllnessModel for TchIllnessModel {
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput> {
        use tch::{Device, IValue, Tensor};
        // Every input the script passes is float32 (np.float32 columns,
        // torch.tensor(..., dtype=torch.float32) scalars), and so are ours.
        let mk = |v: &[f32], shape: &[i64]| {
            Tensor::from_slice(v).reshape(shape).to_device(Device::Cpu)
        };
        let mut ivs: Vec<IValue> = Vec::with_capacity(23);
        for s in &inputs.series {
            ivs.push(IValue::Tensor(mk(s, &[N_DAYS as i64, 1])));
        }
        for s in &inputs.scalars {
            ivs.push(IValue::Tensor(mk(&[*s], &[1, 1])));
        }
        // The graph returns an 18-tuple, so the ivalue entry point is required.
        let out = self.inner.forward_is(&ivs)?;
        let items: Vec<IValue> = match out {
            IValue::Tuple(t) => t,
            IValue::TensorList(t) => t.into_iter().map(IValue::Tensor).collect(),
            _ => bail!("unexpected illness model output type"),
        };
        // Which element kind the graph emits per output is not documented, so read
        // whichever kind is there and widen.
        let read = |i: usize| -> Result<Vec<f64>> {
            let t = match &items[i] {
                IValue::Tensor(t) => t.reshape(&[-1]),
                _ => bail!("illness model output {i} is not a tensor"),
            };
            if let Ok(v) = Vec::<f64>::try_from(&t) {
                return Ok(v);
            }
            if let Ok(v) = Vec::<f32>::try_from(&t) {
                return Ok(v.iter().map(|x| *x as f64).collect());
            }
            if let Ok(v) = Vec::<i64>::try_from(&t) {
                return Ok(v.iter().map(|x| *x as f64).collect());
            }
            bail!("illness model output {i} has an unreadable element kind")
        };
        let score = read(0)?.first().copied().unwrap_or(f64::NAN);
        let decision = read(1)?.first().copied().unwrap_or(f64::NAN);
        let mut biomarkers = Vec::new();
        // Index the vec by the model-tuple position so SHOWN_BIOMARKERS'
        // indices (2,4,5,6) address the right biomarker (Python does out[idx]).
        for i in 0..9 {
            biomarkers.push(read(i)?);
        }
        Ok(ModelOutput { score, decision, biomarkers })
    }
}

// ---------------------------------------------------------------------------
// JSON output (key order and number formatting match json.dumps exactly)
// ---------------------------------------------------------------------------

pub(crate) enum J {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    fn dump(&self, pretty: bool) -> String {
        let mut s = String::new();
        self.write(&mut s, pretty, 0);
        s
    }

    fn write(&self, out: &mut String, pretty: bool, level: usize) {
        let pad = |n: usize| -> String {
            if pretty {
                " ".repeat(2 * n)
            } else {
                String::new()
            }
        };
        let nl = if pretty { "\n" } else { "" };
        match self {
            J::Null => out.push_str("null"),
            J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            J::Num(n) => out.push_str(n),
            J::Str(s) => {
                out.push('"');
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
                out.push('"');
            }
            J::Arr(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        // json.dumps' separators: ", " when indent is None, "," + newline when not
                        if pretty {
                            out.push(',');
                        } else {
                            out.push_str(", ");
                        }
                    }
                    out.push_str(nl);
                    out.push_str(&pad(level + 1));
                    it.write(out, pretty, level + 1);
                }
                out.push_str(nl);
                out.push_str(&pad(level));
                out.push(']');
            }
            J::Obj(items) => {
                if items.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (i, (k, v)) in items.iter().enumerate() {
                    if i > 0 {
                        if pretty {
                            out.push(',');
                        } else {
                            out.push_str(", ");
                        }
                    }
                    out.push_str(nl);
                    out.push_str(&pad(level + 1));
                    out.push('"');
                    out.push_str(k);
                    out.push('"');
                    out.push_str(": ");
                    v.write(out, pretty, level + 1);
                }
                out.push_str(nl);
                out.push_str(&pad(level));
                out.push('}');
            }
        }
    }
}

/// Python `repr(float)`: shortest round-trip, `.0` for integral values, and
/// exponent form outside [1e-4, 1e16) the way CPython prints it.
fn py_float(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let a = v.abs();
    if v != 0.0 && (a < 1e-4 || a >= 1e16) {
        let s = format!("{:e}", v); // shortest mantissa, e.g. 1.2e-5
        if let Some((mant, exp)) = s.split_once('e') {
            if let Ok(exp) = exp.parse::<i32>() {
                return format!("{}e{}{:02}", mant, if exp < 0 { '-' } else { '+' }, exp.abs());
            }
        }
    }
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Python `round(v, n)` then `repr()` of the result (round-half-even on the
/// exact binary value, which is also what Rust's `{:.n$}` does).
fn py_round(v: f64, n: usize) -> String {
    if !v.is_finite() {
        return py_float(v);
    }
    let s = format!("{:.*}", n, v);
    match s.parse::<f64>() {
        Ok(r) => py_float(r),
        Err(_) => s,
    }
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Args {
    pub db: PathBuf,
    pub tz: f64,
    pub profile: Option<PathBuf>,
    pub serial: Option<String>,
    pub pretty: bool,
}

impl Args {
    /// Same argument handling as the script's `main()`: positional db + tz,
    /// `--serial <hex>`, `--profile <path>`, `--json`.
    pub fn parse(argv: &[String]) -> Result<Args> {
        let get = |flag: &str| -> Option<String> {
            argv.iter()
                .position(|a| a == flag)
                .and_then(|i| argv.get(i + 1))
                .cloned()
        };
        let serial = get("--serial");
        let mut drop = vec!["--serial".to_string()];
        if let Some(s) = &serial {
            drop.push(s.clone());
        }
        let positional: Vec<&String> = argv
            .iter()
            .filter(|a| !a.starts_with("--") && !drop.iter().any(|d| d == *a))
            .collect();
        let profile = get("--profile");
        let db = positional
            .first()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("oura.db"));
        let tz: f64 = positional
            .get(1)
            .map(|s| s.parse::<f64>().with_context(|| format!("bad tz {s}")))
            .transpose()?
            .unwrap_or(0.0);
        let profile = match profile {
            Some(p) => Some(PathBuf::from(p)),
            None => {
                let pj = db.parent().unwrap_or(Path::new(".")).join("profile.json");
                if pj.exists() {
                    Some(pj)
                } else {
                    None
                }
            }
        };
        Ok(Args { db, tz, profile, serial, pretty: !argv.iter().any(|a| a == "--json") })
    }
}

/// Everything the pipeline produces before inference.
pub struct Features {
    pub date: String,
    pub inputs: ModelInputs,
    pub days_with_data: i64,
}

pub enum Prepared {
    /// Mirrors the script's early-exit `{"available": false, ...}` states.
    Status(&'static str),
    Ready(Box<Features>),
}

/// The full data + feature pipeline (no inference).
pub fn prepare(args: &Args) -> Result<Prepared> {
    let conn = open_db(&args.db)?;

    // Pass 1: boot epochs from time-sync / rtc / ring-start events.
    let mut clock = Clock::new();
    let mut total = 0usize;
    for_each_row(&conn, args.serial.as_deref(), |ds, tag, dj, cu| {
        total += 1;
        clock.add_row(ds, tag, dj, cu);
    })?;
    if total == 0 {
        return Ok(Prepared::Status("NO_DATA"));
    }
    clock.finish();

    // Pass 2: dated rows only -> bedtime windows, biometrics, activity.
    let raw = collect_raw(&conn, &clock, args.tz, args.serial.as_deref())?;
    let per_day = nightly_biometrics(&raw, args.tz);
    if std::env::var_os("OURA_ILLNESS_DEBUG").is_some() {
        eprintln!(
            "rows={total} boot={} windows={} ibi={} hrv={} temp={} per_day={} sed={} rest={}",
            clock.epoch_count(),
            raw.windows.len(),
            raw.ibi_t.len(),
            raw.hrv_t.len(),
            raw.temp_t.len(),
            per_day.len(),
            raw.sed.len(),
            raw.rest.len()
        );
        if let Some((&d, n)) = per_day.iter().next_back() {
            eprintln!("newest day {d}: {n:?}");
        }
    }
    if per_day.is_empty() {
        return Ok(Prepared::Status("MISSING_SLEEP_DATA"));
    }
    let anchor = *per_day.keys().max().unwrap(); // latest wake day = "today"
    let s = build_series(&per_day, &raw.sed, &raw.rest, anchor);

    // 301 guard: today's temperature deviation must exist.
    if !s.temp_dev[0].is_finite() {
        return Ok(Prepared::Status("MISSING_LAST_NIGHT_SLEEP"));
    }
    // 302 guard: at most 7 of the last 14 nights may be missing.
    if s.temp_dev[..14].iter().filter(|x| x.is_nan()).count() > 7 {
        return Ok(Prepared::Status("MISSING_SLEEP_DATA"));
    }

    // Long-term scalar baselines (avg + std over the available history).
    let rhr = finite(&s.low_hr);
    let hrvb = finite(&s.hrv);
    let tvals = finite64(&s.skin);
    let sedvals = finite(&s.sed);
    let b_rhr_avg = if !rhr.is_empty() { mean(&rhr) } else { 55.0 };
    let b_rhr_dev = if rhr.len() > 1 { std_dev(&rhr) } else { 3.0 };
    let b_hrv_avg = if !hrvb.is_empty() { mean(&hrvb) } else { 45.0 };
    let b_hrv_dev = if hrvb.len() > 1 { std_dev(&hrvb) } else { 6.0 };
    let b_tmp_avg = if s.base_temp.is_finite() { s.base_temp } else { 36.0 };
    let b_tmp_dev = if tvals.len() > 1 { std_dev(&tvals) } else { 0.2 };
    let b_sed_avg = if !sedvals.is_empty() { mean(&sedvals) } else { 30000.0 };
    let b_sed_dev = if sedvals.len() > 1 { std_dev(&sedvals) } else { 4000.0 };

    // Demographics: a flat profile, or the dashboard's serial-keyed map.
    let mut prof: Value = match &args.profile {
        Some(p) if p.exists() => {
            let text = std::fs::read_to_string(p).unwrap_or_default();
            serde_json::from_str::<Value>(&text).unwrap_or(Value::Null)
        }
        _ => Value::Null,
    };
    let has_direct = prof
        .as_object()
        .map(|o| o.contains_key("age") || o.contains_key("sex"))
        .unwrap_or(false);
    if !has_direct {
        let key = args.serial.clone().unwrap_or_default();
        prof = prof.get(&key).cloned().or_else(|| prof.get("").cloned()).unwrap_or(Value::Null);
    }
    let prof = prof.as_object().cloned().unwrap_or_default();
    let pf = |k: &str, d: f64| -> f64 {
        prof.get(k).and_then(|v| json_f64(Some(v))).unwrap_or(d)
    };
    let age = pf("age", 30.0);
    let h = pf("height_m", 1.78);
    let w = pf("weight_kg", 75.0);
    let bmi = w / (h * h);
    let sex_str = match prof.get("sex") {
        Some(Value::String(s)) => s.to_uppercase(),
        Some(v @ Value::Number(_)) => v.to_string().to_uppercase(),
        _ => "M".to_string(),
    };
    let sex = match sex_str.as_str() {
        "F" => -1.0,
        "M" => 1.0,
        _ => 0.0,
    };
    let (y, mo, dd) = civil(anchor);
    let dow = weekday_from_days(anchor) as f64;
    let date = format!("{y:04}-{mo:02}-{dd:02}");

    let nan_col = vec![f32::NAN; N_DAYS];
    let inputs = ModelInputs {
        series: [
            s.breath.clone(),
            s.avg_hr.clone(),
            s.low_hr.clone(),
            s.hrv.clone(),
            s.temp_dev.clone(),
            s.sed.clone(),
            s.rest.clone(),
            nan_col.clone(),
            nan_col,
        ],
        scalars: vec![
            f32::NAN,
            f32::NAN,
            age as f32,
            bmi as f32,
            sex as f32,
            dow as f32,
            b_rhr_avg as f32,
            b_rhr_dev as f32,
            b_hrv_avg as f32,
            b_hrv_dev as f32,
            b_tmp_avg as f32,
            b_tmp_dev as f32,
            b_sed_avg as f32,
            b_sed_dev as f32,
        ],
    };
    let days_with_data = s.temp_dev.iter().filter(|x| x.is_finite()).count() as i64;
    Ok(Prepared::Ready(Box::new(Features { date, inputs, days_with_data })))
}

fn status_json(status: &str) -> J {
    J::Obj(vec![
        ("available".to_string(), J::Bool(false)),
        ("status".to_string(), J::Str(status.to_string())),
    ])
}

/// The same JSON the Python script's `run()` returns.
pub(crate) fn run_model(args: &Args) -> Result<J> {
    let feats = match prepare(args)? {
        Prepared::Status(s) => return Ok(status_json(s)),
        Prepared::Ready(f) => f,
    };

    #[cfg(feature = "torch")]
    let model = match model_path() {
        Some(p) => TchIllnessModel::load(&p)?,
        None => bail!("illness_detection_0_5_1.pt not found under notes/models"),
    };
    #[cfg(not(feature = "torch"))]
    let model = NoInference;

    let out = model.forward(&feats.inputs)?;
    let decision = out.decision.round() as i64;
    let (risk, light) = DECISION
        .iter()
        .find(|(d, _, _)| *d == decision)
        .map(|(_, r, l)| (*r, *l))
        .unwrap_or(("BelowThreshold", "NO_SIGNS"));

    let mut biomarkers = Vec::new();
    for (idx, label) in SHOWN_BIOMARKERS {
        let v = match out.biomarkers.get(idx) {
            Some(v) if v.len() >= 4 => v.clone(),
            _ => continue,
        };
        if v.iter().any(|x| !x.is_finite()) {
            continue; // a biomarker with any NaN element is dropped
        }
        let is_out = v[0].round() as i64 == 1;
        let reason = if is_out {
            J::Str(if v[1] > v[3] { "ELEVATED".to_string() } else { "DECREASED".to_string() })
        } else {
            J::Null
        };
        biomarkers.push(J::Obj(vec![
            ("type".to_string(), J::Str(label.to_string())),
            ("value".to_string(), J::Num(py_round(v[1], 2))),
            ("lower".to_string(), J::Num(py_round(v[2], 2))),
            ("upper".to_string(), J::Num(py_round(v[3], 2))),
            ("indicatesSymptoms".to_string(), J::Bool(is_out)),
            ("reason".to_string(), reason),
        ]));
    }

    let breath_today = if feats.inputs.series[0][0].is_finite() {
        J::Num(py_round(feats.inputs.series[0][0] as f64, 1))
    } else {
        J::Null
    };

    Ok(J::Obj(vec![
        ("available".to_string(), J::Bool(true)),
        ("date".to_string(), J::Str(feats.date.clone())),
        ("score".to_string(), J::Num(py_round(out.score, 4))),
        ("decision".to_string(), J::Num(decision.to_string())),
        ("risk_level".to_string(), J::Str(risk.to_string())),
        ("traffic_light".to_string(), J::Str(light.to_string())),
        ("biomarkers".to_string(), J::Arr(biomarkers)),
        ("breath_today".to_string(), breath_today),
        ("status".to_string(), J::Str(light.to_string())),
        ("days_with_data".to_string(), J::Num(feats.days_with_data.to_string())),
    ]))
}

pub fn run(args: &[String]) -> Result<()> {
    let args = Args::parse(args)?;
    println!("{}", run_json(&args)?);
    Ok(())
}

/// The JSON the script prints, as a string (tests, and any caller that wants the
/// payload instead of stdout).
pub fn run_json(args: &Args) -> Result<String> {
    Ok(run_model(args)?.dump(args.pretty))
}

/// The JSON `run` would print, from raw argv — the in-process entry point for
/// the dashboard's RustRunner (same argv contract as the CLI, golden-tested).
pub fn json_for_args(argv: &[String]) -> Result<String> {
    run_json(&Args::parse(argv)?)
}
