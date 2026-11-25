//! Incremental activity recompute: the Rust port of `tools/event_cache.py`
//! (`EventSource` + `UnitCache`) and `run_incremental` in
//! `tools/run_activity_model.py:387`.
//!
//! The cold path (`activity::prepare`) re-reads every decoded row of the ring
//! and re-runs the AAD model on every local calendar day: 1.45 M rows on the
//! reference ring. That work is unchanged when nothing in the database changed,
//! so the incremental path keeps sidecars next to the DB and recomputes only the
//! days that could have changed:
//!
//! * `<model_key>.<serial>.inc.json` — the ring-clock epoch list `Clock::build`
//!   produced plus the `MAX(id)` watermark of the events table. `events` is
//!   append-only (`UNIQUE(serial, tag, ring_timestamp, body)` dedupes every
//!   sync), so `MAX(id)` is a sound watermark: if it did not move and the epochs
//!   are cached, the events table is not read at all.
//! * `<model_key>.<serial>.units.json` — per-day model output, keyed by a
//!   fingerprint of everything that changes the answer (state version, model
//!   asset, settings, user vector, tz, gait on/off, ACM_SCALE).
//! * `<model_key>.<serial>.<where-hash>.skel.json` — the clock skeleton rows
//!   with their own watermark, so a run that added rows does not re-scan 1.45 M
//!   rows to rebuild the clock.
//!
//! **These files are Rust-owned.** The model key carries `.rs.`
//! (`activity.rs.<serial>.inc.json`), so Python's
//! `activity.<serial>.inc.json` / `.units.json` / `.skel.pkl` are never read or
//! written from here — the pickle in particular is not parseable from Rust,
//! which is why the skeleton cache is a JSON array-of-arrays instead. The
//! settings fingerprint and the skeleton-cache key use FNV-1a-64 rather than
//! sha256: those hashes are only ever compared against themselves, so what has
//! to mirror Python is the *input list* being hashed, not the hash function.
//!
//! Scoring is not duplicated: `recompute_day` calls the same `build_signals`,
//! `collect_step_packets` / `decode_step_packets`, `build_day_request` and
//! `sessions_from` the cold path runs, on a windowed row set instead of the
//! whole table. The guarantee this exists to provide — and which
//! `tests/activity_incremental_test.rs` measures rather than asserts — is that
//! for one database the incremental path returns exactly what the cold path
//! returns.

use crate::activity::{
    self, ActivityInference, Args, Clock, DayOutcome, Epoch, InferenceUnavailable, Session,
    Settings, StepMotionDecoder, StepSeries, WorkoutRow, Date, EPOCH_RESET_SLACK_DS,
};
use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Namespace marker that keeps these sidecars apart from Python's.
pub const MODEL_KEY: &str = "activity.rs";
/// Bumping this resets the incremental state and invalidates the unit cache, the
/// way Python's `event_cache.STATE_VERSION` does; it also feeds the settings
/// fingerprint, so a change in the code contract invalidates cached units.
pub const STATE_VERSION: &str = "activity.rs-inc-1";
/// Ring-counter margin around a requested window (6 h either side), mirroring
/// `event_cache.DS_MARGIN`. Overfetching is harmless — every consumer filters to
/// the window it asked for — while underfetching silently drops data.
pub const DS_MARGIN: i64 = 6 * 3600 * 10;
/// The step-motion row set, which carries no decoded_json and so is invisible to
/// `activity::WHERE`.
pub const STEP_WHERE: &str = "tag IN (126,127) AND body IS NOT NULL";

/// A row as the cache layer sees it: (ring ds, tag, payload, captured unix).
pub type Row = (i64, i64, Option<String>, i64);
/// A step-motion packet row: (ring ds, tag, body, captured unix).
pub type PacketRow = (i64, i64, Vec<u8>, i64);

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// FNV-1a-64, hex. Stands in for Python's sha256 in the settings fingerprint and
/// the skeleton-cache key (see the module note).
pub fn fnv1a64_hex(data: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    format!("{h:016x}")
}

/// Approximate unix seconds for a ring counter, from the nearest anchor. Mirrors
/// `_pred` in event_cache.py; used only for conservative window math.
pub fn pred(epoch: &Epoch, ds: i64) -> f64 {
    if !epoch.anchors.is_empty() {
        // Python's min(key=abs(a[0]-ds)) keeps the first minimum in list order.
        let mut best: Option<(i64, (i64, i64))> = None;
        for a in &epoch.anchors {
            let d = (a.0 - ds).abs();
            if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                best = Some((d, *a));
            }
        }
        let a = best.expect("anchors non-empty").1;
        return a.1 as f64 + (ds - a.0) as f64 / 10.0;
    }
    epoch.fallback_unix as f64 - (epoch.max_ds - ds) as f64 / 10.0
}

/// A boot whose anchor set changed: (epoch index, ds window).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TailWindow {
    pub epoch: usize,
    pub lo: i64,
    pub hi: i64,
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("replace {}", path.display()))?;
    Ok(())
}

fn epoch_to_value(e: &Epoch) -> Value {
    json!([
        e.min_ds,
        e.max_ds,
        e.fallback_unix,
        e.capture_min,
        e.capture_max,
        e.anchors.iter().map(|(d, u)| json!([d, u])).collect::<Vec<Value>>(),
        e.boots,
    ])
}

fn epochs_from_value(values: &[Value]) -> Result<Vec<Epoch>> {
    let mut out = Vec::with_capacity(values.len());
    for v in values {
        let a = v
            .as_array()
            .with_context(|| format!("epoch entry is not an array: {v}"))?;
        if a.len() < 7 {
            bail!("epoch entry too short: {v}");
        }
        let num = |x: &Value, what: &str| -> Result<i64> {
            x.as_i64().with_context(|| format!("epoch {what} is not an integer: {x}"))
        };
        let anchors = a[5]
            .as_array()
            .context("epoch anchors are not an array")?
            .iter()
            .map(|p| {
                let p = p.as_array().context("anchor is not an array")?;
                Ok((p[0].as_i64().context("anchor ds")?, p[1].as_i64().context("anchor unix")?))
            })
            .collect::<Result<Vec<(i64, i64)>>>()?;
        let boots = a[6]
            .as_array()
            .context("epoch boots are not an array")?
            .iter()
            .map(|b| num(b, "boot"))
            .collect::<Result<Vec<i64>>>()?;
        out.push(Epoch {
            min_ds: num(&a[0], "min_ds")?,
            max_ds: num(&a[1], "max_ds")?,
            fallback_unix: num(&a[2], "fallback_unix")?,
            capture_min: num(&a[3], "capture_min")?,
            capture_max: num(&a[4], "capture_max")?,
            anchors,
            boots,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// EventSource
// ---------------------------------------------------------------------------

/// `[anchor count, highest anchor ds]` for one boot, keyed by its first ds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorFp(usize, Option<i64>);

fn anchor_fp_of(e: &Epoch) -> AnchorFp {
    if e.anchors.is_empty() {
        AnchorFp(0, None)
    } else {
        AnchorFp(e.anchors.len(), e.anchors.iter().map(|a| a.0).max())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub version: String,
    pub watermark: i64,
    pub epochs: Vec<Value>,
    pub anchor_fp: BTreeMap<String, AnchorFp>,
}

impl State {
    fn fresh(version: &str) -> State {
        State {
            version: version.to_string(),
            watermark: 0,
            epochs: Vec::new(),
            anchor_fp: BTreeMap::new(),
        }
    }
}

/// One skeleton row: `[ring ds, tag, captured unix, anchor json?]`. Only the
/// clock's anchor tags keep a payload — that is all `Clock::build` reads.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkelRow(
    pub i64,
    pub i64,
    pub i64,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub Option<String>,
);

/// Incremental state for one (model, ring) pair. See the module note for which
/// files this owns.
pub struct EventSource {
    db: PathBuf,
    serial: Option<String>,
    model_key: String,
    version: String,
    con: Connection,
    /// `MAX(id)` with the serial filter is a ~1.9 s scan (no index covers it)
    /// and one run asks for it up to three times. Caching it per run is safe:
    /// the worst a concurrent sync can do is leave the committed watermark a
    /// little old, which re-processes rows rather than dropping them.
    max_id_cache: RefCell<HashMap<String, i64>>,
    pub state: State,
}

impl EventSource {
    pub fn new(db: &Path, serial: Option<&str>, model_key: &str, version: &str) -> Result<Self> {
        let con = Connection::open_with_flags(
            db,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("opening {}", db.display()))?;
        let mut this = EventSource {
            db: db.to_path_buf(),
            serial: serial.map(|s| s.to_string()),
            model_key: model_key.to_string(),
            version: version.to_string(),
            con,
            max_id_cache: RefCell::new(HashMap::new()),
            state: State::fresh(version),
        };
        this.load_state();
        Ok(this)
    }

    fn stem(&self) -> String {
        format!("{}.{}", self.model_key, self.serial.clone().unwrap_or_else(|| "all".into()))
    }

    pub fn state_path(&self) -> PathBuf {
        self.db.with_file_name(format!("{}.inc.json", self.stem()))
    }

    pub fn skel_path(&self, where_sql: &str) -> PathBuf {
        self.db.with_file_name(format!(
            "{}.{}.skel.json",
            self.stem(),
            fnv1a64_hex(where_sql.as_bytes())
        ))
    }

    /// Every file this instance owns, for tests that clean up after themselves.
    pub fn sidecar_paths(&self, where_sql: &str) -> Vec<PathBuf> {
        vec![self.state_path(), self.skel_path(where_sql)]
    }

    fn serial_clause(&self) -> (String, Option<String>) {
        match &self.serial {
            Some(s) => (" AND serial = ?".to_string(), Some(s.clone())),
            None => (String::new(), None),
        }
    }

    fn load_state(&mut self) {
        let Ok(text) = std::fs::read_to_string(self.state_path()) else { return };
        let Ok(value) = serde_json::from_str::<Value>(&text) else { return };
        if value.get("version").and_then(Value::as_str) != Some(self.version.as_str()) {
            return; // a model or contract upgrade invalidates the cached clock
        }
        self.state = serde_json::from_value(value).unwrap_or_else(|_| State::fresh(&self.version));
    }

    pub fn has_state(&self) -> bool {
        !self.state.epochs.is_empty()
    }

    pub fn max_id(&self, where_sql: &str) -> Result<i64> {
        if let Some(v) = self.max_id_cache.borrow().get(where_sql) {
            return Ok(*v);
        }
        let (clause, sp) = self.serial_clause();
        let sql = format!("SELECT MAX(id) FROM events WHERE {where_sql}{clause}");
        let mut stmt = self.con.prepare(&sql)?;
        let value: Option<i64> = match &sp {
            Some(s) => stmt.query_row(params![s], |r| r.get(0)).optional()?,
            None => stmt.query_row([], |r| r.get(0)).optional()?,
        }
        .flatten();
        let value = value.unwrap_or(0);
        self.max_id_cache.borrow_mut().insert(where_sql.to_string(), value);
        Ok(value)
    }

    /// Rows newer than the committed watermark, in capture order — the same shape
    /// as a full load, so a caller can treat the two identically.
    pub fn new_rows(&self, where_sql: &str) -> Result<Vec<Row>> {
        let (clause, sp) = self.serial_clause();
        let sql = format!(
            "SELECT ring_timestamp, tag, decoded_json, captured_unix FROM events \
             WHERE {where_sql}{clause} AND id > ? ORDER BY captured_unix, id"
        );
        let watermark = self.state.watermark;
        let mut stmt = self.con.prepare(&sql)?;
        let map = |r: &rusqlite::Row| -> rusqlite::Result<Row> {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        };
        let rows = match &sp {
            Some(s) => stmt
                .query_map(params![s, watermark], map)?
                .collect::<rusqlite::Result<Vec<Row>>>()?,
            None => stmt
                .query_map(params![watermark], map)?
                .collect::<rusqlite::Result<Vec<Row>>>()?,
        };
        Ok(rows)
    }

    /// Rebuild the ring clock, or replay it when nothing arrived.
    ///
    /// Returns the clock plus the ds windows of every boot whose anchor set
    /// changed since the last commit — a boot that gained a time-sync anchor
    /// re-dates everything after its previous last anchor, and nothing before.
    pub fn clock(&mut self, where_sql: &str) -> Result<(Clock, Vec<TailWindow>)> {
        let now = self.max_id(where_sql)?;
        if now == self.state.watermark && !self.state.epochs.is_empty() {
            let epochs = epochs_from_value(&self.state.epochs)?;
            return Ok((Clock::from_epochs(epochs), Vec::new()));
        }
        let path = self.skel_path(where_sql);
        let (mut skel, skel_wm) = read_skel(&path);
        if now > skel_wm {
            let (clause, sp) = self.serial_clause();
            let sql = format!(
                "SELECT ring_timestamp, tag, \
                 CASE WHEN tag IN (65,66,133) THEN decoded_json END, captured_unix \
                 FROM events WHERE {where_sql}{clause} AND id > ? ORDER BY captured_unix, id"
            );
            let mut stmt = self.con.prepare(&sql)?;
            let map = |r: &rusqlite::Row| -> rusqlite::Result<SkelRow> {
                Ok(SkelRow(r.get(0)?, r.get(1)?, r.get(3)?, r.get(2)?))
            };
            let rows = match &sp {
                Some(s) => stmt
                    .query_map(params![s, skel_wm], map)?
                    .collect::<rusqlite::Result<Vec<SkelRow>>>()?,
                None => stmt
                    .query_map(params![skel_wm], map)?
                    .collect::<rusqlite::Result<Vec<SkelRow>>>()?,
            };
            skel.extend(rows);
            write_skel(&path, now, &skel)?;
        }
        let clock = Clock::build(skel.iter().map(|r| (r.0, r.1, r.3.clone(), r.2)));
        let changed = self.anchor_diff(&clock);
        self.state.epochs = clock.epochs.iter().map(epoch_to_value).collect();
        Ok((clock, changed))
    }

    /// Seed the anchor fingerprints from a clock built the long way round (the
    /// cold path), so the first incremental run after it tails from the last
    /// known anchor instead of re-processing a whole boot. Mirrors `note_epochs`.
    pub fn note_epochs(&mut self, clock: &Clock) {
        self.anchor_diff(clock);
    }

    /// Update per-boot anchor fingerprints; return the boots whose anchor set
    /// changed (their clock mapping — and only theirs — can re-date rows that
    /// were already processed). Mirrors `_anchor_diff`.
    pub fn anchor_diff(&mut self, clock: &Clock) -> Vec<TailWindow> {
        let mut changed = Vec::new();
        for (i, e) in clock.epochs.iter().enumerate() {
            let fp = anchor_fp_of(e);
            let key = e.min_ds.to_string();
            let prev = self.state.anchor_fp.get(&key).copied();
            let (lo, hi) = match prev {
                None => (e.min_ds, if fp.0 > 0 { fp.1 } else { None }),
                Some(prev) => (prev.1.unwrap_or(e.min_ds), fp.1),
            };
            if let Some(hi) = hi {
                if prev.map(|p| p.1 != fp.1).unwrap_or(true) {
                    changed.push(TailWindow {
                        epoch: i,
                        lo: e.min_ds.max(lo - DS_MARGIN),
                        hi: e.max_ds.min(hi + DS_MARGIN),
                    });
                }
            }
            self.state.anchor_fp.insert(key, fp);
        }
        changed
    }

    /// Conservative ds windows in every boot that could contain wall-clock
    /// `[u_lo, u_hi]`. The clock maps ds -> unix; this inverts it, and dating
    /// error is bounded by the gap to the nearest anchor, so the margin is
    /// generous on purpose.
    pub fn ds_windows(&self, u_lo: f64, u_hi: f64, clock: &Clock) -> Vec<(i64, i64)> {
        let mut out = Vec::new();
        for e in &clock.epochs {
            let (a, b) = (pred(e, e.min_ds), pred(e, e.max_ds));
            let (span_lo, span_hi) = (a.min(b), a.max(b));
            if span_lo > u_hi || span_hi < u_lo {
                continue;
            }
            let (lo, hi) = if e.anchors.is_empty() {
                (e.min_ds as f64, e.max_ds as f64)
            } else {
                let lo = e
                    .anchors
                    .iter()
                    .map(|x| x.0 as f64 + (u_lo - x.1 as f64) * 10.0)
                    .fold(f64::INFINITY, f64::min);
                let hi = e
                    .anchors
                    .iter()
                    .map(|x| x.0 as f64 + (u_hi - x.1 as f64) * 10.0)
                    .fold(f64::NEG_INFINITY, f64::max);
                (lo, hi)
            };
            let lo = (e.min_ds - EPOCH_RESET_SLACK_DS).max(lo as i64 - DS_MARGIN);
            let hi = (e.max_ds + EPOCH_RESET_SLACK_DS).min(hi as i64 + DS_MARGIN);
            if lo <= hi {
                out.push((lo, hi));
            }
        }
        out
    }

    fn window_sql(&self, tags: &[i64], columns: &str, where_extra: &str) -> String {
        let (clause, _) = self.serial_clause();
        format!(
            "SELECT id, {columns} FROM events WHERE tag IN ({}) AND ring_timestamp BETWEEN ? AND ?{}{} \
             ORDER BY captured_unix, id",
            vec!["?"; tags.len()].join(","),
            if where_extra.is_empty() { String::new() } else { format!(" AND {where_extra}") },
            clause
        )
    }

    /// Rows in one ds window, id first. `tags` must be explicit: the only index
    /// that can seek a ds range is `(tag, ring_timestamp)`, so a bare BETWEEN
    /// would scan the whole table.
    fn window_query<T>(
        &self,
        sql: &str,
        tags: &[i64],
        lo: i64,
        hi: i64,
        map: impl FnMut(&rusqlite::Row) -> rusqlite::Result<T>,
    ) -> Result<Vec<T>> {
        let mut stmt = self.con.prepare(sql)?;
        let mut args: Vec<&dyn rusqlite::types::ToSql> = Vec::with_capacity(tags.len() + 3);
        for tag in tags {
            args.push(tag);
        }
        args.push(&lo);
        args.push(&hi);
        let (_, serial) = self.serial_clause();
        if let Some(s) = &serial {
            args.push(s);
        }
        let rows = stmt.query_map(rusqlite::params_from_iter(args.iter()), map)?;
        Ok(rows.collect::<rusqlite::Result<Vec<T>>>()?)
    }

    fn rows_for_ds(&self, lo: i64, hi: i64, tags: &[i64], where_extra: &str) -> Result<Vec<(i64, Row)>> {
        let sql =
            self.window_sql(tags, "ring_timestamp,tag,decoded_json,captured_unix", where_extra);
        self.window_query(&sql, tags, lo, hi, |r| {
            Ok((r.get::<_, i64>(0)?, (r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))
        })
    }

    fn packets_for_ds(
        &self,
        lo: i64,
        hi: i64,
        tags: &[i64],
        where_extra: &str,
    ) -> Result<Vec<(i64, PacketRow)>> {
        let sql = self.window_sql(tags, "ring_timestamp,tag,body,captured_unix", where_extra);
        self.window_query(&sql, tags, lo, hi, |r| {
            Ok((r.get::<_, i64>(0)?, (r.get(1)?, r.get(2)?, r.get::<_, Vec<u8>>(3)?, r.get(4)?)))
        })
    }

    /// Union of ds windows with each row exactly once, in the same
    /// `(captured_unix, id)` order a full load uses. `ds_windows` may overlap
    /// when boots share counter space; concatenating them would feed the model
    /// duplicate rows in a different order than the cold path.
    pub fn rows_in_windows(
        &self,
        windows: &[(i64, i64)],
        tags: &[i64],
        where_extra: &str,
    ) -> Result<Vec<Row>> {
        let mut seen: HashMap<i64, Row> = HashMap::new();
        for (lo, hi) in windows {
            for (id, row) in self.rows_for_ds(*lo, *hi, tags, where_extra)? {
                seen.insert(id, row);
            }
        }
        let mut rows: Vec<(i64, Row)> = seen.into_iter().collect();
        rows.sort_by(|a, b| a.1 .3.cmp(&b.1 .3).then(a.0.cmp(&b.0)));
        Ok(rows.into_iter().map(|(_, r)| r).collect())
    }

    /// `rows_in_windows` for the step-motion packets, whose payload is a blob.
    pub fn packets_in_windows(
        &self,
        windows: &[(i64, i64)],
        tags: &[i64],
        where_extra: &str,
    ) -> Result<Vec<PacketRow>> {
        let mut seen: HashMap<i64, PacketRow> = HashMap::new();
        for (lo, hi) in windows {
            for (id, row) in self.packets_for_ds(*lo, *hi, tags, where_extra)? {
                seen.insert(id, row);
            }
        }
        let mut rows: Vec<(i64, PacketRow)> = seen.into_iter().collect();
        rows.sort_by(|a, b| a.1 .3.cmp(&b.1 .3).then(a.0.cmp(&b.0)));
        Ok(rows.into_iter().map(|(_, r)| r).collect())
    }

    /// Persist the watermark and the clock. Only after a pass that covered every
    /// day the caller was asked for — a partial pass must not advance the
    /// watermark, or the days it skipped would be lost.
    pub fn commit(&mut self, clock: &Clock, watermark: i64) -> Result<()> {
        self.state.version = self.version.clone();
        self.state.watermark = watermark;
        self.state.epochs = clock.epochs.iter().map(epoch_to_value).collect();
        let body = serde_json::to_string(&json!({
            "version": self.state.version,
            "watermark": self.state.watermark,
            "epochs": self.state.epochs,
            "anchor_fp": self
                .state
                .anchor_fp
                .iter()
                .map(|(k, v)| (k.clone(), json!([v.0, v.1])))
                .collect::<serde_json::Map<String, Value>>(),
        }))
        .context("serializing incremental state")?;
        write_atomic(&self.state_path(), body.as_bytes())
    }

    /// The skeleton cache is keyed by the WHERE clause, so a changed clause
    /// simply reads a different file; nothing to delete. Mirrors Python.
    pub fn discard_skeleton_cache(&self) {}
}

fn read_skel(path: &Path) -> (Vec<SkelRow>, i64) {
    // A corrupt or unreadable cache is not an error: the events table is the
    // source of truth and the cache is rebuilt from it.
    let Ok(text) = std::fs::read_to_string(path) else { return (Vec::new(), 0) };
    let Ok(value) = serde_json::from_str::<Value>(&text) else { return (Vec::new(), 0) };
    let watermark = value.get("watermark").and_then(Value::as_i64).unwrap_or(0);
    let rows = value
        .get("rows")
        .and_then(|v| serde_json::from_value::<Vec<SkelRow>>(v.clone()).ok())
        .unwrap_or_default();
    (rows, watermark)
}

fn write_skel(path: &Path, watermark: i64, rows: &[SkelRow]) -> Result<()> {
    // Hand-rolled rather than serde_json: 1.45 M rows of JSON through a Value
    // tree costs seconds and a lot of allocator churn.
    let mut out = String::with_capacity(rows.len() * 34 + 64);
    out.push_str("{\"watermark\":");
    out.push_str(&watermark.to_string());
    out.push_str(",\"rows\":[");
    for (i, r) in rows.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('[');
        out.push_str(&r.0.to_string());
        out.push(',');
        out.push_str(&r.1.to_string());
        out.push(',');
        out.push_str(&r.2.to_string());
        if let Some(js) = &r.3 {
            out.push(',');
            out.push_str(&serde_json::to_string(js).context("encoding anchor json")?);
        }
        out.push(']');
    }
    out.push_str("]}");
    write_atomic(path, out.as_bytes())
}

// ---------------------------------------------------------------------------
// UnitCache
// ---------------------------------------------------------------------------

/// A session as stored in the unit cache: field-for-field the output shape, so a
/// cached day replays into byte-identical JSON.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CachedSession {
    pub start: String,
    pub end: String,
    pub duration_min: i64,
    pub is_workout: f64,
    pub label: String,
    pub label_confidence: f64,
    pub top3: Vec<(String, f64)>,
}

impl CachedSession {
    fn from_session(s: &Session) -> CachedSession {
        CachedSession {
            start: s.start.clone(),
            end: s.end.clone(),
            duration_min: s.duration_min,
            is_workout: s.is_workout,
            label: s.label.clone(),
            label_confidence: s.label_confidence,
            top3: s.top3.clone(),
        }
    }

    fn to_session(&self) -> Session {
        Session {
            start: self.start.clone(),
            end: self.end.clone(),
            duration_min: self.duration_min,
            is_workout: self.is_workout,
            label: self.label.clone(),
            label_confidence: self.label_confidence,
            top3: self.top3.clone(),
        }
    }
}

/// Per-day model results, valid while the settings fingerprint holds. Mirrors
/// `event_cache.UnitCache`.
pub struct UnitCache {
    path: PathBuf,
    settings_fp: String,
    units: BTreeMap<String, Vec<Session>>,
}

/// The fingerprint input list, mirrored from run_activity_model.py: state
/// version, model asset and version, the scoring settings, the user vector, the
/// timezone, whether gait is on, the scale env var, and the gait asset.
pub fn settings_fingerprint(
    args: &Args,
    model_name: &str,
    model_version: &str,
    user_vector: &[f64],
    step_model: Option<&Path>,
) -> String {
    let payload = json!([
        STATE_VERSION,
        model_name,
        model_version,
        args.threshold,
        args.min_duration,
        user_vector,
        args.tz,
        args.no_stepmotion,
        std::env::var("ACM_SCALE").unwrap_or_else(|_| "1".to_string()),
        step_model.map(|p| p.display().to_string()),
    ]);
    fnv1a64_hex(payload.to_string().as_bytes())
}

impl UnitCache {
    pub fn new(db: &Path, model_key: &str, serial: Option<&str>, settings_fp: &str) -> UnitCache {
        let stem = format!("{}.{}", model_key, serial.unwrap_or("all"));
        let path = db.with_file_name(format!("{stem}.units.json"));
        let mut units =
            UnitCache { path, settings_fp: settings_fp.to_string(), units: BTreeMap::new() };
        units.load();
        units
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn load(&mut self) {
        let Ok(text) = std::fs::read_to_string(&self.path) else { return };
        let Ok(value) = serde_json::from_str::<Value>(&text) else { return };
        if value.get("settings_fp").and_then(Value::as_str) != Some(self.settings_fp.as_str()) {
            return; // settings changed: every cached unit is stale
        }
        let Some(map) = value.get("units").and_then(Value::as_object) else { return };
        for (day, sessions) in map {
            let Ok(parsed) = serde_json::from_value::<Vec<CachedSession>>(sessions.clone()) else {
                continue;
            };
            self.units.insert(day.clone(), parsed.iter().map(CachedSession::to_session).collect());
        }
    }

    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    pub fn len(&self) -> usize {
        self.units.len()
    }

    pub fn days(&self) -> BTreeSet<String> {
        self.units.keys().cloned().collect()
    }

    pub fn get(&self, day: &str) -> Option<&Vec<Session>> {
        self.units.get(day)
    }

    pub fn set(&mut self, day: &str, sessions: Vec<Session>) {
        self.units.insert(day.to_string(), sessions);
    }

    pub fn clear(&mut self) {
        self.units.clear();
    }

    pub fn save(&self) -> Result<()> {
        let units: serde_json::Map<String, Value> = self
            .units
            .iter()
            .map(|(day, sessions)| {
                let cached: Vec<CachedSession> =
                    sessions.iter().map(CachedSession::from_session).collect();
                (day.clone(), serde_json::to_value(&cached).unwrap_or(Value::Array(Vec::new())))
            })
            .collect();
        let body = serde_json::to_string(&json!({
            "version": STATE_VERSION,
            "settings_fp": self.settings_fp,
            "units": units,
        }))
        .context("serializing unit cache")?;
        write_atomic(&self.path, body.as_bytes())
    }
}

// ---------------------------------------------------------------------------
// counting backends, so "did almost no model work" is measured, not claimed
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
pub struct Counters {
    pub aad_calls: Arc<AtomicUsize>,
    pub decoder_calls: Arc<AtomicUsize>,
    pub decoder_packets: Arc<AtomicUsize>,
}

impl Counters {
    pub fn aad(&self) -> usize {
        self.aad_calls.load(Ordering::Relaxed)
    }
    pub fn decoder(&self) -> usize {
        self.decoder_calls.load(Ordering::Relaxed)
    }
    pub fn packets(&self) -> usize {
        self.decoder_packets.load(Ordering::Relaxed)
    }
    pub fn reset(&self) {
        self.aad_calls.store(0, Ordering::Relaxed);
        self.decoder_calls.store(0, Ordering::Relaxed);
        self.decoder_packets.store(0, Ordering::Relaxed);
    }
}

pub struct Backends {
    pub inference: Box<dyn ActivityInference>,
    pub decoder: Option<Box<dyn StepMotionDecoder>>,
    pub counters: Counters,
}

/// Wrap the real backends so every model call is counted. The cold path loads its
/// gait decoder inside `activity::prepare`, so gait counts cover the incremental
/// path only; AAD calls cover both.
pub fn counted_backends(model: &Path, step_model: Option<&Path>) -> Result<Backends> {
    let counters = Counters::default();
    let inner =
        activity::activity_backend(model).unwrap_or_else(|| Box::new(activity::UnimplementedInference));
    let inference = Box::new(CountingInference { inner, counter: counters.aad_calls.clone() });
    let decoder = step_model.and_then(|path| {
        activity::step_decoder_backend(path).map(|inner| {
            Box::new(CountingDecoder {
                inner,
                calls: counters.decoder_calls.clone(),
                packets: counters.decoder_packets.clone(),
            }) as Box<dyn StepMotionDecoder>
        })
    });
    Ok(Backends { inference, decoder, counters })
}

struct CountingInference {
    inner: Box<dyn ActivityInference>,
    counter: Arc<AtomicUsize>,
}

impl ActivityInference for CountingInference {
    fn run_day(&self, req: &activity::DayRequest) -> Result<Vec<WorkoutRow>> {
        self.counter.fetch_add(1, Ordering::Relaxed);
        self.inner.run_day(req)
    }
}

struct CountingDecoder {
    inner: Box<dyn StepMotionDecoder>,
    calls: Arc<AtomicUsize>,
    packets: Arc<AtomicUsize>,
}

impl StepMotionDecoder for CountingDecoder {
    fn decode(&self, timestamps: &[i64], raw: &[Vec<i64>]) -> Result<(Vec<i64>, Vec<[f32; 11]>)> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.packets.fetch_add(timestamps.len(), Ordering::Relaxed);
        self.inner.decode(timestamps, raw)
    }
}

// ---------------------------------------------------------------------------
// the two paths
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunPath {
    Cold,
    Incremental,
}

impl RunPath {
    pub fn name(&self) -> &'static str {
        match self {
            RunPath::Cold => "cold",
            RunPath::Incremental => "incremental",
        }
    }
}

#[derive(Debug)]
pub struct RunReport {
    pub path: RunPath,
    pub sessions: Vec<Session>,
    pub model_version: String,
    pub days_total: usize,
    pub days_recomputed: usize,
    pub days_replayed: usize,
    pub new_signal_rows: usize,
    pub new_step_rows: usize,
    pub tail_windows: usize,
    pub seconds: f64,
}

/// `run_one_day` up to the model call, and past it: a day the model refuses is an
/// empty day, not a failed run; a missing backend is fatal. Mirrors the
/// try/except in `run_one_day` and `collect_sessions`.
fn run_request(
    inference: &dyn ActivityInference,
    req: &activity::DayRequest,
    day: Date,
    args: &Args,
) -> Result<Vec<Session>> {
    match inference.run_day(req) {
        Ok(rows) => Ok(activity::sessions_from(day, &rows)),
        Err(error) => {
            if error.downcast_ref::<InferenceUnavailable>().is_some() {
                return Err(error);
            }
            if args.verbose {
                eprintln!("{}: model rejected incomplete day ({error})", day.iso());
            }
            Ok(Vec::new())
        }
    }
}

/// `run_full` in run_activity_model.py, reusing `activity::prepare`, with the
/// caches written on the way out.
fn run_full(
    args: &Args,
    db_path: &Path,
    repo: &Path,
    backends: &Backends,
    units: &mut UnitCache,
    src: &mut EventSource,
) -> Result<(Vec<Session>, String, usize)> {
    let pipeline = activity::prepare(args, db_path, repo)?;
    let mut sessions = Vec::new();
    units.clear();
    for day in &pipeline.days {
        let day_sessions = match &day.outcome {
            DayOutcome::NoMet | DayOutcome::Rejected => Vec::new(),
            DayOutcome::Run(req) => run_request(backends.inference.as_ref(), req, day.day, args)?,
        };
        units.set(&day.day.iso(), day_sessions.clone());
        sessions.extend(day_sessions);
    }
    units.save()?;
    src.note_epochs(&pipeline.clock);
    src.commit(&pipeline.clock, src.max_id(activity::WHERE)?)?;
    Ok((sessions, pipeline.model_version.clone(), pipeline.days.len()))
}

/// Recompute one local calendar day from windowed rows, through exactly the same
/// `build_signals` / `build_day_request` / model path the cold path uses.
fn recompute_day(
    args: &Args,
    src: &EventSource,
    clock: &Clock,
    day_key: &str,
    backends: &Backends,
    user_vector: &[f64],
    settings: &Settings,
) -> Result<Vec<Session>> {
    let day = match activity::parse_date(day_key) {
        Some(d) => d,
        None => bail!("bad day key: {day_key}"),
    };
    // `calendar.timegm(day) - int(tz * 3600)`, i.e. local midnight in unix s.
    let u0 = day.days_from_civil() * 86400 - (args.tz * 3600.0) as i64;
    let windows = src.ds_windows(u0 as f64, (u0 + 86400) as f64, clock);
    let rows = src.rows_in_windows(&windows, &activity::SIGNAL_TAGS, activity::WHERE)?;
    let signals = activity::build_signals(rows.into_iter(), clock);
    let steps: StepSeries = match backends.decoder.as_ref() {
        Some(decoder) => {
            let packets =
                src.packets_in_windows(&windows, &activity::STEP_TAGS, "body IS NOT NULL")?;
            let collected = activity::collect_step_packets(&packets, clock)?;
            activity::decode_step_packets(&collected, decoder.as_ref())?
        }
        None => Vec::new(),
    };
    let day_start_min = day.days_from_civil() as f64 * 1440.0 - args.tz * 60.0;
    match activity::build_day_request(day, day_start_min, user_vector, &signals, &steps, settings) {
        DayOutcome::NoMet | DayOutcome::Rejected => Ok(Vec::new()),
        DayOutcome::Run(req) => run_request(backends.inference.as_ref(), &req, day, args),
    }
}

/// `run_incremental`: recompute the days whose inputs changed (new rows, or a
/// boot that gained clock anchors since the last commit); every other day comes
/// from the unit cache. Output equals the cold path on the same data.
fn run_incremental(
    args: &Args,
    src: &mut EventSource,
    units: &mut UnitCache,
    backends: &Backends,
    user_vector: &[f64],
    settings: &Settings,
    force_day: Option<&str>,
) -> Result<RunReport> {
    let started = Instant::now();
    let (clock, tail) = src.clock(activity::WHERE)?;
    let mut dirty: BTreeSet<String> = BTreeSet::new();
    let mut new_signal_rows = 0;
    let mut new_step_rows = 0;
    // Signal rows and step packets are separate row sets (step packets carry no
    // decoded_json); either can dirty a day.
    for (where_sql, counter) in [
        (activity::WHERE, &mut new_signal_rows),
        (STEP_WHERE, &mut new_step_rows),
    ] {
        for row in src.new_rows(where_sql)? {
            *counter += 1;
            if clock.is_dated(row.0, Some(row.3)) {
                if let Some(unix) = clock.unix_s(row.0, Some(row.3)) {
                    dirty.insert(activity::Date::from_local_unix(unix, args.tz).iso());
                }
            }
        }
    }
    for window in &tail {
        let epoch = &clock.epochs[window.epoch];
        let (a, b) = (pred(epoch, window.lo), pred(epoch, window.hi));
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let mut day = activity::Date::from_local_unix(lo, args.tz);
        let last = activity::Date::from_local_unix(hi, args.tz);
        loop {
            dirty.insert(day.iso());
            if day >= last {
                break;
            }
            day = day.shift_days(1);
        }
    }
    if let Some(day) = force_day {
        dirty.insert(day.to_string());
    }

    let keys: BTreeSet<String> = units.days().union(&dirty).cloned().collect();
    let mut sessions = Vec::new();
    let mut recomputed = 0;
    let mut replayed = 0;
    for key in &keys {
        if !dirty.contains(key) {
            if let Some(cached) = units.get(key) {
                sessions.extend(cached.iter().cloned());
            }
            replayed += 1;
            continue;
        }
        recomputed += 1;
        let day_sessions = recompute_day(args, src, &clock, key, backends, user_vector, settings)?;
        units.set(key, day_sessions.clone());
        sessions.extend(day_sessions);
    }
    units.save()?;
    // A forced single-day pass did not cover the other days, so it must not
    // advance the watermark.
    if force_day.is_none() {
        src.commit(&clock, src.max_id(activity::WHERE)?)?;
    }
    Ok(RunReport {
        path: RunPath::Incremental,
        sessions,
        model_version: String::new(),
        days_total: keys.len(),
        days_recomputed: recomputed,
        days_replayed: replayed,
        new_signal_rows,
        new_step_rows,
        tail_windows: tail.len(),
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// The dispatch `main()` performs: cold when asked for it or when nothing is
/// cached, incremental otherwise.
pub fn run_model(args: &Args, db_path: &Path, repo: &Path, backends: &Backends) -> Result<RunReport> {
    let started = Instant::now();
    let model = activity::find_model(repo)?;
    if !model.exists() {
        bail!("error: model not found: {}", model.display());
    }
    let model_name = model.file_name().unwrap().to_string_lossy().to_string();
    let version_key = activity::numeric_key(&model_name);
    let version = version_key.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(".");
    let step_model = if args.no_stepmotion {
        None
    } else {
        model.parent().and_then(|d| activity::find_step_decoder(d))
    };
    let user_vector = activity::profile_vector(db_path, args.serial.as_deref());
    let settings = activity::model_settings(&version_key, args.threshold, args.min_duration);
    let fingerprint =
        settings_fingerprint(args, &model_name, &version, &user_vector, step_model.as_deref());

    let mut src = EventSource::new(db_path, args.serial.as_deref(), MODEL_KEY, STATE_VERSION)?;
    let mut units = UnitCache::new(db_path, MODEL_KEY, args.serial.as_deref(), &fingerprint);

    let mut report = if args.full || !src.has_state() || units.is_empty() {
        let (sessions, model_version, days) =
            run_full(args, db_path, repo, backends, &mut units, &mut src)?;
        RunReport {
            path: RunPath::Cold,
            sessions,
            model_version,
            days_total: days,
            days_recomputed: days,
            days_replayed: 0,
            new_signal_rows: 0,
            new_step_rows: 0,
            tail_windows: 0,
            seconds: 0.0,
        }
    } else {
        run_incremental(
            args,
            &mut src,
            &mut units,
            backends,
            &user_vector,
            &settings,
            args.date.as_deref(),
        )?
    };
    if report.model_version.is_empty() {
        report.model_version = version;
    }
    // --date narrows the answer to one day, exactly as the Python CLI does.
    if let Some(date) = &args.date {
        report.sessions.retain(|s| s.start.starts_with(date));
    }
    report.seconds = started.elapsed().as_secs_f64();
    Ok(report)
}

/// The JSON the CLI/dashboard expects, warm path included. Same argv contract as
/// `activity::json_for_args`.
pub fn json_for_args(argv: &[String]) -> Result<String> {
    let mut args = activity::parse_args(argv)?;
    let repo = activity::repo_root();
    let db_path = activity::resolve_db(args.db.as_deref(), &repo)?;
    args.user_vector = activity::profile_vector(&db_path, args.serial.as_deref());
    let model = activity::find_model(&repo)?;
    let step_model = if args.no_stepmotion {
        None
    } else {
        model.parent().and_then(|d| activity::find_step_decoder(d))
    };
    let backends = counted_backends(&model, step_model.as_deref())?;
    let report = run_model(&args, &db_path, &repo, &backends)?;
    if args.verbose {
        eprintln!(
            "path={} days_total={} recomputed={} replayed={} aad_calls={} gait_calls={}",
            report.path.name(),
            report.days_total,
            report.days_recomputed,
            report.days_replayed,
            backends.counters.aad(),
            backends.counters.decoder()
        );
    }
    Ok(activity::render_result(&report.model_version, &report.sessions))
}
