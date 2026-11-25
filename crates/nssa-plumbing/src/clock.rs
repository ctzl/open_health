//! Port of `tools/epoch_time.py` (which itself mirrors
//! `crates/oura-summary/src/ring_time.rs` and EventStore.RingClock on iOS).
//!
//! Every float here is the same f64 operation the Python does: anchors are integer
//! unix seconds, `predicted = anchor_unix + (ds - anchor_ds) / 10.0`.
//!
//! `Clock::from_epochs_json` reads the exact list layout `build_epochs()` produces
//! (`[min_ds, max_ds, fallback_unix, capture_min, capture_max, [[ds, unix], …],
//! [boot ds, …]]`), which is what `tools/nssa_plumbing_golden.py` dumps as
//! `clock.json` and what a shim would send once per worker start.

use serde_json::Value;

pub const EPOCH_RESET_SLACK_DS: i64 = 6 * 3600 * 10;
pub const FUTURE_SLACK_S: f64 = 6.0 * 3600.0;
pub const ANCHOR_AGREEMENT_S: f64 = 30.0 * 60.0;
pub const ANCHOR_AGREEMENT_FRACTION: f64 = 0.02;

/// One boot epoch, in `build_epochs()`'s list layout.
#[derive(Clone, Debug, Default)]
pub struct Epoch {
    pub min_ds: i64,
    pub max_ds: i64,
    pub fallback_unix: i64,
    pub capture_min: i64,
    pub capture_max: i64,
    /// time-sync / RTC anchors: `(ds, unix_time)`, in insertion order.
    pub anchors: Vec<(i64, i64)>,
    /// ring_start (0x41) counters seen during this boot.
    pub boots: Vec<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Undated {
    MissingAnchor,
    AcceleratedCounter,
    AmbiguousRebootStall,
}

/// Anchors on either side of a counter value, as `bracket()` reports them.
enum Bracket {
    Consistent,
    Erratic,
    Stalled((i64, i64, usize), (i64, i64, usize)),
}

pub struct Clock {
    epochs: Vec<Epoch>,
    /// per epoch: anchors indexed by (ds, unix, insertion order), sorted, plus the
    /// parallel ds list bisected by every lookup (`_prepared()` in Python).
    indexed: Vec<Vec<(i64, i64, usize)>>,
    dsl: Vec<Vec<i64>>,
}

fn bisect_left(list: &[i64], x: i64) -> usize {
    let (mut lo, mut hi) = (0usize, list.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        if list[mid] < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

impl Clock {
    pub fn new(epochs: Vec<Epoch>) -> Clock {
        let mut indexed = Vec::with_capacity(epochs.len());
        let mut dsl = Vec::with_capacity(epochs.len());
        for e in &epochs {
            let mut idx: Vec<(i64, i64, usize)> = e
                .anchors
                .iter()
                .enumerate()
                .map(|(i, (ds, unix))| (*ds, *unix, i))
                .collect();
            idx.sort();
            dsl.push(idx.iter().map(|a| a.0).collect());
            indexed.push(idx);
        }
        Clock {
            epochs,
            indexed,
            dsl,
        }
    }

    /// Parse the Python epoch list layout (see `tools/nssa_plumbing_golden.py`).
    pub fn from_epochs_json(text: &str) -> Result<Clock, String> {
        let value: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let list = value.as_array().ok_or("clock: expected an array of epochs")?;
        let mut epochs = Vec::with_capacity(list.len());
        for item in list {
            let e = item.as_array().ok_or("clock: expected an epoch array")?;
            if e.len() < 7 {
                return Err("clock: short epoch record".to_string());
            }
            let int = |v: &Value| v.as_i64().unwrap_or(0);
            let mut epoch = Epoch {
                min_ds: int(&e[0]),
                max_ds: int(&e[1]),
                fallback_unix: int(&e[2]),
                capture_min: int(&e[3]),
                capture_max: int(&e[4]),
                ..Default::default()
            };
            if let Some(anchors) = e[5].as_array() {
                for a in anchors {
                    if let Some(a) = a.as_array() {
                        epoch.anchors.push((int(&a[0]), int(&a[1])));
                    }
                }
            }
            if let Some(boots) = e[6].as_array() {
                epoch.boots = boots.iter().map(int).collect();
            }
            epochs.push(epoch);
        }
        Ok(Clock::new(epochs))
    }

    pub fn epochs(&self) -> &[Epoch] {
        &self.epochs
    }

    fn capture_distance(&self, e: &Epoch, captured: i64) -> i64 {
        if captured < e.capture_min {
            e.capture_min - captured
        } else if captured > e.capture_max {
            captured - e.capture_max
        } else {
            0
        }
    }

    fn pick_epoch(&self, ds: i64, captured: Option<i64>) -> usize {
        let candidates: Vec<usize> = (0..self.epochs.len())
            .filter(|i| {
                let e = &self.epochs[*i];
                e.min_ds - EPOCH_RESET_SLACK_DS <= ds && ds <= e.max_ds + EPOCH_RESET_SLACK_DS
            })
            .collect();
        if candidates.is_empty() {
            return self.epochs.len().saturating_sub(1);
        }
        // Python's min() keeps the first minimum, over ascending index order.
        let mut best = candidates[0];
        match captured {
            Some(c) => {
                let mut bd = self.capture_distance(&self.epochs[best], c);
                for &i in &candidates[1..] {
                    let d = self.capture_distance(&self.epochs[i], c);
                    if d < bd {
                        best = i;
                        bd = d;
                    }
                }
            }
            None => {
                let mut bd = self.epochs[best].max_ds - self.epochs[best].min_ds;
                for &i in &candidates[1..] {
                    let d = self.epochs[i].max_ds - self.epochs[i].min_ds;
                    if d < bd {
                        best = i;
                        bd = d;
                    }
                }
            }
        }
        best
    }

    fn bracket(&self, ei: usize, ds: i64) -> Bracket {
        let anchors = &self.indexed[ei];
        let dsl = &self.dsl[ei];
        let idx = bisect_left(dsl, ds);
        if idx == 0 || idx >= anchors.len() {
            return Bracket::Consistent;
        }
        let prev = anchors[idx - 1];
        let next = anchors[idx];
        if next.0 == ds {
            return Bracket::Consistent;
        }
        let wall_s = next.1 - prev.1;
        let counter_s = (next.0 - prev.0) as f64 / 10.0;
        let tolerance = ANCHOR_AGREEMENT_S.max(counter_s * ANCHOR_AGREEMENT_FRACTION);
        if (wall_s as f64) < counter_s - tolerance {
            Bracket::Erratic
        } else if (wall_s as f64) > counter_s + tolerance {
            Bracket::Stalled(prev, next)
        } else {
            Bracket::Consistent
        }
    }

    /// Anchor closest to `ds`; ties resolve to the earliest insertion index, like
    /// Python's `min(key=abs)` over the original order.
    fn nearest(&self, ei: usize, ds: i64) -> (i64, i64) {
        let anchors = &self.indexed[ei];
        let dsl = &self.dsl[ei];
        let idx = bisect_left(dsl, ds);
        if idx == 0 {
            return (anchors[0].0, anchors[0].1);
        }
        if idx >= anchors.len() {
            let last = anchors[anchors.len() - 1];
            return (last.0, last.1);
        }
        let before = anchors[idx - 1];
        let after = anchors[idx];
        if (ds - before.0, before.2) <= (after.0 - ds, after.2) {
            (before.0, before.1)
        } else {
            (after.0, after.1)
        }
    }

    /// `make_unix_s(...)`'s inner `unix_s`.
    pub fn unix_s(&self, ds: i64, captured: Option<i64>) -> Option<f64> {
        let ei = self.pick_epoch(ds, captured);
        let e = &self.epochs[ei];
        if !e.anchors.is_empty() {
            let (anchor_ds, anchor_unix) = self.nearest(ei, ds);
            let predicted = anchor_unix as f64 + (ds - anchor_ds) as f64 / 10.0;
            match self.bracket(ei, ds) {
                Bracket::Stalled(before, after) => {
                    let late = after.1 as f64 - (after.0 - ds) as f64 / 10.0;
                    let early = before.1 as f64 + (ds - before.0) as f64 / 10.0;
                    // A ring_start between the anchors marks the stall exactly.
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
                            return None; // lost time cannot be assigned to one of several reboots
                        }
                        return Some(if ds >= hi { late } else { early });
                    }
                    return Some(match captured {
                        None => late,
                        Some(c) => {
                            if late <= c as f64 + FUTURE_SLACK_S {
                                late
                            } else {
                                early
                            }
                        }
                    });
                }
                Bracket::Erratic => return Some(predicted), // undated; callers check is_dated()
                Bracket::Consistent => {
                    if captured.is_none()
                        || predicted <= captured.unwrap_or(0) as f64 + FUTURE_SLACK_S
                    {
                        return Some(predicted);
                    }
                }
            }
            if let Some(c) = captured {
                // Only borrow a boot's clock when this ds continues that boot's
                // counter; a rebooted ring restarts near zero and must not be
                // projected through an older boot that only ran at higher counts.
                let mut best: Option<f64> = None;
                for epoch in &self.epochs {
                    if !(epoch.min_ds - EPOCH_RESET_SLACK_DS <= ds
                        && ds <= epoch.max_ds + EPOCH_RESET_SLACK_DS)
                    {
                        continue;
                    }
                    for (a_ds, unix) in &epoch.anchors {
                        let v = *unix as f64 + (ds - *a_ds) as f64 / 10.0;
                        if v <= c as f64 + FUTURE_SLACK_S {
                            best = Some(best.map_or(v, |b| b.max(v)));
                        }
                    }
                }
                if let Some(v) = best {
                    return Some(v);
                }
            }
        }
        let fallback = e.fallback_unix as f64 - (e.max_ds - ds) as f64 / 10.0;
        match captured {
            Some(c) => Some(fallback.min(c as f64 + FUTURE_SLACK_S)),
            None => Some(fallback),
        }
    }

    /// `run_sleep_model.ms`: epoch milliseconds, falling back to the raw counter.
    pub fn ms(&self, ds: i64, captured: Option<i64>) -> i64 {
        match self.unix_s(ds, captured) {
            Some(u) => (u * 1000.0).round_ties_even() as i64,
            None => ds * 100,
        }
    }

    /// `undated_reason`.
    pub fn undated_reason(&self, ds: i64, captured: Option<i64>) -> Option<Undated> {
        let candidates: Vec<usize> = (0..self.epochs.len())
            .filter(|i| {
                let e = &self.epochs[*i];
                e.min_ds - EPOCH_RESET_SLACK_DS <= ds && ds <= e.max_ds + EPOCH_RESET_SLACK_DS
            })
            .collect();
        if candidates.is_empty() {
            return Some(Undated::MissingAnchor);
        }
        let ei = self.pick_epoch(ds, captured);
        let e = &self.epochs[ei];
        if e.anchors.is_empty() {
            return Some(Undated::MissingAnchor);
        }
        match self.bracket(ei, ds) {
            Bracket::Erratic => Some(Undated::AcceleratedCounter),
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
                        return Some(Undated::AmbiguousRebootStall);
                    }
                }
                None
            }
            Bracket::Consistent => {
                let (anchor_ds, anchor_unix) = self.nearest(ei, ds);
                let c = match captured {
                    None => return None,
                    Some(c) => c,
                };
                if anchor_unix as f64 + (ds - anchor_ds) as f64 / 10.0 <= c as f64 + FUTURE_SLACK_S {
                    return None;
                }
                for epoch in &self.epochs {
                    if !(epoch.min_ds - EPOCH_RESET_SLACK_DS <= ds
                        && ds <= epoch.max_ds + EPOCH_RESET_SLACK_DS)
                    {
                        continue;
                    }
                    for (a_ds, unix) in &epoch.anchors {
                        if *unix as f64 + (ds - *a_ds) as f64 / 10.0 <= c as f64 + FUTURE_SLACK_S {
                            return None;
                        }
                    }
                }
                Some(Undated::AcceleratedCounter)
            }
        }
    }

    pub fn is_dated(&self, ds: i64, captured: Option<i64>) -> bool {
        self.undated_reason(ds, captured).is_none()
    }
}

/// `build_epochs` over `(ds, tag, decoded_json, captured_unix)` rows in DB order.
pub fn build_epochs(rows: &[(i64, u8, Option<&str>, i64)]) -> Vec<Epoch> {
    let mut epochs: Vec<Epoch> = Vec::new();
    // Callers query by (captured_unix, id). Do not sort by ds here: thousands of
    // rows share one capture second, and sorting those rows would erase reboot jumps.
    for (ds, tag, js, cu) in rows {
        let ds = *ds;
        let fresh = match epochs.last() {
            None => true,
            Some(e) => ds < e.max_ds - EPOCH_RESET_SLACK_DS,
        };
        if fresh {
            epochs.push(Epoch {
                min_ds: ds,
                max_ds: ds,
                fallback_unix: *cu,
                capture_min: *cu,
                capture_max: *cu,
                ..Default::default()
            });
        } else {
            let e = epochs.last_mut().unwrap();
            if ds >= e.max_ds {
                e.max_ds = ds;
                e.fallback_unix = *cu;
            }
            e.min_ds = e.min_ds.min(ds);
            e.capture_min = e.capture_min.min(*cu);
            e.capture_max = e.capture_max.max(*cu);
        }
        let e = epochs.last_mut().unwrap();
        if *tag == 0x41 {
            e.boots.push(ds);
        }
        if (*tag == 0x42 || *tag == 0x85) && js.is_some() {
            if let Ok(value) = serde_json::from_str::<Value>(js.unwrap_or("")) {
                if let Some(unix) = value.get("unix_time").and_then(|v| v.as_i64()) {
                    e.anchors.push((ds, unix));
                }
            }
        }
    }
    epochs
}

/// `run_sleep_model.hm`: local HH:MM for an epoch-ms value at a fixed UTC offset.
pub fn hm(ms: i64, tz_hours: f64) -> String {
    // datetime.utcfromtimestamp rounds to the nearest microsecond, ties to even.
    let micros = ((ms as f64 / 1000.0) + tz_hours * 3600.0) * 1_000_000.0;
    let micros = micros.round_ties_even() as i128;
    let secs = micros.div_euclid(1_000_000);
    let h = secs.div_euclid(3600).rem_euclid(24);
    let m = secs.div_euclid(60).rem_euclid(60);
    format!("{h:02}:{m:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hm_matches_utc_plus_offset() {
        // 1795343767000 ms = 2026-11-22T10:36:07Z; TZ=+1 -> 11:36.
        assert_eq!(hm(1795343767000, 1.0), "11:36");
        assert_eq!(hm(1795343767000, 0.0), "10:36");
        // Negative offsets and pre-epoch values follow Python's divmod semantics.
        assert_eq!(hm(1795343767000, -5.5), "05:06");
    }

    #[test]
    fn epoch_split_on_counter_reset() {
        // A counter drop only starts a new boot epoch when it exceeds
        // EPOCH_RESET_SLACK_DS (60 h of deciseconds); smaller dips stay merged.
        let rows: Vec<(i64, u8, Option<&str>, i64)> = vec![
            (300_000, 0x60, None, 100),
            (300_100, 0x42, Some(r#"{"unix_time": 500}"#), 101),
            (1000, 0x60, None, 102), // inside the slack: same boot
            (50, 0x60, None, 103),   // past the slack: new boot
        ];
        let epochs = build_epochs(&rows);
        assert_eq!(epochs.len(), 2);
        assert_eq!(epochs[0].anchors, vec![(300_100, 500)]);
        assert_eq!((epochs[0].min_ds, epochs[0].max_ds), (300_000, 300_100));
        assert_eq!((epochs[1].min_ds, epochs[1].max_ds), (50, 1000));
    }
}
