//! Port of `run_sleep_model.ring_bedtimes`.

use std::collections::HashMap;

use crate::clock::Clock;
use crate::rows::{EventRow, TAG_BEDTIME};

/// The ring's bedtime periods inside a summary window: the latest 0x76 per start
/// (the app replaces an interim result whenever the ring extends its bedtime).
pub fn ring_bedtimes(
    clock: &Clock,
    rows: &[EventRow],
    start_ds: i64,
    end_ds: i64,
    captured: i64,
) -> Vec<(i64, i64, i64)> {
    let mut latest: HashMap<i64, (i64, i64, i64)> = HashMap::new();
    for row in rows {
        if row.tag != TAG_BEDTIME {
            continue;
        }
        let (s, e) = match (row.bedtime_start_ds, row.bedtime_end_ds) {
            (Some(s), Some(e)) => (s, e),
            _ => continue,
        };
        if e <= s {
            continue;
        }
        if (clock.ms(s, Some(row.cu)) - clock.ms(s, Some(captured))).abs() > 300_000 {
            continue; // another boot's counter
        }
        if s < end_ds && e > start_ds {
            latest.insert(s, (s, e, row.cu));
        }
    }
    let mut out: Vec<(i64, i64, i64)> = latest.values().copied().collect();
    out.sort(); // sorted(latest.values())
    out
}

/// `score_window`'s `bed_cu`: the capture time of the newest 0x76 row that starts
/// this window, else the last row's capture time (else 0).
pub fn bed_capture(rows: &[EventRow], start_ds: i64, captured_unix: Option<i64>) -> i64 {
    if let Some(c) = captured_unix {
        return c;
    }
    for row in rows.iter().rev() {
        if row.tag == TAG_BEDTIME && row.bedtime_start_ds == Some(start_ds) {
            return row.cu;
        }
    }
    rows.last().map(|r| r.cu).unwrap_or(0)
}
