//! Port of `tools/nssa_inputs.py` — the ACM / IBI / temperature half of the NSSA
//! inputs. Every float32 operation numpy does in float32 is an `f32` here.

use crate::clock::Clock;
use crate::rows::{EventRow, TAG_ACM, TAG_IBI, TAG_TEMP};

pub const TEMP_SAMPLE_MS: i64 = 30_000;
/// `collect_nssa_inputs`'s boot-overlap guard.
pub const DS_SLACK: i64 = 6000;
/// Reject events whose clock disagrees with the window's own dating by more than 5 min.
pub const CLOCK_DRIFT_MS: i64 = 300_000;

pub type Beat = (i64, i64, i64);
pub type CorrectedBeat = (i64, i64, i64, u32);

#[derive(Debug, Default)]
pub struct NssaInputs {
    /// (start_ms, end_ms) of the ring bedtime, in epoch milliseconds.
    pub bedtime: (i64, i64),
    /// One 30 s window per sleep_acm_period: (epoch ms, [madIqr, madMax,
    /// madTrimmedMean, fingerAngleIqr, fingerAngleMax, fingerAngleMean]).
    pub acm: Vec<(i64, [f32; 6])>,
    /// ecore-corrected IBI: (epoch ms, ibi, amplitude, validity).
    pub ibi: Vec<CorrectedBeat>,
    /// The pre-correction beat stream `correct_ibis` was called with (kept for
    /// golden-vector comparison; the shim does not send it anywhere).
    pub beats_pre: Vec<Beat>,
    /// One sample per 30 s of sleep_temp_event: (epoch ms, degrees).
    pub temp: Vec<(i64, f32)>,
}

/// `EventParser::parse_api_sleep_acm_period`: bytes 0..5 are three `int +
/// frac/255` finger angles (tmean, max, iqr), bytes 6..11 three `12-bit/4095 +
/// high nibble` MAD statistics (tmean, max, iqr), all float32. Returned in the
/// NSSA model's column order; None for a malformed body.
pub fn decode_acm_period(body: &[u8]) -> Option<[f32; 6]> {
    if body.len() != 12 {
        return None;
    }
    let angle = [0, 2, 4].map(|i| body[i] as f32 / 255.0f32 + body[i + 1] as f32);
    let mad = [0usize, 2, 4].map(|i| {
        let lo = body[6 + i] as u32;
        let hi = body[7 + i] as u32;
        ((lo | ((hi & 0x0F) << 8)) as f32) / 4095.0f32 + (hi >> 4) as f32
    });
    let [fa_mean, fa_max, fa_iqr] = angle;
    let [mad_mean, mad_max, mad_iqr] = mad;
    Some([mad_iqr, mad_max, mad_mean, fa_iqr, fa_max, fa_mean])
}

/// `parse_api_sleep_temp_event`: an even-length 2..14 byte body of little-endian
/// centi-degrees; sample i of n is stamped event - (n-1-i)*30 s.
pub fn expand_temp_event(body: &[u8], event_ms: i64) -> Vec<(i64, f32)> {
    if body.len() < 2 || body.len() > 15 || body.len() % 2 != 0 {
        return Vec::new();
    }
    let n = body.len() / 2;
    (0..n)
        .map(|i| {
            let raw = body[2 * i] as u32 | ((body[2 * i + 1] as u32) << 8);
            let value = raw as f32 / 100.0f32;
            (event_ms - (n as i64 - 1 - i as i64) * TEMP_SAMPLE_MS, value)
        })
        .collect()
}

/// `parse_api_ibi_and_amplitude_event`: beat k starts at event time minus
/// ibi[k..5]; each stamp is clamped to the previous one (the parser's session
/// keeps the last stamp across events). Returns (beats, new_last).
pub fn expand_ibi_event(ibis: &[i64; 6], amps: &[i64], event_ms: i64, last_ms: i64) -> (Vec<Beat>, i64) {
    let mut beats = Vec::with_capacity(6);
    let mut stamp = event_ms - ibis.iter().sum::<i64>();
    let mut last = last_ms;
    for (i, ibi) in ibis.iter().enumerate() {
        let clamped = if stamp > last { stamp } else { last };
        let amp = amps.get(i).copied().unwrap_or(0);
        beats.push((clamped, *ibi, amp));
        last = clamped;
        stamp += ibi;
    }
    (beats, last)
}

/// `RingEventParserExtKt.indexOfFirstAndLastOrNull` + `subListInclusive`: keep
/// everything from the first to the last item stamped inside [start, end].
pub fn inclusive_window<T: Clone, K: Fn(&T) -> i64>(
    items: &[T],
    start_ms: i64,
    end_ms: i64,
    key: K,
) -> Vec<T> {
    let inside = |i: usize| {
        let t = key(&items[i]);
        start_ms <= t && t <= end_ms
    };
    let first = (0..items.len()).find(|&i| inside(i));
    match first {
        None => Vec::new(),
        Some(first) => {
            let last = (0..items.len()).rev().find(|&i| inside(i)).expect("first implies last");
            items[first..=last].to_vec()
        }
    }
}

/// Assemble one bedtime's NSSA inputs (see `tools/nssa_inputs.py`).
///
/// `rows` are the decoded events in DB order; only 0x60/0x72/0x75 matter. IBI
/// correction is delegated to `ibi_correct::correct_ibis` — the same Rust code the
/// Python shim in `tools/ibi_correction.py` calls.
pub fn collect_nssa_inputs(
    rows: &[EventRow],
    start_ds: i64,
    end_ds: i64,
    captured_unix: i64,
    clock: &Clock,
) -> NssaInputs {
    let start_ms = clock.ms(start_ds, Some(captured_unix));
    let end_ms = clock.ms(end_ds, Some(captured_unix));
    let mut acm_events: Vec<(i64, [f32; 6])> = Vec::new();
    let mut ibi_events: Vec<(i64, [i64; 6], Vec<i64>)> = Vec::new();
    let mut temp_events: Vec<(i64, Vec<u8>)> = Vec::new();

    for row in rows {
        if row.tag != TAG_ACM && row.tag != TAG_IBI && row.tag != TAG_TEMP {
            continue;
        }
        // ds restarts after a reboot: reject rows from another boot epoch whose
        // relative counter happens to overlap this night's range.
        if !(start_ds - DS_SLACK <= row.ds && row.ds <= end_ds + DS_SLACK) {
            continue;
        }
        let event_ms = clock.ms(row.ds, Some(row.cu));
        if (event_ms - (start_ms + (row.ds - start_ds) * 100)).abs() > CLOCK_DRIFT_MS {
            continue;
        }
        // The app queries DbRawEvents by event time inside the bedtime.
        if !(start_ms <= event_ms && event_ms <= end_ms) {
            continue;
        }
        match row.tag {
            TAG_ACM => {
                if let Some(body) = &row.body {
                    if let Some(stats) = decode_acm_period(body) {
                        acm_events.push((event_ms, stats));
                    }
                }
            }
            TAG_TEMP => {
                if let Some(body) = &row.body {
                    temp_events.push((event_ms, body.clone()));
                }
            }
            _ => {
                if let Some(ibis) = row.ibi_ms {
                    ibi_events.push((event_ms, ibis, row.amplitude.clone().unwrap_or_default()));
                }
            }
        }
    }

    // Python's list.sort is stable; so is Rust's sort_by_key.
    acm_events.sort_by_key(|e| e.0);
    temp_events.sort_by_key(|e| e.0);
    ibi_events.sort_by_key(|e| e.0);

    let acm = inclusive_window(&acm_events, start_ms, end_ms, |e| e.0);

    let mut temp: Vec<(i64, f32)> = Vec::new();
    for (event_ms, body) in &temp_events {
        temp.extend(expand_temp_event(body, *event_ms));
    }
    temp = inclusive_window(&temp, start_ms, end_ms, |e| e.0);

    let mut beats: Vec<Beat> = Vec::new();
    let mut last = 0i64;
    for (event_ms, ibis, amps) in &ibi_events {
        let (expanded, new_last) = expand_ibi_event(ibis, amps, *event_ms, last);
        beats.extend(expanded);
        last = new_last;
    }
    beats = inclusive_window(&beats, start_ms, end_ms, |b| b.0);

    let ibi = if beats.is_empty() {
        Vec::new()
    } else {
        let raw: Vec<(u64, i64, i64)> =
            beats.iter().map(|(ts, ibi, amp)| (*ts as u64, *ibi, *amp)).collect();
        ibi_correct::correct_ibis(&raw)
            .into_iter()
            .map(|(ts, ibi, amp, validity)| (ts as i64, ibi, amp, validity))
            .collect()
    };

    NssaInputs { bedtime: (start_ms, end_ms), acm, ibi, beats_pre: beats, temp }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acm_decode_matches_numpy_float32_order() {
        // Body of all zeros decodes to six zeros; a 12-byte body with known bytes
        // exercises both the angle and MAD branches. The bits below are
        // tools/nssa_inputs.decode_acm_period under numpy float32, in the model's
        // column order [mad_iqr, mad_max, mad_mean, fa_iqr, fa_max, fa_mean].
        let body = [1u8, 2, 3, 4, 5, 6, 7, 0x12, 9, 0x03, 11, 0x4F];
        let v = decode_acm_period(&body).unwrap();
        assert_eq!(
            v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            vec![1084102626, 1044532261, 1066416388, 1086365857, 1082155104, 1073758272]
        );
        assert_eq!(decode_acm_period(&[0u8; 11]), None);
    }

    #[test]
    fn ibi_stamps_clamp_forward() {
        // Every beat starts at event minus the IBIs that follow it, then each stamp
        // is clamped up to the previous one — so a late-arriving event replays at
        // `last_ms` until its own counter catches up (Python's `expand_ibi_event`).
        let (beats, last) = expand_ibi_event(&[100, 100, 100, 100, 100, 100], &[7], 1000, 900);
        assert_eq!(beats[0], (900, 100, 7)); // clamped up to last_ms
        assert_eq!(beats[1..], [(900, 100, 0); 5]);
        assert_eq!(last, 900);
        // Without a clamp the stamps are the plain running sum, and amplitudes
        // past the end of the list decode to 0.
        let (free, last) = expand_ibi_event(&[100; 6], &[7, 8, 9], 1000, 0);
        assert_eq!(
            free,
            [(400, 100, 7), (500, 100, 8), (600, 100, 9), (700, 100, 0), (800, 100, 0), (900, 100, 0)]
        );
        assert_eq!(last, 900);
    }

    #[test]
    fn temp_expansion_is_centi_degrees_f32() {
        let out = expand_temp_event(&[0x10, 0x27], 100_000);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 100_000);
        assert_eq!(out[0].1.to_bits(), (10000u32 as f32 / 100.0f32).to_bits());
    }

    #[test]
    fn inclusive_window_spans_first_to_last() {
        let items = vec![(0i64, 'a'), (5, 'b'), (50, 'c'), (95, 'd'), (120, 'e')];
        let w = inclusive_window(&items, 10, 100, |i| i.0);
        assert_eq!(w, vec![(50, 'c'), (95, 'd')]);
        assert!(inclusive_window(&items, 200, 300, |i| i.0).is_empty());
    }
}
