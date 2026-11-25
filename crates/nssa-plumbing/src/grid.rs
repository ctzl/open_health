//! Port of `run_sleep_model.py`'s window plumbing: `score_window`, `grid_stages`
//! and `nssa_stager.demographics`.
//!
//! The model itself is not here: `score_window` takes a `Stager` callback, exactly
//! like the Python composes `stage_bedtime` (which is `collect_nssa_inputs` +
//! `breath_series` + the torch model). Everything that is not the model — clock
//! dating, bedtime discovery, the 30 s grid, and every aggregate the dashboard
//! reads — is here and is bit-exact against the golden vectors.

use serde_json::{Map, Value};

use crate::bedtimes::{bed_capture, ring_bedtimes};
use crate::clock::{hm, Clock};
use crate::num::{round0, round1};
use crate::rows::EventRow;

pub const EPOCH_MS: i64 = 30_000;
/// `run_sleep_model.STAGE`
pub const STAGE: [(i32, &str); 4] = [(1, "DEEP"), (2, "LIGHT"), (3, "REM"), (4, "WAKE")];

/// `nssa_stager.demographics`: [age, bmi, sex] as the model wants them.
pub fn demographics(age: f64, height_m: f64, weight_kg: f64, sex: &str) -> [f64; 3] {
    let bmi = weight_kg / (height_m * height_m);
    let sex = match sex.to_uppercase().chars().next() {
        Some('F') => -1.0,
        Some('M') => 1.0,
        _ => 0.0,
    };
    [age.trunc(), bmi, sex]
}

/// One staged ring bedtime: `stage_bedtime`'s return value.
#[derive(Clone, Debug)]
pub struct Staged {
    /// epoch start ms (NSSA's own grid: floor30(bedtime start) + n * 30 s)
    pub t: Vec<i64>,
    pub codes: Vec<i32>,
    /// BDI apnea epochs, or None when the BDI model could not run.
    pub apnea: Option<Vec<(i64, f64)>>,
    pub breath: Vec<(i64, f64)>,
}

pub trait Stager {
    /// `stage_bedtime(b_start, b_end, bed_cu, demo)`; None = unstaged night.
    fn stage(&mut self, b_start_ds: i64, b_end_ds: i64, bed_cu: i64, demo: &[f64; 3]) -> Option<Staged>;
}

/// Python's `//` is floor division; `-(-x // d)` is its ceiling.
fn ceil_div(x: i64, d: i64) -> i64 {
    -((-x).div_euclid(d))
}

/// `grid_stages`: paint NSSA epochs onto the summary's 30 s grid.
pub fn grid_stages(
    periods: &[(Vec<i64>, Vec<i32>)],
    failed: &[(i64, i64)],
    start_ms: i64,
    end_ms: i64,
) -> Vec<i32> {
    let n = std::cmp::max(1, ceil_div(end_ms - start_ms, EPOCH_MS) as usize);
    let mut out = vec![4i32; n];
    for (b_start_ms, b_end_ms) in failed {
        for idx in 0..n {
            let mid = start_ms + idx as i64 * EPOCH_MS + EPOCH_MS / 2;
            if b_start_ms <= &mid && mid < *b_end_ms {
                out[idx] = 0;
            }
        }
    }
    for (ts, codes) in periods {
        for (t, code) in ts.iter().zip(codes.iter()) {
            // the cell whose midpoint falls inside [t, t + 30 s)
            let idx = ceil_div(t - start_ms - EPOCH_MS / 2, EPOCH_MS);
            if (0..n as i64).contains(&idx) && (1..=4).contains(code) {
                out[idx as usize] = *code;
            }
        }
    }
    out
}

/// `score_window`: stage one summary window. Err carries the Python error string.
pub fn score_window(
    clock: &Clock,
    rows: &[EventRow],
    start_ds: i64,
    end_ds: i64,
    captured_unix: Option<i64>,
    tz: f64,
    demo: &[f64; 3],
    stager: &mut dyn Stager,
) -> Result<Value, String> {
    let bed_cu = bed_capture(rows, start_ds, captured_unix);
    if !clock.is_dated(start_ds, Some(bed_cu)) || !clock.is_dated(end_ds, Some(bed_cu)) {
        return Err("sleep window clock is undated or ambiguous".to_string());
    }
    let mut beds = ring_bedtimes(clock, rows, start_ds, end_ds, bed_cu);
    let mut bedtime_source = "ring";
    if beds.is_empty() {
        // no ring bedtime (signal-derived night): stage the summary window
        beds = vec![(start_ds, end_ds, bed_cu)];
        bedtime_source = "summary";
    }

    let mut periods: Vec<(Vec<i64>, Vec<i32>)> = Vec::new();
    let mut failed: Vec<(i64, i64)> = Vec::new();
    let mut apnea_events: Vec<(i64, f64)> = Vec::new();
    let mut apnea_partial = false;
    let mut breath_windows: Vec<(i64, f64)> = Vec::new();
    for (b_start, b_end, _cu) in beds {
        match stager.stage(b_start, b_end, bed_cu, demo) {
            Some(s) => {
                breath_windows.extend(s.breath.iter().copied());
                match &s.apnea {
                    None => apnea_partial = true, // BDI unavailable for this bedtime
                    Some(a) => apnea_events.extend(a.iter().copied()),
                }
                periods.push((s.t, s.codes));
            }
            None => failed.push((clock.ms(b_start, Some(bed_cu)), clock.ms(b_end, Some(bed_cu)))),
        }
    }
    if periods.is_empty() {
        return Err(
            "NSSA could not stage this window (missing ACM, IBI or sleep temperature)".to_string(),
        );
    }

    let start_ms = clock.ms(start_ds, Some(bed_cu));
    let end_ms = clock.ms(end_ds, Some(bed_cu));
    let stages = grid_stages(&periods, &failed, start_ms, end_ms);
    let n = stages.len();
    let known = stages.iter().filter(|c| **c != 0).count();
    let mut mins: [f64; 5] = [0.0; 5]; // index 1..=4, like STAGE
    for (code, _) in STAGE {
        mins[code as usize] = stages.iter().filter(|c| **c == code).count() as f64 * 0.5;
    }
    let asleep = mins[1] + mins[2] + mins[3];
    // max(periods, key=len(codes)): the first maximum wins.
    let main = periods
        .iter()
        .max_by(|a, b| a.1.len().cmp(&b.1.len()).then_with(|| std::cmp::Ordering::Greater))
        .expect("periods is non-empty");

    let mut out = Map::new();
    out.insert("start_ds".into(), Value::from(start_ds));
    out.insert("end_ds".into(), Value::from(end_ds));
    out.insert("captured_unix".into(), Value::from(bed_cu));
    out.insert("start_local".into(), Value::from(hm(start_ms, tz)));
    out.insert("end_local".into(), Value::from(hm(end_ms, tz)));
    out.insert("epochs".into(), Value::from(n as i64));
    out.insert("in_bed_min".into(), Value::from(n as f64 * 0.5));
    out.insert("asleep_min".into(), Value::from(asleep));
    out.insert(
        "efficiency_pct".into(),
        if known == n {
            Value::from(round0(100.0 * asleep / (n as f64 * 0.5)))
        } else {
            Value::Null
        },
    );
    out.insert("source".into(), Value::from("nssa"));
    out.insert("bedtime_source".into(), Value::from(bedtime_source));
    out.insert(
        "stages".into(),
        Value::Array(stages.iter().map(|c| Value::from(*c as i64)).collect()),
    );
    out.insert("stages_t0_unix".into(), Value::from(main.0[0] as f64 / 1000.0));
    out.insert(
        "nssa_periods".into(),
        Value::Array(
            periods
                .iter()
                .map(|(ts, codes)| {
                    let mut m = Map::new();
                    m.insert("t0_ms".into(), Value::from(ts[0]));
                    m.insert(
                        "stages".into(),
                        Value::Array(codes.iter().map(|c| Value::from(*c as i64)).collect()),
                    );
                    Value::Object(m)
                })
                .collect(),
        ),
    );
    if apnea_partial {
        out.insert("apnea".into(), Value::Null);
    } else {
        apnea_events.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let mut m = Map::new();
        m.insert(
            "events".into(),
            Value::Array(
                apnea_events
                    .iter()
                    .map(|(t, s)| Value::Array(vec![Value::from(*t), Value::from(*s)]))
                    .collect(),
            ),
        );
        m.insert("flagged_epochs".into(), Value::from(apnea_events.len() as i64));
        m.insert("flagged_min".into(), Value::from(round1(apnea_events.len() as f64 * 0.5)));
        out.insert("apnea".into(), Value::Object(m));
    }
    if breath_windows.len() >= 10 {
        breath_windows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let mut vals: Vec<f64> = breath_windows.iter().map(|(_, v)| *v).collect();
        vals.sort_by(|a, b| a.total_cmp(b));
        let median = round1(vals[vals.len() / 2]);
        let mut m = Map::new();
        m.insert(
            "series".into(),
            Value::Array(
                breath_windows
                    .iter()
                    .map(|(t, v)| Value::Array(vec![Value::from(*t), Value::from(*v)]))
                    .collect(),
            ),
        );
        m.insert("median".into(), Value::from(median));
        m.insert("windows".into(), Value::from(breath_windows.len() as i64));
        out.insert("breath".into(), Value::Object(m));
    } else {
        out.insert("breath".into(), Value::Null);
    }
    out.insert(
        "demographics".into(),
        Value::Array(demo.iter().map(|v| Value::from(*v)).collect()),
    );
    for (code, name) in STAGE {
        let key = name.to_lowercase();
        out.insert(format!("{key}_min"), Value::from(mins[code as usize]));
        out.insert(
            format!("{key}_pct"),
            Value::from(round0(100.0 * mins[code as usize] / (n as f64 * 0.5))),
        );
    }
    Ok(Value::Object(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_div_follows_python_floor() {
        assert_eq!(ceil_div(59_999, 30_000), 2);
        assert_eq!(ceil_div(60_000, 30_000), 2);
        assert_eq!(ceil_div(60_001, 30_000), 3);
        assert_eq!(ceil_div(-1, 30_000), 0);
        assert_eq!(ceil_div(-30_000, 30_000), -1);
        assert_eq!(ceil_div(-30_001, 30_000), -1);
    }

    #[test]
    fn grid_paints_by_cell_midpoint() {
        // One epoch 15 s before the grid start still owns cell 0 (midpoint 15 s).
        let periods = vec![(vec![15_000i64, 45_000], vec![2i32, 3])];
        let g = grid_stages(&periods, &[], 0, 90_000);
        assert_eq!(g, vec![2, 3, 4]);
        // An unstaged bedtime paints its own cells 0 (unknown), not WAKE.
        let g = grid_stages(&[], &[(0i64, 60_000)], 0, 90_000);
        assert_eq!(g, vec![0, 0, 4]);
    }

    #[test]
    fn demographics_matches_python() {
        let d = demographics(34.9, 1.75, 70.0, "female");
        assert_eq!(d[0], 34.0);
        assert_eq!(d[1], 70.0 / (1.75f64 * 1.75));
        assert_eq!(d[2], -1.0);
        assert_eq!(demographics(1.0, 1.0, 1.0, "")[2], 0.0);
    }
}
