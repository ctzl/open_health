//! Bit-exact golden tests against the vectors written by
//! `tools/nssa_plumbing_golden.py` (which runs the real Python functions).
//!
//! Every float comparison here is on bits: f32 values compare `to_bits()` (that is
//! what numpy computes), f64 values compare `to_bits()` after the JSON round-trip
//! (Python's `repr` and serde's shortest formatter both round-trip exactly).
//! Nothing in this file uses a tolerance.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use nssa_plumbing::bedtimes::{bed_capture, ring_bedtimes};
use nssa_plumbing::breath::breath_series;
use nssa_plumbing::clock::Clock;
use nssa_plumbing::grid::{score_window, Staged, Stager};
use nssa_plumbing::inputs::{collect_nssa_inputs, NssaInputs};
use nssa_plumbing::rows::EventRow;
use serde_json::{Number, Value};

fn vectors() -> &'static PathBuf {
    static V: OnceLock<PathBuf> = OnceLock::new();
    V.get_or_init(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors"))
}

/// The golden vectors are captured from the real (personal) DB and are
/// local-only — they are gitignored. Without them these tests skip.
fn require_vectors() -> bool {
    if vectors().join("manifest.json").exists() {
        return true;
    }
    eprintln!(
        "SKIP: real-data golden vectors absent (local-only; regenerate with          tools/nssa_plumbing_golden.py)"
    );
    false
}

fn load(name: &str) -> Value {
    let path = vectors().join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// Rebuild a `[first, delta, delta, …]` series the oracle harness compacted.
fn rebuild(deltas: &Value) -> Vec<Value> {
    let mut cur = 0i64;
    deltas
        .as_array()
        .unwrap()
        .iter()
        .map(|d| {
            cur += d.as_i64().expect("delta");
            Value::from(cur)
        })
        .collect()
}

/// A JSON number that carries an f32 exactly (Python wrote these as doubles).
fn f32_json(x: f32) -> Value {
    Value::Number(Number::from_f64(x as f64).expect("finite"))
}

fn f64_json(x: f64) -> Value {
    Value::Number(Number::from_f64(x).expect("finite"))
}

/// Deep JSON comparison with exact float bits and useful failure text.
fn diff(a: &Value, b: &Value, path: &str, out: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().filter(|k| !y.contains_key(k.as_str())) {
                out.push(format!("{path}.{k}: absent in golden"));
            }
            for k in y.keys().filter(|k| !x.contains_key(k.as_str())) {
                out.push(format!("{path}.{k}: absent in rust"));
            }
            for k in x.keys().filter(|k| y.contains_key(k.as_str())) {
                diff(&x[k], &y[k], &format!("{path}.{k}"), out);
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                out.push(format!("{path}: length {} != golden {}", x.len(), y.len()));
                return;
            }
            for (i, (u, v)) in x.iter().zip(y.iter()).enumerate() {
                diff(u, v, &format!("{path}[{i}]"), out);
            }
        }
        (Value::Number(x), Value::Number(y)) => {
            // Exact bits; integers of this size are exact as f64 either way.
            let u = x.as_f64().unwrap_or(f64::NAN);
            let v = y.as_f64().unwrap_or(f64::NAN);
            if u.to_bits() != v.to_bits() {
                out.push(format!("{path}: {x} != golden {y}"));
            }
        }
        (x, y) => {
            if x != y {
                out.push(format!("{path}: {x} != golden {y}"));
            }
        }
    }
}

fn report(case: &str, mut diffs: Vec<String>) {
    diffs.sort();
    diffs.dedup();
    if diffs.is_empty() {
        println!("PASS {case}: bit-exact");
    } else {
        let shown = diffs
            .iter()
            .take(12)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ");
        panic!("FAIL {case}: {} differing values\n  {}", diffs.len(), shown);
    }
}

fn clock() -> Clock {
    Clock::from_epochs_json(&std::fs::read_to_string(vectors().join("clock.json")).unwrap())
        .expect("clock.json")
}

fn rows_of(doc: &Value) -> Vec<EventRow> {
    serde_json::from_value(doc["rows"].clone()).expect("event rows")
}

fn bedtime_rows() -> Vec<EventRow> {
    serde_json::from_value(load("bedtime_rows.json")).expect("bedtime rows")
}

fn manifest() -> Value {
    load("manifest.json")
}

fn night_count() -> usize {
    manifest()["nights"].as_array().unwrap().len()
}

/// night index by (bedtime start_ds, end_ds), from the manifest.
fn night_index() -> BTreeMap<(i64, i64), usize> {
    manifest()["nights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            (
                (
                    m["bedtime_ds"][0].as_i64().unwrap(),
                    m["bedtime_ds"][1].as_i64().unwrap(),
                ),
                m["i"].as_i64().unwrap() as usize,
            )
        })
        .collect()
}

// ---------------------------------------------------------------- clock ------

#[test]
fn clock_is_dated_matches_python() {
    if !require_vectors() {
        return;
    }
    let clock = clock();
    let windows = load("windows.json");
    let mut diffs = Vec::new();
    let mut checked = 0usize;
    for w in windows.as_array().unwrap() {
        for call in w["is_dated"].as_array().unwrap() {
            let (ds, cu) = (call[0].as_i64().unwrap(), call[1].as_i64().unwrap());
            let want = call[2].as_bool().unwrap();
            let got = clock.is_dated(ds, Some(cu));
            if got != want {
                diffs.push(format!("is_dated({ds}, {cu}): {got} != golden {want}"));
            }
            checked += 1;
        }
    }
    assert!(checked >= 200, "only {checked} is_dated calls recorded");
    report(&format!("is_dated ({checked} calls)"), diffs);
}

// ----------------------------------------------------------- ring_bedtimes ---

#[test]
fn ring_bedtimes_matches_python() {
    if !require_vectors() {
        return;
    }
    let clock = clock();
    let beds = bedtime_rows();
    let windows = load("windows.json");
    let mut diffs = Vec::new();
    let mut total = 0usize;
    let mut skipped = 0usize;
    for w in windows.as_array().unwrap() {
        // The harness only fills `bedtimes` when score_window actually reaches
        // ring_bedtimes. For a window it rejects first ("clock is undated or
        // ambiguous") the recorded [] is the per-window default, not an
        // observation, so there is nothing to compare against there.
        let observed = w["bedtimes_called"]
            .as_bool()
            .unwrap_or_else(|| w["error"].is_null());
        if !observed {
            skipped += 1;
            continue;
        }
        let wi = w["i"].as_i64().unwrap();
        let (s, e) = (
            w["window"][0].as_i64().unwrap(),
            w["window"][1].as_i64().unwrap(),
        );
        // score_window's own bed_cu rule, over the same 0x76 rows.
        let cu = bed_capture(&beds, s, w["captured_unix_arg"].as_i64());
        let got: Vec<Value> = ring_bedtimes(&clock, &beds, s, e, cu)
            .into_iter()
            .map(|(a, b, c)| Value::from(vec![a, b, c]))
            .collect();
        total += got.len();
        diff(
            &Value::Array(got),
            &w["bedtimes"],
            &format!("window {wi} bedtimes"),
            &mut diffs,
        );
    }
    let nwin = windows.as_array().unwrap().len();
    assert!(total > 0, "no bedtimes found at all");
    report(
        &format!(
            "ring_bedtimes ({total} bedtimes over {} windows, {skipped} windows \
             where score_window errored before calling it)",
            nwin - skipped
        ),
        diffs,
    );
}

// ------------------------------------------------ collect_nssa_inputs + breath

/// Run the ported input collection for one recorded night.
fn collect_for(doc: &Value, clock: &Clock) -> NssaInputs {
    let (s, e) = (
        doc["bedtime_ds"][0].as_i64().unwrap(),
        doc["bedtime_ds"][1].as_i64().unwrap(),
    );
    collect_nssa_inputs(
        &rows_of(doc),
        s,
        e,
        doc["captured_unix"].as_i64().unwrap(),
        clock,
    )
}

fn compare_night(
    i: usize,
    doc: &Value,
    clock: &Clock,
    diffs: &mut Vec<String>,
) -> [usize; 4] {
    let out = collect_for(doc, clock);
    let p = format!("night {i}");

    let bedtime = Value::Array(vec![Value::from(out.bedtime.0), Value::from(out.bedtime.1)]);
    diff(&bedtime, &doc["bedtime_ms"], &format!("{p} bedtime_ms"), diffs);

    let acm_t: Vec<Value> = out.acm.iter().map(|a| Value::from(a.0)).collect();
    diff(
        &Value::Array(acm_t),
        &Value::Array(rebuild(&doc["acm"]["t"])),
        &format!("{p} acm.t"),
        diffs,
    );
    let acm_v: Vec<Value> = out
        .acm
        .iter()
        .map(|a| Value::Array(a.1.iter().map(|x| f32_json(*x)).collect()))
        .collect();
    diff(&Value::Array(acm_v), &doc["acm"]["v"], &format!("{p} acm.v"), diffs);

    let beats: Vec<Value> = out
        .beats_pre
        .iter()
        .map(|b| Value::from(vec![b.0, b.1, b.2]))
        .collect();
    diff(&Value::Array(beats), &doc["beats_pre"], &format!("{p} beats_pre"), diffs);

    let ibi: Vec<Value> = out
        .ibi
        .iter()
        .map(|b| Value::from(vec![b.0, b.1, b.2, b.3 as i64]))
        .collect();
    diff(&Value::Array(ibi), &doc["ibi"], &format!("{p} ibi"), diffs);

    let temp_t: Vec<Value> = out.temp.iter().map(|a| Value::from(a.0)).collect();
    diff(
        &Value::Array(temp_t),
        &Value::Array(rebuild(&doc["temp"]["t"])),
        &format!("{p} temp.t"),
        diffs,
    );
    let temp_v: Vec<Value> = out.temp.iter().map(|a| f32_json(a.1)).collect();
    diff(&Value::Array(temp_v), &doc["temp"]["v"], &format!("{p} temp.v"), diffs);

    [out.acm.len(), out.beats_pre.len(), out.ibi.len(), out.temp.len()]
}

#[test]
fn collect_nssa_inputs_matches_python() {
    if !require_vectors() {
        return;
    }
    let clock = clock();
    let n = night_count();
    let mut diffs = Vec::new();
    let mut totals = [0usize; 4];
    for i in 0..n {
        let doc = load(&format!("night_{i}.json"));
        let got = compare_night(i, &doc, &clock, &mut diffs);
        for (t, g) in totals.iter_mut().zip(got) {
            *t += g;
        }
    }
    assert!(totals[0] > 1000 && totals[1] > 10000, "suspiciously little data: {totals:?}");
    report(
        &format!(
            "collect_nssa_inputs over {n} nights ({} acm, {} beats, {} corrected ibi, {} temp)",
            totals[0], totals[1], totals[2], totals[3]
        ),
        diffs,
    );
}

#[test]
fn breath_series_matches_python() {
    if !require_vectors() {
        return;
    }
    let clock = clock();
    let n = night_count();
    let mut diffs = Vec::new();
    let mut total = 0usize;
    let mut empty = 0usize;
    for i in 0..n {
        let doc = load(&format!("night_{i}.json"));
        let out = collect_for(&doc, &clock);
        let got: Vec<Value> = breath_series(&out.ibi)
            .into_iter()
            .map(|(t, v)| Value::Array(vec![Value::from(t), f64_json(v)]))
            .collect();
        if got.is_empty() {
            empty += 1;
        }
        total += got.len();
        diff(&Value::Array(got), &doc["breath"], &format!("night {i} breath"), &mut diffs);
    }
    report(
        &format!("breath_series over {n} nights ({total} windows, {empty} nights with none)"),
        diffs,
    );
}

// ------------------------------------------------------------- score_window --

struct Scripted {
    calls: Vec<(i64, i64, Option<Staged>)>,
    i: usize,
}

impl Stager for Scripted {
    fn stage(&mut self, b_start: i64, b_end: i64, _cu: i64, _demo: &[f64; 3]) -> Option<Staged> {
        let (s, e, staged) = &self.calls[self.i];
        assert_eq!((*s, *e), (b_start, b_end), "stage() called out of order");
        self.i += 1;
        staged.clone()
    }
}

/// The staged epochs / apnea / breath the model layer produced for this night.
fn staged_of(doc: &Value) -> Option<Staged> {
    let stage = doc["stage"].as_object()?;
    let t: Vec<i64> = stage["t"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let codes: Vec<i32> = stage["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let apnea = doc["apnea"].as_array().map(|a| {
        a.iter()
            .map(|e| (e[0].as_i64().unwrap(), e[1].as_f64().unwrap()))
            .collect::<Vec<_>>()
    });
    let breath = doc["breath"]
        .as_array()
        .map(|b| {
            b.iter()
                .map(|e| (e[0].as_i64().unwrap(), e[1].as_f64().unwrap()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Some(Staged { t, codes, apnea, breath })
}

#[test]
fn score_window_matches_python() {
    if !require_vectors() {
        return;
    }
    let clock = clock();
    let beds = bedtime_rows();
    let nights = night_index();
    let windows = load("windows.json");
    let mut docs: BTreeMap<usize, Value> = BTreeMap::new();
    let mut doc_for = |i: usize| -> Value {
        if let Some(d) = docs.get(&i) {
            return d.clone();
        }
        let d = load(&format!("night_{i}.json"));
        docs.insert(i, d.clone());
        d
    };

    let mut diffs = Vec::new();
    let mut ok = 0usize;
    let mut err = 0usize;
    for w in windows.as_array().unwrap() {
        let wi = w["i"].as_i64().unwrap();
        let (s, e) = (
            w["window"][0].as_i64().unwrap(),
            w["window"][1].as_i64().unwrap(),
        );
        let cu = w["captured_unix_arg"].as_i64();

        // The rows this window's plumbing sees: every 0x76 (ring_bedtimes and the
        // bed_cu scan look at the whole table) plus the decoded rows of its beds.
        let mut rows = beds.clone();
        let mut calls = Vec::new();
        for b in w["bedtimes"].as_array().unwrap() {
            let (bs, be) = (b[0].as_i64().unwrap(), b[1].as_i64().unwrap());
            if let Some(n) = nights.get(&(bs, be)) {
                let doc = doc_for(*n);
                rows.extend(rows_of(&doc));
                calls.push((bs, be, staged_of(&doc)));
            } else {
                calls.push((bs, be, None));
            }
        }
        if w["bedtimes"].as_array().unwrap().is_empty() {
            // no ring bedtime: score_window stages the summary window itself
            if let Some(n) = nights.get(&(s, e)) {
                let doc = doc_for(*n);
                rows.extend(rows_of(&doc));
                calls.push((s, e, staged_of(&doc)));
            }
        }

        let demo = w["result"]
            .get("demographics")
            .and_then(|d| d.as_array())
            .map(|d| d.iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![0.0, 0.0, 0.0]);
        let demo = [demo[0], demo[1], demo[2]];

        let mut stager = Scripted { calls, i: 0 };
        let got = score_window(&clock, &rows, s, e, cu, 1.0, &demo, &mut stager);
        match &w["result"] {
            Value::Object(_) => match &got {
                Ok(value) => {
                    ok += 1;
                    diff(value, &w["result"], &format!("window {wi}"), &mut diffs);
                }
                Err(message) => diffs.push(format!(
                    "window {wi}: error {message:?} where the golden has a result"
                )),
            },
            Value::Null => match &got {
                Err(message) => {
                    err += 1;
                    if Some(message.as_str()) != w["error"].as_str() {
                        diffs.push(format!(
                            "window {wi} error: {message:?} != golden {:?}",
                            w["error"]
                        ));
                    }
                }
                Ok(_) => {
                    diffs.push(format!("window {wi}: a result where the golden has an error"))
                }
            },
            other => diffs.push(format!("window {wi}: unexpected golden shape {other}")),
        }
    }
    assert!(ok > 100, "only {ok} windows compared");
    report(&format!("score_window ({ok} results, {err} error paths)"), diffs);
}
