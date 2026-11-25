//! Timing harness for the ported plumbing: the same 121 nights the golden suite
//! reads, but measured instead of compared.
//!
//! ```text
//! cargo run --release -p nssa-plumbing --example plumbing_bench
//! ```
//!
//! Prints the Rust counterpart of `tools/nssa_plumbing_golden.py`'s
//! `manifest.json -> timings_s`: `collect_nssa_inputs` and `breath_series` over
//! every recorded night, plus `ring_bedtimes` over every recorded summary window.
//! Vector loading and JSON parsing stay outside the timed region on purpose —
//! the Python harness pays for its DB read in a separate phase
//! (`import_read_rows_and_clock`) and excludes it from `plumbing_total`.
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use nssa_plumbing::bedtimes::{bed_capture, ring_bedtimes};
use nssa_plumbing::breath::breath_series;
use nssa_plumbing::clock::Clock;
use nssa_plumbing::inputs::collect_nssa_inputs;
use nssa_plumbing::rows::EventRow;
use serde_json::Value;

fn vectors() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors")
}

fn load(name: &str) -> Value {
    let path = vectors().join(name);
    serde_json::from_str(&std::fs::read_to_string(&path).expect(name)).expect(name)
}

/// Rows of one night, decoded once and reused by both phases.
struct Night {
    rows: Vec<EventRow>,
    bedtime: (i64, i64),
    captured: i64,
}

fn main() {
    let t_load = Instant::now();
    let clock = Clock::from_epochs_json(
        &std::fs::read_to_string(vectors().join("clock.json")).expect("clock.json"),
    )
    .expect("epochs");
    let n = load("manifest.json")["nights"].as_array().unwrap().len();
    let nights: Vec<Night> = (0..n)
        .map(|i| {
            let doc = load(&format!("night_{i}.json"));
            Night {
                rows: serde_json::from_value(doc["rows"].clone()).expect("rows"),
                bedtime: (
                    doc["bedtime_ds"][0].as_i64().unwrap(),
                    doc["bedtime_ds"][1].as_i64().unwrap(),
                ),
                captured: doc["captured_unix"].as_i64().unwrap(),
            }
        })
        .collect();
    let beds: Vec<EventRow> = serde_json::from_value(load("bedtime_rows.json")).expect("bed rows");
    let windows = load("windows.json");
    let load_s = t_load.elapsed().as_secs_f64();

    // ---- collect_nssa_inputs --------------------------------------------------
    let t = Instant::now();
    let inputs: Vec<_> = nights
        .iter()
        .map(|nt| {
            black_box(collect_nssa_inputs(
                black_box(&nt.rows),
                nt.bedtime.0,
                nt.bedtime.1,
                nt.captured,
                &clock,
            ))
        })
        .collect();
    let collect_s = t.elapsed().as_secs_f64();

    // ---- breath_series --------------------------------------------------------
    let t = Instant::now();
    let mut breath_points = 0usize;
    for out in &inputs {
        breath_points += breath_series(black_box(&out.ibi)).len();
    }
    let breath_s = t.elapsed().as_secs_f64();

    // ---- ring_bedtimes over every summary window ------------------------------
    let t = Instant::now();
    let mut bedtimes = 0usize;
    for w in windows.as_array().unwrap() {
        let (s, e) = (
            w["window"][0].as_i64().unwrap(),
            w["window"][1].as_i64().unwrap(),
        );
        // score_window's own bed_cu rule, exactly as tests/golden.rs drives it.
        let cu = bed_capture(&beds, s, w["captured_unix_arg"].as_i64());
        bedtimes += ring_bedtimes(&clock, &beds, s, e, cu).len();
    }
    let ring_s = t.elapsed().as_secs_f64();

    println!("nights              {n}");
    println!("windows             {}", windows.as_array().unwrap().len());
    println!(
        "acm rows            {}",
        inputs.iter().map(|o| o.acm.len()).sum::<usize>()
    );
    println!(
        "ibi rows            {}",
        inputs.iter().map(|o| o.ibi.len()).sum::<usize>()
    );
    println!("breath points     {breath_points}");
    println!("bedtimes          {bedtimes}");
    println!("timings_s vectors_load (excluded from the totals below) {load_s:.4}");
    println!("timings_s collect_nssa_inputs {collect_s:.4}");
    println!("timings_s breath_series {breath_s:.4}");
    println!("timings_s ring_bedtimes {ring_s:.4}");
    println!(
        "timings_s plumbing_total (collect + breath) {:.4}",
        collect_s + breath_s
    );
}
