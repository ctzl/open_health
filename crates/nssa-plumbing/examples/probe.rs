//! Debug probe: print my ACM decode next to the golden's, as exact f64 bits.
//!
//!   cargo run --release --offline -p nssa-plumbing --example probe -- 0 10

use nssa_plumbing::clock::Clock;
use nssa_plumbing::inputs::collect_nssa_inputs;
use nssa_plumbing::rows::EventRow;
use serde_json::Value;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let night: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(0);
    let from: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(0);
    let to: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(from + 6);

    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors");
    let text = std::fs::read_to_string(dir.join(format!("night_{night}.json"))).unwrap();
    let doc: Value = serde_json::from_str(&text).unwrap();
    let clock = Clock::from_epochs_json(&std::fs::read_to_string(dir.join("clock.json")).unwrap())
        .expect("clock");
    let rows: Vec<EventRow> = serde_json::from_value(doc["rows"].clone()).unwrap();
    let (s, e) = (
        doc["bedtime_ds"][0].as_i64().unwrap(),
        doc["bedtime_ds"][1].as_i64().unwrap(),
    );
    let cu = doc["captured_unix"].as_i64().unwrap();
    let out = collect_nssa_inputs(&rows, s, e, cu, &clock);

    println!("rows {} -> mine {} golden {}", rows.len(), out.acm.len(), doc["acm"]["v"].as_array().unwrap().len());
    for i in from..to {
        let mine_t = out.acm.get(i).map(|a| a.0);
        let gold_t = doc["acm"]["t"].get(i).and_then(|v| v.as_i64());
        let mine_bits: Vec<String> = out
            .acm
            .get(i)
            .map(|a| a.1.iter().map(|x| format!("{:08x}", x.to_bits())).collect())
            .unwrap_or_default();
        let gold_bits: Vec<String> = doc["acm"]["v"]
            .get(i)
            .map(|v| {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| {
                        let d = x.as_f64().unwrap();
                        format!("{:016x}|{:08x}", d.to_bits(), (d as f32).to_bits())
                    })
                    .collect()
            })
            .unwrap_or_default();
        println!("[{i}] t mine={mine_t:?} gold={gold_t:?}\n     mine={mine_bits:?}\n     gold={gold_bits:?}");
    }
}
