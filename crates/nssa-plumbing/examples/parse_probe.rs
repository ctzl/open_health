//! Does serde_json parse these decimals to the same f64 as Rust's std parser?
use std::path::Path;

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors");
    let text = std::fs::read_to_string(dir.join("night_0.json")).unwrap();
    let needle = "0.09019608050584793";
    let at = text.find(needle).unwrap();
    let snippet = &text[at..at + needle.len() + 1]; // include the following delimiter
    println!("file bytes: {snippet:?}");

    for s in ["0.09019608050584793", "0.019607843831181526"] {
        let std: f64 = s.parse().unwrap();
        let via_value: serde_json::Value = serde_json::from_str(s).unwrap();
        let via_f64: f64 = serde_json::from_str(s).unwrap();
        println!(
            "{s}\n  std        {:016x}\n  Value      {:016x}\n  typed f64  {:016x}",
            std.to_bits(),
            via_value.as_f64().unwrap().to_bits(),
            via_f64.to_bits()
        );
    }
}
