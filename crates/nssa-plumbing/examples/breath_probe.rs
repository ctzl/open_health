//! Per-window spectral diagnostics, same shape as the Python probe script, so
//! the two can be diffed window by window.
//!
//!   cargo run --release --offline -p nssa-plumbing --example breath_probe -- 84

use nssa_plumbing::breath::{band_bins, spectrum_diag, FS, RESP_HI, RESP_LO, STEP_S, WIN_S};
use nssa_plumbing::inputs::CorrectedBeat;
use serde_json::Value;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let night: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(0);

    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors");
    let text = std::fs::read_to_string(dir.join(format!("night_{night}.json"))).unwrap();
    let doc: Value = serde_json::from_str(&text).unwrap();
    let beats: Vec<CorrectedBeat> = doc["ibi"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| {
            let b = b.as_array().unwrap();
            (
                b[0].as_i64().unwrap(),
                b[1].as_i64().unwrap(),
                b[2].as_i64().unwrap(),
                b[3].as_u64().unwrap() as u32,
            )
        })
        .collect();

    let ts: Vec<f64> = beats.iter().map(|b| b.0 as f64 / 1000.0).collect();
    let ms: Vec<f64> = beats.iter().map(|b| b.1 as f64).collect();
    let Some((grid, resampled)) = nssa_plumbing::breath::tachogram(&ts, &ms) else {
        println!("no tachogram");
        return;
    };
    println!(
        "night {night}: beats={} grid={} span={:.1}s",
        beats.len(),
        grid.len(),
        grid[grid.len() - 1] - grid[0]
    );
    let x = nssa_plumbing::breath::bandpass(&resampled);
    let win = (WIN_S as f64 * FS) as usize;
    let step = (STEP_S as f64 * FS) as usize;
    let bins = band_bins(win);
    println!(
        "win={win} step={step} windows={} len(x)={} band bins {}..{} ({})",
        (x.len() - win + step) / step,
        x.len(),
        bins.first().map(|b| b.k).unwrap_or(0),
        bins.last().map(|b| b.k).unwrap_or(0),
        bins.len()
    );

    let mut out = Vec::new();
    let mut i = 0usize;
    while i + win <= x.len() {
        let d = spectrum_diag(&x[i..i + win], &bins);
        out.push(serde_json::json!({
            "i": i,
            "k": bins[d.best].k,
            "gate": !(d.max < 4.0 * d.med),
            "ratio": d.ratio,
            "gap": d.gap,
            "br": bins[d.best].k as f64 / ((1.0 / FS) * win as f64) * 60.0,
        }));
        i += step;
    }
    if args.get(2).map(|s| s.as_str()) == Some("json") {
        println!("{}", serde_json::to_string(&out).unwrap());
        return;
    }
    if args.get(2).map(|s| s.as_str()) == Some("tapdiag") {
        // per-tap intermediates: lo, hi, hi-lo, hanning taper, pre-mean kernel
        let n = nssa_plumbing::breath::FIR_TAPS;
        let t: Vec<f64> = (0..n).map(|i| i as f64 - (n as f64 - 1.0) / 2.0).collect();
        let taper = nssa_plumbing::breath::hanning(n);
        let lo_f = (2.0 * nssa_plumbing::breath::RESP_LO) / nssa_plumbing::breath::FS;
        let hi_f = (2.0 * nssa_plumbing::breath::RESP_HI) / nssa_plumbing::breath::FS;
        let lo: Vec<f64> = t.iter().map(|x| lo_f * nssa_plumbing::breath::sinc(lo_f * x)).collect();
        let hi: Vec<f64> = t.iter().map(|x| hi_f * nssa_plumbing::breath::sinc(hi_f * x)).collect();
        let pre: Vec<f64> = (0..n).map(|i| (hi[i] - lo[i]) * taper[i]).collect();
        println!(
            "{}",
            serde_json::json!({"lo": lo, "hi": hi, "diff": (0..n).map(|i| hi[i] - lo[i]).collect::<Vec<_>>(), "taper": taper, "pre": pre})
        );
        return;
    }
    if args.get(2).map(|s| s.as_str()) == Some("kern") {
        let bits: Vec<String> = nssa_plumbing::breath::fir_kernel()
            .iter()
            .map(|v| format!("{:016x}", v.to_bits()))
            .collect();
        println!("{}", serde_json::to_string(&bits).unwrap());
        return;
    }
    if args.get(2).map(|s| s.as_str()) == Some("dc") {
        // after DC removal, before the FIR: isolates np.mean from np.convolve
        let m = nssa_plumbing::breath::dc_mean(&resampled);
        let bits: Vec<String> = resampled
            .iter()
            .map(|v| format!("{:016x}", (v - m).to_bits()))
            .collect();
        println!("{}", serde_json::to_string(&bits).unwrap());
        return;
    }
    if args.get(2).map(|s| s.as_str()) == Some("raw") {
        let bits: Vec<String> = resampled.iter().map(|v| format!("{:016x}", v.to_bits())).collect();
        println!("{}", serde_json::to_string(&bits).unwrap());
        return;
    }
    if args.get(2).map(|s| s.as_str()) == Some("signal") {
        // exact bit patterns of the band-passed signal, for the x-vs-x diff
        let bits: Vec<String> = x
            .iter()
            .map(|v| format!("{:016x}", v.to_bits()))
            .collect();
        println!("{}", serde_json::to_string(&bits).unwrap());
        return;
    }
    if args.get(2).map(|s| s.as_str()) == Some("bp") {
        // in-band powers for one window: same order as `bp = power[band]`
        let i: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(0);
        let bp = nssa_plumbing::breath::in_band_powers(&x[i..i + win], &bins);
        println!("{}", serde_json::to_string(&bp).unwrap());
        return;
    }
    if args.get(2).map(|s| s.as_str()) == Some("grid") {
        let bits: Vec<String> = grid.iter().map(|v| format!("{:016x}", v.to_bits())).collect();
        println!("{}", serde_json::to_string(&bits).unwrap());
        return;
    }
    println!("{}", serde_json::to_string(&out[..out.len().min(12)]).unwrap());
    let mut tight: Vec<&Value> = out.iter().filter(|o| o["gap"].is_number()).collect();
    tight.sort_by(|a, b| {
        a["gap"]
            .as_f64()
            .unwrap()
            .partial_cmp(&b["gap"].as_f64().unwrap())
            .unwrap()
    });
    println!(
        "tightest top-2 gaps: {}",
        serde_json::to_string(&tight[..tight.len().min(6)]).unwrap()
    );
    println!(
        "total windows: {} gated in: {}",
        out.len(),
        out.iter().filter(|o| o["gate"].as_bool().unwrap()).count()
    );
    let _ = (RESP_LO, RESP_HI);
}
