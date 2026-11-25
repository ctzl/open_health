//! Cold vs warm-incremental wall time for the activity path, in the shape the
//! dashboard calls it: default arguments (no `--full`), one serial.
//!
//! Run with the same environment the tests need (venv active, torch DLLs on
//! PATH):
//!
//! ```text
//! source .venv/Scripts/activate && export PATH="$VIRTUAL_ENV/Lib/site-packages/torch/lib:$PATH"
//! cargo run -p oura-models --features torch --example activity_incremental_bench
//! ```
//!
//! It deletes the Rust sidecars, runs the cold path (which writes them), then
//! runs the incremental path against them and prints both wall times plus the
//! model-call counts. The sidecars it leaves behind are the ones the dashboard
//! would use; delete `activity.rs.<serial>.*.json` next to the DB to start over.

use anyhow::Result;
use oura_models::{activity, incremental};
use std::path::PathBuf;

/// The bench measures the real ring, so it reads the same local-only serial as
/// the tests (tests/vectors/test-env.json, gitignored); OURA_SERIAL overrides.
fn bench_serial(repo: &std::path::Path) -> Result<String> {
    if let Ok(v) = std::env::var("OURA_SERIAL") {
        return Ok(v);
    }
    let p = repo.join("crates/oura-models/tests/vectors/test-env.json");
    let txt = std::fs::read_to_string(&p)
        .map_err(|_| anyhow::anyhow!("no serial: set OURA_SERIAL or capture {}", p.display()))?;
    let v: serde_json::Value = serde_json::from_str(&txt)?;
    Ok(v["serial"].as_str().unwrap_or_default().to_string())
}
const TZ: f64 = -5.0;

fn main() -> Result<()> {
    let repo = activity::repo_root();
    let serial = bench_serial(&repo)?;
    let serial = serial.as_str();
    // Same lookup as the tests: OURA_DB, then <repo>/oura.db, then the sibling
    // open_oura checkout this machine's data lives in.
    let db = match std::env::var("OURA_DB") {
        Ok(v) => PathBuf::from(v),
        Err(_) => {
            let alt = repo
                .parent()
                .map(|p| p.join("open_oura").join("oura.db"))
                .filter(|p| p.is_file());
            match alt {
                Some(p) => p,
                None => activity::resolve_db(None, &repo)?,
            }
        }
    };
    if !db.is_file() {
        anyhow::bail!("database not found: {} (set OURA_DB)", db.display());
    }
    let model = activity::find_model(&repo)?;
    if activity::activity_backend(&model).is_none() {
        anyhow::bail!(
            "no TorchScript backend linked ({}); run with --features torch",
            activity::tch_status()
        );
    }
    let step = model.parent().and_then(|d| activity::find_step_decoder(d));

    // Start from no Rust cache at all, so the first run is the cold path.
    let src = incremental::EventSource::new(&db, Some(serial), incremental::MODEL_KEY, incremental::STATE_VERSION)?;
    for path in src.sidecar_paths(activity::WHERE) {
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
    }
    drop(src);
    for entry in std::fs::read_dir(db.parent().unwrap())? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with(&format!("{}.{serial}.", incremental::MODEL_KEY))
            && name.ends_with(".skel.json")
        {
            std::fs::remove_file(entry.path())?;
        }
    }

    let argv = vec![
        db.to_string_lossy().into_owned(),
        "--tz".into(),
        TZ.to_string(),
        "--serial".into(),
        serial.into(),
    ];
    let args = activity::parse_args(&argv)?;

    let mut rows = Vec::new();
    for (label, _) in [("cold (no cache)", 0), ("warm incremental", 1)] {
        let backends = incremental::counted_backends(&model, step.as_deref())?;
        let report = incremental::run_model(&args, &db, &repo, &backends)?;
        println!(
            "{label:<18} {:>8.2} s  path={:<11} sessions={:<3} days total={:<3} recomputed={:<3} \
             replayed={:<3} aad_calls={:<3} gait_calls={:<3} new_rows={}/{}",
            report.seconds,
            report.path.name(),
            report.sessions.len(),
            report.days_total,
            report.days_recomputed,
            report.days_replayed,
            backends.counters.aad(),
            backends.counters.decoder(),
            report.new_signal_rows,
            report.new_step_rows,
        );
        rows.push((label, report));
    }
    let (cold, warm) = (&rows[0].1, &rows[1].1);
    println!(
        "\nsessions identical: {} ({} vs {})",
        cold.sessions.len() == warm.sessions.len()
            && activity::render_result(&cold.model_version, &cold.sessions)
                == activity::render_result(&warm.model_version, &warm.sessions),
        cold.sessions.len(),
        warm.sessions.len()
    );
    println!(
        "speedup: {:.1}x  ({:.2} s -> {:.2} s)",
        cold.seconds / warm.seconds.max(0.001),
        cold.seconds,
        warm.seconds
    );
    Ok(())
}
