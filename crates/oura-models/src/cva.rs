//! Port of `tools/run_cva_model.py` — cardiovascular age + pulse-wave velocity
//! from the ring's raw PPG (BLE tag 0x81). Golden vector:
//! `tests/vectors/cva_golden.json`; plan: docs/python-to-rust-plan.md (T1a).
//!
//! Pipeline, 1:1 with the Python oracle:
//!   events(tag=0x81) -> stateful delta-stream decode -> run split at >2 s gaps
//!   -> 1500-sample segments -> cva_1_*.pt TorchScript -> rounded JSON.
//!
//! Inference sits behind [`Inference`]: [`TorchModel`] (tch, `torch` feature)
//! or [`NoInference`] (stub), so the data pipeline stays testable without
//! libtorch.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use rusqlite::Connection;
use serde_json::{Map, Value};

/// Samples per segment (hard model constant).
pub const SEG_LEN: usize = 1500;
/// >2 s (ring deciseconds) splits two PPG measurements.
pub const GAP_DS: i64 = 20;

// ---------------------------------------------------------------------------
// CLI (argparse mirror: db positional, --sex/--age/--height/--weight/--ring,
// --since-cursor, --serial, --json)
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct Args {
    pub db: Option<String>,
    pub sex: Option<String>,
    pub age: Option<f64>,
    pub height: Option<f64>,
    /// Parsed for CLI parity; the cva v1 model does not consume weight.
    #[allow(dead_code)]
    pub weight: Option<f64>,
    pub ring: Option<f64>,
    pub since_cursor: i64,
    pub serial: Option<String>,
    pub json: bool,
}

fn flag_and_inline(arg: &str) -> (&str, Option<String>) {
    if !arg.starts_with('-') {
        return (arg, None);
    }
    match arg.split_once('=') {
        Some((f, v)) => (f, Some(v.to_string())),
        None => (arg, None),
    }
}

fn take_value(args: &[String], i: &mut usize, flag: &str, inline: Option<String>) -> Result<String> {
    if let Some(v) = inline {
        return Ok(v);
    }
    *i += 1;
    args.get(*i)
        .cloned()
        .with_context(|| format!("expected a value after {flag}"))
}

fn parse_num(s: &str, flag: &str) -> Result<f64> {
    s.parse::<f64>()
        .with_context(|| format!("expected a number after {flag}, got {s:?}"))
}

pub fn parse_args(args: &[String]) -> Result<Args> {
    let mut a = Args::default();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let (flag, inline) = flag_and_inline(arg);
        match flag {
            "--sex" => a.sex = Some(take_value(args, &mut i, flag, inline)?),
            "--age" => a.age = Some(parse_num(&take_value(args, &mut i, flag, inline)?, flag)?),
            "--height" => a.height = Some(parse_num(&take_value(args, &mut i, flag, inline)?, flag)?),
            "--weight" => a.weight = Some(parse_num(&take_value(args, &mut i, flag, inline)?, flag)?),
            "--ring" => a.ring = Some(parse_num(&take_value(args, &mut i, flag, inline)?, flag)?),
            "--since-cursor" => {
                a.since_cursor = take_value(args, &mut i, flag, inline)?
                    .parse::<i64>()
                    .with_context(|| format!("expected an integer after {flag}"))?;
            }
            "--serial" => a.serial = Some(take_value(args, &mut i, flag, inline)?),
            "--json" => a.json = true,
            "-h" | "--help" => {
                println!(
                    "usage: oura-models cva [DB] [--sex M|F|O] [--age Y] [--height M] \
                     [--weight KG] [--ring N] [--since-cursor DS] [--serial HEX] [--json]"
                );
                return Ok(a);
            }
            other => {
                if other.starts_with('-') || a.db.is_some() {
                    bail!("unrecognized argument: {other}");
                }
                a.db = Some(other.to_string());
            }
        }
        i += 1;
    }
    if let Some(sex) = &a.sex {
        if sex != "M" && sex != "F" && sex != "O" {
            bail!("invalid choice: {sex:?} (choose from 'M', 'F', 'O')");
        }
    }
    Ok(a)
}

// ---------------------------------------------------------------------------
// Path resolution (mirror of tools/_common.py resolve_db / resolve_models_dir
// and run_cva_model.find_model)
// ---------------------------------------------------------------------------

/// Repo roots that may own `notes/models`: the built binary's ancestors, the
/// source tree this crate was compiled from, and the cwd.
fn repo_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        for up in exe.ancestors().skip(1).take(6) {
            if up.join("tools").join("run_cva_model.py").is_file() {
                roots.push(up.to_path_buf());
                break;
            }
        }
    }
    // crates/oura-models -> crates -> repo
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    if let Some(repo) = manifest.parent().and_then(|p| p.parent()) {
        roots.push(repo.to_path_buf());
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd.clone());
        roots.push(cwd.join(".."));
    }
    roots
}

/// Model search dirs, same order as `_common.resolve_models_dir` /
/// `run_cva_model.find_model`: OURA_MODELS_DIR wins, else this checkout's
/// notes/models, then the sibling open_oura checkout.
fn model_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(d) = std::env::var("OURA_MODELS_DIR") {
        if !d.is_empty() {
            dirs.push(PathBuf::from(d));
        }
    }
    for repo in repo_roots() {
        dirs.push(repo.join("notes").join("models"));
        if let Some(parent) = repo.parent() {
            dirs.push(parent.join("open_oura").join("notes").join("models"));
        }
    }
    dirs
}

/// Newest decrypted raw-PPG cva model (cva_1_*, not the from-emb cva_2_*).
pub fn find_model() -> Result<PathBuf> {
    for dir in model_dirs() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        let mut cands: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("cva_1_") && n.ends_with(".pt"))
                    .unwrap_or(false)
            })
            .collect();
        cands.sort();
        if let Some(last) = cands.pop() {
            return Ok(last);
        }
    }
    bail!("no decrypted cva_1_*.pt in notes/models — decrypt with model_key.hex first")
}

/// Resolve the SQLite events DB (explicit arg wins, then the _common.py defaults).
pub fn resolve_db(arg: Option<&str>) -> Result<PathBuf> {
    let db = match arg {
        Some(a) => PathBuf::from(a),
        None => {
            let mut cands: Vec<PathBuf> = Vec::new();
            if let Ok(cwd) = std::env::current_dir() {
                cands.push(cwd.join("oura.db"));
            }
            for repo in repo_roots() {
                cands.push(repo.join("oura.db"));
                cands.push(repo.join("captures").join("ring5.db"));
            }
            let fallback = repo_roots()
                .first()
                .map(|r| r.join("oura.db"))
                .unwrap_or_else(|| PathBuf::from("oura.db"));
            cands.into_iter().find(|p| p.is_file()).unwrap_or(fallback)
        }
    };
    if !db.is_file() {
        bail!("error: database not found: {} (run `oura sync` first)", db.display());
    }
    Ok(db)
}

/// Demographics from `profile.json` next to the DB (mirror of
/// `_common.read_profile`): a per-serial map whose `""` key is the default, or a
/// legacy flat object meaning the only profile. Empty map when absent/bad.
pub fn read_profile(db: &Path, serial: Option<&str>) -> Map<String, Value> {
    let dir = db.parent().unwrap_or(Path::new("."));
    let Ok(text) = std::fs::read_to_string(dir.join("profile.json")) else {
        return Map::new();
    };
    let Ok(Value::Object(mut obj)) = serde_json::from_str::<Value>(&text) else {
        return Map::new();
    };
    if obj.contains_key("age") || obj.contains_key("sex") {
        return obj;
    }
    let key = serial.unwrap_or("");
    let hit = obj.remove(key).or_else(|| obj.remove("")).unwrap_or(Value::Null);
    match hit {
        Value::Object(o) => o,
        _ => Map::new(),
    }
}

fn prof_str(prof: &Map<String, Value>, key: &str) -> Option<String> {
    match prof.get(key)? {
        Value::String(s) => Some(s.clone()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn prof_num(prof: &Map<String, Value>, key: &str) -> Option<f64> {
    match prof.get(key)? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// PPG decode + segmentation (run_cva_model.decode_cva_ppg_bodies/build_segments)
// ---------------------------------------------------------------------------

/// Reconstruct one measurement's PPG samples from its 0x81 event bodies.
///
/// Stateful delta stream: 0x80 marks the next three bytes as a signed 24-bit
/// absolute ADC sample; every other byte is a signed int8 delta from the
/// previous sample. `acc` starts at 0 per run, exactly like the Python.
pub fn decode_cva_ppg_bodies(bodies: &[&[u8]]) -> Vec<f32> {
    let mut samples: Vec<f32> = Vec::new();
    let mut acc: i64 = 0;
    for data in bodies {
        let mut i = 0;
        while i < data.len() {
            let b = data[i];
            if b == 0x80 && i + 3 < data.len() {
                let mut raw =
                    (data[i + 1] as i64) | ((data[i + 2] as i64) << 8) | ((data[i + 3] as i64) << 16);
                if raw & 0x80_0000 != 0 {
                    raw -= 0x100_0000;
                }
                acc = raw;
                samples.push(acc as f32);
                i += 4;
            } else {
                acc += if b > 127 { b as i64 - 256 } else { b as i64 };
                samples.push(acc as f32);
                i += 1;
            }
        }
    }
    samples
}

#[derive(Debug, Clone)]
pub struct Segments {
    /// 1500-sample f32 segments, in model input order.
    pub waves: Vec<Vec<f32>>,
    /// Segment start, ring deciseconds (a shape-only input to the model).
    pub ts: Vec<i64>,
    /// Number of measurement runs (>2 s gap separated) — the "measurements" key.
    pub runs: usize,
}

/// Read tag-0x81 events, split runs at >2 s gaps, decode, chunk into SEG_LEN.
pub fn build_segments(db: &Path, since_ds: i64, serial: Option<&str>) -> Result<Segments> {
    let con = Connection::open(db).with_context(|| format!("open {}", db.display()))?;
    let sql = if serial.is_some() {
        "SELECT ring_timestamp, body FROM events WHERE tag=129 AND ring_timestamp>? AND serial = ? ORDER BY ring_timestamp"
    } else {
        "SELECT ring_timestamp, body FROM events WHERE tag=129 AND ring_timestamp>? ORDER BY ring_timestamp"
    };
    let mut stmt = con
        .prepare_cached(sql)
        .context("prepare cva event query (tag 0x81)")?;
    let map = |r: &rusqlite::Row| -> rusqlite::Result<(i64, Vec<u8>)> { Ok((r.get(0)?, r.get(1)?)) };
    let rows: Vec<(i64, Vec<u8>)> = match serial {
        Some(s) => stmt
            .query_map(rusqlite::params![since_ds, s], map)?
            .collect::<rusqlite::Result<Vec<_>>>()?,
        None => stmt
            .query_map(rusqlite::params![since_ds], map)?
            .collect::<rusqlite::Result<Vec<_>>>()?,
    };
    if rows.is_empty() {
        bail!("no cva_raw_ppg_data (tag 0x81) events — is cva_ppg enabled? (oura feature-status)");
    }

    // Contiguous measurement runs: a >GAP_DS gap between event timestamps ends one.
    let mut runs: Vec<Vec<usize>> = vec![vec![0]];
    for i in 1..rows.len() {
        if rows[i].0 - rows[i - 1].0 > GAP_DS {
            runs.push(Vec::new());
        }
        runs.last_mut().unwrap().push(i);
    }

    let mut segs = Segments { waves: Vec::new(), ts: Vec::new(), runs: runs.len() };
    for run in &runs {
        let bodies: Vec<&[u8]> = run.iter().map(|k| rows[*k].1.as_slice()).collect();
        let wave = decode_cva_ppg_bodies(&bodies);
        let start_ts = rows[run[0]].0;
        let mut s = 0;
        while s + SEG_LEN <= wave.len() {
            segs.waves.push(wave[s..s + SEG_LEN].to_vec());
            segs.ts.push(start_ts + s as i64);
            s += SEG_LEN;
        }
    }
    Ok(segs)
}

// ---------------------------------------------------------------------------
// Inference behind a trait
// ---------------------------------------------------------------------------

/// Model demographics row: [sex(-1 F / +1 M / 0 other), height_m, age_years, ring_size].
#[derive(Debug, Clone, Copy)]
pub struct Demographics {
    pub sex: f32,
    pub height_m: f32,
    pub age_years: f32,
    pub ring_size: f32,
}

/// Raw (unrounded) model outputs for the whole batch.
#[derive(Debug, Clone, Copy)]
pub struct ModelOutput {
    pub daily_cva: f64,
    pub quality: f64,
    pub variance: f64,
    pub daily_pwv: f64,
}

pub trait Inference {
    fn run(&self, segs: &Segments, demo: &Demographics) -> Result<ModelOutput>;
}

/// Used when the crate is built without the `torch` feature: the pipeline runs,
/// inference is refused loudly rather than guessed at.
pub struct NoInference;

impl Inference for NoInference {
    fn run(&self, _segs: &Segments, _demo: &Demographics) -> Result<ModelOutput> {
        bail!(
            "cva inference not available: this build has no torch backend \
             (rebuild with --features torch)"
        )
    }
}

/// cva_1_*.pt TorchScript through tch/libtorch (`torch` feature).
#[cfg(feature = "torch")]
pub struct TorchModel {
    pub path: PathBuf,
}

#[cfg(feature = "torch")]
impl TorchModel {
    pub fn new(path: impl AsRef<Path>) -> Self {
        TorchModel { path: path.as_ref().to_path_buf() }
    }
}

#[cfg(feature = "torch")]
impl Inference for TorchModel {
    fn run(&self, segs: &Segments, demo: &Demographics) -> Result<ModelOutput> {
        use tch::{no_grad_guard, IValue, CModule, Tensor};

        let n = segs.waves.len() as i64;
        let mut flat: Vec<f32> = Vec::with_capacity(segs.waves.len() * SEG_LEN);
        for w in &segs.waves {
            flat.extend_from_slice(w);
        }
        let rates: Vec<f32> = vec![50.0; segs.waves.len()]; // the ring streams at 50 Hz
        let tst: Vec<f32> = segs.ts.iter().map(|t| *t as f32).collect();

        // Bound to a name, not `_`: an unnamed guard drops immediately.
        let _nograd = no_grad_guard();
        let mut m = CModule::load(&self.path)
            .with_context(|| format!("torch.jit.load({})", self.path.display()))?;
        m.set_eval(); // torch.jit.load(...).eval()

        let ppg = Tensor::from_slice(&flat).view([n, SEG_LEN as i64]);
        let rates = Tensor::from_slice(&rates).view([n, 1]);
        let tst = Tensor::from_slice(&tst).view([n, 1]);
        let demo_t =
            Tensor::from_slice(&[demo.sex, demo.height_m, demo.age_years, demo.ring_size]).view([1, 4]);
        let inputs = [ppg, rates, tst, demo_t]
            .iter()
            .map(|t| IValue::Tensor(t.shallow_clone()))
            .collect::<Vec<_>>();
        let out = m
            .forward_is(&inputs)
            .context("cva model forward(ppg, rates, timestamps, demographics)")?;

        // (daily_cva, quality, variance, daily_pwv, ppg_segment_metrics[n,11])
        let outs: Vec<tch::Tensor> = match out {
            IValue::Tuple(v) => v.into_iter().map(tensor_of).collect::<Result<_>>()?,
            IValue::TensorList(v) => v,
            IValue::Tensor(t) => vec![t],
            other => bail!("unexpected cva model return type: {other:?}"),
        };
        if outs.len() < 4 {
            bail!("cva model returned {} outputs, expected 5", outs.len());
        }
        Ok(ModelOutput {
            daily_cva: scalar(&outs[0])?,
            quality: scalar(&outs[1])?,
            variance: scalar(&outs[2])?,
            daily_pwv: scalar(&outs[3])?,
        })
    }
}

#[cfg(feature = "torch")]
fn tensor_of(iv: tch::IValue) -> Result<tch::Tensor> {
    match iv {
        tch::IValue::Tensor(t) => Ok(t),
        _ => bail!("cva model returned a non-tensor element"),
    }
}

#[cfg(feature = "torch")]
fn scalar(t: &tch::Tensor) -> Result<f64> {
    Ok(t.double_value(&[]))
}

// ---------------------------------------------------------------------------
// Output (json.dumps key order + CPython rounding/formatting)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CvaJson {
    pub segments: usize,
    pub measurements: usize,
    pub vascular_age: f64,
    pub chronological_age: f64,
    pub pwv_ms: f64,
    pub quality: i64,
    pub variance: f64,
}

impl CvaJson {
    /// Byte-for-byte the object `json.dumps({...})` prints in run_cva_model.py
    /// (same key order, same float repr).
    pub fn to_json_line(&self) -> String {
        format!(
            "{{\"segments\": {}, \"measurements\": {}, \"vascular_age\": {}, \"chronological_age\": {}, \"pwv_ms\": {}, \"quality\": {}, \"variance\": {}}}",
            self.segments,
            self.measurements,
            py_repr(self.vascular_age),
            py_repr(self.chronological_age),
            py_repr(self.pwv_ms),
            self.quality,
            py_repr(self.variance),
        )
    }
}

/// CPython's `repr(float)`: shortest round-trip digits, always with a decimal
/// point (json.dumps uses this).
fn py_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let mut s = format!("{x}");
    if !s.contains(['.', 'e', 'E']) {
        s.push_str(".0");
    }
    s
}

/// CPython's `round(x, digits)`: round-half-even on the *exact* binary value.
/// `{:.60}` is that exact expansion for |x| < 1e20 (a double needs ~55
/// fractional digits there), so the tail decides ties the way CPython does —
/// rounding the shortest repr instead gets round(2.675, 2) wrong (2.68, where
/// CPython says 2.67 because that double is really 2.67499999999999982).
fn py_round(x: f64, digits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    round_decimal(&format!("{x:.60}"), digits)
        .parse()
        .unwrap_or(x)
}

/// CPython's `round(x)` -> int (half-even).
fn py_round_int(x: f64) -> i64 {
    round_decimal(&format!("{x:.60}"), 0).parse().unwrap_or(0)
}

/// Round an exact decimal string to `digits` fractional digits, half-to-even.
fn round_decimal(s: &str, digits: usize) -> String {
    let neg = s.starts_with('-');
    let body = s.trim_start_matches('-');
    let (ip, fp) = body.split_once('.').unwrap_or((body, ""));
    let mut ds: Vec<u8> = ip.bytes().chain(fp.bytes()).map(|b| b - b'0').collect();
    let mut ip_len = ip.len();
    let cut = ip_len + digits;
    if cut >= ds.len() {
        return s.to_string(); // already exact at this precision
    }
    let first_drop = ds[cut];
    let tail_nonzero = ds[cut + 1..].iter().any(|&d| d != 0);
    let kept_last = if cut == 0 { 0 } else { ds[cut - 1] };
    let round_up = first_drop > 5 || (first_drop == 5 && (tail_nonzero || kept_last % 2 == 1));
    ds.truncate(cut);
    if round_up {
        let mut i = ds.len();
        loop {
            if i == 0 {
                ds.insert(0, 1);
                ip_len += 1;
                break;
            }
            i -= 1;
            if ds[i] == 9 {
                ds[i] = 0;
            } else {
                ds[i] += 1;
                break;
            }
        }
    }
    let digits_str: String = ds.iter().map(|d| (b'0' + d) as char).collect();
    let (head_all, tail) = digits_str.split_at(ip_len.max(1));
    let head = head_all.trim_start_matches('0');
    let head = if head.is_empty() { "0" } else { head };
    let out = if digits == 0 {
        head.to_string()
    } else {
        format!("{head}.{tail}")
    };
    if neg && out.bytes().any(|b| b != b'0' && b != b'.') {
        format!("-{out}")
    } else {
        out
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Estimate {
    pub json: CvaJson,
    /// Unrounded model outputs, for diagnostics/tests.
    pub raw: ModelOutput,
    pub segments: usize,
    pub measurements: usize,
}

/// Full pipeline with the inference backend this build was compiled with.
pub fn estimate(args: &[String]) -> Result<Estimate> {
    #[cfg(feature = "torch")]
    let backend: Box<dyn Inference> = Box::new(TorchModel::new(find_model()?));
    #[cfg(not(feature = "torch"))]
    let backend: Box<dyn Inference> = Box::new(NoInference);
    estimate_with(args, backend.as_ref())
}

/// Full pipeline with an injected backend (tests use this to swap inference).
pub fn estimate_with(args: &[String], backend: &dyn Inference) -> Result<Estimate> {
    let a = parse_args(args)?;
    let db = resolve_db(a.db.as_deref())?;
    let prof = read_profile(&db, a.serial.as_deref());
    // CLI wins, then profile.json, then the built-in defaults (Python's `pick`).
    let sex_s = a
        .sex
        .clone()
        .or_else(|| prof_str(&prof, "sex"))
        .unwrap_or_else(|| "M".to_string());
    let sex = match sex_s.as_str() {
        "F" => -1.0f32,
        "M" => 1.0,
        "O" => 0.0,
        other => bail!("invalid sex value: {other:?} (expected 'M', 'F' or 'O')"),
    };
    let age = a.age.or(prof_num(&prof, "age")).unwrap_or(30.0);
    let height = a.height.or(prof_num(&prof, "height_m")).unwrap_or(1.78);
    let ring = a.ring.or(prof_num(&prof, "ring_size")).unwrap_or(10.0);

    let segs = build_segments(&db, a.since_cursor, a.serial.as_deref())?;
    if segs.waves.is_empty() {
        bail!(
            "no full {SEG_LEN}-sample PPG segment available ({} measurement runs, all too short)",
            segs.runs
        );
    }
    let demo = Demographics {
        sex,
        height_m: height as f32,
        age_years: age as f32,
        ring_size: ring as f32,
    };
    let out = backend.run(&segs, &demo)?;

    let json = CvaJson {
        segments: segs.waves.len(),
        measurements: segs.runs,
        vascular_age: py_round(out.daily_cva, 1),
        chronological_age: age,
        pwv_ms: py_round(out.daily_pwv, 2),
        quality: py_round_int(out.quality),
        variance: py_round(out.variance, 2),
    };
    Ok(Estimate { json, raw: out, segments: segs.waves.len(), measurements: segs.runs })
}

pub fn run(args: &[String]) -> Result<()> {
    let a = parse_args(args)?;
    if a.json {
        println!("{}", json_for_args(args)?);
        return Ok(());
    }
    let est = estimate(args)?;
    let age = est.json.chronological_age;
    println!(
        "CVA (cardiovascular age) — {} PPG segments from {} measurements",
        est.segments, est.measurements
    );
    println!(
        "  demographics: sex={} age={:.0} height={} ring={}",
        a.sex.clone().unwrap_or_else(|| "M".into()),
        age,
        py_repr(a.height.unwrap_or(1.78)),
        py_repr(a.ring.unwrap_or(10.0)),
    );
    println!(
        "  vascular age : {:.1} years   (anchored to input age; here {:+.1}y vs chronological)",
        est.raw.daily_cva,
        est.raw.daily_cva - age
    );
    println!(
        "  pulse-wave velocity: {:.2} m/s   (lower = less arterial stiffness)",
        est.raw.daily_pwv
    );
    println!(
        "  quality: {:.0}  variance: {:.2}",
        est.raw.quality, est.raw.variance
    );
    Ok(())
}

/// The JSON `run` would print, as a string — the in-process entry point for the
/// dashboard's RustRunner (same argv contract as the CLI, golden-tested).
pub fn json_for_args(args: &[String]) -> Result<String> {
    Ok(estimate(args)?.json.to_json_line())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_absolute_and_delta() {
        // 0x80 + 3-byte little-endian signed24 absolute, then int8 deltas,
        // continuing across bodies (acc carries over within a run).
        let b1: &[u8] = &[0x80, 0x00, 0x01, 0x00, 0x05, 0xFF];
        let b2: &[u8] = &[0x7F];
        assert_eq!(decode_cva_ppg_bodies(&[b1, b2]), vec![256.0, 261.0, 260.0, 387.0]);
    }

    #[test]
    fn decode_trailing_marker_falls_back_to_delta() {
        // Python's `i + 3 < len(data)` guard: a 0x80 without three following
        // bytes decodes as a -128 delta instead.
        let b: &[u8] = &[0x80, 0x01, 0x02];
        assert_eq!(decode_cva_ppg_bodies(&[b]), vec![-128.0, -127.0, -125.0]);
    }

    #[test]
    fn decode_negative_signed24() {
        let b: &[u8] = &[0x80, 0xFF, 0xFF, 0xFF];
        assert_eq!(decode_cva_ppg_bodies(&[b]), vec![-1.0]);
    }

    #[test]
    fn rounding_matches_cpython() {
        // Half-even on the exact binary value, not on the shortest repr.
        // (35.65 is really 35.6499999999999985, 2.675 really 2.6749999999999998.)
        assert_eq!(py_round(35.65, 1), 35.6);
        assert_eq!(py_round(2.675, 2), 2.67);
        assert_eq!(py_round(0.125, 2), 0.12);
        assert_eq!(py_round(164.175, 2), 164.18);
        assert_eq!(py_round(6.75, 2), 6.75);
        assert_eq!(py_round_int(2.5), 2);
        assert_eq!(py_round_int(3.5), 4);
        assert_eq!(py_round_int(2.4999), 2);
        assert_eq!(py_round(-0.04, 1), -0.0);
    }

    #[test]
    fn float_repr_matches_python() {
        assert_eq!(py_repr(37.0), "37.0");
        assert_eq!(py_repr(35.6), "35.6");
        assert_eq!(py_repr(6.8), "6.8");
        assert_eq!(py_repr(164.18), "164.18");
    }

    #[test]
    fn json_line_shape() {
        let j = CvaJson {
            segments: 461,
            measurements: 637,
            vascular_age: 35.6,
            chronological_age: 37.0,
            pwv_ms: 6.8,
            quality: 3,
            variance: 164.18,
        };
        let line = j.to_json_line();
        let v: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["segments"], 461);
        assert_eq!(v["quality"], 3);
        assert_eq!(v["chronological_age"], 37.0);
        assert!(line.starts_with("{\"segments\": 461, \"measurements\": 637"));
    }

    /// The rounding + repr layer against the raw model outputs the Python oracle
    /// actually produced (torch 2.13.0+cpu, serial <serial>, 463
    /// segments / 640 runs). The expected string is that oracle's own stdout.
    #[test]
    fn json_matches_python_on_real_model_values() {
        let raw = ModelOutput {
            daily_cva: 35.63494873046875,
            quality: 3.0,
            variance: 164.3849639892578,
            daily_pwv: 6.801610469818115,
        };
        let j = CvaJson {
            segments: 463,
            measurements: 640,
            vascular_age: py_round(raw.daily_cva, 1),
            chronological_age: 37.0,
            pwv_ms: py_round(raw.daily_pwv, 2),
            quality: py_round_int(raw.quality),
            variance: py_round(raw.variance, 2),
        };
        assert_eq!(
            j.to_json_line(),
            "{\"segments\": 463, \"measurements\": 640, \"vascular_age\": 35.6, \"chronological_age\": 37.0, \"pwv_ms\": 6.8, \"quality\": 3, \"variance\": 164.38}"
        );
    }

    #[test]
    fn arg_parsing() {
        let a = parse_args(&[
            "db.sqlite".into(),
            "--sex".into(),
            "F".into(),
            "--age=41".into(),
            "--ring".into(),
            "8".into(),
            "--since-cursor".into(),
            "10".into(),
            "--json".into(),
        ])
        .unwrap();
        assert_eq!(a.db.as_deref(), Some("db.sqlite"));
        assert_eq!(a.sex.as_deref(), Some("F"));
        assert_eq!(a.age, Some(41.0));
        assert_eq!(a.ring, Some(8.0));
        assert_eq!(a.since_cursor, 10);
        assert!(a.json);
        assert!(parse_args(&["--sex".into(), "X".into()]).is_err());
        assert!(parse_args(&["--nope".into()]).is_err());
    }
}
