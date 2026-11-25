//! T2c: answer staged sleep nights from the shared model cache, in-process.
//!
//! `tools/run_sleep_model.py` keys its night cache by
//! `sha256(json.dumps([CACHE_VERSION, serial, demo, inputs], sort_keys=True))`.
//! The inputs this crate collects are bit-identical to Python's (golden-tested),
//! so Rust can compute the same key and serve a night straight from
//! `nssa_cache.json` — no torch process at all. A window whose cache cannot
//! answer every bedtime is reported as a miss so the caller delegates only that
//! window to Python, which fills the cache for the next run.
//!
//! The fingerprint text must match CPython's `json.dumps` byte for byte:
//! default separators (`", "`, `": "`), `sort_keys=True` (the inputs dict keys
//! sort as acm < bedtime < ibi < temp), and `float.__repr__` (shortest
//! round-trip, always with a `.0`, exponent form below 1e-4 / at or above 1e16).
//! Rust's `{}` on f64 is the same shortest round-trip algorithm, so only the
//! float *syntax* needs adapting.

use crate::breath::breath_series;
use crate::clock::Clock;
use crate::grid::{self, Staged, Stager};
use crate::inputs::{self, NssaInputs};
use crate::rows::EventRow;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// `run_sleep_model.CACHE_VERSION`.
pub const CACHE_VERSION: &str = "nssa-2.6.0-r2";

/// `run_sleep_model._cache_path`: one shared file next to the DB.
pub fn cache_path(db: &Path) -> PathBuf {
    db.parent()
        .unwrap_or(Path::new("."))
        .join("nssa_cache.json")
}

// ── the fingerprint ───────────────────────────────────────────────────────────

/// CPython `repr(float)` as `json.dumps` writes it: the shortest decimal that
/// round-trips, ties broken half-to-even against the value's EXACT decimal
/// expansion (David Gay's dtoa mode 0). Rust's `{}` shares the shortest
/// property but breaks exact ties half-up, so the digits are re-rounded here
/// from the exact expansion (`{:.1074}` is exact for every f64).
pub fn py_float(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let neg = v.is_sign_negative();
    // Exact decimal expansion of |v| as "int.frac" (finite for f64).
    let exact = format!("{:.1074}", v.abs());
    let (int_part, frac_part) = exact.split_once('.').unwrap();
    // All digits with the decimal-point position, then leading zeros dropped.
    let all = format!("{int_part}{frac_part}");
    let lead = all.find(|c: char| c != '0').unwrap_or(all.len() - 1);
    let digits = &all[lead..];
    let decpt = int_part.len() as i64 - lead as i64;

    // Shortest significant-digit count that round-trips (<= 17 for f64).
    let mut out = String::new();
    for n in 1..=17usize {
        let (mut d, mut dp) = (digits[..n.min(digits.len())].to_string(), decpt);
        if n < digits.len() {
            // Round half-to-even at position n against the exact tail.
            let kept_last = d.as_bytes()[d.len() - 1] - b'0';
            let next = digits.as_bytes()[n] - b'0';
            let tail_nonzero = digits.as_bytes()[n + 1..].iter().any(|&c| c != b'0');
            let up = next > 5 || (next == 5 && (tail_nonzero || kept_last % 2 == 1));
            if up {
                let (nd, shift) = increment(&d);
                d = nd;
                dp = decpt + shift;
            }
        }
        let s = render(&d, dp);
        if s.parse::<f64>().is_ok_and(|p| p.to_bits() == v.abs().to_bits()) {
            out = s;
            break;
        }
    }
    if out.is_empty() {
        out = render(digits, decpt); // unreachable for finite f64
    }
    if neg {
        out.insert(0, '-');
    }
    out
}

/// Decimal string +1 with carry (may grow the digit count; dp tracks it).
fn increment(d: &str) -> (String, i64) {
    let mut b: Vec<u8> = d.bytes().collect();
    let mut i = b.len();
    while i > 0 {
        if b[i - 1] == b'9' {
            b[i - 1] = b'0';
            i -= 1;
        } else {
            b[i - 1] += 1;
            return (String::from_utf8(b).unwrap(), 0);
        }
    }
    b.insert(0, b'1'); // 999 -> 1000: one more digit, decimal point shifts
    (String::from_utf8(b).unwrap(), 1)
}

/// Positional or scientific form per CPython: scientific iff decpt < -4+1 or
/// decpt > 16 (i.e. |v| < 1e-4 or >= 1e16), exponent signed with >=2 digits.
fn render(d: &str, decpt: i64) -> String {
    let d = d.trim_end_matches('0');
    let d = if d.is_empty() { "0" } else { d };
    if decpt < -3 || decpt > 16 {
        let mant = if d.len() == 1 {
            d.to_string()
        } else {
            format!("{}.{}", &d[..1], &d[1..])
        };
        let e = decpt - 1;
        format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    } else if decpt <= 0 {
        format!("0.{}{d}", "0".repeat((-decpt) as usize))
    } else if decpt as usize >= d.len() {
        format!("{}{}.0", d, "0".repeat(decpt as usize - d.len()))
    } else {
        format!("{}.{}", &d[..decpt as usize], &d[decpt as usize..])
    }
}

/// `json.dumps` string literal (ensure_ascii; callers pass ASCII only).
fn json_str(s: &str) -> String {
    debug_assert!(s.is_ascii(), "fingerprint strings are ASCII");
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `json.dumps([CACHE_VERSION, SERIAL, demo, inputs], sort_keys=True)` — the
/// exact bytes Python hashes. `beats_pre` is Rust-only and not part of the
/// Python inputs dict, so it is not serialized.
pub fn fingerprint_text(serial: Option<&str>, demo: &[f64; 3], inputs: &NssaInputs) -> String {
    let mut s = String::new();
    s.push('[');
    s.push_str(&json_str(CACHE_VERSION));
    s.push_str(", ");
    match serial {
        Some(v) => s.push_str(&json_str(v)),
        None => s.push_str("null"),
    }
    s.push_str(", [");
    for (i, v) in demo.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&py_float(*v));
    }
    s.push_str("], {\"acm\": [");
    for (i, (ts, stats)) in inputs.acm.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push('[');
        s.push_str(&ts.to_string());
        s.push_str(", [");
        for (j, v) in stats.iter().enumerate() {
            if j > 0 {
                s.push_str(", ");
            }
            // f32 widened to f64 is exact; CPython stores the same widened
            // double (float(np.float32)) and reprs it identically.
            s.push_str(&py_float(*v as f64));
        }
        s.push_str("]]");
    }
    s.push_str("], \"bedtime\": [");
    s.push_str(&inputs.bedtime.0.to_string());
    s.push_str(", ");
    s.push_str(&inputs.bedtime.1.to_string());
    s.push_str("], \"ibi\": [");
    for (i, (ts, ibi, amp, validity)) in inputs.ibi.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("[{ts}, {ibi}, {amp}, {validity}]"));
    }
    s.push_str("], \"temp\": [");
    for (i, (ts, v)) in inputs.temp.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("[{ts}, {}]", py_float(*v as f64)));
    }
    s.push_str("]}]");
    s
}

/// `stage_bedtime`'s cache key: sha256 hex of the fingerprint text.
pub fn fingerprint(serial: Option<&str>, demo: &[f64; 3], inputs: &NssaInputs) -> String {
    let mut h = Sha256::new();
    h.update(fingerprint_text(serial, demo, inputs).as_bytes());
    let mut out = String::with_capacity(64);
    for b in h.finalize() {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

// ── the cache file ────────────────────────────────────────────────────────────

/// One cached night. `failed` = the `{}` entry Python writes when the stager
/// could not stage the inputs; it is a hit that means "no stages".
pub struct NightEntry {
    pub failed: bool,
    pub t: Vec<i64>,
    pub s: Vec<i32>,
    pub a: Option<Vec<(i64, f64)>>,
}

fn parse_entry(v: &Value) -> Option<NightEntry> {
    let obj = v.as_object()?;
    if obj.is_empty() {
        return Some(NightEntry {
            failed: true,
            t: Vec::new(),
            s: Vec::new(),
            a: None,
        });
    }
    let t = obj
        .get("t")?
        .as_array()?
        .iter()
        .map(|x| x.as_i64())
        .collect::<Option<Vec<i64>>>()?;
    let s = obj
        .get("s")?
        .as_array()?
        .iter()
        .map(|x| x.as_i64().map(|v| v as i32))
        .collect::<Option<Vec<i32>>>()?;
    let a = match obj.get("a") {
        None | Some(Value::Null) => None,
        Some(arr) => Some(
            arr.as_array()?
                .iter()
                .map(|p| {
                    let p = p.as_array()?;
                    Some((p.first()?.as_i64()?, p.get(1)?.as_f64()?))
                })
                .collect::<Option<Vec<(i64, f64)>>>()?,
        ),
    };
    Some(NightEntry {
        failed: false,
        t,
        s,
        a,
    })
}

pub struct Cache {
    serial: Option<String>,
    nights: HashMap<String, NightEntry>,
}

impl Cache {
    /// Load `nssa_cache.json` next to `db`; `None` when missing, unreadable, or
    /// written by a different CACHE_VERSION (Python would rewrite it; we should
    /// not trust it).
    pub fn load(db: &Path, serial: Option<&str>) -> Option<Cache> {
        let text = std::fs::read_to_string(cache_path(db)).ok()?;
        let doc: Value = serde_json::from_str(&text).ok()?;
        if doc.get("version").and_then(|v| v.as_str()) != Some(CACHE_VERSION) {
            return None;
        }
        let nights = doc.get("nights")?.as_object()?;
        let mut map = HashMap::with_capacity(nights.len());
        for (k, v) in nights {
            if let Some(e) = parse_entry(v) {
                map.insert(k.clone(), e);
            }
        }
        Some(Cache {
            serial: serial.map(|s| s.to_string()),
            nights: map,
        })
    }

    pub fn len(&self) -> usize {
        self.nights.len()
    }

    pub fn has_night(&self, key: &str) -> bool {
        self.nights.contains_key(key)
    }

    pub fn is_empty(&self) -> bool {
        self.nights.is_empty()
    }
}

// ── serving a window without torch ────────────────────────────────────────────

/// A [`Stager`] that only reads the cache. Any bedtime the cache cannot answer
/// sets `missed`; the caller must then delegate the whole window to Python
/// (partial windows would mix two staging runs on one hypnogram).
pub struct CacheStager<'a> {
    clock: &'a Clock,
    rows: &'a [EventRow],
    cache: &'a Cache,
    pub missed: bool,
}

impl<'a> CacheStager<'a> {
    pub fn new(clock: &'a Clock, rows: &'a [EventRow], cache: &'a Cache) -> Self {
        CacheStager {
            clock,
            rows,
            cache,
            missed: false,
        }
    }
}

impl Stager for CacheStager<'_> {
    fn stage(&mut self, b_start: i64, b_end: i64, bed_cu: i64, demo: &[f64; 3]) -> Option<Staged> {
        // Same order as stage_bedtime: inputs, then breath (outside the cache),
        // then the key lookup.
        let inputs = inputs::collect_nssa_inputs(self.rows, b_start, b_end, bed_cu, self.clock);
        let breath = breath_series(&inputs.ibi);
        let key = fingerprint(self.cache.serial.as_deref(), demo, &inputs);
        match self.cache.nights.get(&key) {
            None => {
                self.missed = true;
                None
            }
            Some(e) if e.failed => None,
            Some(e) => Some(Staged {
                t: e.t.clone(),
                codes: e.s.clone(),
                apnea: e.a.clone(),
                breath,
            }),
        }
    }
}

/// What [`serve_window`] produced.
pub enum Served {
    /// The full `score_window` output, identical to what Python would return.
    Ok(Value),
    /// A `score_window` error (same string Python raises) — not a cache miss.
    Err(String),
    /// The cache could not answer every bedtime; delegate this window.
    Missed,
}

pub fn serve_window(
    clock: &Clock,
    rows: &[EventRow],
    cache: &Cache,
    start_ds: i64,
    end_ds: i64,
    captured_unix: Option<i64>,
    tz: f64,
    demo: &[f64; 3],
) -> Served {
    let mut stager = CacheStager::new(clock, rows, cache);
    let out = grid::score_window(clock, rows, start_ds, end_ds, captured_unix, tz, demo, &mut stager);
    if stager.missed {
        return Served::Missed;
    }
    match out {
        Ok(v) => Served::Ok(v),
        Err(e) => Served::Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::demographics;

    /// The same f32 expressions `nssa_inputs.decode_acm_period` runs on the
    /// bodies [12,3,200,1,7,0,123,45,250,15,1,0] and [0,0,0,0,0,0,255,255,15,0,8,7].
    fn acm1() -> [f32; 6] {
        [
            1f32 / 4095f32 + 0f32,
            4090f32 / 4095f32 + 0f32,
            3451f32 / 4095f32 + 2f32,
            7f32 / 255f32 + 0f32,
            200f32 / 255f32 + 1f32,
            12f32 / 255f32 + 3f32,
        ]
    }
    fn acm2() -> [f32; 6] {
        [
            1800f32 / 4095f32,
            15f32 / 4095f32,
            4095f32 / 4095f32 + 15f32,
            0f32 / 255f32 + 0f32,
            0f32 / 255f32 + 0f32,
            0f32 / 255f32 + 0f32,
        ]
    }

    /// py_float must equal CPython repr() on adversarial + random doubles
    /// (tests/vectors/py_float_reprs.json; ties like 2.49411773681640625 come
    /// from f32->f64 widening, which is exactly what ACM stats produce).
    #[test]
    fn py_float_matches_cpython() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/vectors/py_float_reprs.json"
        );
        let Ok(raw) = std::fs::read_to_string(path) else {
            eprintln!("skip: no py_float vectors");
            return;
        };
        let cases: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap();
        for c in &cases {
            let v = c["v"].as_f64().unwrap();
            assert_eq!(py_float(v), c["repr"].as_str().unwrap(), "py_float({v:?})");
        }
        assert_eq!(py_float(f64::NAN), "NaN");
        assert_eq!(py_float(f64::INFINITY), "Infinity");
        assert_eq!(py_float(f64::NEG_INFINITY), "-Infinity");
        assert_eq!(py_float(-0.0), "-0.0");
        assert_eq!(py_float(0.0), "0.0");
    }

    fn vector_inputs() -> NssaInputs {
        NssaInputs {
            bedtime: (1732000000000, 1732028000000),
            acm: vec![(1732000000000, acm1()), (1732000030000, acm2())],
            ibi: vec![(1732000000100, 800, 1234, 1), (1732000800200, 950, 2000, 0)],
            beats_pre: vec![],
            temp: vec![(1732000000000, 3456f32 / 100f32), (1732000030000, 0f32 / 100f32)],
        }
    }

    /// Expected hashes computed by CPython on the identical inputs (the oracle
    /// script lives in the T2c work session; values are synthetic).
    #[test]
    fn fingerprint_matches_cpython() {
        let demo = demographics(34.0, 1.78, 75.0, "M");
        assert_eq!(
            fingerprint(Some("AABBCCDDEEFF0011"), &demo, &vector_inputs()),
            "cc06d30b2ac16767a0fcdf2df264b1e59d421ddb67d6888585063b71a7106ccc"
        );
        assert_eq!(
            fingerprint(None, &demo, &vector_inputs()),
            "c785a4b98a34ba452167876de117c458db5146ea15c27ebf1adf7b55e01410f9"
        );
        let empty = NssaInputs {
            bedtime: (0, 0),
            acm: vec![],
            ibi: vec![],
            beats_pre: vec![],
            temp: vec![],
        };
        assert_eq!(
            fingerprint(None, &[0.0, 0.0, 0.0], &empty),
            "b527b73bf27e5a7981843d7b7c238dd39d343ad92012ec865e6680e5bae15719"
        );
    }

    #[test]
    fn py_float_matches_cpython_repr() {
        // Values CPython reprs with exponent form, plus the .0 rule.
        assert_eq!(py_float(0.0), "0.0");
        assert_eq!(py_float(-0.0), "-0.0");
        assert_eq!(py_float(34.0), "34.0");
        assert_eq!(py_float(-1.0), "-1.0");
        assert_eq!(py_float(1e-5), "1e-05");
        assert_eq!(py_float(2.44e-5), "2.44e-05");
        assert_eq!(py_float(1e16), "1e+16");
        assert_eq!(py_float(1.0 / 3.0), "0.3333333333333333");
        assert_eq!(py_float((3456f32 / 100f32) as f64), "34.560001373291016");
    }

    #[test]
    fn cache_parses_entries_and_rejects_versions() {
        let dir = std::env::temp_dir().join(format!("nssa_cache_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("oura.db");
        let body = r#"{"version":"nssa-2.6.0-r2","nights":{
            "k1":{"t":[1,31],"s":[3,2],"a":[[1,0.5]]},
            "k2":{}
        }}"#;
        std::fs::write(cache_path(&db), body).unwrap();
        let c = Cache::load(&db, Some("S")).unwrap();
        assert_eq!(c.len(), 2);
        let e = c.nights.get("k1").unwrap();
        assert!(!e.failed);
        assert_eq!(e.t, vec![1, 31]);
        assert_eq!(e.s, vec![3, 2]);
        assert_eq!(e.a.as_ref().unwrap()[0].0, 1);
        assert!(c.nights.get("k2").unwrap().failed);
        std::fs::write(cache_path(&db), r#"{"version":"older","nights":{}}"#).unwrap();
        assert!(Cache::load(&db, Some("S")).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}

