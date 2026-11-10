//! Backend for the `/blood` explorer page.
//!
//! The dashboard renders the JSON this module computes. PDF imports are local:
//! `pdftotext -layout` extracts text, deterministic parsers normalize known
//! SYNLAB/CUF rows, and the results are cached in `blood.db`, separate from the
//! ring's `oura.db`.
//!
//! Layout: `<blood_dir>/blood.db` plus one document directory per profile
//! (`<blood_dir>/<ring-serial>/*.pdf`). Inside a profile directory every PDF
//! belongs to that person (any `*.pdf` is scanned); files sitting at the root
//! predate profiles and keep the legacy `blood *.pdf` scan filter.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use regex::Regex;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// Which direction is healthier - controls trend colour + which side is
/// "concerning".
#[derive(Clone, Copy, PartialEq)]
enum Dir {
    Up,   // higher is better (HDL, eGFR, vitamin D...)
    Down, // lower is better (LDL, CRP, urea...)
    Mid,  // a window is best (sodium, calcium...)
}
use Dir::*;

#[derive(Clone, Copy)]
struct MarkerDef {
    key: &'static str,
    name: &'static str,
    unit: &'static str,
    panel: &'static str,
    low: Option<f64>,
    high: Option<f64>,
    ref_text: Option<&'static str>,
    good: Dir,
    about: &'static str,
    advice: Option<&'static str>,
    aliases: &'static [&'static str],
}

#[derive(Clone, Debug)]
struct ParsedMarker {
    key: String,
    raw_name: String,
    value: f64,
    unit: String,
    low: Option<f64>,
    high: Option<f64>,
    ref_text: Option<String>,
}

#[derive(Clone, Debug)]
struct ParsedReport {
    filename: String,
    path: String,
    sha: String,
    size: u64,
    lab: String,
    collection_date: String,
    text_hash: String,
    status: String,
    warnings: Vec<String>,
    markers: Vec<ParsedMarker>,
}

#[derive(Clone, Debug)]
struct ImportRow {
    sha: String,
    filename: String,
    path: String,
    lab: String,
    collection_date: String,
    size: u64,
    text_hash: String,
    status: String,
    warnings: Vec<String>,
    serial: String,
}

#[derive(Clone, Debug)]
struct MarkerRow {
    import_sha: String,
    key: String,
    raw_name: String,
    value: f64,
    unit: String,
    low: Option<f64>,
    high: Option<f64>,
    ref_text: Option<String>,
    date: String,
}

/// Panels in display order.
const PANELS: [&str; 8] = [
    "Lipids & cardiovascular",
    "Metabolic",
    "Kidney",
    "Liver",
    "Complete blood count",
    "Vitamins & minerals",
    "Hormones",
    "Inflammation",
];

const MIN_EXPECTED_MARKERS: usize = 4;
static BLOOD_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Called from `main.rs` with the `--blood-files` value, if any.
pub fn set_files_dir(dir: Option<PathBuf>) {
    let _ = BLOOD_DIR.set(dir.map(expand_tilde));
}

fn expand_tilde(p: PathBuf) -> PathBuf {
    if let Ok(rest) = p.strip_prefix("~") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    p
}

fn default_health_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let p = PathBuf::from(home).join("Documents/official/health");
    p.exists().then_some(p)
}

fn blood_dir() -> Option<PathBuf> {
    if let Some(Some(d)) = BLOOD_DIR.get() {
        return Some(d.clone());
    }
    if let Some(env) = std::env::var_os("OURA_BLOOD_FILES") {
        if !env.is_empty() {
            return Some(expand_tilde(PathBuf::from(env)));
        }
    }
    default_health_dir()
}

fn db_path_for(dir: &Path) -> PathBuf {
    dir.join("blood.db")
}

fn open_db(dir: &Path) -> Result<Connection> {
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let conn = Connection::open(db_path_for(dir))?;
    init_db(&conn)?;
    Ok(conn)
}

fn init_db(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS imports (
            sha TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            path TEXT NOT NULL,
            lab TEXT NOT NULL,
            collection_date TEXT NOT NULL,
            size INTEGER NOT NULL,
            text_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            warnings_json TEXT NOT NULL,
            imported_at INTEGER NOT NULL,
            serial TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS marker_rows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            import_sha TEXT NOT NULL REFERENCES imports(sha) ON DELETE CASCADE,
            key TEXT NOT NULL,
            raw_name TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            ref_low REAL,
            ref_high REAL,
            ref_text TEXT,
            UNIQUE(import_sha, key)
        );
        "#,
    )?;
    // Profiles feature (docs/profiles.md): existing DBs get the serial column.
    let has_serial: bool = conn
        .prepare("SELECT 1 FROM pragma_table_info('imports') WHERE name = 'serial'")?
        .exists([])?;
    if !has_serial {
        conn.execute("ALTER TABLE imports ADD COLUMN serial TEXT NOT NULL DEFAULT ''", [])?;
    }
    Ok(())
}

fn marker_defs() -> Vec<MarkerDef> {
    let n = |x: f64| Some(x);
    vec![
        md("chol_total", "Total cholesterol", "mg/dL", PANELS[0], None, n(190.0), Some("< 190"), Down,
          "All cholesterol carried in the blood; a coarse first look at lipid health.",
          Some("Above range and the main thing to keep working on. ApoB and LDL are the sharper targets to watch alongside it."),
          &["colesterol total", "cholesterol total"]),
        md("ldl", "LDL cholesterol", "mg/dL", PANELS[0], None, n(116.0), Some("< 116"), Down,
          "The cholesterol most directly linked to arterial plaque.",
          Some("Still above the general target. For low overall risk aim under 116 mg/dL; diet, exercise, and ApoB are the sharper context to watch alongside it."),
          &["colesterol da fraccao ldl", "colesterol ldl", "ldl"]),
        md("hdl", "HDL cholesterol", "mg/dL", PANELS[0], n(40.0), None, Some("> 40"), Up,
          "The protective fraction that helps clear cholesterol.",
          Some("Comfortably protective. Kept high by aerobic exercise and unsaturated fats."),
          &["colesterol da fraccao hdl", "colesterol hdl", "hdl"]),
        md("trig", "Triglycerides", "mg/dL", PANELS[0], None, n(150.0), Some("< 150"), Down,
          "Circulating fat; sensitive to refined carbs, alcohol and recent meals.",
          Some("Excellent and well under range. A marker of a low-sugar, low-alcohol pattern."),
          &["trigliceridos", "triglycerides"]),
        md("apob", "Apolipoprotein B", "mg/dL", PANELS[0], n(46.0), n(174.0), None, Down,
          "One ApoB per atherogenic particle - the truest count of plaque-forming particles.",
          Some("Inside the lab range. Many prevention guidelines aim under 100 mg/dL; trend this alongside LDL."),
          &["apolipoproteina b", "apo b", "apolipoprotein b", "apob"]),
        md("lpa", "Lipoprotein(a)", "nmol/L", PANELS[0], None, n(75.0), Some("< 75"), Down,
          "A largely genetic, independent cardiovascular risk particle.",
          Some("Mostly set by genetics and barely moves with lifestyle. It adds to risk independently of LDL: a reason to keep the modifiable markers tight."),
          &["lipoproteina a", "lipoprotein a", "lp(a)", "lp a"]),
        md("glucose", "Fasting glucose", "mg/dL", PANELS[1], n(74.0), n(106.0), None, Down,
          "Blood sugar after an overnight fast.", None, &["glucose", "glicemia"]),
        md("hba1c", "HbA1c", "%", PANELS[1], None, n(5.7), Some("< 5.7"), Down,
          "Average blood sugar over about 3 months.",
          Some("Firmly in the healthy range. No action needed."),
          &["hemoglobina a1c", "hemoglobin a1c", "hba1c"]),
        md("insulin", "Fasting insulin", "µUI/mL", PANELS[1], n(3.0), n(25.0), None, Down,
          "How hard the pancreas works to hold glucose steady.",
          Some("Low-normal readings often reflect high insulin sensitivity when glucose and HbA1c are healthy."),
          &["insulina", "insulin"]),
        md("homa", "HOMA-IR", "", PANELS[1], n(1.92), n(2.2), Some("1.92 - 2.20"), Down,
          "Insulin-resistance index (lower means more insulin-sensitive).",
          Some("Well below the reference - excellent insulin sensitivity."),
          &["indice de resistencia a insulina", "homa"]),
        md("uric", "Uric acid", "mg/dL", PANELS[1], n(3.5), n(7.2), None, Mid,
          "By-product of purine metabolism; high levels associate with gout.", None, &["acido urico", "uric acid"]),
        md("creatinine", "Creatinine", "mg/dL", PANELS[2], n(0.70), n(1.30), None, Mid,
          "Muscle by-product cleared by the kidneys; a core kidney marker.", None, &["creatinina", "creatininemia", "creatinine"]),
        md("egfr", "eGFR", "mL/min/1.73m²", PANELS[2], n(60.0), None, Some(">= 60"), Up,
          "Estimated kidney filtration rate.",
          Some("Healthy filtration. Values naturally wobble with hydration and recent protein/creatine intake."),
          &["taxa de filtracao glomerular", "tfge", "egfr", "gfr", "ckd-epi"]),
        md("urea", "Urea (BUN)", "mg/dL", PANELS[2], n(19.0), n(49.0), None, Down,
          "Nitrogen waste from protein; rises with high protein intake or dehydration.",
          Some("Often reflects a high-protein diet or being under-hydrated at the draw when eGFR is normal."),
          &["ureia", "bun", "urea nitrogen"]),
        md("ast", "AST (GOT)", "UI/L", PANELS[3], None, n(34.0), Some("< 34"), Down,
          "Enzyme released by liver (and muscle) cells.",
          Some("Near the upper limit can follow hard exercise in the days before the draw; ALT and GGT give useful context."),
          &["aspartato-aminotransferase", "aspartato aminotransferase", "ast/got", "ast", "sgot", "(ast)"]),
        md("alt", "ALT (GPT)", "UI/L", PANELS[3], None, n(49.0), Some("< 49"), Down,
          "The most liver-specific of the routine enzymes.", None, &["alanina-aminotransferase", "alanina aminotransferase", "alt/gpt", "alt", "sgpt", "(alt)"]),
        md("ggt", "GGT", "UI/L", PANELS[3], None, n(73.0), Some("< 73"), Down,
          "Sensitive to alcohol and bile flow.", None, &["gamaglutamil transferase", "ggt"]),
        md("ck", "Creatine kinase", "U/L", PANELS[3], None, n(439.0), Some("< 439"), Mid,
          "Muscle enzyme; rises after hard or unfamiliar training and muscle injury.",
          Some("A high reading right after heavy training is expected; retest after a few easy days before reading anything clinical."),
          &["creatine kinase", "creatine", "ck"]),
        md("bilirubin", "Total bilirubin", "mg/dL", PANELS[3], None, n(1.20), Some("< 1.20"), Down,
          "Heme breakdown product processed by the liver.", None, &["bilirrubina total", "bilirubin total", "bilirubin"]),
        md("albumin", "Albumin", "g/dL", PANELS[3], n(3.2), n(4.8), None, Up,
          "The main blood protein; a marker of liver synthesis and nutrition.", None, &["albumina", "albuminemia", "albumin"]),
        md("hemoglobin", "Hemoglobin", "g/dL", PANELS[4], n(13.7), n(17.2), None, Mid,
          "Oxygen-carrying protein in red cells.", None, &["hemoglobina", "hemoglobin"]),
        md("hematocrit", "Hematocrit", "%", PANELS[4], n(40.0), n(50.0), None, Mid,
          "Fraction of blood volume that is red cells.", None, &["hematocrito", "hematocrit"]),
        md("rbc", "Red blood cells", "x10^12/L", PANELS[4], n(4.50), n(5.60), None, Mid,
          "Red cell count.", Some("Dipped just below range on the first draw and has climbed into it since."),
          &["eritrocitos", "rbc", "red blood cells"]),
        md("wbc", "White blood cells", "x10^9/L", PANELS[4], n(3.70), n(9.50), None, Mid,
          "Immune cell count; the body's baseline defence level.", None, &["leucocitos", "wbc", "white blood cells"]),
        md("platelets", "Platelets", "x10^9/L", PANELS[4], n(170.0), n(430.0), None, Mid,
          "Clotting cells.", None, &["plaquetas", "platelets"]),
        md("rdw", "RDW", "%", PANELS[4], n(11.6), n(14.1), None, Down,
          "Variation in red-cell size; an early flag for some anaemias.", None, &["rdw"]),
        md("vit_d", "Vitamin D (25-OH)", "ng/mL", PANELS[5], n(30.0), n(100.0), None, Up,
          "Vitamin/hormone for bone, immune and muscle function.",
          Some("Track this through winter; the useful target band is usually interpreted against season, sun exposure, and supplementation."),
          &["vitamina d", "vitamin d"]),
        md("ferritin", "Ferritin", "µg/L", PANELS[5], n(39.3), n(439.4), None, Mid,
          "Iron stores.", None, &["ferritina", "ferritin"]),
        md("b12", "Vitamin B12", "ng/L", PANELS[5], n(211.0), n(911.0), None, Up,
          "Needed for nerves and red-cell formation.", None, &["vitamina b12", "vitamin b12", "b12"]),
        md("folate", "Folate", "ng/mL", PANELS[5], n(5.4), None, Some("> 5.4"), Up,
          "B-vitamin for DNA synthesis and red cells.", None, &["acido folico", "folatos", "folate", "folic acid"]),
        md("magnesium", "Magnesium", "mg/dL", PANELS[5], n(1.6), n(2.6), None, Mid,
          "Cofactor for hundreds of enzymes, including muscle and nerve function.", None, &["magnesio", "magnesium"]),
        md("calcium", "Calcium", "mg/dL", PANELS[5], n(8.7), n(10.4), None, Mid,
          "Tightly regulated mineral for bone, nerves and muscle.", None, &["calcio", "calcemia", "calcium"]),
        md("tsh", "TSH", "mUI/L", PANELS[6], n(0.55), n(4.78), None, Mid,
          "Pituitary signal that sets thyroid output.", None, &["hormona tirostimulante", "tsh"]),
        md("ft4", "Free T4", "pmol/L", PANELS[6], n(10.3), n(34.7), None, Mid,
          "The circulating thyroid hormone reserve.", None, &["tiroxina livre", "free t4", "ft4"]),
        md("ft3", "Free T3", "pmol/L", PANELS[6], n(3.5), n(6.5), None, Mid,
          "The active thyroid hormone.", None, &["triiodotironina livre", "free t3", "ft3"]),
        md("test_free", "Testosterone, free", "pg/mL", PANELS[6], n(12.30), n(46.60), None, Up,
          "The biologically active fraction.", Some("Follows the same pattern as total testosterone."),
          &["testosterona livre", "testosterone free", "free testosterone"]),
        md("test_total", "Testosterone, total", "ng/dL", PANELS[6], n(197.4), n(669.6), None, Up,
          "Total circulating testosterone (bound + free).",
          Some("Sleep, energy balance, stress, and training load can move this marker substantially."),
          &["testosterona total", "testosterona total plasmatica", "testosterone total", "testosterone"]),
        md("dheas", "DHEA-S", "µg/dL", PANELS[6], n(35.0), n(569.0), None, Up,
          "Adrenal androgen and a rough vitality/recovery marker.", None, &["dehidroepiandrosterona sulfato", "dhea s", "dheas", "dhea"]),
        md("hscrp", "hs-CRP", "mg/dL", PANELS[7], None, n(0.33), Some("<= 0.33"), Down,
          "High-sensitivity marker of systemic inflammation and vascular risk.",
          Some("Very low readings are a good sign for cardiovascular risk; interpret spikes against infections or hard training."),
          &["proteina c reactiva ultra-sensivel", "proteina c reativa ultra-sensivel", "pcr ultra", "c reactive protein", "hs crp", "crp"]),
    ]
}

#[allow(clippy::too_many_arguments)]
fn md(
    key: &'static str,
    name: &'static str,
    unit: &'static str,
    panel: &'static str,
    low: Option<f64>,
    high: Option<f64>,
    ref_text: Option<&'static str>,
    good: Dir,
    about: &'static str,
    advice: Option<&'static str>,
    aliases: &'static [&'static str],
) -> MarkerDef {
    MarkerDef {
        key,
        name,
        unit,
        panel,
        low,
        high,
        ref_text,
        good,
        about,
        advice,
        aliases,
    }
}

/// Resolve the pdftotext binary once per process. Plain PATH lookup first;
/// on Windows the dashboard is often started from a shell whose PATH lacks
/// poppler (git-bash ships one, but its PATH doesn't survive into a detached
/// server), so fall back to $OURA_PDFTOTEXT and known install locations.
fn pdftotext_bin() -> &'static Path {
    static BIN: OnceLock<PathBuf> = OnceLock::new();
    BIN.get_or_init(|| {
        if let Some(env) = std::env::var_os("OURA_PDFTOTEXT") {
            let p = PathBuf::from(&env);
            if p.exists() {
                return p;
            }
        }
        if works("pdftotext") {
            return PathBuf::from("pdftotext");
        }
        for cand in pdftotext_candidates() {
            if cand.to_str().is_some_and(|s| works(s)) {
                return cand;
            }
        }
        PathBuf::from("pdftotext") // keep the original error message
    })
}

fn works(cmd: &str) -> bool {
    // Only the spawn error matters: poppler's `-v` prints the version but
    // exits non-zero (99), so the exit status must not be part of the probe.
    Command::new(cmd).arg("-v").output().is_ok()
}

fn pdftotext_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    // Git for Windows ships poppler in mingw64/bin: the standard install, plus
    // Hermes-managed git checkouts under ~/.hermes-tools style tool dirs.
    let mut roots = vec![PathBuf::from(r"C:\Program Files\Git")];
    if let Ok(pf) = std::env::var("ProgramFiles") {
        roots.push(PathBuf::from(pf).join("Git"));
    }
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        let tools = PathBuf::from(local).join("hermes").join("tools");
        if let Ok(rd) = fs::read_dir(&tools) {
            for e in rd.flatten() {
                roots.push(e.path());
            }
        }
    }
    for root in roots {
        for rel in [
            PathBuf::from("mingw64").join("bin").join("pdftotext.exe"),
            PathBuf::from("usr").join("bin").join("pdftotext.exe"),
        ] {
            let p = root.join(&rel);
            if p.exists() {
                out.push(p);
            }
        }
    }
    out
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn extract_pdf_text(path: &Path) -> Result<String> {
    let out = Command::new(pdftotext_bin())
        .arg("-layout")
        .arg(path)
        .arg("-")
        .output()
        .with_context(|| {
            "running pdftotext -layout (install poppler, or set $OURA_PDFTOTEXT to its path)"
        })?;
    if !out.status.success() {
        return Err(anyhow!(
            "pdftotext failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn parse_pdf(path: &Path) -> Result<ParsedReport> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text = extract_pdf_text(path)?;
    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let lab = detect_lab(&text);
    let collection_date = collection_date(&text)
        .or_else(|| date_from_filename(&filename))
        .unwrap_or_else(|| "unknown".to_string());
    let mut warnings = Vec::new();
    if collection_date == "unknown" {
        warnings.push("collection date not found".to_string());
    }
    let mut markers = parse_markers(&text, &mut warnings);
    if markers.len() < MIN_EXPECTED_MARKERS {
        merge_fallback_markers(&mut markers, llm_fallback_markers(&text, &mut warnings));
    }
    let status = if markers.len() >= MIN_EXPECTED_MARKERS {
        "ok"
    } else {
        "partial"
    }
    .to_string();
    if markers.len() < MIN_EXPECTED_MARKERS {
        warnings.push(format!("only {} known markers extracted", markers.len()));
    }
    Ok(ParsedReport {
        filename,
        path: path.display().to_string(),
        sha: sha256_hex(&bytes),
        size: bytes.len() as u64,
        lab,
        collection_date,
        text_hash: sha256_hex(text.as_bytes()),
        status,
        warnings,
        markers,
    })
}

fn detect_lab(text: &str) -> String {
    let n = normalize(text);
    if n.contains("synlab") {
        "SYNLAB".to_string()
    } else if n.contains("hospital cuf") || n.contains("cuf tejo") {
        "CUF".to_string()
    } else {
        "unknown".to_string()
    }
}

fn collection_date(text: &str) -> Option<String> {
    let syn = Regex::new(r"Data Colheita\s+(\d{2})-(\d{2})-(\d{4})").unwrap();
    if let Some(c) = syn.captures(text) {
        return Some(format!("{}-{}-{}", &c[3], &c[2], &c[1]));
    }
    let cuf = Regex::new(r"Data de inscrição:\s*(\d{2})/(\d{2})/(\d{4})").unwrap();
    if let Some(c) = cuf.captures(text) {
        return Some(format!("{}-{}-{}", &c[3], &c[2], &c[1]));
    }
    // US reports (Labcorp-style): "Order date: 09/29/2026" (MM/DD/YYYY).
    let us = Regex::new(r"(?i)order date:\s*(\d{1,2})/(\d{1,2})/(\d{4})").unwrap();
    if let Some(c) = us.captures(text) {
        return Some(format!("{}-{:0>2}-{:0>2}", &c[3], &c[1], &c[2]));
    }
    None
}

fn date_from_filename(name: &str) -> Option<String> {
    let re = Regex::new(r"(\d{2})-(\d{2})-(\d{4})").unwrap();
    let c = re.captures(name)?;
    Some(format!("{}-{}-{}", &c[3], &c[2], &c[1]))
}

fn parse_markers(text: &str, warnings: &mut Vec<String>) -> Vec<ParsedMarker> {
    let defs = marker_defs();
    let mut out: BTreeMap<String, ParsedMarker> = BTreeMap::new();
    for line in text.lines() {
        let Some((raw_name, cells)) = split_result_line(line) else {
            continue;
        };
        let norm = normalize(&raw_name);
        if norm.contains("bun creatinine") {
            continue; // derived ratio, not a tracked marker
        }
        if cells.first().is_some_and(|c| is_qualitative(&normalize(c))) {
            continue; // urinalysis-style qualitative rows
        }
        let Some(def) = match_def(&raw_name, &defs) else {
            continue;
        };
        if out.contains_key(def.key) {
            continue;
        }
        if def.key == "hemoglobin" && norm.contains("a1c") {
            continue;
        }
        // Only warn for rows that look like data: the value cell must start
        // with a digit or a < > = prefix. Section headers ("Vitamin D,
        // 25-Hydroxy  <footnote prose>"), stray "." lines, and qualitative
        // urinalysis rows are skipped silently.
        let looks_numeric = cells.first().is_some_and(|c| {
            let t = c.trim();
            t.starts_with(['<', '>', '=']) || t.chars().next().is_some_and(|ch| ch.is_ascii_digit())
        });
        match parse_cells(&cells, *def) {
            Some(parsed) => {
                out.insert(
                    def.key.to_string(),
                    ParsedMarker {
                        key: def.key.to_string(),
                        raw_name,
                        value: parsed.value,
                        unit: parsed.unit,
                        low: parsed.low,
                        high: parsed.high,
                        ref_text: parsed.ref_text,
                    },
                );
            }
            None if looks_numeric => warnings.push(format!("could not parse row: {raw_name}")),
            None => {}
        }
    }
    out.into_values().collect()
}

fn merge_fallback_markers(markers: &mut Vec<ParsedMarker>, fallback: Vec<ParsedMarker>) {
    for row in fallback {
        if !markers.iter().any(|m| m.key == row.key) {
            markers.push(row);
        }
    }
}

fn llm_fallback_markers(text: &str, warnings: &mut Vec<String>) -> Vec<ParsedMarker> {
    let Ok(key) = std::env::var("AGENT_BACKEND_API_KEY") else {
        warnings.push(
            "deterministic parse was sparse; AGENT_BACKEND_API_KEY not set for fallback normalizer"
                .to_string(),
        );
        return Vec::new();
    };
    let candidates = fallback_candidate_lines(text);
    if candidates.is_empty() {
        return Vec::new();
    }
    let allowed: Vec<&str> = marker_defs().iter().map(|d| d.key).collect();
    let prompt = format!(
        "Extract blood test marker rows from these pdftotext lines. Return strict JSON only, no prose, shaped as {{\"markers\":[{{\"key\":\"ldl\",\"raw_name\":\"...\",\"value\":158.0,\"unit\":\"mg/dL\",\"ref_low\":null,\"ref_high\":116.0,\"ref_text\":\"< 116\"}}]}}. Allowed keys: {}. Omit rows without a numeric result.\n\n{}",
        allowed.join(", "),
        candidates.join("\n")
    );
    let body = json!({
        "messages": [
            { "role": "system", "content": "You normalize lab PDF text into strict JSON." },
            { "role": "user", "content": prompt }
        ],
        "response_format": { "type": "json_object" }
    });
    let auth = format!("Bearer {key}");
    let resp = ureq::post("https://agent-backend.thomas.md/v1")
        .set("Authorization", &auth)
        .set("Content-Type", "application/json")
        .send_string(&body.to_string());
    let text = match resp {
        Ok(r) => match r.into_string() {
            Ok(s) => s,
            Err(e) => {
                warnings.push(format!(
                    "fallback normalizer response could not be read: {e}"
                ));
                return Vec::new();
            }
        },
        Err(e) => {
            warnings.push(format!("fallback normalizer failed: {e}"));
            return Vec::new();
        }
    };
    parse_llm_marker_json(&text, warnings)
}

fn fallback_candidate_lines(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let (name, cells) = split_result_line(line)?;
            (cells.iter().any(|c| parse_value_unit(c).is_some()) && !looks_like_header(&name))
                .then(|| line.trim().to_string())
        })
        .take(120)
        .collect()
}

fn parse_llm_marker_json(text: &str, warnings: &mut Vec<String>) -> Vec<ParsedMarker> {
    let parsed: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => {
            warnings.push(format!("fallback normalizer did not return JSON: {e}"));
            return Vec::new();
        }
    };
    let defs = marker_defs();
    let Some(rows) = parsed["markers"].as_array() else {
        warnings.push("fallback normalizer JSON had no markers array".to_string());
        return Vec::new();
    };
    rows.iter()
        .filter_map(|r| {
            let key = r["key"].as_str()?;
            let def = defs.iter().find(|d| d.key == key)?;
            let value = r["value"].as_f64()?;
            Some(ParsedMarker {
                key: key.to_string(),
                raw_name: r["raw_name"].as_str().unwrap_or(def.name).to_string(),
                value,
                unit: r["unit"].as_str().unwrap_or(def.unit).to_string(),
                low: r["ref_low"].as_f64().or(def.low),
                high: r["ref_high"].as_f64().or(def.high),
                ref_text: r["ref_text"]
                    .as_str()
                    .map(str::to_string)
                    .or_else(|| def.ref_text.map(str::to_string)),
            })
        })
        .collect()
}

fn split_result_line(line: &str) -> Option<(String, Vec<String>)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with("Mét.:") || line.starts_with("Met.:") {
        return None;
    }
    let splitter = Regex::new(r"\s{2,}").unwrap();
    let parts: Vec<String> = splitter
        .split(line)
        .map(clean_cell)
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() < 2 {
        return None;
    }
    // US-style reports glue the value to the name with a single space
    // ("HDL Cholesterol 83"); move a trailing number into the cells.
    let mut parts = parts;
    let glued = Regex::new(r"^(.+?)\s+(\d+(?:[.,]\d+)?)$").unwrap();
    if let Some(c) = glued.captures(&parts[0]) {
        let name = c[1].to_string();
        let num = c[2].to_string();
        parts[0] = num;
        parts.insert(0, name);
    }
    let name = clean_name(&parts[0]);
    if name.is_empty() || looks_like_header(&name) {
        return None;
    }
    Some((name, parts[1..].to_vec()))
}

fn clean_cell(s: &str) -> String {
    s.trim().trim_matches(':').trim().to_string()
}

fn clean_name(s: &str) -> String {
    s.trim()
        .trim_start_matches("(*)")
        .trim_start_matches("(1)")
        .trim()
        .trim_start_matches('-')
        .trim()
        .trim_end_matches(':')
        .trim()
        .to_string()
}

fn looks_like_header(name: &str) -> bool {
    let n = normalize(name);
    n.contains("impressao")
        || n.contains("pagina")
        || n.contains("data colheita")
        || n.contains("resultado")
        || n.contains("hematologia")
        || n.contains("bioquimica")
        || n.contains("patologia quimica")
        || n.contains("val ref")
}

fn match_def<'a>(name: &str, defs: &'a [MarkerDef]) -> Option<&'a MarkerDef> {
    let n = normalize(name);
    let tokens: Vec<&str> = n.split_whitespace().collect();
    defs.iter()
        .find(|d| d.aliases.iter().any(|a| token_seq(&tokens, a)))
}

/// An alias matches only as a contiguous run of whole tokens, so "vldl"
/// never matches "ldl" and "creatinine" never matches "creatine kinase".
fn token_seq(tokens: &[&str], alias: &str) -> bool {
    let alias_norm = normalize(alias);
    let pat: Vec<&str> = alias_norm.split_whitespace().collect();
    !pat.is_empty() && pat.len() <= tokens.len() && tokens.windows(pat.len()).any(|w| w == pat)
}

struct ParsedCells {
    value: f64,
    unit: String,
    low: Option<f64>,
    high: Option<f64>,
    ref_text: Option<String>,
}

fn parse_cells(cells: &[String], def: MarkerDef) -> Option<ParsedCells> {
    let first = cells.first()?.trim();
    if normalize(first).contains("em curso") {
        return None;
    }
    let (value, mut unit, mut next) = parse_value_unit(first)?;
    let mut low = None;
    let mut high = None;
    let mut ref_text = None;
    // Some layouts glue value and reference into one cell ("6.0 4.8-5.6 %"):
    // a unit that starts with a digit is really the reference range.
    if unit.starts_with(|c: char| c.is_ascii_digit()) {
        if let Some((l, h, t)) = parse_ref(&unit) {
            low = l;
            high = h;
            ref_text = Some(t);
        }
        unit = unit_token(&unit).unwrap_or_default();
    } else if let Some((l, h, t)) = parse_ref(&unit) {
        // ">39 mg/dL" glued to the value: reference, not unit.
        low = l;
        high = h;
        ref_text = Some(t);
        unit = unit_token(&unit).unwrap_or_default();
    }
    if unit.is_empty() && cells.get(1).is_some_and(|c| !is_ref_cell(c)) {
        unit = cells[1].trim().to_string();
        next = 2;
    }
    if low.is_none() && high.is_none() {
        for c in cells.iter().skip(next) {
            if let Some((l, h, t)) = parse_ref(c) {
                low = l;
                high = h;
                ref_text = Some(t);
                break;
            }
        }
    }
    if unit.is_empty() {
        // US reports attach the unit to the reference cell ("0-10 mg/L").
        unit = cells
            .iter()
            .find_map(|c| unit_token(c))
            .unwrap_or_else(|| def.unit.to_string());
    }
    if low.is_none() && high.is_none() {
        low = def.low;
        high = def.high;
        ref_text = def.ref_text.map(str::to_string);
    }
    Some(ParsedCells {
        value,
        unit: unit.trim().to_string(),
        low,
        high,
        ref_text,
    })
}

fn parse_value_unit(s: &str) -> Option<(f64, String, usize)> {
    let re = Regex::new(r"(?i)^\s*(?:[<>]=?\s*)?(\d+(?:[.,]\d+)?)\s*(.*?)\s*$").unwrap();
    let c = re.captures(s)?;
    let value = c[1].replace(',', ".").parse().ok()?;
    let unit = c
        .get(2)
        .map(|m| m.as_str().trim().to_string())
        .unwrap_or_default();
    Some((value, unit, 1))
}

fn is_ref_cell(s: &str) -> bool {
    parse_ref(s).is_some()
}

fn unit_token(s: &str) -> Option<String> {
    let re = Regex::new(r"[A-Za-z\u{00b5}%][A-Za-z0-9\u{00b5}%/^.]*").ok()?;
    let tokens: Vec<String> = re.find_iter(s).map(|m| m.as_str().to_string()).collect();
    tokens
        .into_iter()
        .find(|t| t == "%" || t.contains('/') || t.ends_with('L'))
}

fn is_qualitative(n: &str) -> bool {
    matches!(
        n,
        "negative" | "negative trace" | "trace" | "small" | "moderate" | "large" | "normal"
            | "clear" | "yellow" | "not estab" | "not established"
    )
}

fn parse_ref(s: &str) -> Option<(Option<f64>, Option<f64>, String)> {
    let s = s.trim();
    let norm = s.replace(',', ".");
    let range = Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)").unwrap();
    if let Some(c) = range.captures(&norm) {
        let l = c[1].parse().ok()?;
        let h = c[2].parse().ok()?;
        return Some((Some(l), Some(h), format!("{} - {}", trim(l), trim(h))));
    }
    let cmp = Regex::new(r"(?i)(<=|>=|<|>)\s*(\d+(?:\.\d+)?)").unwrap();
    if let Some(c) = cmp.captures(&norm) {
        let op = &c[1];
        let v: f64 = c[2].parse().ok()?;
        return match op {
            "<" | "<=" => Some((None, Some(v), format!("{op} {}", trim(v)))),
            ">" | ">=" => Some((Some(v), None, format!("{op} {}", trim(v)))),
            _ => None,
        };
    }
    None
}

fn normalize(s: &str) -> String {
    let lower = s.to_lowercase();
    let mapped: String = lower
        .chars()
        .map(|c| match c {
            'á' | 'à' | 'ã' | 'â' | 'ä' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' => 'i',
            'ó' | 'ò' | 'õ' | 'ô' | 'ö' => 'o',
            'ú' | 'ù' | 'û' | 'ü' => 'u',
            'ç' => 'c',
            other => other,
        })
        .collect();
    let cleaned: String = mapped
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' { c } else { ' ' })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn store_report(conn: &mut Connection, report: &ParsedReport, serial: &str) -> Result<bool> {
    let exists: Option<String> = conn
        .query_row(
            "SELECT sha FROM imports WHERE sha = ?1",
            params![report.sha],
            |r| r.get(0),
        )
        .optional()?;
    if exists.is_some() {
        return Ok(false);
    }
    let tx = conn.transaction()?;
    let imported_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
    tx.execute(
        "INSERT INTO imports
         (sha, filename, path, lab, collection_date, size, text_hash, status, warnings_json, imported_at, serial)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            report.sha,
            report.filename,
            report.path,
            report.lab,
            report.collection_date,
            report.size as i64,
            report.text_hash,
            report.status,
            serde_json::to_string(&report.warnings)?,
            imported_at,
            serial,
        ],
    )?;
    for m in &report.markers {
        tx.execute(
            "INSERT INTO marker_rows
             (import_sha, key, raw_name, value, unit, ref_low, ref_high, ref_text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![report.sha, m.key, m.raw_name, m.value, m.unit, m.low, m.high, m.ref_text,],
        )?;
    }
    tx.commit()?;
    Ok(true)
}

fn read_imports(conn: &Connection, serial: Option<&str>) -> Result<Vec<ImportRow>> {
    let mut st = conn.prepare(
        "SELECT sha, filename, path, lab, collection_date, size, text_hash, status, warnings_json, serial
         FROM imports
         WHERE (?1 = '' OR serial = ?1)
         ORDER BY collection_date DESC, filename DESC",
    )?;
    let rows = st
        .query_map([serial.unwrap_or("")], |r| {
            let warnings_json: String = r.get(8)?;
            let warnings = serde_json::from_str(&warnings_json).unwrap_or_default();
            Ok(ImportRow {
                sha: r.get(0)?,
                filename: r.get(1)?,
                path: r.get(2)?,
                lab: r.get(3)?,
                collection_date: r.get(4)?,
                size: r.get::<_, i64>(5)? as u64,
                text_hash: r.get(6)?,
                status: r.get(7)?,
                warnings,
                serial: r.get(9)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn read_marker_rows(conn: &Connection, serial: Option<&str>) -> Result<Vec<MarkerRow>> {
    let mut st = conn.prepare(
        "SELECT m.import_sha, m.key, m.raw_name, m.value, m.unit, m.ref_low, m.ref_high, m.ref_text, i.collection_date
         FROM marker_rows m
         JOIN imports i ON i.sha = m.import_sha
         WHERE (?1 = '' OR i.serial = ?1)
         ORDER BY i.collection_date ASC, m.key ASC",
    )?;
    let rows = st
        .query_map([serial.unwrap_or("")], |r| {
            Ok(MarkerRow {
                import_sha: r.get(0)?,
                key: r.get(1)?,
                raw_name: r.get(2)?,
                value: r.get(3)?,
                unit: r.get(4)?,
                low: r.get(5)?,
                high: r.get(6)?,
                ref_text: r.get(7)?,
                date: r.get(8)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn scan_pdf_paths(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_ascii_lowercase();
            if name.starts_with("blood ") && name.ends_with(".pdf") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Every PDF in a directory. Used for per-profile document dirs, where any
/// lab PDF belongs to that person regardless of the legacy `blood ` prefix.
fn scan_all_pdfs(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_file()
                && e.file_name()
                    .to_string_lossy()
                    .to_ascii_lowercase()
                    .ends_with(".pdf")
            {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Per-profile document directory: `<root>/<ring-serial>/`. Ring serials are
/// alphanumeric; anything else is rejected so a serial can never escape the
/// blood root. `""` (unassigned) has no directory — those files stay at the
/// root.
fn profile_dir(root: &Path, serial: &str) -> Option<PathBuf> {
    (!serial.is_empty()
        && serial
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'))
    .then(|| root.join(serial))
}

/// Move an imported PDF into the profile's document dir (root for ""),
/// returning the new path. No-op when the file is already there.
fn move_import_pdf(root: &Path, from: &str, serial: &str) -> Result<PathBuf> {
    let src = PathBuf::from(from);
    let target = profile_dir(root, serial).unwrap_or_else(|| root.to_path_buf());
    if src.parent() == Some(target.as_path()) {
        return Ok(src);
    }
    fs::create_dir_all(&target).with_context(|| format!("creating {}", target.display()))?;
    let dest = target.join(src.file_name().context("imported pdf has no file name")?);
    fs::rename(&src, &dest).with_context(|| {
        format!("moving {} to {}", src.display(), dest.display())
    })?;
    Ok(dest)
}

fn import_path_at(db_dir: &Path, path: &Path, serial: &str) -> Result<Value> {
    let mut conn = open_db(db_dir)?;
    let report = parse_pdf(path)?;
    let inserted = store_report(&mut conn, &report, serial)?;
    Ok(json!({
        "ok": true,
        "imported": inserted,
        "duplicate": !inserted,
        "file": report.filename,
        "date": report.collection_date,
        "lab": report.lab,
        "sha": report.sha,
        "markers": report.markers.len(),
        "status": report.status,
        "warnings": report.warnings,
    }))
}

fn import_dir_at(scan_dir: &Path, db_root: &Path, serial: &str, any_pdf: bool) -> Result<Value> {
    let paths = if any_pdf {
        scan_all_pdfs(scan_dir)
    } else {
        scan_pdf_paths(scan_dir)
    };
    let mut imported = 0usize;
    let mut duplicates = 0usize;
    let mut failed = 0usize;
    let mut rows = Vec::new();
    for p in paths {
        match import_path_at(db_root, &p, serial) {
            Ok(v) => {
                if v["imported"].as_bool().unwrap_or(false) {
                    imported += 1;
                } else {
                    duplicates += 1;
                }
                rows.push(v);
            }
            Err(e) => {
                failed += 1;
                rows.push(json!({
                    "ok": false,
                    "file": p.file_name().unwrap_or_default().to_string_lossy(),
                    "message": e.to_string(),
                }));
            }
        }
    }
    Ok(json!({
        "ok": failed == 0,
        "dir": scan_dir.display().to_string(),
        "scanned": rows.len(),
        "imported": imported,
        "duplicates": duplicates,
        "failed": failed,
        "imports": rows,
    }))
}

/// `POST /api/blood/import-dir` - scan the active profile's document
/// directory (`<blood_dir>/<serial>/`, created on demand) for PDFs, stamping
/// new imports with `serial`. With no profile (""), falls back to the legacy
/// root scan (`blood *.pdf` only).
pub fn import_dir(serial: &str) -> Value {
    let Some(root) = blood_dir() else {
        return json!({ "ok": false, "message": "no blood directory configured; pass --blood-files DIR" });
    };
    let (dir, any_pdf) = match profile_dir(&root, serial) {
        Some(d) => {
            if let Err(e) = fs::create_dir_all(&d) {
                return json!({ "ok": false, "message": format!("creating {}: {e}", d.display()) });
            }
            (d, true)
        }
        None => (root.clone(), false),
    };
    match import_dir_at(&dir, &root, serial, any_pdf) {
        Ok(v) => v,
        Err(e) => json!({ "ok": false, "message": e.to_string() }),
    }
}

/// `POST /api/blood/import {"path": "/path/to/report.pdf"}` — new import is
/// stamped with `serial` (the active profile).
pub fn import_path(path: &str, serial: &str) -> Value {
    let Some(dir) = blood_dir() else {
        return json!({ "ok": false, "message": "no blood directory configured; pass --blood-files DIR" });
    };
    let path = expand_tilde(PathBuf::from(path));
    if path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("pdf"))
        != Some(true)
    {
        return json!({ "ok": false, "message": "expected a PDF path" });
    }
    match import_path_at(&dir, &path, serial) {
        Ok(v) => v,
        Err(e) => json!({ "ok": false, "message": e.to_string() }),
    }
}

/// `GET /api/blood/report` - computed panel from imported PDF rows, scoped to
/// `serial` (None = every profile; "" matches only unassigned rows).
pub fn report(serial: Option<&str>) -> Value {
    let Some(dir) = blood_dir() else {
        return empty_report("no blood directory configured; pass --blood-files DIR");
    };
    match report_from_dir(&dir, serial) {
        Ok(v) => v,
        Err(e) => json!({ "error": e.to_string() }),
    }
}

/// One-time migration: legacy rows (serial='') move to the primary profile.
/// Returns the number of rows reassigned.
pub fn backfill_serial(primary: &str) -> Value {
    let Some(dir) = blood_dir() else {
        return json!({"ok": false, "message": "no blood directory configured"});
    };
    match open_db(&dir).and_then(|conn| {
        conn.execute(
            "UPDATE imports SET serial = ?1 WHERE serial = ''",
            params![primary],
        )
        .map_err(anyhow::Error::from)
    }) {
        Ok(n) => json!({"ok": true, "reassigned": n}),
        Err(e) => json!({"ok": false, "message": e.to_string()}),
    }
}

/// `POST /api/blood/assign {"sha": ..., "serial": ...}` — move an import (and
/// its marker rows, via the join) to another profile, and move its PDF into
/// the target profile's document directory (unassigned → back to the root).
pub fn assign(sha: &str, serial: &str) -> Value {
    let Some(root) = blood_dir() else {
        return json!({"ok": false, "message": "no blood directory configured"});
    };
    match open_db(&root).and_then(|conn| {
        let from = conn
            .query_row(
                "SELECT path FROM imports WHERE sha LIKE ?1 || '%'",
                params![sha],
                |r| r.get::<_, String>(0),
            )
            .optional()?;
        let n = conn.execute(
            "UPDATE imports SET serial = ?2 WHERE sha LIKE ?1 || '%'",
            params![sha, serial],
        )?;
        if n == 0 {
            return Ok(None);
        }
        let mut move_error = String::new();
        if let Some(from) = from {
            match move_import_pdf(&root, &from, serial) {
                Ok(dest) => {
                    conn.execute(
                        "UPDATE imports SET path = ?2 WHERE path = ?1",
                        params![from, dest.to_string_lossy().as_ref()],
                    )?;
                }
                // The reassignment itself stands; a failed file move only
                // leaves the PDF in the old location.
                Err(e) => move_error = e.to_string(),
            }
        }
        Ok(Some(move_error))
    }) {
        Ok(None) => json!({"ok": false, "message": format!("no import matching sha prefix {sha}")}),
        Ok(Some(move_error)) => json!({
            "ok": true,
            "file_moved": move_error.is_empty(),
            "message": move_error,
        }),
        Err(e) => json!({"ok": false, "message": e.to_string()}),
    }
}

/// One-time layout migration: move imported PDFs that still sit at the blood
/// root into their profile's document directory, updating stored paths.
/// Idempotent — files already inside a subdirectory are left alone.
pub fn migrate_profile_dirs() -> Value {
    let Some(root) = blood_dir() else {
        return json!({"ok": false, "message": "no blood directory configured"});
    };
    match open_db(&root).and_then(|conn| {
        let rows: Vec<(String, String, String)> = conn
            .prepare("SELECT sha, path, serial FROM imports WHERE serial != ''")
            .and_then(|mut st| {
                st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                    .map(|m| m.filter_map(|x| x.ok()).collect::<Vec<_>>())
            })?;
        let mut moved = 0usize;
        for (sha, path, serial) in rows {
            if Path::new(&path).parent() != Some(root.as_path()) {
                continue; // already inside a document dir
            }
            match move_import_pdf(&root, &path, &serial) {
                Ok(dest) => {
                    conn.execute(
                        "UPDATE imports SET path = ?2 WHERE sha = ?1",
                        params![sha, dest.to_string_lossy().as_ref()],
                    )?;
                    moved += 1;
                }
                Err(e) => eprintln!("blood: profile-dir migration skipped {path}: {e}"),
            }
        }
        Ok(moved)
    }) {
        Ok(n) => json!({"ok": true, "moved": n}),
        Err(e) => json!({"ok": false, "message": e.to_string()}),
    }
}

/// Create empty per-profile document dirs so it is obvious where new PDFs go.
pub fn ensure_profile_dirs(serials: &[String]) {
    let Some(root) = blood_dir() else { return };
    for s in serials {
        if let Some(d) = profile_dir(&root, s) {
            let _ = fs::create_dir_all(d);
        }
    }
}

fn report_from_dir(dir: &Path, serial: Option<&str>) -> Result<Value> {
    let conn = open_db(dir)?;
    // Auto-import only when the DB holds nothing at all — never per-profile,
    // or an empty profile would rescan the PDFs on every request.
    if read_imports(&conn, None)?.is_empty() {
        drop(conn);
        // First run for a profile: scan that profile's document dir; an
        // unassigned request keeps the legacy root scan.
        let (scan_dir, any_pdf) = match serial.and_then(|s| profile_dir(dir, s)) {
            Some(d) => {
                let _ = fs::create_dir_all(&d);
                (d, true)
            }
            None => (dir.to_path_buf(), false),
        };
        let _ = import_dir_at(&scan_dir, dir, serial.unwrap_or(""), any_pdf);
    }
    let conn = open_db(dir)?;
    let imports = read_imports(&conn, serial)?;
    let rows = read_marker_rows(&conn, serial)?;
    Ok(build_report_json(&imports, &rows, &db_path_for(dir)))
}

fn empty_report(message: &str) -> Value {
    json!({
        "mocked": false,
        "source": "pdf",
        "panels": PANELS,
        "markers": [],
        "imports": [],
        "draws": [],
        "summary": {
            "markers_total": 0,
            "flagged": 0,
            "imports": 0,
            "first_date": null,
            "latest_date": null,
        },
        "warnings": [message],
    })
}

fn build_report_json(imports: &[ImportRow], rows: &[MarkerRow], db_path: &Path) -> Value {
    let defs = marker_defs();
    let mut by_key: HashMap<String, Vec<&MarkerRow>> = HashMap::new();
    for r in rows {
        by_key.entry(r.key.clone()).or_default().push(r);
    }
    let markers_json: Vec<Value> = defs
        .iter()
        .filter_map(|d| by_key.get(d.key).map(|points| marker_json(d, points)))
        .collect();
    let flagged = markers_json
        .iter()
        .filter(|m| m["flagged"].as_bool().unwrap_or(false))
        .count();
    let mut draws: Vec<String> = imports.iter().map(|i| i.collection_date.clone()).collect();
    draws.sort();
    draws.dedup();
    let imports_json: Vec<Value> = imports
        .iter()
        .map(|i| {
            let markers = rows.iter().filter(|r| r.import_sha == i.sha).count();
            json!({
                "file": i.filename,
                "path": i.path,
                "date": i.collection_date,
                "lab": i.lab,
                "size": i.size,
                "sha": &i.sha[..12.min(i.sha.len())],
                "text_hash": &i.text_hash[..12.min(i.text_hash.len())],
                "markers": markers,
                "status": i.status,
                "warnings": i.warnings,
                "serial": &i.serial,
                "source": "pdf",
            })
        })
        .collect();
    json!({
        "mocked": false,
        "source": "pdf",
        "db": db_path.display().to_string(),
        "panels": PANELS,
        "markers": markers_json,
        "imports": imports_json,
        "draws": draws,
        "summary": {
            "markers_total": markers_json.len(),
            "flagged": flagged,
            "imports": imports.len(),
            "first_date": imports.iter().map(|i| i.collection_date.as_str()).min(),
            "latest_date": imports.iter().map(|i| i.collection_date.as_str()).max(),
        },
    })
}

fn round(v: f64, dp: i32) -> f64 {
    let f = 10f64.powi(dp);
    (v * f).round() / f
}

fn ref_text(def: &MarkerDef, latest: Option<&MarkerRow>) -> String {
    if let Some(t) = latest.and_then(|r| r.ref_text.clone()) {
        return t;
    }
    if let Some(t) = def.ref_text {
        return t.to_string();
    }
    match (
        latest.and_then(|r| r.low).or(def.low),
        latest.and_then(|r| r.high).or(def.high),
    ) {
        (Some(l), Some(h)) => format!("{} - {}", trim(l), trim(h)),
        (Some(l), None) => format!("> {}", trim(l)),
        (None, Some(h)) => format!("< {}", trim(h)),
        (None, None) => "-".into(),
    }
}

fn trim(v: f64) -> String {
    if (v.fract()).abs() < 1e-9 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn marker_json(def: &MarkerDef, points_in: &[&MarkerRow]) -> Value {
    let mut points = points_in.to_vec();
    points.sort_by(|a, b| a.date.cmp(&b.date));
    let latest_row = *points.last().expect("marker has at least one row");
    let latest = latest_row.value;
    let prev = if points.len() >= 2 {
        Some(points[points.len() - 2].value)
    } else {
        None
    };
    let low = latest_row.low.or(def.low);
    let high = latest_row.high.or(def.high);

    let out_low = low.map(|l| latest < l).unwrap_or(false);
    let out_high = high.map(|h| latest > h).unwrap_or(false);
    let status = if out_high {
        "high"
    } else if out_low {
        "low"
    } else {
        "normal"
    };
    let bad = match def.good {
        Up => out_low,
        Down => out_high,
        Mid => out_low || out_high,
    };
    let severity = if bad {
        "watch"
    } else if out_low || out_high {
        "good"
    } else {
        "neutral"
    };
    let (trend, trend_good, delta, delta_pct) = match prev {
        Some(p) => {
            let d = latest - p;
            let eps = (latest.abs() * 0.005).max(1e-9);
            let dir = if d.abs() <= eps {
                "flat"
            } else if d > 0.0 {
                "up"
            } else {
                "down"
            };
            let good = if dir == "flat" {
                true
            } else {
                match def.good {
                    Up => d > 0.0,
                    Down => d < 0.0,
                    Mid => match (low, high) {
                        (Some(l), Some(h)) => {
                            let c = (l + h) / 2.0;
                            (latest - c).abs() <= (p - c).abs()
                        }
                        _ => true,
                    },
                }
            };
            (
                dir,
                good,
                Some(round(d, 3)),
                if p != 0.0 {
                    Some(round(d / p * 100.0, 1))
                } else {
                    None
                },
            )
        }
        None => ("flat", true, None, None),
    };
    let pts: Vec<Value> = points
        .iter()
        .map(|r| json!({ "date": r.date, "value": round(r.value, 3) }))
        .collect();

    json!({
        "key": def.key,
        "name": def.name,
        "raw_name": latest_row.raw_name,
        "unit": if latest_row.unit.is_empty() { def.unit } else { latest_row.unit.as_str() },
        "panel": def.panel,
        "about": def.about,
        "advice": def.advice,
        "low": low,
        "high": high,
        "ref_text": ref_text(def, Some(latest_row)),
        "points": pts,
        "latest": round(latest, 3),
        "latest_date": latest_row.date,
        "prev": prev.map(|p| round(p, 3)),
        "delta": delta,
        "delta_pct": delta_pct,
        "status": status,
        "flagged": bad,
        "severity": severity,
        "trend": trend,
        "trend_good": trend_good,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "open_oura_blood_test_{name}_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn parses_synlab_rows() {
        let text = r#"
Data Colheita                   09-06-2026 09:13
      Glucose                                                                          87 mg/dL                        74 - 106               79             76
(*) Taxa de filtração glomerular, estimada (TFGe)                                      79 mL/min/1,73m2                  >= 60                87             94
      Colesterol total                                                                235 mg/dL                               < 190                 229           278
      Colesterol da fracção HDL                                                         62 mg/dL                              > 40                  65            76
(*) Colesterol da fracção LDL                                                         158 mg/dL                               < 116                 154           185
      Creatinina                                                                      1.26 mg/dL                     0.70 - 1.30             1.17           1.10
"#;
        let mut warnings = Vec::new();
        let rows = parse_markers(text, &mut warnings);
        let by_key: HashMap<_, _> = rows.iter().map(|r| (r.key.as_str(), r.value)).collect();
        assert_eq!(by_key["chol_total"], 235.0);
        assert_eq!(by_key["ldl"], 158.0);
        assert_eq!(by_key["hdl"], 62.0);
        assert_eq!(by_key["glucose"], 87.0);
        assert_eq!(by_key["creatinine"], 1.26);
        assert_eq!(by_key["egfr"], 79.0);
    }

    #[test]
    fn parses_cuf_rows() {
        let text = r#"
Data de inscrição: 09/09/2025   09:16:41
 Glicémia                             88     mg/dL                      70 - 110
 Colesterol Total                  248       mg/dL                         < 190
 Colesterol LDL                    166       mg/dL                         < 115
 Colesterol HDL                       64     mg/dL                       35 - 55
 Creatininémia                          1.02 mg/dL                      0.70 - 1.30
  TFGe [CKD-EPI 2009]                 103    ml/min/1,73 m 2                     60
 Testosterona Total Plasmática (TT)       144.7     ng/dl                 241.0 - 827.0
"#;
        assert_eq!(collection_date(text).as_deref(), Some("2025-09-09"));
        let mut warnings = Vec::new();
        let rows = parse_markers(text, &mut warnings);
        let by_key: HashMap<_, _> = rows.iter().map(|r| (r.key.as_str(), r.value)).collect();
        assert_eq!(by_key["glucose"], 88.0);
        assert_eq!(by_key["ldl"], 166.0);
        assert_eq!(by_key["egfr"], 103.0);
        assert_eq!(by_key["test_total"], 144.7);
    }

    #[test]
    fn normalizes_aliases() {
        let defs = marker_defs();
        assert_eq!(match_def("Ácido úrico", &defs).unwrap().key, "uric");
        assert_eq!(
            match_def("Colesterol da fracção LDL", &defs).unwrap().key,
            "ldl"
        );
        assert_eq!(match_def("Creatininémia", &defs).unwrap().key, "creatinine");
    }

    #[test]
    fn parses_reference_ranges() {
        assert_eq!(
            parse_ref("< 190").unwrap(),
            (None, Some(190.0), "< 190".into())
        );
        assert_eq!(
            parse_ref("> 40").unwrap(),
            (Some(40.0), None, "> 40".into())
        );
        assert_eq!(
            parse_ref("<= 0.33").unwrap(),
            (None, Some(0.33), "<= 0.33".into())
        );
        assert_eq!(
            parse_ref(">= 60").unwrap(),
            (Some(60.0), None, ">= 60".into())
        );
        assert_eq!(
            parse_ref("0.70 - 1.30").unwrap(),
            (Some(0.70), Some(1.30), "0.7 - 1.3".into())
        );
    }

    #[test]
    fn dedupes_by_file_hash() {
        let dir = temp_dir("dedupe");
        let mut conn = open_db(&dir).unwrap();
        let report = ParsedReport {
            filename: "blood fake.pdf".into(),
            path: "/tmp/blood fake.pdf".into(),
            sha: "abc".into(),
            size: 10,
            lab: "SYNLAB".into(),
            collection_date: "2026-06-09".into(),
            text_hash: "def".into(),
            status: "ok".into(),
            warnings: vec![],
            markers: vec![ParsedMarker {
                key: "glucose".into(),
                raw_name: "Glucose".into(),
                value: 87.0,
                unit: "mg/dL".into(),
                low: Some(74.0),
                high: Some(106.0),
                ref_text: Some("74 - 106".into()),
            }],
        };
        assert!(store_report(&mut conn, &report, "").unwrap());
        assert!(!store_report(&mut conn, &report, "").unwrap());
        assert_eq!(read_imports(&conn, None).unwrap().len(), 1);
        assert_eq!(read_marker_rows(&conn, None).unwrap().len(), 1);
        // per-profile reads see their own rows only (distinct sha per file)
        let mut report2 = report.clone();
        report2.sha = "deadbeef0000".into();
        store_report(&mut conn, &report2, "S1").unwrap();
        assert_eq!(read_imports(&conn, Some("S1")).unwrap().len(), 1);
        assert_eq!(read_imports(&conn, Some("S2")).unwrap().len(), 0);
        assert_eq!(read_marker_rows(&conn, Some("S1")).unwrap().len(), 1);
        assert_eq!(read_marker_rows(&conn, Some("S2")).unwrap().len(), 0);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn profile_dirs_and_pdf_scans() {
        let root = temp_dir("profile_dirs");
        // serials are validated so a path can never escape the root
        assert_eq!(profile_dir(&root, ""), None);
        assert_eq!(profile_dir(&root, "../escape"), None);
        assert_eq!(
            profile_dir(&root, "40160B2500000001"),
            Some(root.join("40160B2500000001"))
        );
        let pd = root.join("S1");
        fs::create_dir_all(&pd).unwrap();
        for name in ["LK blood 09-11-2025.pdf", "blood 27-07-2022.pdf", "notes.txt"] {
            fs::write(pd.join(name), "x").unwrap();
        }
        fs::write(root.join("LK blood root.pdf"), "x").unwrap();
        fs::write(root.join("blood root.pdf"), "x").unwrap();
        // inside a profile dir any PDF counts; at the root only the legacy prefix
        assert_eq!(scan_all_pdfs(&pd).len(), 2);
        assert_eq!(scan_pdf_paths(&root).len(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn move_import_pdf_relocates_and_is_idempotent() {
        let root = temp_dir("move_pdf");
        let src = root.join("blood x.pdf");
        fs::write(&src, "x").unwrap();
        let dest = move_import_pdf(&root, &src.to_string_lossy(), "S1").unwrap();
        assert_eq!(dest, root.join("S1").join("blood x.pdf"));
        assert!(dest.exists() && !src.exists());
        // already in place: no-op
        let again = move_import_pdf(&root, &dest.to_string_lossy(), "S1").unwrap();
        assert_eq!(again, dest);
        // unassigned moves back to the root
        let back = move_import_pdf(&root, &dest.to_string_lossy(), "").unwrap();
        assert_eq!(back, src);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn report_shape_matches_frontend() {
        let imports = vec![ImportRow {
            sha: "abcdef123456".into(),
            filename: "blood fake.pdf".into(),
            path: "/tmp/blood fake.pdf".into(),
            lab: "SYNLAB".into(),
            collection_date: "2026-06-09".into(),
            size: 100,
            text_hash: "feedbeef1234".into(),
            status: "ok".into(),
            warnings: vec![],
            serial: "".into(),
        }];
        let rows = vec![MarkerRow {
            import_sha: "abcdef123456".into(),
            key: "glucose".into(),
            raw_name: "Glucose".into(),
            value: 87.0,
            unit: "mg/dL".into(),
            low: Some(74.0),
            high: Some(106.0),
            ref_text: Some("74 - 106".into()),
            date: "2026-06-09".into(),
        }];
        let v = build_report_json(&imports, &rows, Path::new("/tmp/blood.db"));
        assert_eq!(v["mocked"], false);
        assert_eq!(v["source"], "pdf");
        assert!(v["summary"].get("markers_total").is_some());
        assert!(v["imports"].as_array().unwrap()[0]
            .get("warnings")
            .is_some());
        assert!(v["markers"].as_array().unwrap()[0].get("points").is_some());
    }

    #[test]
    fn real_june_pdf_matches_golden_when_present() {
        // the PDF lives in the owner's profile dir since the layout
        // migration; check the legacy root location too
        let root = PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
            .join("Documents/official/health");
        let mut candidates = vec![root.join("blood 09-06-2026.pdf")];
        if let Ok(rd) = fs::read_dir(&root) {
            for e in rd.flatten() {
                let p = e.path();
                if p.file_name().is_some_and(|n| n == "blood 09-06-2026.pdf") {
                    candidates.push(p);
                }
            }
        }
        let Some(path) = candidates.iter().find(|p| p.exists()).cloned() else {
            return;
        };
        let expected: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/blood_june_2026_key_rows.json"
        ))
        .unwrap();
        let report = parse_pdf(&path).unwrap();
        assert_eq!(
            report.collection_date,
            expected["collection_date"].as_str().unwrap()
        );
        let by_key: HashMap<_, _> = report
            .markers
            .iter()
            .map(|r| (r.key.as_str(), r.value))
            .collect();
        for (key, value) in expected["rows"].as_object().unwrap() {
            assert_eq!(by_key[key.as_str()], value.as_f64().unwrap());
        }
    }
}
