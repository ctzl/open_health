//! The row shape every function here consumes.
//!
//! One event row, in the compact form `tools/nssa_plumbing_golden.py` writes and a
//! future shim would send over the worker pipe:
//!
//! ```json
//! [ring_timestamp, tag, captured_unix, x, y]
//! ```
//!
//! `x`/`y` are tag-dependent, exactly like the decoded payloads the Python side
//! already holds:
//!
//! * `0x60` (ibi_and_amplitude): `x` = `decoded_json["ibi_ms"]` (6 ints),
//!   `y` = `decoded_json["amplitude"]`;
//! * `0x72` (sleep_acm_period) / `0x75` (sleep_temp_event): `y` = the raw event
//!   body as a lowercase hex string (`null` when the DB body is NULL);
//! * `0x76` (bedtime_period): `x` = `bedtime_start_ds`, `y` = `bedtime_end_ds`.

use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::Deserialize;

pub const TAG_IBI: u8 = 0x60;
pub const TAG_ACM: u8 = 0x72;
pub const TAG_TEMP: u8 = 0x75;
pub const TAG_BEDTIME: u8 = 0x76;

#[derive(Clone, Debug, Default)]
pub struct EventRow {
    pub ds: i64,
    pub tag: u8,
    pub cu: i64,
    /// `ibi_ms` for tag 0x60; only rows with exactly six intervals are kept by
    /// `collect_nssa_inputs`, and the oracle harness drops the others.
    pub ibi_ms: Option<[i64; 6]>,
    pub amplitude: Option<Vec<i64>>,
    /// Raw event body for tags 0x72 / 0x75.
    pub body: Option<Vec<u8>>,
    /// bedtime_start_ds / bedtime_end_ds for tag 0x76.
    pub bedtime_start_ds: Option<i64>,
    pub bedtime_end_ds: Option<i64>,
}

impl<'de> Deserialize<'de> for EventRow {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct V;

        impl<'de> Visitor<'de> for V {
            type Value = EventRow;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("[ds, tag, captured_unix, payload...]")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<EventRow, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let ds: i64 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &"event row"))?;
                let tag: u8 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &"event row"))?;
                let cu: i64 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &"event row"))?;
                let mut row = EventRow {
                    ds,
                    tag,
                    cu,
                    ..Default::default()
                };

                let x: Option<serde_json::Value> = seq.next_element()?;
                let y: Option<serde_json::Value> = seq.next_element()?;
                match tag {
                    TAG_IBI => {
                        if let Some(serde_json::Value::Array(a)) = x {
                            if a.len() == 6 {
                                let mut ibis = [0i64; 6];
                                for (slot, v) in ibis.iter_mut().zip(a.iter()) {
                                    *slot = v.as_i64().unwrap_or(0);
                                }
                                row.ibi_ms = Some(ibis);
                            }
                        }
                        if let Some(serde_json::Value::Array(a)) = y {
                            row.amplitude = Some(a.iter().map(|v| v.as_i64().unwrap_or(0)).collect());
                        }
                    }
                    TAG_BEDTIME => {
                        row.bedtime_start_ds = x.and_then(|v| v.as_i64());
                        row.bedtime_end_ds = y.and_then(|v| v.as_i64());
                    }
                    _ => {
                        if let Some(serde_json::Value::String(hex)) = y {
                            row.body = Some(unhex(&hex));
                        }
                    }
                }
                Ok(row)
            }
        }

        deserializer.deserialize_seq(V)
    }
}

pub fn unhex(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len() / 2);
    let nib = |c: u8| -> u32 {
        match c {
            b'0'..=b'9' => (c - b'0') as u32,
            b'a'..=b'f' => (c - b'a' + 10) as u32,
            b'A'..=b'F' => (c - b'A' + 10) as u32,
            _ => 0,
        }
    };
    let mut i = 0;
    while i + 1 < b.len() {
        out.push((nib(b[i]) << 4 | nib(b[i + 1])) as u8);
        i += 2;
    }
    out
}

pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
