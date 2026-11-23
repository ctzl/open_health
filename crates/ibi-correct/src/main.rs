//! CLI wrapper around the IBI correction port, used by the Python tools in
//! open_health (tools/ibi_correction.py shells out to this binary when it
//! exists and falls back to the pure-Python reference otherwise).
//!
//!   ibi-correct           one-shot: read one JSON array of [ts,ibi,amp]
//!                         triples on stdin, write [[ts,ibi,amp,validity],...]
//!                         on stdout.
//!   ibi-correct --jsonl   persistent worker: one window per input line,
//!                         one result array per output line (flushed).

use std::io::{self, BufRead, Write};

fn main() {
    let jsonl = std::env::args().any(|a| a == "--jsonl");
    if jsonl {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut out = stdout.lock();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            let reply = process(&line);
            if writeln!(out, "{reply}").is_err() {
                break;
            }
            out.flush().unwrap();
        }
    } else {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf).unwrap();
        print!("{}", process(&buf));
        println!();
    }
}

fn process(line: &str) -> String {
    let parsed: Result<Vec<[i64; 3]>, _> = serde_json::from_str(line);
    match parsed {
        Ok(window) => {
            let raw: Vec<(u64, i64, i64)> =
                window.iter().map(|[ts, ibi, amp]| (*ts as u64, *ibi, *amp)).collect();
            let corrected = ibi_correct::correct_ibis(&raw);
            let json: Vec<[u64; 4]> = corrected
                .iter()
                .map(|(ts, ibi, amp, val)| [*ts, *ibi as u64, *amp as u64, *val as u64])
                .collect();
            serde_json::to_string(&json).unwrap_or_else(|_| "[]".to_string())
        }
        Err(e) => format!("{{\"error\":{}}}", serde_json::to_string(&e.to_string()).unwrap()),
    }
}

use std::io::Read;
