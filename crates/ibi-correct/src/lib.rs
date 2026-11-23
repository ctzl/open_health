//! Exact port of Oura's on-phone IBI correction (official Android app).
//!
//! Rust twin of `tools/ibi_correction.py` in open_health — read that file's
//! docstring for the reverse-engineering provenance (libappecore.so,
//! rr_correction_init_ex / rr_correction_run_ex). Every operation mirrors the
//! arm64 original: integer fields are u16/u8/u32/u64 and all floating-point
//! arithmetic is IEEE float32. `tests/golden.rs` verifies bit-exact agreement
//! with the Python reference on real and synthetic streams; keep the two in
//! sync when touching either.

const CAP: usize = 64;
const K60000: f32 = 60000.0f32;

/// float32 -> int, round to nearest, ties away from zero (arm64 FCVTAS).
/// The intermediate math runs in f64 exactly like the Python reference.
fn fcvtas(x: f32) -> i64 {
    let xf = x as f64;
    if xf.is_nan() {
        return 0;
    }
    let r = xf.abs() + 0.5;
    let r = r.floor();
    if xf >= 0.0 {
        r as i64
    } else {
        -(r as i64)
    }
}

/// float32 -> int, truncate toward zero (arm64 FCVTZS); NaN -> 0.
fn fcvtzs(x: f32) -> i64 {
    let xf = x as f64;
    if xf.is_nan() {
        return 0;
    }
    xf.trunc() as i64
}

/// s_ibi_data_t (32 bytes). Copy semantics match the Python Entry.copy().
#[derive(Clone, Copy, Debug)]
struct Entry {
    ts: u64,
    ts2: u64,
    ibi: u32,
    amp: u32,
    ibi_out: u32,
    amp_out: u32,
    validity: u32,
    ok: u32,
    extra: u32,
}

impl Entry {
    const ZERO: Entry =
        Entry { ts: 0, ts2: 0, ibi: 0, amp: 0, ibi_out: 0, amp_out: 0, validity: 0, ok: 0, extra: 0 };
}

/// ibirbuffer_t: fixed 64 slots (never cleared), head = next write, tail = oldest.
#[derive(Clone, Debug)]
struct RingBuf {
    slots: [Entry; CAP],
    head: usize,
    tail: usize,
    full: bool,
}

impl RingBuf {
    fn new() -> Self {
        RingBuf { slots: [Entry::ZERO; CAP], head: 0, tail: 0, full: false }
    }

    fn push(&mut self, e: Entry) {
        self.slots[self.head] = e;
        if self.full {
            self.tail = if self.tail + 1 == CAP { 0 } else { self.tail + 1 };
        }
        self.head = if self.head + 1 == CAP { 0 } else { self.head + 1 };
        self.full = self.head == self.tail;
    }

    fn is_empty(&self) -> bool {
        !self.full && self.head == self.tail
    }

    fn pop(&mut self) -> Option<Entry> {
        if self.is_empty() {
            return None;
        }
        let e = self.slots[self.tail];
        self.full = false;
        self.tail = if self.tail + 1 == CAP { 0 } else { self.tail + 1 };
        Some(e)
    }

    fn count(&self) -> usize {
        if self.full {
            CAP
        } else {
            (self.head + CAP - self.tail) % CAP
        }
    }

    /// read_from_head(signed i): 0 = newest, -1 = previous ... -> slot index.
    fn rfh_idx(&self, i: i32) -> usize {
        if self.is_empty() || i.abs() >= CAP as i32 {
            panic!("native abort in ibirbuffer_read_from_head");
        }
        let idx = i as i64 + CAP as i64 + self.head as i64 - 1;
        if idx < 0 {
            panic!("native abort in ibirbuffer_read_from_head");
        }
        (idx as usize) % CAP
    }

    fn rfh(&self, i: i32) -> Entry {
        let idx = self.rfh_idx(i);
        self.slots[idx]
    }
}

/// Transliteration of median7_u16 (values u16; 0xffff treated as "missing").
fn median7_u16(v: &[u32; 7]) -> u32 {
    const M: i32 = 0xffff;
    let mut w8 = v[0] as i32;
    let mut w9 = v[1] as i32;
    let mut w12 = v[2] as i32;
    let mut w10 = if w9 > w8 { w9 } else { w8 };
    let w13 = if w9 < w8 { w9 } else { w8 };
    if w9 == M {
        w10 = M;
        w9 = w8;
    } else {
        w9 = w13;
    }
    // third
    if w12 == M {
        w8 = w10;
        w10 = M;
    } else if w12 >= w10 {
        w8 = w10;
        w10 = w12;
    } else {
        w8 = if w12 > w9 { w12 } else { w9 };
        w9 = if w12 < w9 { w12 } else { w9 };
    }
    // fourth
    w12 = v[3] as i32;
    let mut w11;
    if w12 == M {
        w11 = w8;
        w8 = w10;
        w10 = M;
    } else if w12 >= w10 {
        w11 = w8;
        w8 = w10;
        w10 = w12;
    } else if w12 >= w8 {
        w11 = w8;
        w8 = w12;
    } else {
        w11 = if w12 > w9 { w12 } else { w9 };
        w9 = if w12 < w9 { w12 } else { w9 };
    }
    // fifth
    w12 = v[4] as i32;
    if w12 >= w10 {
        w9 = w8;
        w8 = w10;
    } else if w12 >= w8 {
        w9 = w8;
        w8 = w12;
    } else {
        let w10_ = w11;
        let t = if w12 > w9 { w12 } else { w9 };
        w11 = if w12 < w10_ { t } else { w11 };
        w9 = if w12 > w10_ { w12 } else { w10_ };
    }
    // sixth
    w10 = v[5] as i32;
    if w10 < w8 {
        let a8 = w11;
        let a11 = w9;
        let t = if w10 > a8 { w10 } else { a8 };
        w9 = if w10 < a11 { t } else { w9 };
        w8 = if w10 > a11 { w10 } else { a11 };
    }
    // seventh
    w10 = v[6] as i32;
    let t = if w10 > w9 { w10 } else { w9 };
    (if w10 < w8 { t } else { w8 }) as u32
}

fn s16(x: i64) -> i32 {
    let x = (x as u32 & 0xffff) as i32;
    if x & 0x8000 != 0 {
        x - 0x10000
    } else {
        x
    }
}

/// Transliteration of median7(short*) (signed; 0x7fff treated as "missing").
fn median7_s16(v: &[i64; 7]) -> i32 {
    const M: i32 = 0x7fff;
    let mut w8 = s16(v[0]);
    let w9r = s16(v[1]);
    let mut w12 = s16(v[2]);
    let w10u = (w9r as u32 & 0xffff) as i32;
    let w13 = if w9r > w8 { w9r } else { w8 };
    let w14 = if w9r < w8 { w9r } else { w8 };
    let mut w9;
    let mut w10;
    if w10u == M {
        w9 = w10u;
        w10 = w8;
    } else {
        w9 = w13;
        w10 = w14;
    }
    if w12 == M {
        w8 = w9;
        w9 = M;
    } else if w12 >= s16(w9 as i64) {
        w8 = w9;
        w9 = w12;
    } else {
        let w10s = s16(w10 as i64);
        w8 = if w12 > w10s { w12 } else { w10s };
        w10 = if w12 < w10s { w12 } else { w10s };
    }
    w12 = s16(v[3]);
    let mut w11;
    if w12 == M {
        w11 = w8;
        w8 = w9;
        w9 = M;
    } else if w12 >= s16(w9 as i64) {
        w11 = w8;
        w8 = w9;
        w9 = w12;
    } else if w12 >= s16(w8 as i64) {
        w11 = w8;
        w8 = w12;
    } else {
        let w10s = s16(w10 as i64);
        w11 = if w12 > w10s { w12 } else { w10s };
        w10 = if w12 < w10s { w12 } else { w10s };
    }
    w12 = s16(v[4]);
    if w12 >= s16(w9 as i64) {
        w10 = w8;
        w8 = w9;
    } else if w12 >= s16(w8 as i64) {
        w10 = w8;
        w8 = w12;
    } else {
        // Faithful to the assembly quirk: these temps read w10/w11, not w9/w10.
        let w9s = s16(w10 as i64);
        let w10s = s16(w11 as i64);
        let t = if w12 > w9s { w12 } else { w9s };
        w11 = if w12 < w10s { t } else { w11 };
        w10 = if w12 > w10s { w12 } else { w10s };
    }
    w9 = s16(v[5]);
    if w9 < s16(w8 as i64) {
        let a8 = s16(w11 as i64);
        let a11 = s16(w10 as i64);
        let t = if w9 > a8 { w9 } else { a8 };
        w10 = if w9 < a11 { t } else { w10 };
        w8 = if w9 > a11 { w9 } else { a11 };
    }
    w9 = s16(v[6]);
    let w10s = s16(w10 as i64);
    let t = if w9 > w10s { w9 } else { w10s };
    if w9 < s16(w8 as i64) {
        t
    } else {
        w8
    }
}

fn hr(ibi: i64) -> f32 {
    K60000 / (ibi as f32)
}

/// One corrected beat: (timestamp_ms, ibi_ms, amplitude, validity).
pub type CorrectedIbi = (u64, i64, i64, u32);

pub struct IbiCorrection {
    buf1: RingBuf, // raw beats
    buf2: RingBuf, // pending outputs
    count: i64,
    max_out: u32,
    max_gap: u32,
    prev_ibi: u32,
    n_out: u32,
    n_valid: u32,
    interp_ts: bool,
    hr_mode: bool,
    prev_hr: f32,
    max_hr: f32,
    bad_run: u32,
    saved_bad_run: u32,
    dl_ibi: [i64; 7],
    dl_raw_amp: [i64; 7],
    dl_amp: [i64; 7],
    dl_val: [i64; 7],
    dl_ts: [u64; 7],
    dl_extra: [i64; 7],
    // median_error state
    me_prev_hr: f32,
    me_max_hr: f32,
    me_flag: bool,
    me_prev_ibi: i64,
    now_ts: u64,
    ms_flag: bool,
    out: Vec<CorrectedIbi>,
}

impl Default for IbiCorrection {
    fn default() -> Self {
        Self::new()
    }
}

impl IbiCorrection {
    pub fn new() -> Self {
        IbiCorrection {
            buf1: RingBuf::new(),
            buf2: RingBuf::new(),
            count: 0,
            max_out: 4,
            max_gap: 5000,
            prev_ibi: 1000,
            n_out: 0,
            n_valid: 0,
            interp_ts: false,
            hr_mode: true,
            prev_hr: 60.0f32,
            max_hr: 150.0f32,
            bad_run: 0,
            saved_bad_run: 0,
            dl_ibi: [0; 7],
            dl_raw_amp: [0; 7],
            dl_amp: [0; 7],
            dl_val: [0; 7],
            dl_ts: [0; 7],
            dl_extra: [0; 7],
            me_prev_hr: 0.0f32,
            me_max_hr: 150.0f32,
            me_flag: true,
            me_prev_ibi: 1000,
            now_ts: 0,
            ms_flag: true,
            out: Vec::new(),
        }
    }

    /// 0x1770dc: variance of HR over buf1[-6..0] (no zero guard).
    fn var7_buf1(&self) -> f32 {
        let mut hs = [0.0f32; 7];
        for (k, i) in (-6i32..=0).enumerate() {
            hs[k] = hr(self.buf1.rfh(i).ibi as i64);
        }
        let mut s = hs[0];
        for h in &hs[1..] {
            s = s + h;
        }
        let mean = s / 7.0f32;
        let mut acc = 0.0f32;
        for (k, h) in hs.iter().enumerate() {
            let d = h - mean;
            let d2 = d * d;
            acc = if k == 0 { d2 } else { acc + d2 };
        }
        acc / 6.0f32
    }

    fn median_error(&mut self) {
        let before = self.dl_val[3];
        static CNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let idx = CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dbg = std::env::var("IBI_ME_DEBUG")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            == Some(idx);
        self.median_error_inner(dbg);
        if std::env::var_os("IBI_TRACE").is_some() {
            eprintln!(
                "ME {idx} ts={} A={:?} V={:?} flag={} prev_hr={:.6} prev_ibi={} \
                 max_hr={:.6} extra={} val {}->{}",
                self.dl_ts[3],
                self.dl_ibi,
                self.dl_val,
                self.me_flag as u8,
                self.me_prev_hr,
                self.me_prev_ibi,
                self.me_max_hr,
                self.dl_extra[3],
                before,
                self.dl_val[3],
            );
        }
    }

    fn median_error_inner(&mut self, dbg: bool) {
        if self.dl_val[3] != 1 {
            return;
        }
        let a = self.dl_ibi;
        let w22 = a[3];
        let s8 = hr(w22);
        if self.me_prev_hr == 0.0f32 {
            self.me_prev_hr = s8;
        }
        let med = s16(median7_s16(&a) as i64);
        let tot: i64 = a.iter().map(|x| s16(*x) as i64).sum();
        let mean = (tot as f32) / 7.0f32;
        let s1 = K60000 / (med as f32);
        let s0 = K60000 / (fcvtzs(mean) as f32);
        if dbg {
            let v7 = self.var7_dl();
            let d = s0 - s8;
            eprintln!(
                "w22={w22} s8={s8:.9} med={med} tot={tot} mean={mean:.9} s1={s1:.9} \
                 s0={s0:.9} v7={v7:.9} d2={:.9} v7x5={:.9} flag={} prev_hr_pre={:.9} \
                 prev_ibi_pre={} max_hr={:.9}",
                d * d,
                v7 * 5.0f32,
                self.me_flag,
                self.me_prev_hr,
                self.me_prev_ibi,
                self.me_max_hr,
            );
        }
        let mut bad = false;
        if self.me_flag {
            let v7 = self.var7_dl();
            if s8 <= 30.0f32 || s8 > self.me_max_hr || (s8 - s1).abs() > 18.0f32 {
                bad = true;
            } else if s8 > (self.me_prev_hr + 23.0f32) + 4.0f32 {
                bad = true;
            } else if s8 < self.me_prev_hr * 0.625f32 {
                bad = true;
            } else if s8 < (self.me_prev_hr + -16.0f32) + -4.0f32 {
                bad = true;
            } else {
                let d = s0 - s8;
                if d * d > v7 * 5.0f32 {
                    bad = true;
                }
            }
        } else {
            if s8 <= 30.0f32 || s8 > self.me_max_hr || (s8 - s1).abs() > 18.0f32 {
                bad = true;
            } else if (w22 - self.me_prev_ibi).abs() * 100 > self.me_prev_ibi * 25 {
                bad = true;
            } else {
                let d = s0 - s8;
                let v7 = self.var7_dl();
                if d * d > v7 * 5.0f32 {
                    bad = true;
                } else if !(self.dl_extra[3] < 7) {
                    bad = true;
                }
            }
        }
        if bad {
            self.dl_val[3] = 2;
        }
        self.me_prev_hr = s8;
        self.me_prev_ibi = w22;
    }

    /// median_error_ex helper: HR variance over the 7-beat delay line (0 for ibi==0).
    fn var7_dl(&self) -> f32 {
        let a = self.dl_ibi;
        let mut hs = [0.0f32; 7];
        for i in 0..7 {
            hs[i] = if a[i] != 0 { hr(a[i]) } else { 0.0f32 };
        }
        let mut s = if a[0] == 0 { 0.0f32 } else { hs[0] };
        for i in 1..7 {
            if a[i] != 0 {
                s = s + hs[i];
            }
        }
        let mean = s / 7.0f32;
        let mut acc = 0.0f32;
        for (k, h) in hs.iter().enumerate() {
            let d = h - mean;
            let d2 = d * d;
            acc = if k == 0 { d2 } else { acc + d2 };
        }
        acc / 6.0f32
    }

    fn flush(&mut self, start: i32, n: u32, dec: bool) {
        let mut w20 = start;
        let mut w23 = n as i64;
        loop {
            let idx = self.buf1.rfh_idx(w20);
            let e = self.buf1.slots[idx];
            let mut o = e;
            o.ibi_out = e.ibi;
            o.amp_out = 0;
            o.validity = 2;
            o.ts2 = e.ts;
            self.buf2.push(o);
            self.buf1.slots[idx].ibi = 0;
            self.buf1.slots[idx].amp = 0;
            self.buf1.slots[idx].validity = 0;
            self.buf1.pop();
            if dec {
                self.bad_run = self.bad_run.wrapping_sub(1) & 0xff;
            }
            w23 = (w23 - 1) & 0xffffffff;
            w20 += 1;
            if (w23 & 0xff) == 0 {
                break;
            }
        }
    }

    /// L870 (w25 > -64 always here) -> flush long bad runs, then L924.
    fn after_gap(&mut self) {
        if self.bad_run > 60 {
            self.flush(-63, self.bad_run, true);
        }
        self.emit_last();
    }

    fn emit_last(&mut self) {
        let idx = self.buf1.rfh_idx(-3);
        let e = self.buf1.slots[idx];
        if e.ok == 1 {
            let mut o = e;
            o.validity = 1;
            o.ibi_out = e.ibi;
            o.amp_out = e.amp;
            o.ts2 = e.ts;
            self.buf2.push(o);
        }
    }

    pub fn run(&mut self, ibi: i64, amp: i64, ts: u64, extra: i64) {
        let ibi = (ibi as u64 & 0xffff) as u32;
        let amp = (amp as u64 & 0xffff) as u32;
        self.now_ts = ts;
        if ibi != 0 {
            self.ingest(ibi, amp, extra);
        }
        self.output();
    }

    fn ingest(&mut self, ibi: u32, amp: u32, extra: i64) {
        self.count += 1;
        let new = Entry { ts: self.now_ts, ibi, amp, ok: 1, extra: extra as u32, ..Entry::ZERO };
        self.buf1.push(new);
        if self.count == 1 {
            let mut o = new;
            o.ibi_out = ibi;
            o.amp_out = amp;
            o.validity = 0;
            o.ts2 = self.now_ts;
            if new.ts.wrapping_sub(1) <= 99999999998 {
                self.ms_flag = false;
            }
            self.buf2.push(o);
            return;
        }
        if self.count < 5 {
            return;
        }
        let idx = self.buf1.rfh_idx(-3);
        let e = self.buf1.slots[idx];
        let s8 = if e.ibi != 0 { hr(e.ibi as i64) } else { K60000 };
        let n = self.buf1.count();
        let update;
        if n <= 6 {
            let mut bad = false;
            if self.hr_mode {
                if s8 <= 30.0f32 || s8 > self.max_hr {
                    bad = true;
                } else if s8 < self.prev_hr * 0.625f32 || s8 > self.prev_hr + 18.0f32 {
                    bad = true;
                } else if s8 < self.prev_hr + -25.0f32 {
                    bad = true;
                }
            } else {
                if s8 <= 30.0f32 || s8 > self.max_hr {
                    bad = true;
                } else if ((e.ibi as i64 - self.prev_ibi as i64).abs()) * 100
                    > self.prev_ibi as i64 * 20
                {
                    bad = true;
                } else if !(e.extra < 7) {
                    bad = true;
                }
            }
            if bad {
                self.buf1.slots[idx].ok = 0;
                self.bad_run = (self.bad_run + 1) & 0xff;
                update = true;
            } else {
                self.buf1.slots[idx].ok = 1;
                self.saved_bad_run = self.bad_run;
                self.bad_run = 0;
                update = true;
            }
        } else {
            let mut vv = [0u32; 7];
            let mut tot: i64 = 0;
            for (k, i) in (-6i32..=0).enumerate() {
                let ib = self.buf1.rfh(i).ibi;
                vv[k] = ib;
                tot += ib as i64;
            }
            let m = median7_u16(&vv);
            let mm = if m & 0xfffe != 0 { m } else { 1 };
            let s9 = K60000 / (mm as f32);
            let mean = (tot as f32) / 7.0f32;
            let mut s0 = if mean != 0.0f32 { K60000 / mean } else { 0.0f32 };
            s0 = s0 - s8;
            let mut bad = false;
            if s8 <= 30.0f32 || s8 > self.max_hr || (s8 - s9).abs() > 14.0f32 {
                bad = true;
            } else if self.hr_mode {
                if s8 < self.prev_hr * 0.625f32 || s8 > self.prev_hr + 23.0f32 {
                    bad = true;
                } else if s8 < self.prev_hr + -16.0f32 {
                    bad = true;
                } else if s0 * s0 > self.var7_buf1() * 1.5f32 {
                    bad = true;
                }
            } else {
                if ((e.ibi as i64 - self.prev_ibi as i64).abs()) * 100 > self.prev_ibi as i64 * 20
                {
                    bad = true;
                } else if s0 * s0 > self.var7_buf1() * 1.5f32 {
                    bad = true;
                } else if !(e.extra < 7) {
                    bad = true;
                }
            }
            if bad {
                self.buf1.slots[idx].ok = 0;
                self.bad_run = (self.bad_run + 1) & 0xff;
                update = self.bad_run >= 4;
            } else {
                self.buf1.slots[idx].ok = 1;
                self.saved_bad_run = self.bad_run;
                self.bad_run = 0;
                update = true;
            }
        }
        if update {
            self.prev_hr = s8;
            self.prev_ibi = self.buf1.slots[idx].ibi;
        }
        let a = self.buf1.rfh(-3);
        let b = self.buf1.rfh(-4);
        if a.ok == 1 && b.ok == 0 {
            self.gap();
            return;
        }
        if self.bad_run > 60 {
            self.flush(-63, self.bad_run, true);
        }
        self.emit_last();
    }

    fn gap(&mut self) {
        let mut w24: i32 = 0;
        let g: i32;
        loop {
            let gi = w24 - 4;
            if self.buf1.rfh(gi).ok != 0 {
                g = gi;
                break;
            }
            let w8 = w24 - 5;
            w24 -= 1;
            if !(-63..=-1).contains(&w8) {
                // no earlier valid beat in the buffer
                if self.bad_run > 60 {
                    self.flush(-63, self.bad_run, true);
                } else if self.saved_bad_run != 0 {
                    self.flush(-3 - self.saved_bad_run as i32, self.saved_bad_run, false);
                }
                self.emit_last();
                return;
            }
        }
        let mut w21 = w24 - 3;
        let ge = self.buf1.rfh(g);
        let (gi, gts) = (ge.ibi as i64, ge.ts);
        let le = self.buf1.rfh(-3);
        let (li, lts) = (le.ibi as i64, le.ts);
        let mut sumbad: i64 = 0;
        if g <= -5 {
            let mut k = w21;
            while k < -3 {
                sumbad += self.buf1.rfh(k).ibi as i64;
                k += 1;
            }
        }
        let w11 = li + gi;
        let mut nfill = fcvtas((sumbad as f32) / ((w11 >> 1) as f32));
        if nfill <= 1 {
            nfill = 1;
        }
        let nbad = -w24 as i64;
        let same = nfill == nbad;
        if sumbad > self.max_gap as i64 || (1 - w24 as i64) > self.max_out as i64 {
            let mut last_ts: u64 = 0;
            let mut e = self.buf1.rfh(w21);
            if e.ok == 0 {
                loop {
                    e = self.buf1.rfh(w21);
                    let mut o = e;
                    o.ibi_out = e.ibi;
                    o.amp_out = 0;
                    o.validity = 2;
                    o.ts2 = e.ts;
                    self.buf2.push(o);
                    w21 += 1;
                    last_ts = e.ts;
                    if self.buf1.rfh(w21).ok != 0 {
                        break;
                    }
                }
            }
            if self.max_gap != 0 {
                self.buf2.push(Entry { ts2: last_ts, validity: 2, ..Entry::ZERO });
            }
            self.after_gap();
            return;
        }
        // fill / redistribute the gap
        let step = fcvtas(((li - gi) as f32) / ((nfill + 1) as f32));
        let end = (nfill + w24 as i64 - 4) as i32;
        let mut cur = step + gi;
        let mut cum: i64 = 0;
        let mut last_ts: u64 = 0;
        let mut w20 = g;
        loop {
            let w22 = w20 + 1;
            let e = self.buf1.rfh(w22);
            let mut o = e;
            if nbad >= nfill || w22 < -3 {
                last_ts = e.ts;
            } else {
                o = Entry { ts: last_ts, ..Entry::ZERO };
            }
            o.ibi_out = (cur as u64 & 0xffff) as u32;
            o.amp_out = 0;
            o.validity = 3;
            self.buf2.push(o);
            w20 += 1;
            cum += cur as i64;
            cur += step;
            if w20 >= end {
                break;
            }
        }
        if self.interp_ts && ((w11 as u64 & 0xffffffff) >= 2) {
            let keep = same; // (the interp-ts branch; unused by the app: interp_ts=false)
            self.restore_or_validate(keep, nfill, g, end, w21);
            return;
        }
        let diff = sumbad - cum;
        let mut q = (diff as f32) / (nfill as f32);
        q = if diff < 0 { q.ceil() } else { q.floor() };
        let w11q = fcvtzs(q);
        let scale: f32 = if !self.ms_flag { 1000.0f32 } else { 1.0f32 };
        let mut r = diff - nfill * w11q;
        let mut w23: i32 = (1 - nfill) as i32;
        let mut keep = same;
        let mut cts: u64 = 0;
        let mut w24b = g;
        loop {
            let idx2 = self.buf2.rfh_idx(w23);
            let e2 = self.buf2.slots[idx2];
            let v: i64 = if r == 0 {
                e2.ibi_out as i64 + w11q
            } else if r < 0 {
                r += 1;
                (w11q - 1) + e2.ibi_out as i64
            } else {
                r -= 1;
                (w11q + 1) + e2.ibi_out as i64
            };
            let v = (v as u64 & 0xffffffff) as u32;
            self.buf2.slots[idx2].ibi_out = v & 0xffff;
            cts = cts.wrapping_add((v & 0xffff) as u64);
            let t = (fcvtas((cts as f32) / scale) as i64)
                .wrapping_add(gts as i64) as u64;
            self.buf2.slots[idx2].ts2 = if t > lts { lts } else { t };
            if keep {
                let d = (self.buf1.rfh(end).ibi as i64 - self.buf2.slots[idx2].ibi_out as i64).abs();
                keep = d < 126;
            } else {
                keep = false;
            }
            w24b += 1;
            w23 += 1;
            if w24b >= end {
                break;
            }
        }
        if keep {
            self.restore_or_validate(true, nfill, g, end, w21);
            return;
        }
        self.after_gap();
    }

    fn restore_or_validate(&mut self, keep: bool, nfill: i64, g: i32, end: i32, w21: i32) {
        if !keep {
            self.after_gap();
            return;
        }
        let mut w23 = g;
        let mut w24 = (1 - nfill) as i32;
        loop {
            w23 += 1;
            let v = self.buf1.rfh(w23).ibi;
            let idx = self.buf2.rfh_idx(w24);
            self.buf2.slots[idx].ibi_out = v;
            w24 += 1;
            if !(w23 < end) {
                break;
            }
        }
        let li = self.buf1.rfh(-3).ibi as i64;
        let gi = self.buf1.rfh(g).ibi as i64;
        let s9 = K60000 / (li.min(gi) as f32) + 34.5f32;
        let mut mn: u32 = 0xffff;
        let mut mx: u32 = 0;
        if w21 <= -4 {
            let mut i = w21;
            while i < -3 {
                let v = self.buf1.rfh(i).ibi;
                if mn >= v {
                    mn = v;
                }
                if mx < v {
                    mx = v;
                }
                i += 1;
            }
        } else {
            mn = 0xffff;
            mx = 0;
        }
        let s8 = K60000 / (mn as f32);
        let mut good = false;
        let mut s0 = f32::NAN;
        let mut s1 = f32::NAN;
        if s8 < s9 {
            s0 = K60000 / (li.max(gi) as f32) + -24.0f32;
            let lim = f32::from_bits(0x41f0_cccd);
            if s0 < lim {
                s0 = lim;
            }
            s1 = K60000 / (mx as f32);
            if s1 > s0 {
                good = true;
            }
        }
        if std::env::var_os("IBI_TRACE").is_some() {
            eprintln!(
                "ROV nfill={nfill} g={g} end={end} w21={w21} li={li} gi={gi} \
                 mn={mn} mx={mx} s8={s8} s9={s9} s0={s0} s1={s1} good={good}"
            );
        }
        let mut w22 = (1 - nfill) as i32;
        let mut w23 = g;
        loop {
            let idx = self.buf2.rfh_idx(w22);
            self.buf2.slots[idx].validity = if good { 1 } else { 2 };
            if good {
                let e2 = self.buf2.slots[idx];
                self.buf2.slots[idx].amp_out = e2.amp;
                self.buf2.slots[idx].ts2 = e2.ts;
            }
            w23 += 1;
            w22 += 1;
            if w23 >= end {
                break;
            }
        }
        self.after_gap();
    }

    fn output(&mut self) {
        if self.buf2.is_empty() {
            return;
        }
        let mut outs = 0u32;
        loop {
            if outs >= self.max_out {
                return;
            }
            let e = self.buf2.pop().expect("native assert: pop failed");
            if e.ibi_out != 0 && self.count >= 0 {
                for i in 0..6 {
                    self.dl_ibi[i] = self.dl_ibi[i + 1];
                    self.dl_raw_amp[i] = self.dl_raw_amp[i + 1];
                    self.dl_amp[i] = self.dl_amp[i + 1];
                    self.dl_val[i] = self.dl_val[i + 1];
                    self.dl_ts[i] = self.dl_ts[i + 1];
                    self.dl_extra[i] = self.dl_extra[i + 1];
                }
                self.dl_ibi[6] = e.ibi_out as i64;
                self.dl_raw_amp[6] = e.amp as i64;
                self.dl_amp[6] = e.amp_out as i64;
                self.dl_val[6] = e.validity as i64;
                self.dl_ts[6] = e.ts2;
                self.dl_extra[6] = e.extra as i64;
                if self.dl_ibi[0] != 0 {
                    self.median_error();
                }
                if self.dl_ibi[3] != 0 {
                    self.n_out = (self.n_out + 1) & 0xffff;
                    if self.dl_extra[3] <= 1 && self.dl_val[3] == 1 {
                        self.n_valid = (self.n_valid + 1) & 0xffff;
                    }
                    self.out
                        .push((self.dl_ts[3], self.dl_ibi[3], self.dl_amp[3], (self.dl_val[3] & 0xff) as u32));
                    outs += 1;
                }
            }
            if self.buf2.is_empty() {
                return;
            }
        }
    }

    pub fn take_output(&mut self) -> Vec<CorrectedIbi> {
        std::mem::take(&mut self.out)
    }
}

/// raw: iterable of (timestamp_ms, ibi_ms, amplitude) in ring order.
/// Returns [(timestamp_ms, ibi, amplitude, validity)] exactly as the official
/// app's ecore.correctIbiAndAmplitudeEvents does (fresh state per call).
pub fn correct_ibis(raw: &[(u64, i64, i64)]) -> Vec<CorrectedIbi> {
    let mut c = IbiCorrection::new();
    for &(ts, ibi, amp) in raw {
        c.run(ibi, amp, ts, 0);
    }
    c.take_output()
}
