//! Port of `tools/respiratory_rate.py`'s series path plus
//! `run_sleep_model.breath_series`.
//!
//! What is bit-exact and what is not, stated honestly:
//!
//! * the ectopic filter, the 4 Hz grid (`np.arange`), `np.interp`, the FIR kernel
//!   and the 5-window median filter are reproduced with the same float64 formulas;
//! * each reported value is `round(k / 60.0 * 60.0, 1)` where `k` is the in-band
//!   spectral peak *index* — so the series is bit-identical as long as the peak
//!   index and the `max < 4 * median` gate agree;
//! * the spectrum itself is NOT bit-identical to numpy's pocketfft: this port
//!   evaluates only the 16 in-band DFT bins directly (they are all `_dominant_freq`
//!   ever reads) and sums them in a different order than pocketfft's plan. Same for
//!   `np.mean` on the ~100k-sample resampled series, where numpy's block
//!   decomposition is not fully pinned down. Both feed only comparisons.

use crate::inputs::CorrectedBeat;
use crate::num::round1;

pub const FS: f64 = 4.0;
pub const RESP_LO: f64 = 0.15;
pub const RESP_HI: f64 = 0.40;
pub const WIN_S: usize = 60;
pub const STEP_S: usize = 30;
pub const IBI_LO: f64 = 300.0;
pub const IBI_HI: f64 = 2000.0;
pub const FIR_TAPS: usize = 129;

/// `run_sleep_model.breath_series`: [] when the night has too few beats.
pub fn breath_series(ibi: &[CorrectedBeat]) -> Vec<(i64, f64)> {
    if ibi.len() < 60 {
        return Vec::new();
    }
    let ts: Vec<f64> = ibi.iter().map(|b| b.0 as f64 / 1000.0).collect();
    let ms: Vec<f64> = ibi.iter().map(|b| b.1 as f64).collect();
    respiratory_series(&ts, &ms)
}

/// `respiratory_series`: [[window_start_unix_s, br/min], …].
pub fn respiratory_series(ts_s: &[f64], ibi_ms: &[f64]) -> Vec<(i64, f64)> {
    let Some((grid, resampled)) = tachogram(ts_s, ibi_ms) else {
        return Vec::new();
    };
    let x = bandpass(&resampled);
    let win = (WIN_S as f64 * FS) as usize;
    let step = (STEP_S as f64 * FS) as usize;
    let bins = band_bins(win);

    let mut rr: Vec<(i64, f64)> = Vec::new();
    let mut i = 0usize;
    while i + win <= x.len() {
        if let Some(f) = dominant_freq(&x[i..i + win], &bins) {
            if 6.0 <= f * 60.0 && f * 60.0 <= 35.0 {
                rr.push((grid[i] as i64, round1(f * 60.0)));
            }
        }
        i += step;
    }
    if rr.len() < 3 {
        return Vec::new();
    }
    // 5-window median filter (upper median, as the Python lambda sorts and indexes).
    let vals: Vec<f64> = rr.iter().map(|(_, v)| *v).collect();
    rr.iter()
        .enumerate()
        .map(|(i, (t, _))| {
            let lo = i.saturating_sub(2);
            let hi = (i + 3).min(vals.len());
            (*t, round1(upper_median(&vals[lo..hi])))
        })
        .collect()
}

/// `_tachogram`: ectopic-filtered, then cubic-resampled to FS.
/// (`pub` only so the diagnostic example can replay the stages; not API.)
#[doc(hidden)]
pub fn tachogram(ts_s: &[f64], ibi_ms: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut ts: Vec<f64> = Vec::with_capacity(ts_s.len());
    let mut ibi: Vec<f64> = Vec::with_capacity(ibi_ms.len());
    for (t, v) in ts_s.iter().zip(ibi_ms.iter()) {
        if *v >= IBI_LO && *v <= IBI_HI {
            ts.push(*t);
            ibi.push(*v);
        }
    }
    if ibi.len() < 30 {
        return None;
    }
    let med = median(&ibi);
    let keep: Vec<(f64, f64)> = ts
        .iter()
        .zip(ibi.iter())
        .filter(|(_, v)| (*v - med).abs() <= 0.30 * med)
        .map(|(t, v)| (*t, *v))
        .collect();
    if keep.len() < 30 || keep[keep.len() - 1].0 - keep[0].0 < WIN_S as f64 {
        return None;
    }
    let (ts2, ibi2): (Vec<f64>, Vec<f64>) = keep.into_iter().unzip();
    let grid = arange(ts2[0], ts2[ts2.len() - 1], 1.0 / FS);
    let values = interp(&grid, &ts2, &ibi2);
    Some((grid, values))
}

/// `np.arange(start, stop, step)` for step > 0.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil();
    let n = if n.is_finite() && n > 0.0 { n as usize } else { 0 };
    (0..n).map(|i| start + i as f64 * step).collect()
}

/// `np.interp`: numpy's `interpolate()` is `fp1 + slope * (x - xp1)`.
fn interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    let n = xp.len();
    x.iter()
        .map(|&v| {
            if v < xp[0] {
                return fp[0];
            }
            if v >= xp[n - 1] {
                return fp[n - 1];
            }
            // last index with xp[i] <= v
            let mut lo = 0usize;
            let mut hi = n - 1;
            while lo < hi {
                let mid = (lo + hi + 1) / 2;
                if xp[mid] <= v {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            let i = lo.min(n - 2);
            let slope = (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i]);
            fp[i] + slope * (v - xp[i])
        })
        .collect()
}

/// numpy's `median`: sort, middle element, or the mean of the two middle ones.
fn median(a: &[f64]) -> f64 {
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// The series' own filter: `sorted(a)[len(a) // 2]` (upper median).
fn upper_median(a: &[f64]) -> f64 {
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    s[s.len() / 2]
}

/// numpy's `DOUBLE_pairwise_sum` (loops_utils.h.src, PW_BLOCKSIZE=128), verified
/// against the *installed* numpy 2.2.6 bit-for-bit at n=64/65/128/129/136/256/
/// 257/1000/7498/8192 and on the real 129-tap FIR kernel — see the probe in
/// `tests::pairwise_sum_matches_numpy_mean_rule`. Exactly three branches:
///
/// * `n < 8`: naive accumulation from **-0.0** (so an all-`-0` array sums to -0);
/// * `8 <= n <= 128`: eight accumulators seeded with `a[0..8]`, strided by 8,
///   combined `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))`, then the `n % 8` tail added
///   one at a time to that result;
/// * `n > 128`: split at `n/2` rounded *down* to a multiple of 8, recurse.
///
/// The associativity is the whole point: every other grouping drifts by 1 ULP on
/// the 129-tap kernel, and that one bit moves ~1/3 of the `_dominant_freq`
/// decisions downstream.
fn pairwise_sum(a: &[f64]) -> f64 {
    const PW_BLOCKSIZE: usize = 128;

    /// The `8 <= n <= PW_BLOCKSIZE` block, verbatim.
    fn block(a: &[f64]) -> f64 {
        let n = a.len();
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8usize;
        let stop = n - (n % 8);
        while i < stop {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    }

    fn naive(a: &[f64]) -> f64 {
        // numpy starts at -0.0, not 0.0.
        let mut res = f64::from_bits(0x8000_0000_0000_0000);
        for &v in a {
            res += v;
        }
        res
    }

    fn rec(a: &[f64]) -> f64 {
        let n = a.len();
        if n < 8 {
            return naive(a);
        }
        if n <= PW_BLOCKSIZE {
            return block(a);
        }
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        rec(&a[..n2]) + rec(&a[n2..])
    }

    rec(a)
}

/// `np.add.reduce` on a contiguous float64 array. numpy's ufunc machinery buffers
/// its inner loop at `NBUFSIZE = 8192` elements, so an array longer than that is
/// reduced as a **left-to-right sum of per-8192-element pairwise sums**, not as
/// one pairwise sum over the whole array. Verified bit-for-bit against the
/// installed numpy 2.2.6 at n=8191/8192/8193/8200/10000/20000/58617/95109: the
/// un-chunked form drifts by 1 ULP from n=8200 up (which is exactly the size of a
/// night's 10 Hz tachogram, so it decides the DC term `_bandpass` removes).
fn reduce(a: &[f64]) -> f64 {
    const NBUFSIZE: usize = 8192;
    if a.len() <= NBUFSIZE {
        return pairwise_sum(a);
    }
    let mut res = 0.0f64;
    for chunk in a.chunks(NBUFSIZE) {
        res += pairwise_sum(chunk);
    }
    res
}

/// `np.mean`: the buffered reduce, then a true divide by n (numpy's `_methods.py`
/// does `um.true_divide(sum, count)`; multiplying by `1/n` is a different bit
/// pattern — it already differs at n=1000 and n=10000).
fn mean(a: &[f64]) -> f64 {
    reduce(a) / a.len() as f64
}

/// `np.mean` of the signal, the DC term `_bandpass` removes.
#[doc(hidden)]
pub fn dc_mean(a: &[f64]) -> f64 {
    mean(a)
}

/// `np.hanning(M)` as numpy 2.x computes it — NOT the textbook formula:
/// `n = arange(1-M, M, 2)` then `0.5 + 0.5*cos(pi*n/(M-1))`. Mathematically the
/// same as `0.5 - 0.5*cos(2*pi*i/(M-1))`, numerically not: the argument is built
/// from an odd integer, so the last bits differ, and this window feeds every
/// spectral decision downstream.
#[doc(hidden)]
pub fn hanning(m: usize) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![1.0];
    }
    let denom = m as f64 - 1.0;
    (0..m)
        .map(|i| {
            let n = 1.0 - m as f64 + 2.0 * i as f64;
            0.5 + 0.5 * (std::f64::consts::PI * n / denom).cos()
        })
        .collect()
}

/// `np.sinc(y) = sin(pi y) / (pi y)`, 1 at y = 0.
#[doc(hidden)]
pub fn sinc(y: f64) -> f64 {
    let x = std::f64::consts::PI * y;
    if x == 0.0 {
        1.0
    } else {
        x.sin() / x
    }
}

/// The FIR taps of `_bandpass`'s no-scipy branch, built exactly as numpy does:
/// `t = arange(129) - 64`, `kern = (hi - lo) * hanning(129)`, `kern -= mean(kern)`.
#[doc(hidden)]
pub fn fir_kernel() -> Vec<f64> {
    let n = FIR_TAPS;
    let t: Vec<f64> = (0..n).map(|i| i as f64 - (n as f64 - 1.0) / 2.0).collect();
    let taper = hanning(n);
    let lo_f = (2.0 * RESP_LO) / FS;
    let hi_f = (2.0 * RESP_HI) / FS;
    let mut kern: Vec<f64> = (0..n)
        .map(|i| (hi_f * sinc(hi_f * t[i]) - lo_f * sinc(lo_f * t[i])) * taper[i])
        .collect();
    let km = mean(&kern);
    for v in kern.iter_mut() {
        *v -= km;
    }
    kern
}

/// `_bandpass`'s no-scipy branch (the venv has no scipy): DC removal, then a
/// 129-tap windowed-sinc band-pass in `np.convolve(..., mode="same")`.
#[doc(hidden)]
pub fn bandpass(x: &[f64]) -> Vec<f64> {
    let m = mean(x);
    let x: Vec<f64> = x.iter().map(|v| v - m).collect();
    convolve_same(&x, &fir_kernel())
}

/// OpenBLAS 0.3.29's Haswell `ddot` (`kernel/x86_64/ddot.c` +
/// `ddot_microk_haswell-2.c`) — the thing numpy's `DOUBLE_dot` calls, and
/// therefore the thing `np.convolve` actually executes.
///
/// Four 4-lane accumulators updated with a **fused** multiply-add, 16 elements
/// per iteration; then the fixed horizontal fold (`vextractf128`/`vaddpd` pairs,
/// `vhaddpd`); then the `n % 16` tail as plain multiply-then-add — the C tail
/// loop is *not* contracted, and modelling it as an FMA breaks 18/300 samples.
///
/// This is a CPU/build dependency, not a numpy-source one: it is what the
/// installed wheel (OpenBLAS 0.3.29, DYNAMIC_ARCH → Haswell) computes, which is
/// what the golden vectors were captured against.
fn ddot(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    debug_assert_eq!(n, y.len());
    if n == 0 {
        return 0.0;
    }
    let n1 = n & !15usize; // n & -16
    let mut i = 0usize;
    let mut dot = 0.0f64;
    if n1 > 0 {
        // a[j][l]: accumulator j, lane l. FMA: acc = x*y + acc, one rounding.
        let mut a = [[0.0f64; 4]; 4];
        while i < n1 {
            for j in 0..4 {
                for l in 0..4 {
                    let p = i + j * 4 + l;
                    a[j][l] = x[p].mul_add(y[p], a[j][l]);
                }
            }
            i += 16;
        }
        // Fold each ymm's high half onto its low half, pairwise-add the four
        // results, then collapse the remaining 2-vector.
        let s = [
            [a[0][0] + a[0][2], a[0][1] + a[0][3]],
            [a[1][0] + a[1][2], a[1][1] + a[1][3]],
            [a[2][0] + a[2][2], a[2][1] + a[2][3]],
            [a[3][0] + a[3][2], a[3][1] + a[3][3]],
        ];
        let t = [s[0][0] + s[1][0], s[0][1] + s[1][1]];
        let u = [s[2][0] + s[3][0], s[2][1] + s[3][1]];
        let w = [t[0] + u[0], t[1] + u[1]];
        dot = w[0] + w[1];
    }
    while i < n {
        dot += x[i] * y[i];
        i += 1;
    }
    dot
}

/// `np.convolve(x, kern, mode="same")`, exactly. numpy routes it through
/// `multiarray.correlate(x, kern[::-1], 'same')` → `_pyarray_correlate`
/// (`multiarraymodule.c`), which for `same` walks a reversed kernel `v` in three
/// phases: `m/2` ramp-up dots of growing length (signal pinned at 0, kernel
/// walking backwards), then the full-length dot at every signal position, then
/// `m/2` ramp-down dots of shrinking length. Every dot is `DOUBLE_dot` →
/// `cblas_ddot`, i.e. [`ddot`].
///
/// Requires `x.len() >= kern.len()`; the caller guarantees it (the tachogram
/// refuses spans under `WIN_S`, so the resampled grid is always >= 241 samples,
/// and below that numpy would swap its arguments and return a different length).
fn convolve_same(x: &[f64], kern: &[f64]) -> Vec<f64> {
    let m = kern.len();
    let nx = x.len();
    debug_assert!(nx >= m);
    let v: Vec<f64> = kern.iter().rev().copied().collect();
    let n_left = m / 2;
    let n_right = m - n_left - 1;
    let mut out = Vec::with_capacity(nx);
    for i in 0..n_left {
        let n = m - n_left + i;
        out.push(ddot(&x[..n], &v[n_left - i..n_left - i + n]));
    }
    for j in 0..=nx - m {
        out.push(ddot(&x[j..j + m], &v));
    }
    for k in 0..n_right {
        let n = m - 1 - k;
        let base = nx - m + 1 + k;
        out.push(ddot(&x[base..base + n], &v[..n]));
    }
    out
}

/// In-band DFT bins for a window of `win` samples: (k, per-sample cos/sin).
/// Built once per night; `_dominant_freq` only ever reads these bins.
pub struct BandBin {
    pub k: usize,
    pub table: Vec<(f64, f64)>,
}

pub fn band_bins(win: usize) -> Vec<BandBin> {
    let d = 1.0 / FS;
    let denom = d * win as f64;
    (0..=win / 2)
        .filter_map(|k| {
            let f = k as f64 / denom;
            if f < RESP_LO || f > RESP_HI {
                return None;
            }
            let table = (0..win)
                .map(|i| {
                    let ang = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / win as f64;
                    (ang.cos(), ang.sin())
                })
                .collect();
            Some(BandBin { k, table })
        })
        .collect()
}

/// What `_dominant_freq` decided, plus how close the decision was — the golden
/// tests use this to tell a real divergence from a coin-flip tie.
pub struct SpecDiag {
    /// Index into `bins` of the in-band maximum (np.argmax: first maximum).
    pub best: usize,
    /// winning power / median in-band power; diagnostics only.
    pub ratio: f64,
    /// `bp.max()` and `np.median(bp)` as such — the gate compares these two
    /// directly, and `max/med < 4.0` is *not* the same predicate.
    pub max: f64,
    pub med: f64,
    /// relative gap to the runner-up bin.
    pub gap: f64,
    pub n_bins: usize,
}

/// In-band powers for one window, in `bins` order — the exact array numpy
/// builds as `power[band]`. Exposed for the diagnostic probe.
#[doc(hidden)]
pub fn in_band_powers(seg: &[f64], bins: &[BandBin]) -> Vec<f64> {
    let win = seg.len();
    let m = mean(seg);
    let window = hanning(win);
    let seg: Vec<f64> = seg.iter().zip(window.iter()).map(|(v, w)| (v - m) * w).collect();

    let mut bp: Vec<f64> = Vec::with_capacity(bins.len());
    for bin in bins.iter() {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (v, (c, s)) in seg.iter().zip(bin.table.iter()) {
            re += v * c;
            im -= v * s;
        }
        // `np.abs(z) ** 2`, not `re*re + im*im`: numpy's absolute of complex128
        // is the C `hypot`, and its power ufunc special-cases exponent 2 to a
        // plain square. The two forms disagree on ~68% of samples.
        bp.push(cabs(re, im) * cabs(re, im));
    }
    bp
}

/// numpy's `npy_hypot` for the absolute value of a complex128 — the C runtime
/// `hypot` numpy links against, not Rust's `f64::hypot` (they differ in the
/// last bit; see `tests::hypot_matches_numpy_absolute`).
fn cabs(x: f64, y: f64) -> f64 {
    let (mut a, mut b) = (x.abs(), y.abs());
    if b > a {
        std::mem::swap(&mut a, &mut b);
    }
    if a.is_infinite() {
        return a;
    }
    if a == 0.0 {
        return if b == 0.0 { 0.0 } else { b };
    }
    // Scale out of the exponent range, then hypot(a,b) = a*sqrt(1+(b/a)^2).
    if a > 1.0e300 {
        return cabs(a * 0.25, b * 0.25) * 4.0;
    }
    if a < 1.0e-300 {
        return cabs(a * 4.0e300, b * 4.0e300) * 0.25e-300;
    }
    (a * a + b * b).sqrt()
}

pub fn spectrum_diag(seg: &[f64], bins: &[BandBin]) -> SpecDiag {
    let bp = in_band_powers(seg, bins);
    if bp.is_empty() {
        return SpecDiag {
            best: 0,
            ratio: f64::NAN,
            max: f64::NAN,
            med: f64::NAN,
            gap: f64::NAN,
            n_bins: 0,
        };
    }
    let max = bp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let med = median(&bp);
    // np.argmax: the first maximum.
    let mut best = 0usize;
    for (i, v) in bp.iter().enumerate() {
        if *v > bp[best] {
            best = i;
        }
    }
    let runner = bp
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != best)
        .map(|(_, v)| *v)
        .fold(f64::NEG_INFINITY, f64::max);
    SpecDiag {
        best,
        ratio: max / med,
        max,
        med,
        gap: if max != 0.0 { (max - runner) / max } else { f64::NAN },
        n_bins: bp.len(),
    }
}

/// `_dominant_freq`: only the in-band bins are ever read (max, median, argmax),
/// so evaluate exactly those.
fn dominant_freq(seg: &[f64], bins: &[BandBin]) -> Option<f64> {
    if bins.is_empty() {
        return None;
    }
    let d = spectrum_diag(seg, bins);
    // `if bp.max() < 4.0 * np.median(bp)` — the multiply is on the median, not
    // a division on the max; the two predicates disagree near the boundary.
    if d.max < 4.0 * d.med {
        return None; // require a real peak
    }
    let denom = (1.0 / FS) * seg.len() as f64;
    Some(bins[d.best].k as f64 / denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same LCG as the probe that pinned numpy's reduction rule
    /// (`scratch/mean_rule_final.py` in the porting session): u64 multiply-add,
    /// scaled into 2^-6..2^6 with a random sign, so partial sums lose bits and a
    /// one-ULP regrouping is visible.
    fn lcg(n: usize, seed: u64) -> Vec<f64> {
        let mut x = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = x as f64 / 1.8446744073709552e19; // 2^64, round-to-nearest
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut v = u * 2f64.powi((x % 13) as i32 - 6);
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if x & 1 == 1 {
                v = -v;
            }
            out.push(v);
        }
        out
    }

    /// `np.add.reduce` / `np.mean` bit patterns captured from the installed
    /// numpy 2.2.6. The n >= 8193 rows are what pin the NBUFSIZE=8192 chunking:
    /// the un-chunked pairwise form is 1 ULP off there, and a night's tachogram
    /// is ~58k samples, so that bit is the DC term `_bandpass` subtracts.
    #[test]
    fn pairwise_sum_matches_numpy_mean_rule() {
        const CASES: &[(usize, u64, u64, u64)] = &[
            (64, 0x5DEECE6AD, 0xC04DE716F831CC76, 0xBFEDE716F831CC76),
            (65, 0x5DEECE6AE, 0xC05254FB601AF162, 0xBFF20CC83F1E7784),
            (128, 0x5DEECE6ED, 0x40538AE3F8308636, 0x3FE38AE3F8308636),
            (129, 0x5DEECE6EE, 0xC05DB5D0249D2D34, 0xBFED7ADA6FBDB1D0),
            (136, 0x5DEECE6F5, 0x4042DDFD25A58830, 0x3FD1C1DF327DAD5A),
            (1000, 0x5DEECEA55, 0xC04CEE50A0A1F66A, 0xBFADA01106CAB6B2),
            (7498, 0x5DEED03B7, 0x408F18E210447800, 0x3FC0FCDD012B9DF1),
            (8192, 0x5DEED066D, 0x408C69B1737C20AD, 0x3FBC69B1737C20AD),
            (8193, 0x5DEED066E, 0x408163F8BB0BE41F, 0x3FB1636D9F9EE728),
            (10000, 0x5DEED0D7D, 0xC09A72316784847A, 0xBFC5AA2460453C9A),
            (58617, 0x5DEEDCB66, 0x409629B60AFBDE86, 0x3F98C76BDF81A7C3),
        ];
        for &(n, seed, want_sum, want_mean) in CASES {
            let a = lcg(n, seed);
            assert_eq!(
                reduce(&a).to_bits(),
                want_sum,
                "np.add.reduce drifts at n={n}: got {:#018x}",
                reduce(&a).to_bits()
            );
            assert_eq!(
                mean(&a).to_bits(),
                want_mean,
                "np.mean drifts at n={n}: got {:#018x}",
                mean(&a).to_bits()
            );
        }
    }

    #[test]
    fn arange_matches_numpy_length_and_values() {
        let g = arange(100.0, 102.5, 0.25);
        assert_eq!(g.len(), 10);
        assert_eq!(g[9], 102.25);
        assert_eq!(arange(100.0, 100.0, 0.25).len(), 0);
    }

    #[test]
    fn interp_is_linear_between_knots() {
        let xp = [0.0, 1.0, 2.0];
        let fp = [10.0, 20.0, 30.0];
        assert_eq!(interp(&[0.5, 1.0, 3.0, -1.0], &xp, &fp), vec![15.0, 20.0, 30.0, 10.0]);
    }

    #[test]
    fn hypot_probe_print_bits() {
        let re = [1e-200f64, 1e200, 1e-16, 1e16, 0.0, 3.0, 1e-300, 1e300];
        let im = [1e-200f64, 1e200, 1.0, 1.0, 1e-300, 4.0, 1e-300, 1e300];
        let mut o = Vec::new();
        for i in 0..re.len() {
            o.push(format!(
                "std={:016x} mine={:016x}",
                re[i].hypot(im[i]).to_bits(),
                cabs(re[i], im[i]).to_bits()
            ));
        }
        println!("HYPOT {}", o.join(" | "));
    }

    #[test]
    fn medians_follow_numpy() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert_eq!(upper_median(&[1.0, 2.0, 3.0, 4.0]), 3.0);
    }

    #[test]
    fn hanning_endpoints() {
        let h = hanning(5);
        assert_eq!(h[0], 0.0);
        assert_eq!(h[4], 0.0);
        assert_eq!(h[2], 1.0);
    }
}
