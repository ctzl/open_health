# SpO2: turning the ring's R-ratio into a percentage

The ring's **`spo2_r_pi_event`** (tag `0x8b`) carries, per sample, an `r`
(ratio-of-ratios) and a perfusion index (`pi`) — **not** an SpO2 percentage. Two
paths convert it:

1. **Production nightly feature** — uses `API_SPO2_EVENT` (0x6f) /
   `API_SPO2_SMOOTHED_EVENT` (0x70), which are **percentages already computed in
   the ring firmware**. We don't capture those, and that R→% math lives in firmware.
2. **"SpO2 Simple" path** — converts the `r` of `spo2_r_pi_event` in-app with a
   per-hardware quadratic. **This is the one we can reproduce** from our data.

## The formula

```
SpO2(%) = a·r² + b·r + c          clamped to [85, 100]
```

The polynomial runs inside `libecore` (`EcoreWrapper.nativeCalculateSpO2Simple`),
but the **coefficients are in the app** (`com/ouraring/oura/workitem/data/items/d.java`),
selected by ring hardware type:

| hardware | a | b | c |
| --- | --- | --- | --- |
| gen4 / oreo | −13.4 | −5.1 | 105.2 |
| cooper | −12.1 | −6.9 | 106.3 |

`pi` (perfusion index) is passed to the native algo as a quality input; there is no
in-app PI threshold. The daily value is clamped to integer **85–100**.

## The dashboard does this natively (Rust)

Since commit `4373a47` the summary pipeline computes the conversion in Rust — no
Python, no shell-out. `crates/oura-summary/src/lib.rs` (`spo2_pct`) calls the ported
ecore polynomial with the **gen4/oreo coefficients and the same clamp**:

```rust
oura_analysis::ported::spo2::spo2_simple(r, 105.2, -5.1, -13.4).clamp(85.0, 100.0)
```

Every `spo2_r_pi_event` sample is converted as it is folded into the night, and the
result feeds: the per-night SpO2 lane in the summary series, `spo2_mean` /
`spo2_min`, low-desaturation episodes below 90% (`spo2_low_events`), and the
perfusion-index keys (`perf_mean` / `perf_min` and the `perf` lane) taken from the
same events' decoded `perfusion_index` arrays. Nothing about the coefficients, the
clamp or the tags changes between the two paths — `tools/run_spo2.py` below is the
standalone oracle/checker.

**Ring 5 caveat:** its exact hardware→coefficient mapping isn't pinned down in the
decompiled app (v7.18 predates Ring 5 in that table); the two sets differ by <1% on
our data, so the result is robust either way. Default is gen4/oreo.

## Running it

```
python tools/run_spo2.py [DB] [--hw gen4|cooper] [--night]
```

On a representative night the distribution is **centred in the low 90s** (mean and
median a few points above 90), with **a few percent of samples below 90%** and a
floor pinned at the 85% clamp. (A crude generic `110−25·r` gives ~91%, which is
wrong — use the calibrated quadratic.)
