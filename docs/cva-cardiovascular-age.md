# Cardiovascular age from raw PPG (`cva_ppg` → `cva_raw_ppg_data` → CVA model)

When the **`cva_ppg`** feature (`CAP_CVA_PPG_SAMPLER`, id `0x0d`) is enabled, the
ring records short raw-PPG bursts overnight and emits them as **`cva_raw_ppg_data`**
events (BLE tag **`0x81`**). This is the raw photoplethysmography waveform the
cardiovascular-age (CVA) model needs — the one signal the PPG-gated models used to
be blocked on. With it captured, we can run Oura's own CVA model offline.

## Decoding tag `0x81`

Each event body is a stateful delta stream. Byte `0x80` marks the next three bytes
as a signed 24-bit **absolute** ADC sample; every other byte is a signed int8 delta
from the previous sample. The Rust decoder (`oura-protocol`,
`decode_cva_raw_ppg`) now understands these absolute markers and emits
`{"ppg_samples": [...], "n": N, "absolute_markers": M}` per event.

A full measurement spans a *burst* of consecutive events **<2 s apart** (~1500
samples ≈ 30 s — the ring streams this PPG at **50 Hz**, measured ~50.0–50.3 Hz
across runs). For exact waveform
continuity, concatenate the burst while carrying the last absolute value across
adjacent `0x81` records; the generic event-body decoder is intentionally
stateless and restarts from zero for records that do not contain an absolute
marker.

Validation that the decode is correct: the reconstructed waveform is pulsatile and
its **pulse rate was cross-checked against the independent IBI-derived heart rate
computed over the same window** — the two agree.

## The model (`cva_1_*`, the raw-PPG v1 pipeline)

Both runners pick the **newest decrypted `cva_1_*.pt`** (`find_model` in
`crates/oura-models/src/cva.rs` and in `tools/run_cva_model.py`). The newer
`cva_2_*` asset in the APK is the "cva-from-emb" variant: it takes **128-dim
embeddings**, not raw PPG, and the upstream embedding model that produces them is
not in the asset set — so it cannot run on our data and nothing here calls it.

```
forward(ppg [n, 1500] f32, sampling_rates [n, 1] f32, timestamps [n, 1] f32,
        demographics [1, 4] f32)
  -> (daily_cva, quality, variance, daily_pwv, ppg_segment_metrics[n, 11])
```

- **`ppg`**: groups of **exactly 1500** raw samples (the app drops any
  group that isn't 1500). No filtering/normalization in the app — preprocessing is
  inside the `.pt`. Feed raw reconstructed samples.
- **`sampling_rates`**: one rate per segment; the model **hard-requires 50**, which
  is the ring's streaming rate for this feature.
- **`timestamps`**: shape-validated only — they do not change the output.
- **`demographics` = `[sex, height_m, age_years, ring_size]`** — four values, where
  **sex = −1 female / +1 male / 0 other**. The runners accept a `--weight` flag for
  CLI parity with the other model runners, but the v1 model never consumes weight.
  These come from `profile.json`, not from the ring. The ring's `user_information`
  event (tag `0x5c`) *does* carry an anthropometric profile — four bytes read as
  age / weight kg / sex / height cm (`decode_user_information`), emitted with
  `_status: "inferred"` because those field meanings were inferred rather than
  recovered from a native parser — but a captured sample read as firmware defaults,
  so the CVA runners keep sourcing demographics from `profile.json` (see
  [`profiles.md`](profiles.md) §3.4).

### Outputs
- **`daily_cva`** = **vascular age (years)**. It is *anchored to the input age* and
  adjusted by PPG morphology, so on our data it lands within roughly a year of the
  chronological age. The much-younger figures in earlier drafts of this doc came
  from the retired `cva_2_*` path and are not reproducible here. **Pass real
  demographics for a meaningful absolute number.**
- **`daily_pwv`** = **pulse-wave velocity (m/s)**, PPG-only (lower = less arterial
  stiffness). The current pipeline reports well above the ~4.5 m/s the old
  `cva_2_*` path produced.
- **`quality`** = integer signal-quality score and **`variance`** = model variance;
  there is no `raw_quality` output. `ppg_segment_metrics` = 11
  per-segment features.
- The runner's JSON object (`CvaJson` in `cva.rs`, same shape from
  `tools/run_cva_model.py --json`) carries `segments`, `measurements`,
  `vascular_age`, `chronological_age`, `pwv_ms`, `quality`, `variance`.

## Running it

Native (Rust + LibTorch) — the same entry point the dashboard uses:

```
oura-models cva [DB] --sex M --age 30 --height 1.78 --ring 10 --json
```

`crates/oura-models/src/main.rs` dispatches to `oura_models::cva` (`--weight KG` is
accepted and ignored). A build with `--features torch` makes the dashboard run the
CVA model **in-process** through that code path via its `RustRunner`, falling back
to the Python script per model if the Rust path fails — see
[`model-runners.md`](model-runners.md).

Python oracle / fallback:

```
python tools/run_cva_model.py [DB] --sex M --age 30 --height 1.78 --ring 10
```

Single-night output from the old `cva_2_*` path looked implausibly young; those
results are superseded by the v1 pipeline above. The comparison vectors under
`crates/oura-models/tests/vectors/` are what pins the Rust port against the Python
runner — they are personal data, so this doc does not quote their numbers.

## Caveats
- Absolute `daily_cva` is only meaningful with the user's real demographics.
- We segment by a 2 s event-gap heuristic; the app groups by the protobuf
  `measurementStartTimestamp` (decoded by the native lib we don't run), so a borderline
  burst could be split differently. The model still produces a stable value.
- The model hard-requires `sampling_rates == 50` and the ring streams at 50 Hz, but
  the exact *training* rate isn't confirmed from the Java; segments are fed as raw
  samples (the model bakes in its canonical rate). See `tools/run_cva_model.py`.
