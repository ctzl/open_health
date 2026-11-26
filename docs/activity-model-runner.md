# Running Oura's activity-detection model on our data

`tools/run_activity_model.py` feeds our stored ring events into Oura's decrypted
TorchScript activity model — the newest `automatic_activity_detection_3_1_*.pt`
found in `notes/models/` — and prints detected activity segments — **no raw IMU /
RData needed**, it runs on the windowed signals we already sync. The filename is
never pinned: the asset set rotates with every app version, and the runner detects
its calling convention from the loaded schema.

**This is the reference implementation of the activity engine.** With a
`--features torch` build, `oura sessions` and the dashboard run the same model
**in-process in Rust** (see below); only non-torch builds shell out to this
script (the interpreter probe checks `.venv/bin/python`,
`.venv/Scripts/python.exe`, and the sibling `open_oura` venvs before falling
back to `python3`). Use the CLI for the normal path and this script directly for
`--json`, `--verbose`, or a custom `--threshold`/`--min-duration`.

```
python tools/run_activity_model.py [DB] [--tz HOURS] [--threshold P] [--date YYYY-MM-DD] [--json]
oura sessions --tz-offset 1            # same thing, via the CLI
```
`DB` defaults to `./oura.db` then `captures/ring5.db`. Requires `torch` (CPU is
fine) in the venv. The model lives in `notes/models/`.

## In-process (Rust + LibTorch, no Python at runtime)

Built with `--features torch`, `oura sessions` and the web dashboard run the
model **in-process** via the `tch` crate (`crates/oura-models/src/activity.rs`).
Under that feature **both TorchScript calls are live**: the AAD CModule's
`forward_is` *and* the `steps_motion_decoder` gait decoder — the same
Android-parity labels documented here, no Python at runtime. (The dashboard
defaults to this Rust path, with a per-model Python fallback and an
`OURA_DASHBOARD_PYTHON=1` escape hatch.) Only the legacy `oura-cli` path
(`crates/oura-cli/src/activity_model.rs`) still stubs stepmotion with NaN
features; non-torch builds keep using the Python runner.

The repo links against **its own venv's torch** rather than a system libtorch,
with **no version-check bypass**. Current alignment (the fragile part of the
build — keep all of it in lockstep):

- `tch =0.26.0` (crates/oura-cli + crates/oura-models) ↔ **torch 2.13.0+cpu**
  in the venv (Python 3.10.6);
- a **vendored torch-sys** via `[patch.crates-io]` (`vendor/torch-sys`: the
  libtch typedef `module` is renamed `tch_module` because MSVC C++20 reserves
  `module`, and torch ≥2.10 headers need C++20 even though tch 0.26 passes
  `/std:c++17`);
- `.cargo/config.toml` sets `CXXFLAGS=/std:c++20` and `LIBTORCH_USE_PYTORCH=1`.

Windows (the repo's current host) — activate the venv before building, or cargo
can't find torch's DLLs via python discovery:

```
.venv\Scripts\activate
cargo build --features torch
# at runtime the torch DLL dir must be on PATH:
#   .venv\Lib\site-packages\torch\lib   (run-dashboard.ps1 does this)
```

On macOS the only libtorch is likewise the pip wheel; there `build.rs` embeds
the venv torch `lib/` as an rpath instead of using PATH:

```
export PATH="$PWD/.venv/bin:$PATH"
export LIBTORCH_USE_PYTORCH=1
cargo build --features torch
oura sessions --tz-offset 1     # no DYLD_LIBRARY_PATH needed — build.rs bakes the rpath
```

The default build (no feature) needs no libtorch and keeps using the Python
runner. Note LibTorch prints a one-time `searchsorted` perf warning to stderr
(harmless; stdout/JSON stay clean).

Version coupling: `tch X` pins one exact libtorch (0.20→2.7, 0.22→2.9,
0.24→2.11, **0.26→2.13** ← the current pin). tch 0.27 needs torch 2.15, for
which there is no cp310 wheel, so it's not an option for this venv. `tch` and
`torch-sys` share the `links = "tch"` key: if you bump one, bump both together.

## Android parity

The decompiled Android app ships the same model family; the runner is
version-agnostic (newest `3_1_*` asset) and detects the calling convention from
the loaded schema. Its important preprocessing
steps are now mirrored here and in iOS:

- one inference per local calendar day; model time is minutes since local midnight;
- a 10-minute minimum duration;
- all three gait sub-windows from `steps_motion_decoder_2_0_0`;
- the decoder's columns reordered through Android's `StepMotionFeature` schema.

The ring exposes the decoder input directly as paired `real_step` events (0x7e/0x7f),
so raw RData is not required. On a retained replay, real sessions relabel correctly:
a hike flips from `cycling` to `hiking`, and an immersion interval becomes
`swimming`.

## Days the model cannot evaluate

Two input shapes make AAD throw inside its preprocessor instead of returning no
workouts, so both runners (`tools/run_activity_model.py` and iOS `ActivityModel`) guard
them identically:

- **A channel with no sample inside the MET span.** The model clips every series to its
  valid window and then indexes the (now empty) tensor. A NaN placeholder row is placed
  at the *first MET minute* (not at minute 0) so the channel stays present and is treated
  as missing data.
- **Fewer than 10 worn MET minutes before the last motion sample** (`min_valid_mets`,
  `non_wear_met_threshold = 0.2`). The valid window collapses to minute 0 and `met[0, 0]`
  is read from an empty tensor when the day's MET starts later. `model_would_reject` /
  `isDegenerate` mirror `get_last_valid_time` and return no sessions for such a day.
- **Duplicate MET minutes** (the ring resends a bucket across a clock re-anchor) trip the
  resampler with `index_copy_()`; both runners keep one value per minute, last write wins.

On iOS a day the model still rejects is cached as failed under its input fingerprint and
skipped, so one bad day never blocks the rest of the history or reruns on every launch.

## I/O contract (as implemented)

Leading forward args (TorchScript order): `context, user, met, stepmotion,
motion, temperature, heartrate, location=None, past_activities=None`. The
trailing settings args depend on the loaded model generation, and both runners
detect the schema (`global_settings` in the forward signature) instead of
pinning a version:

- **AAD ≤3.1.12**: three scalar tensors positionally — `probability_threshold,
  minimum_duration_minutes, allow_non_wear`.
- **AAD 3.1.15**: one optional `global_settings` 72-column row read by column —
  col 3 `allow_non_wear`, col 12 is_workout probability threshold, col 26
  min workout duration; NaN falls back to the model's baked-in default, so only
  the three known slots are set.

All series are float32 2-D, **column 0 = time in minutes** on one shared axis.

| input | cols | source (decoded events) |
| --- | --- | --- |
| met | `[t, met]` | 0x50 `met[]` (1 value/min, expanded) |
| motion | `[t, orient, motion_s, ax, ay, az, NaN(regular_motion), low_int, high_int]` | 0x47 |
| temperature | `[t, temps_c[0]]` | 0x46 |
| heartrate | `[t, mean(hr_bpm)]` | 0x80 |
| stepmotion | 12 cols — timestamp + 11 gait features | paired 0x7e/0x7f → `steps_motion_decoder` |

Output `workouts[n,9]` = `[start_min, end_min, is_workout_prob, id1,p1, id2,p2,
id3,p3]` (corrected from the spec, which called col 2 "duration"). `id`→name via
the behavior table in the script (swimming=13, walking=14, cycling=5, …).

## Gotchas learned the hard way

- **Run each day independently.** This is both the Android contract and keeps the
  minute axis exactly representable in float32.
- **Decoder order is not AAD order.** `steps_motion_decoder` emits
  `sum,y,z,total,strideFreq,…`; Android's `px.b` maps that to
  `firstFreq,firstAmp,high,mid,gait,…,sum,total,y,z` before AAD.
- **stepmotion boundary rows must span the day's MET range.** The model derives
  `last_valid_time` from every series; missing boundaries can truncate a day.
- **Ring `ring_timestamp` is device-relative deciseconds**, not unix — anchor to
  the latest event's `captured_unix` (as `oura sessions` does).
- Open calibration unknowns: ACM `avg_*` scaling (env `ACM_SCALE`, default 1) and
  temperature-probe choice (using index 0).

GPS and past-activity inputs are still absent, so sports without gait can remain
ambiguous. `--no-stepmotion` exists only to compare against the old partial pipeline.
