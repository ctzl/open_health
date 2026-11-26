# Running Oura's on-device models on your synced data

The ring ships Oura's analysis as encrypted on-device PyTorch models. Once
decrypted locally (into the gitignored `notes/models/`), several of them run on
the signals we already sync into `oura.db`. This page indexes the runners and what
each can and can't do.

**Engine:** the dashboard (and `oura sessions`, in a `--features torch` build)
runs **cva / activity / illness / stress in-process in Rust** by default —
`RustRunner` in `crates/oura-cli/src/dashboard.rs`, with a per-model Python
fallback; `run-dashboard.ps1` builds `--features torch` whenever `.venv` exists.
`OURA_DASHBOARD_PYTHON=1` forces the Python path; the plain Python
`tools/run_*.py` runners are the fallback, not the default engine.

## Tools

| tool | model(s) | output | status |
| --- | --- | --- | --- |
| `oura sessions` / `tools/run_activity_model.py` | `automatic_activity_detection_3_1_*` (newest; unpinned — the asset set rotates) | activity/workout segments + type label | ✅ runs |
| `tools/run_sleep_model.py` | `sleepstaging_2_6_0` (NSSA, the official app's stager) | DEEP/LIGHT/REM/WAKE hypnogram per ring bedtime + efficiency | ✅ runs (custom ops + classifier, see [sleep-staging.md](algorithms/sleep-staging.md)) |
| `tools/run_models.py bdi` | `sleepnet_bdi_0_4_0` | hypnogram + apnea/breathing-disturbance | ✅ runs |
| `tools/run_cva_model.py` | `cva_1_*` (newest raw-PPG variant; the from-emb `cva_2_*` needs embeddings we can't produce) | cardiovascular (vascular) age + pulse-wave velocity | ✅ runs (needs `cva_ppg` on — see [cva-cardiovascular-age.md](cva-cardiovascular-age.md)) |
| `tools/run_spo2.py` | (no model — Oura's calibration) | overnight SpO2 % | ✅ runs (see [spo2-calibration.md](spo2-calibration.md)) |
| `tools/run_models.py daily_medians` | `daily_medians_1_1_0` | HRV/HR/temp/MET daily medians | ⚠️ runs but needs **awake** HRV (ours is nocturnal) |
| `tools/inspect_models.py` | all | dump each model's `forward()` schema | helper |

The two sleep models cross-validate (close DEEP/LIGHT/REM/WAKE %); that agreement
also fixes the BDI stage-column order to `[AWAKE, LIGHT, REM, DEEP]`.

## What's blocked, and why (data availability, not wiring)

Of the ~26 newest-version models, most are blocked on a signal we can't get:

- **raw PPG waveform** — `cva_1_*` was here until `cva_ppg` was enabled (now
  `cva_raw_ppg_data` 0x81 supplies it). Still blocked for `halite` (needs the same
  raw PPG wired) and `whr` (needs raw PPG + raw accel).
- **raw ACM / stepmotion gait** — `step_counter`, `awhr_imputation`,
  `awhr_profile_selector` (RData-gated; entitlement-locked). `steps_motion_decoder`
  is **not** blocked: the activity pipeline decodes the paired `real_step` events
  (0x7e/0x7f) and runs `steps_motion_decoder_2_0_0` (see
  [activity-model-runner.md](activity-model-runner.md)).
- **bioZ + EDA sensors** — `atlas` (not in our event stream).
- **cloud-computed scores** — `stress_resilience`, `training_stress_score`.
- **cycle / pregnancy inputs** — `popsicle`, `pregnancy_biometrics`.
- **pipeline deps** (need another model's / cloud output first) — `cumulative_stress`,
  `cva_calibrator`, `stress_daytime_sensing`.

(The full per-model matrix lives in the local-only `notes/model-usage-map.md`.)

## Notes
- All runners read `oura.db` by default and resolve model weights from
  `notes/models/`, the sibling `../open_oura/notes/models/`, or
  `$OURA_MODELS_DIR` (all gitignored — no model weights or personal data are
  committed).
- The models are Oura's proprietary, decrypted artifacts; they are **not** included
  in the repo. You supply your own locally.
