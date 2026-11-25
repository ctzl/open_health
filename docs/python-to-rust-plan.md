# Python → Rust port plan (tasks 1 + 2, approved by user 2026-11-23)

Goal: kill the recurring Python cost in the enrich path. Task 1 = port the four
uncached model scripts to Rust. Task 2 = port the run_sleep_model.py plumbing.
Playbook = the ibi-correct one (crates/ibi-correct): golden vectors from the
Python oracle, bit-exact/f32-mirrored Rust, shim or server wiring, bench.

## Measured baseline (cold, full history, real DB <path-to>/oura.db, serial <ring-serial>)

| script | time | notes |
|---|---|---|
| run_sleep_model.py | 169.8 s (121 nights) | cached per-night via nssa_cache.json; ~1.4 s/night cold |
| run_cva_model.py | 12.8 s | NO cache, full history every run |
| run_illness_model.py | 10.7 s | NO cache |
| run_stress_events.py | 7.7 s | NO cache; chains run_stress_model.py subprocess |
| run_activity_model.py | 5.7 s | NO cache |

`import torch` floor = 2.0 s per script (numpy-only floor = 0.16 s).
Server runs all five in parallel per dirty summary (dashboard.rs:503-510,
PythonRunner at dashboard.rs:452, invoked from oura_summary::build_summary at
dashboard.rs:585). ModelRunner trait exists — implement a Rust runner beside it.

## Key facts / code locations

- All 4 models are TorchScript .pt (torch.jit.load): cva_1_*.pt,
  automatic_activity_detection_3_1_*.pt, illness_detection_0_5_1.pt,
  astd_event_detection_0_1_1.pt (+ stress_daytime_sensing via
  run_stress_model.py). notes/models/ (decrypted), sibling open_oura/notes/models.
- NO candle precedent (hourly_hr.rs "candles" = chart candles, false lead).
  The real precedent for TorchScript-in-Rust: oura-cli Cargo.toml feature
  `torch = ["dep:tch"]`, tch 0.22 aligned to venv torch 2.9.0, build.rs rpath
  comment near line 40-50. Windows: torch DLLs live in
  .venv/Lib/site-packages/torch/lib (PATH at runtime).
- astd_event_detection internals are documented in run_stress_events.py
  docstring (classify thresholds ±0.5/±0.4, N=4 bins in 65-min window,
  merge <30 min, int32 day-relative timestamps, zero-events model bug →
  treat as empty). Candidate for hand-port without tch.
- cva decode: tag 0x81 delta-stream (0x80 + 3-byte signed24 absolute, else
  int8 delta), runs split at >2 s gaps, 1500-sample segments, rates=50.0,
  demographics [sex, height_m, age, ring_size].
- run_sleep_model.py hotspots (cProfile, 121 bedtimes, profiler-inflated):
  ring_bedtimes 6.2 s tottime, collect_nssa_inputs 3.3 s + children,
  score_window 3.3 s, row JSON decode 2.2 s, breath_series/_dominant_freq
  2.0 s, correct_ibis 4.0 s (already Rust via persistent worker shim in
  tools/ibi_correction.py — the pattern to copy).
- Server already has a Rust signal-based fallback stager:
  oura_summary::stage_nights_from_signals (dashboard.rs:524).
- run_sleep_model.py args: `db -5 --json --batch --serial <hex>` (tz is a bare
  float like -5, NOT -05:00), bedtime ranges on stdin (pairs.json format
  [[start,end],...] ring deciseconds).
- Golden vectors from real DB are OK to commit (precedent:
  crates/ibi-correct/tests/vectors/real_*.json).

## Task 1 — four model scripts → Rust (recurring ~13 s wall per enrich)

1. New crate `crates/oura-models`: one module per model (cva, activity,
   illness, stress), binary subcommand per model emitting the SAME JSON shape
   as the Python script (server contract = ModelOutputs, dashboard.rs:455-515).
2. Capture golden outputs (EXACT commands, run from open_health root, venv
   python; DB = ../open_oura/oura.db, profile.json beside it):
     python tools/run_cva_model.py ../open_oura/oura.db --json      > cva_golden.json
     python tools/run_stress_events.py ../open_oura/oura.db --tz -5 --json --serial <ring-serial> > stress_golden.json
     python tools/run_illness_model.py ../open_oura/oura.db -5 --json --serial <ring-serial> > illness_golden.json
     python tools/run_activity_model.py ../open_oura/oura.db --tz -5 --json --full --serial <ring-serial> > activity_golden.json
   (illness needs the -5 tz positional + --serial — its test uses exactly those;
    activity WITHOUT --json prints a table, not JSON. Each model's capture args
    must match that test's oracle_args() — check the test before capturing.)
3. Port data plumbing (DB reads, PPG decode, feature builders, epoch dating
   via epoch_time equivalent — check for existing Rust epoch code in
   oura-core/oura-summary first) + inference via tch behind `torch` feature.
   Hand-port astd segmentation logic if tch proves painful on Windows.
4. Server: RustRunner impl of ModelRunner; try Rust, fall back to PythonRunner
   on any error (same fallback philosophy as stage_nights_from_signals).
5. Verify: Rust output == golden vectors (ints exact; floats tol 1e-4 rel).
   Bench: target ~2-3 s total vs 37 s CPU / 13 s wall now.

## Task 2 — run_sleep_model.py plumbing → Rust (backfill/ring-switch win)

1. Port ring_bedtimes, collect_nssa_inputs (incl. breath_series/
   respiratory_series/_dominant_freq), score_window into a crate (extend
   ibi-correct or new `nssa-plumbing` module).
2. Mirror numpy float32 exactly (the ibi lesson: f64 diverges at tolerance
   boundaries).
3. Shim: Python calls a persistent Rust worker over JSON stdio (copy
   tools/ibi_correction.py _rust_worker pattern).
4. Golden vectors: capture collect_nssa_inputs + score_window outputs for the
   121 real-DB bedtimes from the real DB; require bit-exact.
5. Bench cold full run: target 170 s → ~60 s (plumbing only; torch inference
   stays Python).

## Execution order + progress

- [x] P0: golden JSON captured for all 4 scripts → crates/oura-models/tests/vectors/
      (background job proc_28690add56ad; activity/illness scripts read by their port agents)
- [x] skeleton: crates/oura-models added to workspace; Cargo.toml (rusqlite bundled,
      serde, optional tch =0.22.0 behind `torch` feature), stub modules cva/activity/
      illness/stress/common/torchmod; main.rs dispatches subcommands.
      tch Windows linkage probe: proc_a68dbdd72d0e (TORCH_PYTHON=venv python).
- [x] T1a cva: DONE + parent-verified 2026-11-23 (8/8 lib + 4/4 integration
      tests pass vs fresh golden 651/879; real tch inference, byte-identical
      JSON). Agent: crates/oura-models/src/cva.rs + tests/cva_test.rs.
- [x] T1d illness: DONE + parent-verified 2026-11-23 (23 model inputs
      bit-identical; golden JSON matches under torch). Parent fixed an
      off-by-two: biomarkers vec was built from tuple idx 2 but looked up by
      absolute idx (2,4,5,6) -> every slot shifted. Golden MUST be captured
      with `-5 --json --serial ...` (matches the test's oracle_args()).
- [x] T1b stress: DONE + parent-verified 2026-11-23 (26/26 tests; byte-identical
      to oracle incl. 4 real stress events). Parent switched test+golden to
      `--tz -5 --serial <ring-serial>` (no-serial capture had ZERO events —
      multi-ring mixing kills DSA; serial config exercises the event path) and
      fixed the shape guard (looked for event keys at day level, not in
      events[]). DSA NN stays a Python subprocess (or --dsa-json pre-supplied).
      (each agent owns ONLY its src/<model>.rs + tests/<model>_test.rs)
- [x] T1c activity: DONE + parent-verified 2026-11-23 (10/10 tests BOTH with
      and without --features torch; FULL oura-models --features torch suite
      42/42 green: activity 10 + stress 26 + cva 4 + illness 2; byte-identical
      to oracle on FULL history
      incl. real AAD + step-motion gait decode via tch). Agent found+fixed a
      real bug (gait decoder output rows needed decoder_to_aad's 11-col
      permutation — sessions 4->29). Parent re-captured golden with
      `--tz -5 --serial` (original capture used default tz=1 -> 6h session
      offset) and gated the probe test: torch build runs --no-stepmotion to
      match the gait-stubbed probe; packet-collection asserts run non-torch
      only. Note: #[cfg] on expressions is unstable — use cfg!() in tests.
- [x] T1c re-verified 2026-11-24: the "golden 47 vs rust 29" gap was a stale
      comparison, not a port bug — 47 is the *all-serials* count from the Nov 23
      DB state (it survives in hermes/cache/scratch/sidecar_bak/
      activity.all.units.json); the ring-B golden command (`--tz -5 --full
      --serial <ring-serial>`) yields 29 on today's DB and Rust reproduces
      those byte-for-byte. Fresh captures, 0 diffs everywhere (count, order,
      labels, durations): ring B 29/29 — including the dashboard's default args
      (no --full) — both rings 55/55, ring A 33/33. Timings: full history
      Python 54.0 s vs Rust 16.9 s; default window 26.3 s vs 15.8 s.
      DATA DRIFT, not code: these goldens are snapshots of a syncing DB.
      Re-captured activity ORACLE_JSON + its `ran` count (38->39) because the
      ring gained 2 clock anchors since Nov 23 ([1,230,5] -> [1,232,5]), and
      re-captured illness_golden.json (anchor day 2026-11-23 -> 2026-11-24; the
      Rust output was token-identical to a fresh Python run, so the vector was
      stale, not the port). illness feature_builder_matches_python_inputs now
      self-skips (CAPTURED_ANCHOR_DATE is still 2026-11-23) — re-capture its
      23-tensor fixture when convenient. Full oura-models suite green BOTH ways:
      50 tests (lib 8 + activity 10 + cva 4 + illness 2 + stress 26), with and
      without --features torch.
- [x] T1c-inc incremental activity: DONE + parent-verified 2026-11-24
      (crates/oura-models/src/activity_incremental.rs + tests/
      activity_incremental_test.rs, 10/10; full suite 60/0 both feature
      configs). Mirrors Python's sidecar cache (model_state/activity/
      units.json): day-level fingerprint invalidation, per-day state
      round-trip, gait re-decode only when its inputs change. Cold parity
      proven against the full-path RustRunner on the live DB; warm re-run
      2.1 s (vs 17.7 s full Rust, 11.0 s Python). Dashboard now calls this
      path (dashboard.rs activity runner branch); live log after restart:
      `activity (rust): 12583 ms` first run after a 1972-event sync, all
      five models on the Rust path (cva 3.7 s, stress 5.8 s, illness 7.8 s;
      sleep still Python).
- [x] T1c-guard golden drift guards: DONE + parent-verified 2026-11-24
      (activity_test.rs + illness_test.rs only). Read-only rusqlite probes
      (MET/day-window/clock-anchor counts; illness: per-family row counts +
      anchor day) run BEFORE prepare(); on drift the tests print
      "SKIP …: golden stale — re-capture per the module header (…)" and
      pass; real model divergence still fails. ACTIVITY_GUARD=1 /
      ILLNESS_GUARD=1 print the pasteable refreshed constants. Guard cost
      ms-level (activity) / ~0.3-0.8 s (illness), no torch. Goldens were
      already stale and were re-captured (clock now [1,237,5], anchor
      2026-11-24). Guard demo verified: perturbed constants -> both
      binaries short-circuit (25.6 s -> 0.06 s, 24.7 s -> 0.52 s), 0 failed.
- [x] T2: DONE + parent-verified 2026-11-24 (cargo test -p nssa-plumbing
      --release 21/21: 16 unit + 5 golden; all 121 nights bit-exact for
      is_dated/ring_bedtimes/collect_nssa_inputs/breath_series (30,238
      windows)/score_window). np.mean rule pinned empirically (numpy 2.2.6):
      true divide by n; add.reduce buffers >8192 (left-to-right sum of
      per-8192 pairwise sums); DOUBLE_pairwise_sum = 8 accumulators a[0..8]
      strided by 8, ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)) + one-at-a-time n%8
      tail, split at n/2 rounded DOWN to multiple of 8, n<8 naive from -0.0.
      Locked by unit test with hex bit patterns. Timings 121-night plumbing:
      Python 12.37 s vs Rust 5.13 s (collect 3.2x; breath 0.83x slower —
      scalar ddot is the price of bit-exactness; it's 1.5 s of 170 s).
- [ ] T1e: RustRunner wiring in dashboard.rs + fallback + benches + commit (me, after ports)
- [ ] T2c: shim wiring into run_sleep_model.py + bench + commit (me, after T2)

## Pitfalls

- tch on Windows — SOLVED (probe 10): torch 2.13.0+cpu in venv + tch =0.26.0
  + VENDORED torch-sys at vendor/torch-sys ([patch.crates-io] in workspace
  Cargo.toml) with libtch typedef `module` → `tch_module` (MSVC C++20 reserves
  `module`; torch >=2.10 headers need C++20 even though tch 0.26 passes
  /std:c++17). .cargo/config.toml [env] sets CXXFLAGS="/std:c++20"
  LIBTORCH_USE_PYTORCH=1 — build with venv activated.  C++ layer compiles clean; remaining errors were tch 0.26 Rust API usage in
  oura-models (IValue is tch::IValue; Kind is tch::Kind::Float / Kind::Double
  CamelCase variants — there is NO tch::kind::FLOAT64; Tensor::double_value
  takes &[i64]; runtime needs .venv/Lib/site-packages/torch/lib on PATH).
  History: 0.22/0.26 C++17 fail vs >=2.10 headers (probes 3,6,8); 0.27 fails
  vs 2.14.1 (probe 7); torch 2.15 has no cp310 wheel. links="tch" → pins in
  lockstep (oura-cli too). GOLDENS: re-captured under 2.13 (exact commands in
  Task 1 step 2; activity needs --json --full).
  NOTE: `cargo … | tail` hides cargo's exit code — capture $? via a log file.
- TorchScript quirks: astd int32 timestamps + zero-event crash; cva hard
  rate==50 validation; illness NaN handling (torch.full nan inputs).
- Never trust a cargo "Finished" with a running server holding oura.exe —
  kill via powershell Stop-Process -Id (taskkill //F mangles under MSYS).
- Native tools need C:/ paths, not MSYS /c/ paths.
- Commit to `fork` remote (ctzl), not origin.
