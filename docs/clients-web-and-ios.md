# Two clients, one core — keep them in sync

open_health has **two user-facing clients that render the same health data**. When you
add or change a feature, you almost always have to touch **both**. This is the map.

| | Web dashboard | Native iOS app |
| --- | --- | --- |
| Where | `dashboard/web/` (vanilla HTML/CSS/JS) served by `crates/oura-cli/src/dashboard.rs` | `apps/ios/OuraApp/` (SwiftUI) on `crates/oura-core` (UniFFI) |
| Entry | `oura dashboard` → `http://127.0.0.1:8090` | `apps/ios/OuraApp/build_run.sh` (model-free) / `build_run_torch.sh` (on-device models) |
| Render code | five pages under `dashboard/web/`, all routed by `dashboard.rs`: `index.html` + `app.js` + `styles.css` (the day/report page), `ring.{html,js,css}` (ring diagnostics, `GET /api/ring-diagnostics`), `logs.{html,js,css}` (the dashboard's own activity log — syncs, model runs, live-HR sessions, feature toggles), `blood.{html,js,css}`, `dna.{html,js,css}`, plus the shared ring/profile switcher `profile.js` and `icons/` | `OuraApp.swift`, `Reports.swift`, `Components.swift`, `HeartRate.swift`, `Theme.swift` |
| Models run via | in `--features torch` builds (the dashboard included): in-process Rust through `crates/oura-models` (`RustRunner`) for cva/activity/illness/stress; Python torch runners (`tools/run_*_model.py`) for sleep staging + BDI, and for everything in a non-torch build | on-device `.ptl` (`TorchBridge.{h,mm}` + `SleepStaging`/`CvaModel`/`ActivityModel.swift`) |

## The one shared brain: `crates/oura-summary`

`oura_summary::build_summary()` computes **the summary JSON both clients render** — vitals,
per-night stats, the digest, the MET activity profile, steps/kcal, device health. The web
calls it in `dashboard.rs`; iOS calls it through `oura-core`'s `summary_json()` FFI. The
models are injected via the `ModelRunner` trait (web: `RustRunner` in `--features torch`
builds — `OURA_DASHBOARD_PYTHON=1` forces the whole runner back to `PythonRunner` for A/B
benchmarks; `PythonRunner` in non-torch builds; iOS: `NoModelRunner` + the on-device torch
code). Any model that fails in Rust falls back to that model's Python script.

The ported ecore primitives (`ported::{spo2, temperature, metabolic, baseline, sleep_debt,
hrv}`: SpO₂ calibration, nightly skin temperature, Schofield BMR (→ `total_kcal`), Jackson
VO₂max, steps→distance, the annealing-EMA personal baseline behind each vital's `delta_pct`)
come from `oura-analysis`, which is an **upstream git dependency pinned to a rev** — it is not
a workspace crate, so you cannot edit it from this checkout. The shared math you *can* change
lives in this repo's workspace members (`Cargo.toml`): `oura-cli` (dashboard + CLI, DNA/blood
glue), `oura-ffi`, `oura-core` (iOS FFI), **`oura-summary` (the shared summary: sleep
metrics, autonomic-by-stage, ring time, hourly HR, research export)**, `oura-dna`,
`ibi-correct`, `nssa-plumbing` and `oura-models` (the in-process torch models). `oura-store`
is vendored at `vendor/oura-store`; `oura-link`/`oura-protocol` are patched to the sibling
`open_oura` checkout.

**So the rule of thumb:**

- **A new computed metric / field** → add it in `oura-summary` (`build_summary`); both clients
  receive it in the JSON. Then render it in **both** `app.js` and `OuraApp.swift`. Only touch
  the upstream `ported::` primitives through a PR to `open_oura` + a rev bump here.
- **A new visualization / UI** (no new data) → do it in **both** `app.js` and `OuraApp.swift`.
- **A new model** → wire **all three** runners: a `tools/run_*_model.py` (used by
  `PythonRunner`), a module in `crates/oura-models` that `RustRunner` dispatches in-process
  (the default web path; it keeps the same argv contract as the Python script and is
  golden-tested against it, with a per-model Python fallback), **and** an `oura_*` function in
  `TorchBridge.mm` + a Swift `*Model.swift` that builds the same input tensors and folds the
  result into the summary. Sleep staging and the BDI apnea head are still Python-only on the
  web — port them there before dropping that leg.

## Feature ↔ feature correspondence

| Feature | Web (`app.js`) | iOS (`OuraApp.swift`) | Data (JSON key) | Model |
| --- | --- | --- | --- | --- |
| Digest headline | `load()` digest | `RootView` digest | `digest` | — |
| Vitals (HRV/RHR/temp/SpO₂) | `renderTiles` (HRV, resting HR, sleep efficiency, vascular age) + the SpO₂/temperature panels in the day view | `VitalCell` (HRV, heart rate, skin temp, blood O₂) with tap-through trend sheets (`VitalTrendView`) | `vitals`, `nights[]` | — |
| **Sleep score + breathing rate on the report** | — (the day view shows the RSA breathing lane, not a score) | `SleepScoreCard(score:, breathRate:)` in `SleepReport` | `nights[].{sleep_score,breath_rate}` (only for fully staged nights) | — (`oura-summary::sleep_score`, needs a hypnogram) |
| **Unified day (night + activity)** | `renderDay`, `dayCard` | `TodayCard` | `nights[]`, `activity*` | — |
| **Full-page sleep report** (polysomnograph + clinical metrics + interpretation) | `openDayPage`→`sleepSection` (`tlBox` lanes from `nights[].series_t`, `hypnoSvg`) | `DayReportView`→`SleepReport`, `Polysomnograph` (Reports.swift) | `nights[].{stages_full,series,metrics}` | SleepNet |
| **Sleep debt** (14-day card + cumulative debt / total sleep detail) | `renderSleepDebt`→`openSleepDebt` | `SleepDebtCard`→`SleepDebtDetail` | `sleep_debt`, grouped by wake date including naps | SleepNet |
| **Day JSON export** (iOS: the open tab's section; web: the whole day — night + debt, daily stats + MET profile + workouts, and the heart-rate slots) | `openDayPage` "Export JSON" → `exportDayJson` (download) | `DayReportView` export menu → `DayExport` (copy / share sheet) | same sections as the report | — |
| **Research export** (raw CSVs for a time window: every beat, 30 s movement, SpO2 ratio, skin temperature, on the corrected clock; README + manifest in the ZIP) | "Research export" panel → `GET /api/research-export?start&end&streams&tz` | Help & diagnostics → Research export → `ResearchExportView` (share sheet) | `oura_summary::research_export` (`build` + `to_zip`), also `oura research-export` | — |
| **Full-page activity report** (24h MET profile + intensity metrics) | `openDayPage`→`activitySection`, `movementSvg` | `DayReportView`→`ActivityReport`, `MetProfile` (Reports.swift) | `activity_profile`, `activity_daily`, `activity` | AAD |
| **Heart rate across the day** (per-slot 5th–95th band + median tick; beats vs 5-min averages) | `openDayPage`→`heartSection`, `hrTimelineSvg` (`GET /api/hourly-hr?minutes=`; **5 min / 15 min / 1 h switch**, 15 min default) | `HeartRate.swift` (`HourlyHR`, hourly only) | `oura-summary::hourly_hr::{hr_bins, hourly_hr}` — served to the web by `GET /api/hourly-hr` (not in the summary JSON), to iOS by the `hourly_hr_json` FFI | — |
| Stage breakdown | `stageBar` | `StageBar` (Reports.swift) | `nights[].{deep,light,rem,wake}_pct` | SleepNet |
| **Autonomic recovery by stage** (mean HR/HRV in deep/light/REM) | `sleepSection` autonomic grid | `SleepReport` `autonomicGrid` | `nights[].autonomic` | SleepNet (needs hypnogram) |
| **Cardiovascular age** | `renderCardio` | Cardio section | `cardio` | CVA (web: `oura-models` in-process Rust in torch builds, Python otherwise · iOS: `CvaModel`) |
| **VO₂max estimate** | `renderCardio` | Fitness section | `fitness.vo2max` | — (Jackson, model-free) |
| Movement ridge | `ridgeSvg` | `MovementRidge` | `activity_profile` | — (MET, model-free) |
| **Activity sessions / workouts** | `openActDetail` (session) | workouts section | `activity` | AAD (web: `oura-models` in-process Rust in torch builds, Python otherwise · iOS: `ActivityModel`) |
| Steps / active calories / **distance** | activity report stats | activity day stats | `activity_daily` (incl. `distance_m`) | — |
| Previous days browser | `openDaysBrowser` → `openDayPage` | `AllDaysView` → `DayDetailView` | day keys | — |
| Device & data health | `renderDevice` | device section | `device`, `streams` | — |
| **Battery history** (level over the days with readings; charging runs drawn apart) | `renderDevice`→`batteryChart` | — | `device.battery_history` (`[unix, percent]`, 14 days) | — |
| **Ring diagnostics page** (`/ring`: one card per registered ring — identity + key provenance, 14-day wearing-coverage heatmap with hours past the last sync hatched as *unknown*, recording gaps classified power-off / charging / off-finger, per-stream last-seen, battery curve with hover crosshair, the log entries that mention the ring) | `ring.js` (`GET /api/ring-diagnostics`) | — (its pieces live scattered across the device section + Technical reports) | `crates/oura-cli/src/dashboard.rs` (`ring_diagnostics`), sidecar key provenance | — |
| **Stress / restored events** (daytime-HRV DSA chain, 15-min bins vs your own neutral zone) | `renderStress` | — | `stress.days[].events` | stress (web: `oura-models`/`tools/run_stress_model.py` · iOS: none) |
| **Apnea (BDI) + RSA breathing-rate lanes** on the polysomnograph | `apneaSvg`, "Breathing · RSA" lane | — (iOS shows only the single `breath_rate` number in `SleepScoreCard`) | `nights[].{apnea,breath}` | SleepNet-BDI (Python) / IBI-derived RSA |

## The day is one unit — pair night + activity by *wake date*

Both clients render **one "day" = last night's sleep + that day's activity**, drillable
into either half and browsable back through previous days. The hero on each home screen is
the most recent day; "show all N days" (web: `openDaysBrowser`; iOS: `AllDaysView`) opens
the rest, each as a combined night+activity detail.

The **pairing rule matters and must stay identical across clients**: nights are labelled by
their **onset** date (the evening you went to bed), so an overnight sleep that crosses
midnight belongs to the *next* day's morning. A day `D` pairs with the sleep you *woke from*
on the morning of `D` — the night whose **wake date** is `D`, not whose onset date is `D`.
This lives in `wakeYmd()` (web `app.js`) and `Summary.wakeYmd` (iOS `Models.swift`); keep the
two in lockstep. `nightForDay`/`night(forDay:)` pick the longest in-bed night for a morning so
a nap doesn't shadow the real sleep.

## Where the two clients diverge

- **Automatic sleep analysis (iOS)**: opening or syncing the app reuses saved sleep
  results and analyzes only the latest night when its result is missing. A changed
  bedtime window counts as missing. Older missing nights require “Refresh analysis”
  in their Sleep tab. Automatic runs do not prune the historical model cache.
- **iOS compact disclosures**: symptom radar initially shows its status and explanation;
  “View details” reveals measurements and personal ranges together. Ring troubleshooting
  is a "Help & diagnostics" card of grouped rows (check data, share report, export raw
  data, technical reports on their own page) with the reset kept apart as a destructive row.
  These phone-layout changes are intentionally iOS-only; the underlying health data is unchanged.
- **Manual analysis refresh (iOS)**: each day's Sleep and Activity tabs can rerun their
  respective model from saved ring data. Sleep uses the displayed night's bedtime window
  (paired by wake date); activity uses the selected calendar day. The action bypasses
  that result's cache, preserves other days, and publishes successful results to the open
  report and summary cache. This is separate from syncing new data from the ring.
- **Raw ring data export (iOS)**: Help & diagnostics → “Export raw ring data” writes a
  self-contained copy of the phone's SQLite store (`oura-core::export_database` →
  `oura-store`'s `Store::export_to`, a rusqlite page-level `Backup` — deliberately **not**
  `VACUUM INTO`, so it does not rebuild every table and index; the copy runs in one read
  transaction, so a sync happening at the same time cannot tear it, and the output is switched
  to `journal_mode=DELETE` so it stands alone. No auth key — that stays in the Keychain) and
  hands it to the share sheet. On a computer it is a normal `oura --db <file> …` input, so any
  on-phone analysis can be reproduced exactly (`oura --db exported.db dashboard --tz-offset 2`).
- **Ring clock sanity (all three twins)**: two anchors of one boot must agree on the
  counter rate. A counter that *stalled* between them (ring off; wall clock ran ahead)
  takes the later anchor's offset from the stall on, using the download time to pick the
  side. A counter that ran *faster* than wall time (a fresh ring's erratic first days,
  weeks of ds in an hour) makes everything between the two anchors **undated**: the
  summary, the model runners (Rust `oura-models` or Python) and the iOS models all leave
  that data out instead of scattering it over months of phantom days.
- **On-device model caches (iOS)**: every cache file carries a store digest (row count,
  last id, anchor count, last anchor id). When it matches, activity and illness return
  their cached results without streaming the store; illness only scans the tags it
  uses; CVA runs at most once per local day (vascular age moves on a scale of months and
  every sync adds PPG segments). Activity days the model rejects are cached as failed
  under their input fingerprint and skipped until a forced refresh or a pipeline bump.
- **Ring clock diagnostics**: the summary JSON carries a `clock` block (per-boot ds range,
  sync window, anchor count/sources, `undated_nights`, `warnings`) and each night carries
  `wake_ymd`, `start_unix`, `end_unix`, `clock_source`. iOS renders the warnings on Home,
  the per-boot table under Technical reports, and appends it to the shared diagnostic report.
- **Home layout**: same day-unit model on both, but iOS uses thomas.md Quiet Ink (warm paper,
  hairlines, serif titles) while the web still uses its own teal/card theme. Match *data/features*,
  not pixel-for-pixel layout. iOS opens details as sheets; the web as stacked `<dialog>`s.
- **BLE sync**: iOS syncs **natively** — `RingSync.swift` (CoreBluetooth `BLETransport`)
  drives the Rust `RingSession` FFI (`oura-core`) to authenticate + drain into a writable
  DB. The web dashboard has **no** BLE; it reads a DB produced by the desktop `oura sync`.
  Both ultimately run the SAME `oura-link` `OuraClient<T: Transport>` over a different
  transport (btleplug on desktop, CoreBluetooth-over-FFI on iOS). Ring 5 history payloads
  are coalesced into 32 KB chunks before crossing UniFFI; control/summary frames stay
  immediate, and diagnostics log only the aggregate frame/byte count rather than raw
  sensor payloads. This mirrors Android's `ExtGetEvent` raw-buffer accumulation without
  sacrificing the Rust client's 4,096-event cursor checkpoints.

## Sleep metrics: two code paths, one algorithm — keep them in sync

The clinical sleep metrics (onset/REM latency, WASO, awakenings, cycles, fragmentation) and
sleep debt are computed **twice** and must stay identical: once in Rust (`oura-summary`
`sleep_metrics` / `smooth_stages` / `count_bouts` / `count_periods` + `sleep_debt_summary`)
for the web, and once in Swift (`Reports.swift` `Sleep.metrics` / `Sleep.smooth` +
`Summary.stagedSleepDebt`) for iOS. Sleep debt groups every sleep session by wake-date,
including naps in that day's total, then evaluates 14 calendar days with at least five
valid days; this matches the decompiled Android input and UI. The nightly **sleep need**
is personalized like Oura's (`SleepDebtInput.longTermSleepTimeAvgSeconds` ← the long-term
`sleepTimeAvg` baseline): each day's need is the mean of that user's daily totals over the
previous 90 days, IQR-outlier-filtered, clamped to 7–9 h, rounded to 15 min, causal (a
night never sets its own need), with an 8 h fallback below 14 valid history days — see
Rust `sleep_need_s` and its Swift mirror `needS(on:)` in `stagedSleepDebt`. The web reads it from the
summary JSON; iOS recomputes from the
**on-device** SleepNet hypnogram (`NightRow.stages`), because iOS runs `build_summary` with
`NoModelRunner` (no server-side staging), so the FFI `stages_full`/`metrics` are empty there.
The raw signal series (`nights[].series`) DO come from the FFI on both. If you change the
smoothing window or a metric definition, change **both** implementations. Oura's own stagings
(`staging_source` `nssa` or `ring`) are never smoothed or deep-refined (`is_oura_staging` /
`Sleep.isOuraStaging`); only the signal-derived estimate is. Both clients also draw
`stages_5min`, the official app's 5-minute chart (`aggregate_30s_to_5min` / `Sleep.aggregate5Min`).

**Autonomic-by-stage** (mean HR/HRV per sleep stage) is the same story: Rust
`autonomic_by_stage` fills `nights[].autonomic` for the web; iOS recomputes in Swift
(`Sleep.autonomic`) from its on-device hypnogram since that FFI field is null under
`NoModelRunner`. There is **no deliberate timing difference left**: both map each HRV/HR
sample to a stage by its **true timestamp** from `series_t` (`hrv_event` gives
`interval_min`-spaced samples), and iOS falls back to the **index-fraction** mapping only when
`series_t` is missing (a legacy summary). Prefer `series_t` in new code on both clients. We
expose per-stage means (esp. deep-sleep HRV) rather than an overnight HRV "slope":
nocturnal HRV is stage-driven (deep ↑, REM ↓), so a slope tracks stage order, not recovery —
which is why Oura's own app has no per-night HRV trend either.

## Packed HR/HRV samples are stamped at the packet's *last* sample

A firmware `hrv_event` carries several `interval_min`-minute averages (`hr_bpm`,
`rmssd_ms`) under **one** ring timestamp, and the raw beat packets (`0x60`,
`ibi_and_amplitude`) carry several beats under one stamp. In every case the ring
stamps the packet at its **last (newest)** sample and the earlier ones lead up to
it — so element `i` of `len` sits `(len-1-i) * step` *before* the stamp, never
after it. A 7-sample 5-minute packet therefore spans the 30 minutes **ending** at
the stamp.

Placing samples *forward* from the stamp instead pushes the newest one up to a full
packet span into the future of its own emission and shifts the whole nocturnal
HR/HRV series late by that span (~30 min), desynchronising it from the motion
stream and the hypnogram — which under-detects deep sleep (N3 refinement reads the
wrong HR/HRV window). This bit us once; keep every client on the walk-back rule:

- **Rust** (`crates/oura-summary`): go through `packed_series` / `packed_sample_ds`
  (and `hrv_step_ds`); never re-derive `ds + i*step` by hand.
- **Python**: `tools/nssa_inputs.py` (`expand_ibi_event`, temp expansion) and the
  model runners walk back; match them when adding a runner.
- **Reference**: `research_export::beat_times` and its CSV note ("packet time is the
  last beat, earlier beats walk back") are the canonical statement of the rule.

The real end state is to stamp each sample **once in the `open_oura` decoder** so no
client re-derives packet timing at all — tracked as a follow-up.

## Known gaps (iOS-only, not yet on the web)

Now on **both** (do not treat these as gaps any more): the model-free Symptom Radar
(`symptoms`, used when the runner has no `illness`), the ring-clock warnings
(`clock.warnings`), and **the sleep report for a night with no hypnogram** — both clients now
draw the time-in-bed strip and the signal lanes, and add the stage-derived sections only when
stages exist. And **hourly / sub-hour heart rate is not an iOS-only surface either**: the web has
`GET /api/hourly-hr` and the finer bins; iOS is the side still limited to hourly bars (listed
under the web-only gaps below). Still iOS-only:

- **iOS NSSA cache admission** includes the current model demographics before the
  automatic planner reuses a night. The web runner already fingerprints demographics
  with each request; iOS also checks them before its shortcut can skip that request.
  Cache entries bind their mapped window, so a changed bedtime end cannot reuse cells
  from the old interval. Saved NSSA summaries can satisfy the automatic plan; older
  SleepNet summaries cannot substitute for NSSA.
- **Apple Health HRV export** uses the RMSSD type on iOS 27 and later. On older
  systems, ring RMSSD remains visible in the app rather than being mislabeled as
  SDNN. Reconciliation removes only this exporter's legacy `openoura.hrv.*` SDNN
  records, using existing authorization; it requests no new SDNN access.
- **Symptom Radar details**: all four biomarkers with their personal band
  (`lower`/`upper`); the web lists only the flagged ones.

## Known gaps (web-only, not yet on iOS)

Now on **both**: the **time-true overnight lanes** — both clients draw the polysomnograph from
`nights[].series_t` (`[unix, value]` points, gaps kept) and only fall back to spreading the
flat `nights[].series` over the night when a summary predates `series_t`; the same applies to
motion, where `series_t.motion` is preferred and `series.motion_time` is the legacy fallback.
Still web-only:

- **One-page day view**: web shows Sleep, Activity and Heart rate on one scrolling page, every
  chart on one time axis (`dayAxis`) with a shared cursor (`dayCursor`), plus ‹ › / ← → to step
  through days with data; iOS `DayReportView` keeps tabs and reaches days via `AllDaysView`.
- **Sub-hour heart-rate slots**: the web Heart rate tab defaults to 15-minute bars (`hr_bins`,
  with a 5 min / 15 min / 1 h switch); iOS `HeartRate.swift` still shows hourly bars.
- **Ring diagnostics page** (`/ring`, `ring.js` → `GET /api/ring-diagnostics`): the 14-day
  wearing-coverage heatmap (hours after the last sync are hatched as *unknown*, not off), the
  battery curve with a hover crosshair (solid % / dashed pack mV / charging bands), recording
  gaps classified power-off / charging / off-finger, per-stream last-seen and the log entries
  naming the ring. iOS reports clock warnings and the per-boot table under Technical reports but
  has no coverage heatmap, gap classification or battery chart.
- **Stress & restored events** (`renderStress`, from the daytime-HRV DSA chain) and the
  **apnea (BDI) / RSA breathing-rate lanes** on the polysomnograph: no iOS equivalent (iOS
  renders only the nightly `breath_rate` figure in `SleepScoreCard`).
- **Live heart rate** panel: `POST /api/live-hr` streams beats from `oura live-hr`
  (one minute per session, stoppable) as newline-delimited JSON.
- **"Signal estimate" staging notice**: when a night is staged by the Rust signal
  fallback (`staging_source === "signals"`, i.e. Python/torch or the `.pt` model is
  absent on this machine), the web day view shows a note saying so and how to enable
  Oura's model. iOS bundles the on-device model, so this condition is rare there and
  has no matching label yet.

- **Advanced & debugging**: on-ring feature toggles (`/api/feature`) and the per-type
  event stream. Profile editing is now native on iOS, including optional Apple Health
  import for date of birth, biological sex, height, and weight, plus optional export
  of workouts (add/remove as detections change), sleep stages, heart rate, RMSSD HRV (iOS 27+),
  resting HR, steps, calories, and distance.
- **Polysomnograph crosshair**: web has a hover crosshair; iOS uses a touch scrubber
  (drag across the lanes) — same idea, adapted to the input.
- **DNA explorer** (`/dna`): reads genome `*.vcf.gz` files and scores single-SNP **traits**
  against the editable `dna/catalog.json`, plus **polygenic scores** — the illustrative
  built-ins in the catalog *and* real [PGS Catalog](https://www.pgscatalog.org/) scoring
  files (`dna/scores/*.txt.gz`). Parsing/scoring is the `crates/oura-dna` crate
  (`vcf`/`catalog`/`pgs`/`score` modules); the server glue is `crates/oura-cli/src/dna.rs`
  → `dashboard/web/dna.{html,js,css}`. A genome + which scores to apply are chosen with
  selectors; a PGS ID can be fetched on demand (`POST /api/dna/fetch` → EBI) into
  `dna/scores/`. Genomes are read from a **configurable directory** — keep your large,
  private files anywhere via `oura dashboard --dna-files <dir>` (or `$OURA_DNA_FILES`);
  it defaults to the repo's `dna/files/`, while the catalog + fetched PGS scores always
  live in the repo `dna/`. PGS scoring is strict: effect+other-allele matching, strand-flip
  resolution, palindromic-ambiguous exclusion, `weight_type` (OR/HR → `ln`), and coverage
  stats — a raw sum is reported honestly (no population reference is shipped, so no
  percentile). Trait interpretation is **strand-aware** too (reverse-complement fallback for
  non-palindromic SNPs), since a GRCh38 VCF stores e.g. `rs4988235` as A/G while catalogs
  write the classic C/T. **Deliberately web-only** — it has nothing to do with ring data, so
  it does not go through `oura-summary` and is not mirrored on iOS. If it's ever wanted on
  iOS, the `oura-dna` crate is the reusable brain.

  *Whole-genome (gVCF) support:* the reader handles 30x WGS **genomic VCFs** — most of the
  genome is stored as `END=` **reference blocks**, so a single streaming pass resolves any
  trait/PGS locus inside a hom-ref block as homozygous-reference (a per-chromosome merge-join
  cursor). Without this, coverage would collapse to only the sites where the sample carries a
  variant. One 298 MB / 30x gVCF parses in ~6 s (then cached); a `pos_set` gate lets the
  ~tens-of-millions of non-target records skip all lookups. A `.snp-indel` file is the one to
  use — `list_files` classifies each `*.vcf.gz` (`snp-indel` vs `cnv`/`sv`) so the UI prefers
  the scoreable one and explains the copy-number / structural-variant files instead of
  scoring them to noise.

  *Network note:* the app's outbound requests are exactly these: the DNA/PGS fetch above, the
  blood-PDF import's **optional** marker-normalization call to an external LLM endpoint (below,
  only when `$AGENT_BACKEND_API_KEY` is set), and nothing else — no telemetry, analytics or
  tracking of any kind anywhere. Both calls are explicit user actions, and the genome never
  leaves the machine.

- **Blood panel** (`/blood`): tracks lab-test markers over time, reads each against its
  reference range, and surfaces the ones worth attention with plain-language advice. The
  compute — status (in/out of range), trend across draws, which side is "concerning" per
  marker, and the attention list — is real and lives in `crates/oura-cli/src/blood.rs`;
  the front-end is `dashboard/web/blood.{html,js,css}` (each marker card is a
  reference-band sparkline in the main dashboard's graph idiom, with a full time-series +
  advice in the detail dialog). **The inputs are imported, not mocked**: the "Import" button
  calls `POST /api/blood/import-dir` (scan the configured directory) and a single file by path
  goes through `POST /api/blood/import`. Import runs **locally**: `pdftotext -layout` (plus
  `-raw` for the urinalysis block) extracts the text, deterministic parsers recognize known
  SYNLAB/CUF/Labcorp rows, and the result is cached in a small **SQLite `blood.db`, separate
  from the ring's `oura.db`**, keyed by the file's **sha256** — re-importing the same PDF is a
  no-op (`{"duplicate": true}`), and a text hash is kept too so a changed file is detected.
  Reports live per profile (`<blood_dir>/<ring-serial>/*.pdf`, reassignable through
  `POST /api/blood/assign`); the directory is configurable via `oura dashboard --blood-files
  <dir>` / `$OURA_BLOOD_FILES`. When the deterministic parse comes back sparse the importer can
  ask an **external LLM endpoint** to normalize the marker rows — only with
  `$AGENT_BACKEND_API_KEY` set, and it degrades to a warning when unset (this is the second of
  the app's two outbound requests; see the network note above). **Deliberately web-only** — like
  the DNA explorer it has nothing to do with ring data, does not go through `oura-summary`, and
  is not mirrored on iOS; `blood.rs`'s marker model would be the reusable brain if that ever
  changed.

When you close one of these gaps, update this section.

## Ring clock resets → epoch-aware time mapping (every code path)

`ring_timestamp` (ds) is a **per-boot relative deciseconds counter**: it resets to ~0
every time the ring reboots (battery drain, firmware reset). Naively anchoring every ds
to one global `max_ds`/`captured_unix` scatters older boots to wildly wrong dates (a boot
can land months in the past). The fix segments events into boot **epochs** — walk in real
sync order `(captured_unix, then insertion id)`, split on any large backward jump in ds, then use
the epoch's on-ring `time_sync` (0x42) and `rtc_beacon` (0x85)
(`ring_timestamp` ↔ UTC) records as authoritative anchors. RTC beacons must also
retain their JSON in the iOS metadata reader. Ignoring them can shift a whole
night to its download time or project a new boot through an older boot's clock.
`captured_unix` is only an epoch-selection hint and a fallback for legacy data. Every event
resolves with a **source**: `anchor` (its own boot's time_sync/rtc_beacon/phone anchor),
`projected` (another boot's anchor, only when this ds continues that boot's counter — a
rebooted ring restarts near zero and must never be projected through an older boot that
only ran at higher counts), `download_time` (capture-time arithmetic, allowed only for a
boot drained sync after sync so the error is bounded by one sync gap) or `undated` (a boot
downloaded in one go with no anchor). Nights whose bounds are not `anchor`/`projected` are
**withheld** from `nights` and listed in `clock.undated_nights` — showing them dated to the
download would put a 23:00→08:00 sleep at 07:00→15:00 on the wrong day. To make anchors
exist for every boot, the iOS sync (`oura-core` `sync_inner`) now sends the phone clock to
the ring (`sync_time_app`, which makes the ring log a `time_sync`) and, after a drain that saw
new ring time, inserts a synthetic `time_sync` row (`decoded_json.source = "phone"`, body =
unix LE + "phone") pairing the newest drained ds with the phone clock. This lives in
**four places that must stay in sync**:

- `crates/oura-summary/src/ring_time.rs` — the shared `RingClock`; fixes night/activity/
  movement **dates for both clients** at once.
- `crates/oura-models/src/activity.rs` (`Clock` / epoch builder) — the same rule mirrored in
  Rust for the in-process `RustRunner` activity model; its incremental cache stores the epoch
  list so a replay dates rows exactly as a fresh scan would.
- `tools/epoch_time.py` (helper) used by `tools/run_activity_model.py` and
  `tools/run_sleep_model.py` — the **web** on-model session/hypnogram times.
- `apps/ios/OuraApp/EventStore.swift` (`epochs` / `unixSeconds`) used by
  `ActivityModel.swift` and `SleepStaging.swift` — the **iOS** on-device model times.
  iOS must be rebuilt to pick this up.

## Premature sleep ends → evidence-based model windows

The ring can close a raw `bedtime_period` during a brief awakening even though sleep
continues. `oura-summary::normalize_bed_periods` repairs the model boundary in two stages:

- explicit sleep-only ACM, temperature, and SpO₂ packets can extend a raw end by up to
  three hours, but only while they keep coming: the extension walks them in time order
  and stops at the first silence over 30 minutes (`MAX_SLEEP_SUPPORT_GAP_DS`). A Ring 4
  reports no `sleep_state`, so without this a 20-minute still spell 2.5 hours after
  waking — resting SpO₂/temperature packets — was joined onto the night (a 09:03 wake
  shown as 11:51);
- when a long sleep already has at least 30 minutes of that explicit premature-end
  evidence, continuous accepted HR/IBI bursts may carry the candidate window farther.

Pulse evidence is deliberately gated: each burst needs multiple firmware-accepted heart
rate estimates, consecutive bursts can be at most 15 minutes apart, and naps or clean
bedtime ends never use daytime pulse sampling. The resulting canonical `start_ds/end_ds`
is passed unchanged to the Python SleepNet runner and iOS `SleepStaging`, so both clients
score the same recovered window. Regression tests include isolated daytime HR, long gaps,
periodic post-nap sampling, and the extracted Ring 5 brief-wake vector.

The polysomnograph's skin-temperature lane uses only `sleep_temp_event`. Generic
`temp_event` contains multiple device/ambient channels and must never be flattened into
the nocturnal skin-temperature series. `nights[].series.temp_span` records the actual
coverage inside an extended sleep window, so iOS and web leave a visible gap after the
last trustworthy sample instead of stretching or inventing a temperature collapse.

## Ring 5 extended history sync

Ring 5's `ExtGetEvent` batches are self-completing. Do not send the legacy `GetEvent`
ACK (`0x10`) afterward: the ring answers that ACK with another history burst, whose late
notifications race with the next data flush and are discarded. Extended summary result
code `0xff` is a rejected cursor, not a successful empty batch. `oura-link` now rejects
that result explicitly, and `oura-core` checkpoints zero and performs one deduplicated
recovery drain when an existing iOS database contains such a stale cursor.

A from-zero recovery may replay an older boot after the newer boot is already stored.
If that makes the selected epoch project an event more than six hours beyond its phone
capture time, all three implementations fall back to the newest globally plausible
`time_sync` projection. This prevents replay fragments from fabricating future days.

Incremental pulls key off `sync_state.next_cursor` (deciseconds). After a reboot the
ring's ds restarts low, so an empty incremental fetch verifies that the event immediately
before the saved cursor still exists. If that marker is absent, the native iOS core
checkpoints cursor 0 and drains the new boot epoch automatically. This also recovers
cursors poisoned by the pre-`d409f9e` extended-envelope timestamp decoder.
The link layer also rejects any single-batch cursor jump beyond 180 days. A physical
Ring 5 validation exposed malformed tail envelopes with near-`u32::MAX` timestamps;
discarding those impossible records prevents a new poisoned cursor while preserving
the surrounding valid events and terminal summary.

## Android parity notes

The decompiled Android client uses SweetBlue's reference-counted
`PARTIAL_WAKE_LOCK` during BLE work. iOS has no equivalent unrestricted CPU wake
lock, so `IdleTimerLock` keeps the foreground app awake with the same reference-count
ownership semantics and reasserts the idle-timer flag after lifecycle transitions.

Android's legacy NSSA path passes the ring's `BedtimePeriodValue` directly to the
sleep-stage handler. Its newer feature-gated stateless bedtime detector instead skips
ring bedtime events and derives periods from feature-session, state-change, motion,
temperature, time-sync, and alert events at one-minute resolution. That detector's
dynamically delivered model is not embedded in the APK. The shared summary therefore
keeps the raw ring bounds alongside locally adjusted bounds, and only adjusts an end
when adjacent bedtime segments or sleep-only sensor evidence support it.

## Sleep coverage and refresh lifecycle

Unanalysed ring epochs are `0` (unknown), never wake. Both clients leave gaps and
withhold complete-night metrics/scores/debt for partial staging. The summary exports
`staging_source`, `staging_coverage_pct`, `staging_complete` and `captured_unix`.
iOS uses `stages_full` for calculations; `stages` remains the compact preview.
Apple Health skips unknown epochs. Both clients draw motion from timestamped `series_t.motion`
bars, with `series.motion_time` (normalized positions) only as the legacy fallback for summaries
that predate `series_t`; iOS renders those measurements discretely, without interpolation.

SleepNet output timestamps are retained on both paths and mapped onto the bedtime
30-second grid. Input selection checks absolute time as well as relative counters;
Ring 5 quality flags gate beat validity. Batch windows carry capture time to select
the right boot. The latest iOS night rechecks its input fingerprint after store changes.

Automatic lifecycle sync requests are coalesced and concurrent summary reloads queue
one follow-up. iOS reports gate/clock/model stages and has a cooperative 180-second
analysis deadline; a native Torch forward cannot safely be forcibly interrupted, so
its watchdog reports the delay while retaining exclusive database ownership.
Desktop Python runners have a 180-second kill/reap timeout and drain stdout concurrently.

Validation uses synthetic cases; personal exports and model weights remain local.
A report JSON alone cannot replay Gen4 inference or prove the original bedtime.

GitHub iOS validation also compiles the TORCH Swift paths against a validation-only
C bridge and runs pure planning/alignment tests. This checks model plumbing, not
LibTorch inference; release projects never include the stub bridge.


### CLI configuration and community integration

`--db` and `--key-file` override `OURA_DB_FILE` and `OURA_KEY_FILE`. Ring operations
look for adjacent `oura.key`, then a single `oura-<serial>.key`; multiple serial keys
require an explicit selection. Pairing saves its default key beside the database.
Saved-data commands do not load auth keys.

CLI display commands default to the current local UTC offset on Unix (UTC on
other platforms). `--tz-offset` accepts fractional hours, e.g. `5.5`, `5.75`, or
`-3.5`. This remains a fixed offset across the requested history, not a named
zone with historical DST/travel rules; use an explicit offset when needed. The
existing integer-offset iOS FFI contract is retained.

Multiple `ring_start` markers between disagreeing anchors make the interval
between the first and last reboot undatable. Rust, Python, and Swift agree on
this rule; model caches are invalidated for the clock/bedtime changes.

The web BLE actions (sync, live HR, feature mode) share one operation gate. Live
HR drains stderr concurrently, retains only a bounded tail, and kills/reaps a
stalled or disconnected child before releasing the gate. Hardware/firmware live
HR support still depends on the pinned BLE implementation.
