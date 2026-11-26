# Event decoders ported from the native parser — validation status

These decoders were ported from `libringeventparser.so` (the byte layouts are the
parser's ground truth) but most haven't appeared in our captures, so their field
*mapping* is confirmed by code, not by real data. Each unvalidated decoder emits
`"_status":"unvalidated"` in its JSON; drop it once a real sample confirms the
fields. The handler `@ address` is cited in the decoder in the sibling `open_oura`
repo (`crates/oura-protocol/src/events.rs`, `github.com/ctzl/open_oura`), which this
workspace pulls in as a patched path dependency.

## Validated (confirmed against captured bytes)

| Tag / subtype | Event | How confirmed |
| --- | --- | --- |
| `0x61`/`0x24` | `battery_level_changed` | captured: `battery_pct` 92–100 %, `voltage_mv` 4280–4391 mV (Li-ion). |
| `0x6b` | `motion_period` | decodes the captured packed 2-bit streams. |
| `0x6c` | `feature_session` | captured balanced start/stop pairs. |
| `0x7e`/`0x7f` | `real_steps_features` | ~143,900 captured samples of each tag; the bit-unpacked layout is fully mapped. `tools/activity_features.py` `unpack_real_step` expands the two 14-byte packets to the 27 ints the `steps_motion_decoder` takes, and `tools/run_activity_model.py` `STEP_TAGS = (126, 127)` consumes them (same on the Rust side, `oura-models::activity`). The decoder reports `"_status":"part1_raw"` for this path, not `"unvalidated"`. |
| `0x81` | `cva_raw_ppg_data` | ~86,000 captured rows; decoded by `decode_cva_raw_ppg` (the decoder in the sibling `open_oura` repo) and consumed in volume by the cardiovascular-age model (`crates/oura-models/src/cva.rs` reads `events(tag=0x81)`). |

## Unvalidated (layout from parser, awaiting a real sample)

| Tag | Event | Field confidence | How to trigger / validate |
| --- | --- | --- | --- |
| `0x49` | sleep_summary_1 | offsets only (abs time needs header) | after a processed sleep period |
| `0x4c` | sleep_summary_2 | structure only (u64/u16/u32, names TBD) | after a processed sleep period |
| `0x4f` | sleep_summary_3 | structure (3 fields are ÷8 fixed-point) | after a processed sleep period |
| `0x58` | sleep_summary_4 | structure only | after a processed sleep period |
| `0x86` | aohr_event | fields (bpm, quality, 1920 ms interval) | no toggle — rides on daytime HR (enabled); appears when worn |
| `0x84` | ambient_event | i16 @ 5 min, units TBD | appears with ambient sensing |
| `0x87` | atlas_metadata | start-stream control msg | **backend-gated** (`sensing_discovery/atlas` FeatureDefinition, cloud-delivered) — not enableable from an independent client |
| `0x88` | atlas_raw_bioz_data | delta-coded i32 stream | same backend gate as `0x87` |
| `0x61`/`0x11` | charging_time | u32 (units TBD) | **captured (25 samples of subtype `0x11`), awaiting field validation** — the units still have to be checked against a known charge session; the decoder still marks it `"unvalidated"` |

## Not decoded (low value / diagnostic only)

`0x61` debug subtypes other than charging/battery (sleep_statistics `0x09`,
afe_stats `0x28`, ppg_signal_quality `0x35`, fuel_gauge `0x14`, …) are device
diagnostics; they're tagged `{kind:"debug_data", subtype, raw}` and left raw.
Raw-PPG streams (`0x67`/`0x68`) and on-demand measurements (`0x62/0x65/0x66`)
are not decoded yet; `0x81` (the CVA raw-PPG stream) is decoded and consumed — see
the validated table above.
