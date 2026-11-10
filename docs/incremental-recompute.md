# Incremental model recompute

Status: implemented on `explore/rust-models` for the activity runner
(`tools/event_cache.py` + `run_activity_model.py`); other runners adopt the
same primitives. Motivation: every model runner re-reads and re-derives the
whole history on every dashboard sync. Cost grows linearly forever; the
user's data grows daily.

## Ground truth that makes this safe

- `events` is append-only: `UNIQUE(serial, tag, ring_timestamp, body)` dedupes
  every sync, rows are never updated or deleted. `MAX(id)` is a sound watermark.
- Dating is NOT stored in rows: `epoch_time.build_epochs` derives it from the
  ~500 anchor rows (tags 0x41 boot, 0x42 time-sync, 0x85) on every run. So a
  cached raw-row list can never be "wrongly dated" — dating is recomputed
  fresh, and anchors are cheap to load in full.
- Anchors are append-only with monotonic `ring_timestamp` inside a boot, so a
  new anchor can only re-date events with ds *after the previous last anchor
  of that boot* — older days' clock mapping is stable forever.

## Design

Per model key + serial, sidecars next to the db:

1. `<model>.<serial>.inc.json` — run state: `watermark` (MAX(id) of the
   runner's WHERE set), the built `epochs`, and per-boot `anchor_fp`
   (anchor count + max ds).
2. `<model>.<serial>.<where-hash>.skel.pkl` — the cheap columns
   (`ds, tag, anchor payload, captured`) that `build_epochs` needs, with its
   own watermark so a run killed before `commit()` cannot double-append.
3. `<model>.<serial>.units.json` — per-day model results, valid while
   `settings_fp` (model asset hash + runner args + profile demographics +
   STATE_VERSION) holds.

Warm run (`run_incremental`):

- `clock()` returns cached epochs instantly if the watermark is unchanged;
  otherwise it appends only new rows to the skeleton and rebuilds epochs
  (~0.1 s). It diffs `anchor_fp` and returns **tail windows**: ds ranges of
  boots that gained anchors — the only way an already-processed day can change
  without new rows in it.
- Dirty days = days that new rows date into (dated with the *new* epochs) ∪
  days overlapping tail windows. Everything else is copied from the unit cache.
- A dirty day is reloaded through `ds_windows()` + `rows_in_windows()` and
  re-dated; output assembly merges cached + recomputed units into the
  identical full JSON — the dashboard contract does not change.

## Pitfalls found (each is a test in `tools/test_event_cache.py`)

- A bare `ring_timestamp BETWEEN` cannot use any index — full scan per day.
  Window queries must be driven by `tag IN (...)` so SQLite seeks the
  `(tag, ring_timestamp)` index.
- ds windows of different boots overlap (counter space is reused). Naively
  concatenating them feeds the model duplicate rows in a different order than
  the cold path — outputs drift in the third decimal. `rows_in_windows()`
  dedupes by row id and restores `(captured_unix, id)` order.
- The cold path must `note_epochs()` before committing, or the first warm run
  treats every boot as newly anchored and dirties the whole history.
- If the unit cache is missing but run state exists, fall back to the full
  path (a warm pass with no dirty days would emit nothing).

## Invalidation summary

| change | effect |
|---|---|
| new events (normal sync) | only the days those rows date into recompute |
| new time-sync anchor | only days in that boot's tail window re-date → recompute |
| profile.json edited | demographics hash in `settings_fp` → full recompute |
| model asset rotated | model hash in `settings_fp` → full recompute |
| code/contract change | `STATE_VERSION` bump → full recompute |

## Measured (1.9 M events, 201 days, activity runner)

| run | before | after |
|---|---|---|
| cold / `--full` | 32 s | 32 s |
| warm, nothing new | 32 s | 2.4 s |
| warm, signal sync | 32 s | 8.4 s |
| warm, time-sync anchor | 32 s | 9.7 s |

Warm output is byte-identical to `--full` in every scenario. `--date` forces
one day and matches the full output for that day.

## Verification protocol (per runner)

1. cold run → JSON A; 2. warm run → JSON B, `cmp A B` empty; 3. simulate a
   sync (insert signal rows / a new anchor) → warm JSON C vs `--full` JSON D,
   `cmp C D` empty; 4. `--date` one day vs that day's slice of D.
Unit tests for the primitives live in `tools/test_event_cache.py`.
