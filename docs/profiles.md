# User profiles (two-person dashboard) — SPEC

Status: **implemented 2026-11-10** (backend + frontend + migration; verified
per §6 — see §8 for the few places the implementation deviates). Originally
written so the design survives context compaction.

## 1. Goal

The dashboard serves two people (user + wife). Each person = a **profile**.
A profile shows: ring data, blood lab history, DNA reports. A profile
switcher selects whose data every page shows.

## 2. Core decision (user-approved)

**A profile IS a ring registry entry.** No new profiles table. The profile
key is the ring serial already in `rings` (open_oura `storage.rs`,
`list_rings()`, `GET/POST /api/rings`, rename already supported). Blood and
DNA data get tagged with a ring serial. The existing ring switcher becomes
the profile switcher — same cookie, same UI pattern.

Out of scope v1: profiles without a ring; per-profile auth keys (already
per-ring via rings.auth_key_hex); sharing one profile across rings.

## 3. Backend

### 3.1 Blood (`crates/oura-cli/src/blood.rs`)
- Migration (guarded ALTER, same pattern as rings migration):
  `ALTER TABLE imports ADD COLUMN serial TEXT NOT NULL DEFAULT ''`.
- `import_dir_at(dir, serial)` / `import_path_at(path, serial)`: stamp new
  imports with the active profile's serial. `''` = unassigned.
- `report(serial: Option<&str>)`: when Some(s), `WHERE serial = s`; when
  None, all rows (back-compat for curl).
- New: `assign(sha, serial)` → `UPDATE imports SET serial=?1 WHERE sha=?2`.
- Marker history queries join through imports, so filtering imports by
  serial automatically scopes marker history. No marker_rows change.

### 3.2 DNA (`crates/oura-cli/src/dna.rs` + open_oura storage)
- Genomes stay files in the genomes dir (`--dna-files`); assignment is
  metadata, stored in **oura.db** (open_oura `storage.rs`):
  `CREATE TABLE IF NOT EXISTS dna_assignments (filename TEXT PRIMARY KEY,
  serial TEXT NOT NULL, assigned_unix INTEGER NOT NULL)`.
  Store fns: `list_dna_assignments()`, `set_dna_assignment(filename, serial)`.
- `list_files()` attaches `serial` to each `*.vcf.gz` entry (unassigned →
  `""`). Frontend filters by active profile; unassigned files are shown
  under every profile with an "assign" control (safer than guessing).
- `report`/`lookup` unchanged (they take an explicit file).

### 3.3 Routes (`crates/oura-cli/src/dashboard.rs`)
- `GET /api/blood/report?serial=` → scoped report.
- `POST /api/blood/import-dir` body `{serial}` (default: active profile
  from cookie if body omits it — do NOT silently import into '' ).
- `POST /api/blood/assign` body `{sha, serial}`.
- `GET /api/dna/files?serial=` → entries carry `serial`; filtering is
  client-side so the assign UI can show everything.
- Startup auto-import: import into the **primary ring** (first from
  `list_rings()`) — see §5.
- Serial resolution helper: `?serial=` param wins, else `ouradash_ring`
  cookie (name in app.js `RING_COOKIE`), else None.

## 4. Frontend (`dashboard/web/`)
- Share the existing switcher pattern from `app.js` (lines ~2317-2370):
  same `RING_COOKIE` cookie so the choice is ONE state across all pages.
  Extract or copy `loadRings/selectRing/renderSwitcher` into `blood.js`
  and `dna.js` (copy is fine; keep the cookie name + `ringQS()` shape).
- `blood.html/js`: switcher at top; every fetch carries `&serial=`;
  per-import row gets a profile `<select>` (calls /api/blood/assign);
  import button imports into the ACTIVE profile.
- `dna.html/js`: switcher at top; genome picker shows only active-profile
  files + unassigned ones with an assign control.
- `app.js`: relabel switcher "Profile" (it already scopes ring data).

## 5. Migration of existing data (one-time, at dashboard startup)
- If `list_rings()` is non-empty and any `imports.serial=''` exist:
  `UPDATE imports SET serial=<primary> WHERE serial=''` where primary =
  first ring from `list_rings()` (the user's own ring). Log it.
- Unassigned DNA files: NOT backfilled — shown under all profiles with an
  assign control (wife's genome must be assigned explicitly).
- Current state to migrate: 4 blood imports (all user's), 1 VCF in
  dna/files (user's) — verify actual list at implementation time.

## 6. Acceptance tests
1. Fresh build; startup backfills 4 imports to primary ring;
   `GET /api/blood/report?serial=<primary>` → 4 reports;
   `?serial=<wife>` → 0 reports.
2. `POST /api/blood/assign {sha, serial:<wife>}` moves one report;
   counts flip 3/1.
3. Import with `{serial:<wife>}` tags new rows.
4. `/api/dna/files` entries carry serial; assign persists across restart.
5. Browser: switch profile on /, then /blood and /dna show that person
   (same cookie). Single-ring DB behaves exactly as today.

## 7. Deliberate simplifications
- No profile CRUD: rename = rename ring (POST /api/rings), add profile =
  pair a ring (open_oura sync), remove = ring registry.
- No per-profile blood dir: one blood.db, serial column. Keeps the
  import-dir UX and run-dashboard.ps1 unchanged.
- Cookie-based active profile (already the ring-switcher mechanism); no
  server-side session.

## 8. Implementation notes (deviations from the design above)
- **Store crate is vendored**: open_health builds against
  `vendor/oura-store/src/storage.rs` (Cargo patch), NOT the sibling
  `../open_oura` checkout. Both copies were updated to stay in sync; a
  change only in the sibling copy does NOT compile into this dashboard.
- `dna_assignments` table is `(filename TEXT PRIMARY KEY, serial TEXT)` —
  the `assigned_unix` column from §3.2 was dropped (unused).
- DNA filtering is **server-side**: `GET /api/dna/files?serial=` drops
  genomes assigned to another profile and tags each entry with its
  `serial` ("" = unassigned, visible to all). The assign UI is a separate
  "Genome belongs to" select next to the genome picker (`POST
  /api/dna/assign {filename, serial}`; empty serial unassigns).
- Frontend shares one new file, `dashboard/web/profile.js` (cookie read,
  `/api/rings` fetch, `profileUrl()`, header `#profile-sel` switcher —
  hidden when <2 rings), loaded before blood.js/dna.js. Served via the
  `/profile.js` route (embedded + disk-override like other assets).
- Blood rows show a per-report profile `<select>` only when ≥2 rings are
  registered; `POST /api/blood/assign` takes the 12-char sha prefix shown
  in the report JSON (`WHERE sha LIKE ?1 || '%'`).
- Legacy backfill ran: 4 blood imports → primary ring ("Daily ring",
  serial 40160B25…). Verified: primary sees 4, other ring sees 0;
  assign/revert round-trips both stores; unassigned VCF visible to all.
- Main dashboard switcher relabeled "Profile" (index.html); behavior
  unchanged.
