"""Incremental event access + per-unit result cache for the model runners.

See docs/incremental-recompute.md for the design and why this is safe:
`events` is append-only (UNIQUE(serial, tag, ring_timestamp, body) dedupes
every sync), so `MAX(id)` is a sound watermark; dating is never stored in
rows — epoch_time rebuilds it from anchor rows on every run.

A runner uses:
    src = EventSource(db, serial, model_key, version)
    epochs, tail = src.clock(where)             # cached epochs + re-dated ds windows
    new = src.new_rows(where)                  # rows added since the watermark
    windows = src.ds_windows(u_lo, u_hi, epochs)
    rows = src.rows_in_windows(windows, tags)  # each row once, full-load order
    units = UnitCache(db, model_key, serial, settings_fp)   # per-day results
    ...
    src.note_epochs(epochs)                    # cold path only: seed fingerprints
    src.commit(epochs, watermark)              # only after a FULL pass

`version` and `settings_fp` cover everything that invalidates cached results
globally (code contract, model asset, thresholds, profile demographics).
`clock()` reports `tail_ds` windows: ds ranges whose clock mapping changed
because that boot gained time-sync anchors since the last commit — the only
way an already-processed day can change without new rows in it.
"""
import hashlib
import json
import pickle
import sqlite3
from pathlib import Path

from epoch_time import EPOCH_RESET_SLACK_DS, build_epochs

ANCHOR_TAGS = (65, 66, 133)  # 0x41 boot, 0x42 time-sync, 0x85 rtc
# Inverting a wall-clock window to ring-counter ds is approximate (drift,
# stalls, extrapolation); overfetching is harmless (rows are re-dated and
# filtered downstream), underfetching would silently drop data.
DS_MARGIN = 6 * 3600 * 10  # 6 h of counter either side


def _pred(e, ds):
    """Approximate unix for ds in epoch e from its nearest anchor (or the
    capture fallback) — used only for conservative window math."""
    if e[5]:
        a = min(e[5], key=lambda a: abs(a[0] - ds))
        return a[1] + (ds - a[0]) / 10.0
    return e[2] - (e[1] - ds) / 10.0


class EventSource:
    def __init__(self, db_path, serial, model_key, version):
        self.db = str(db_path)
        self.serial = serial
        stem = f"{model_key}.{serial or 'all'}"
        self.state_path = Path(db_path).parent / (stem + ".inc.json")
        self._skel_path = lambda where: Path(db_path).parent / (
            stem + "." + hashlib.sha256(where.encode()).hexdigest()[:12] + ".skel.json")
        st = None
        try:
            st = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            pass
        if not st or st.get("version") != version:
            st = {"version": version, "watermark": 0,
                  "epochs": [], "anchor_fp": {}}
        self.state = st
        self._con = None

    def con(self):
        if self._con is None:
            self._con = sqlite3.connect(self.db)
        return self._con

    def _serial(self):
        return (" AND serial = ?", (self.serial,)) if self.serial else ("", ())

    def max_id(self, where):
        clause, params = self._serial()
        return self.con().execute("SELECT MAX(id) FROM events WHERE " + where + clause,
                                  params).fetchone()[0] or 0

    def _anchor_diff(self, epochs):
        """Update per-boot anchor fingerprints in state; return [epoch_index,
        ds_lo, ds_hi] for boots whose anchor set changed (their clock mapping —
        and only theirs — can re-date already-processed rows)."""
        changed = []
        for i, e in enumerate(epochs):
            key = str(e[0])
            fp = [len(e[5]), max((a[0] for a in e[5]), default=None)]
            prev = self.state["anchor_fp"].get(key)
            if prev is None:
                lo, hi = e[0], fp[1] if fp[0] else None
            else:
                lo, hi = prev[1] if prev[1] is not None else e[0], fp[1]
            if hi is not None and (prev is None or prev[1] != fp[1]):
                changed.append([i, max(e[0], lo - DS_MARGIN), min(e[1], hi + DS_MARGIN)])
            self.state["anchor_fp"][key] = fp
        return changed

    def note_epochs(self, epochs):
        """Seed the anchor fingerprints for a cold-built epoch structure, so the
        first warm run does not treat every boot as newly anchored."""
        self._anchor_diff(epochs)

    def clock(self, where):
        """(epochs, changed_ds_windows). Rebuilds the boot-epoch structure only
        when the row set changed; `changed_ds_windows` lists [ds_lo, ds_hi] for
        boots whose anchor set grew since the last commit (their clock mapping —
        and only theirs — can re-date already-processed rows)."""
        now = self.max_id(where)
        st = self.state
        if now == st["watermark"] and st["epochs"]:
            return [list(e) for e in st["epochs"]], []
        # build_epochs only reads ds/tag/captured plus decoded_json of the
        # anchor tags, so a cheap-column skeleton is exactly equivalent. The
        # skeleton's own watermark lives in the pickle: a run killed between
        # writing it and committing state must not re-append rows to it.
        clause, params = self._serial()
        skel_path = self._skel_path(where).with_suffix(".pkl")
        skel, skel_wm = [], 0
        try:
            with open(skel_path, "rb") as fh:
                data = pickle.load(fh)
            skel, skel_wm = data["rows"], data["watermark"]
        except (OSError, ValueError, TypeError, KeyError, pickle.UnpicklingError):
            pass
        if now > skel_wm:
            skel += self.con().execute(
                "SELECT ring_timestamp, tag, CASE WHEN tag IN (65,66,133) THEN decoded_json END,"
                " captured_unix FROM events WHERE " + where + clause + " AND id > ?"
                " ORDER BY captured_unix, id", tuple(params) + (skel_wm,)).fetchall()
            with open(skel_path, "wb") as fh:
                pickle.dump({"watermark": now, "rows": skel}, fh, protocol=5)
        epochs = build_epochs(skel)
        changed = self._anchor_diff(epochs)
        st["epochs"] = epochs
        return epochs, changed

    def new_rows(self, where, columns="ring_timestamp,tag,decoded_json,captured_unix"):
        clause, params = self._serial()
        return self.con().execute(
            f"SELECT {columns} FROM events WHERE {where}{clause} AND id > ?"
            " ORDER BY captured_unix, id", tuple(params) + (self.state["watermark"],)).fetchall()

    def ds_windows(self, u_lo, u_hi, epochs):
        """Conservative ds windows in every boot that could contain wall-clock
        [u_lo, u_hi)."""
        out = []
        for e in epochs:
            span_lo = min((_pred(e, d) for d in (e[0], e[1])), default=0)
            span_hi = max((_pred(e, d) for d in (e[0], e[1])), default=0)
            if span_lo > u_hi or span_hi < u_lo:
                continue
            if e[5]:
                lo = min(a[0] + (u_lo - a[1]) * 10.0 for a in e[5])
                hi = max(a[0] + (u_hi - a[1]) * 10.0 for a in e[5])
            else:
                lo, hi = e[0], e[1]
            lo = max(e[0] - EPOCH_RESET_SLACK_DS, lo - DS_MARGIN)
            hi = min(e[1] + EPOCH_RESET_SLACK_DS, hi + DS_MARGIN)
            if lo <= hi:
                out.append((int(lo), int(hi)))
        return out

    def rows_for_ds(self, lo, hi, tags, columns="ring_timestamp,tag,decoded_json,captured_unix",
                    where_extra=None):
        """Rows in a ds window. tags must be an explicit tag list: the only
        index that can seek a ds range is (tag, ring_timestamp); a bare BETWEEN
        scans the whole table."""
        clause, params = self._serial()
        sql = (f"SELECT {columns} FROM events "
               "WHERE tag IN (" + ",".join("?" * len(tags)) + ") "
               "AND ring_timestamp BETWEEN ? AND ?")
        if where_extra:
            sql += " AND " + where_extra
        return self.con().execute(sql + clause + " ORDER BY captured_unix, id",
                                  tuple(tags) + (lo, hi) + tuple(params)).fetchall()

    def rows_in_windows(self, windows, tags, columns="id,ring_timestamp,tag,decoded_json,captured_unix",
                        where_extra=None):
        """Union of ds windows with each row exactly once, in the same
        (captured_unix, id) order a full load uses. ds_windows() may overlap
        when boots share counter space; concatenating them would feed the model
        duplicate rows in a different order than the cold path."""
        seen = {}
        for lo, hi in windows:
            for r in self.rows_for_ds(lo, hi, tags, columns=columns, where_extra=where_extra):
                seen[r[0]] = r
        rows = sorted(seen.values(), key=lambda r: (r[4], r[0]))  # captured_unix, id
        return [(r[1], r[2], r[3], r[4]) for r in rows]

    def commit(self, epochs, watermark):
        self.state.update(watermark=watermark, epochs=epochs)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.state_path)

    def discard_skeleton_cache(self):
        pass  # skeleton files are keyed by WHERE hash; nothing to do


class UnitCache:
    """Per-unit (day/night) model results, valid while settings_fp holds."""

    def __init__(self, db_path, model_key, serial, settings_fp):
        self.path = Path(db_path).parent / f"{model_key}.{serial or 'all'}.units.json"
        self.units = {}
        try:
            st = json.loads(self.path.read_text())
            if st.get("settings_fp") == settings_fp:
                self.units = st["units"]
        except (OSError, ValueError):
            pass
        self.settings_fp = settings_fp

    def get(self, key):
        return self.units.get(key)

    def set(self, key, value):
        self.units[key] = value

    def clear(self):
        self.units = {}

    def days(self):
        return sorted(self.units)

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"settings_fp": self.settings_fp, "units": self.units}))
        tmp.replace(self.path)
