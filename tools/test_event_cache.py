import json
import os
import sqlite3
import tempfile
import unittest

from event_cache import DS_MARGIN, EventSource, UnitCache

WHERE = "decoded_json IS NOT NULL"
DAY_DS = 864_000  # ring counter units are 0.1 s


def make_db(path):
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, serial TEXT NOT NULL,
        tag INTEGER NOT NULL, name TEXT NOT NULL, ring_timestamp INTEGER NOT NULL,
        body BLOB NOT NULL, decoded_json TEXT, captured_unix INTEGER NOT NULL,
        UNIQUE(serial, tag, ring_timestamp, body))""")
    return db


def insert(db, tag, ds, captured, decoded=None, serial="S1", body=b"x"):
    db.execute("INSERT INTO events(serial,tag,name,ring_timestamp,body,decoded_json,"
               "captured_unix) VALUES (?,?,?,?,?,?,?)",
               (serial, tag, "t", ds, body, decoded, captured))


def plain(x):
    """Epochs round-trip through JSON state; compare in that normalized form."""
    return json.loads(json.dumps(x))


class EventCacheTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.dir.name, "t.db")
        self.db = make_db(self.db_path)
        self.sources = []
        # One boot: two anchors 1 day apart, signals before/between them.
        insert(self.db, 0x42, 1_000_000, 1_789_000_000, '{"unix_time":1789000000}')
        insert(self.db, 0x42, 1_000_000 + DAY_DS, 1_789_086_400,
               '{"unix_time":1789086400}')
        for ds in (900_000, 1_200_000, 1_800_000):
            insert(self.db, 0x50, ds, 1_789_050_000, '{"met":[1.0]}')
        self.db.commit()

    def tearDown(self):
        for s in self.sources:
            if s._con is not None:
                s._con.close()
        self.db.close()
        self.dir.cleanup()

    def source(self):
        src = EventSource(self.db_path, None, "activity", "test-1")
        self.sources.append(src)
        return src

    def test_no_new_rows_replays_cached_epochs(self):
        src = self.source()
        epochs1, tail1 = src.clock(WHERE)
        src.commit(epochs1, src.max_id(WHERE))
        src2 = self.source()
        epochs2, tail2 = src2.clock(WHERE)
        self.assertEqual(plain(epochs1), plain(epochs2))
        self.assertEqual(tail2, [])

    def test_new_anchor_tails_only_past_the_previous_last_anchor(self):
        src = self.source()
        epochs, _ = src.clock(WHERE)
        src.commit(epochs, src.max_id(WHERE))
        insert(self.db, 0x42, 1_900_000, 1_789_090_000, '{"unix_time":1789090000}')
        self.db.commit()
        src2 = self.source()
        epochs2, tail = src2.clock(WHERE)
        self.assertEqual(len(tail), 1)
        i, lo, hi = tail[0]
        # Re-dating can only affect ds at/after the previous last anchor (± the
        # inversion margin) — never the whole boot.
        self.assertEqual(lo, 1_000_000 + DAY_DS - DS_MARGIN)
        self.assertEqual(hi, min(epochs2[i][1], 1_900_000 + DS_MARGIN))

    def test_cold_run_seeding_prevents_whole_boot_tail(self):
        src = self.source()
        epochs, _ = src.clock(WHERE)
        src.note_epochs(epochs)  # what run_full does before commit
        src.commit(epochs, src.max_id(WHERE))
        insert(self.db, 0x42, 1_900_000, 1_789_090_000, '{"unix_time":1789090000}')
        self.db.commit()
        src2 = self.source()
        _, tail = src2.clock(WHERE)
        self.assertEqual(len(tail), 1)
        self.assertGreater(tail[0][1], 1_000_000)  # not the epoch start

    def test_rows_in_windows_dedupes_and_orders_like_a_full_load(self):
        src = self.source()
        # Genuine duplicates (distinct ids, identical payload) must survive;
        # rows covered by two overlapping windows must appear exactly once.
        insert(self.db, 0x50, 1_200_000, 1_789_050_000, '{"met":[2.0]}', body=b"y")
        self.db.commit()
        full = [(r[1], r[2], r[3], r[4]) for r in self.db.execute(
            "SELECT id,ring_timestamp,tag,decoded_json,captured_unix FROM events "
            "WHERE tag IN (80) AND decoded_json IS NOT NULL ORDER BY captured_unix,id")]
        got = src.rows_in_windows([(900_000, 1_500_000), (1_100_000, 1_800_000)], (0x50,),
                                  where_extra=WHERE)
        self.assertEqual(got, full)

    def test_skeleton_watermark_survives_a_run_killed_before_commit(self):
        src = self.source()
        epochs1, _ = src.clock(WHERE)  # writes the skeleton pickle, no commit
        src2 = self.source()           # state still says watermark=0
        epochs2, _ = src2.clock(WHERE)
        self.assertEqual(plain(epochs1), plain(epochs2))  # rows were not appended twice

    def test_unit_cache_invalidates_on_settings_fingerprint(self):
        units = UnitCache(self.db_path, "activity", None, "fp-a")
        units.set("2026-11-09", [{"start": "x"}])
        units.save()
        self.assertEqual(UnitCache(self.db_path, "activity", None, "fp-a").get("2026-11-09"),
                         [{"start": "x"}])
        self.assertIsNone(UnitCache(self.db_path, "activity", None, "fp-b").get("2026-11-09"))


if __name__ == "__main__":
    unittest.main()
