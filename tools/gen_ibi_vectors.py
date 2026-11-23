"""Generate golden vectors for crates/ibi-correct (the Rust port).

Runs the pure-Python reference (tools/ibi_correction.py) over real HRV streams
from an open_oura DB plus deterministic synthetic streams, and writes
{input, output} JSON pairs into crates/ibi-correct/tests/vectors/. The Rust
test crates/ibi-correct/tests/golden.rs asserts bit-exact agreement.

Usage: python tools/gen_ibi_vectors.py [path-to-oura.db]
"""
import json
import os
import random
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ibi_correction  # noqa: E402
# the pure-Python reference explicitly: once the Rust binary exists,
# ibi_correction.correct_ibis dispatches to it, and these vectors must come
# from the Python side of the equivalence.
correct_ibis = ibi_correction._correct_ibis_python  # noqa: E402
from nssa_inputs import TAG_IBI, expand_ibi_event  # noqa: E402

OUT_DIR = os.path.join(HERE, "..", "crates", "ibi-correct", "tests", "vectors")
REAL_STREAMS = 4
MAX_BEATS = 12000


def real_streams(db):
    con = sqlite3.connect(db)
    days = {}
    for ds, captured, dj in con.execute(
            "SELECT ring_timestamp, captured_unix, decoded_json FROM events "
            "WHERE tag=? ORDER BY captured_unix", (TAG_IBI,)):
        try:
            value = json.loads(dj or "{}")
        except ValueError:
            continue
        ibis = value.get("ibi_ms") or []
        if len(ibis) != 6:
            continue
        days.setdefault(captured // 86400, []).append(
            (captured * 1000, ibis, value.get("amplitude") or []))
    con.close()
    # prefer days that actually exercise the correction machinery
    days = sorted(days.values(), key=lambda evs: -len(evs))[:REAL_STREAMS]
    streams = []
    for events in days:
        events.sort(key=lambda e: e[0])
        beats, last = [], 0
        for event_ms, ibis, amps in events:
            expanded, last = expand_ibi_event(ibis, amps, event_ms, last)
            beats.extend(expanded)
            if len(beats) >= MAX_BEATS:
                break
        streams.append(beats[:MAX_BEATS])
    return streams


def synth_streams():
    rng = random.Random(20260115)
    streams = []

    def walk(n, lo, hi, start, zero_p=0.0, jump_p=0.0, ts0=1_790_000_000_000):
        ts, cur, beats = ts0, start, []
        for _ in range(n):
            if rng.random() < jump_p:
                cur = rng.randint(lo, hi)
            else:
                cur = max(lo, min(hi, cur + rng.randint(-40, 40)))
            amp = rng.randint(0, 60000)
            if rng.random() < zero_p:
                beats.append((ts, 0, amp))  # dropped beat: state machine sees ibi=0
                ts += cur
            else:
                beats.append((ts, cur, amp))
                ts += cur
        return beats

    streams.append(walk(3000, 700, 1100, 900))                       # clean night
    streams.append(walk(2000, 700, 1100, 900, zero_p=0.15))          # dropouts
    streams.append(walk(2000, 300, 2000, 900, jump_p=0.05))          # artifact soup
    streams.append(walk(1500, 430, 2000, 900, jump_p=0.2))           # HR storms
    streams.append(walk(1000, 700, 1100, 900, ts0=50000))            # non-epoch ts (ms_flag=False)
    # long flat bad run -> flush path (bad_run > 60)
    beats, ts = [], 1_790_000_000_000
    for i in range(400):
        beats.append((ts, 900 if i < 50 else 300, 1000))
        ts += beats[-1][1]
    streams.append(beats)
    # alternating extremes -> median/gap machinery
    beats, ts = [], 1_790_000_000_000
    for i in range(600):
        v = 400 if i % 2 else 1400
        beats.append((ts, v, 500))
        ts += v
    streams.append(beats)
    # pathological u16 edges
    streams.append([(1_790_000_000_000 + i * 1000, (65535 if i % 7 == 0 else (1 if i % 11 == 0 else 900)), 65535)
                    for i in range(800)])
    # long gap of zeros between valid beats -> gap fill / interp
    beats, ts = [], 1_790_000_000_000
    for block in range(30):
        for _ in range(20):
            beats.append((ts, 950, 2000))
            ts += 950
        for _ in range(rng.randint(1, 12)):
            beats.append((ts, 0, 0))
        ts += rng.choice([3000, 9000, 30000, 120000])
    streams.append(beats)
    # short random streams for branch fuzzing
    for _ in range(12):
        n = rng.randint(5, 400)
        streams.append([(rng.randint(0, 2) * 1_790_000_000_000 + i * rng.randint(0, 3000),
                         rng.choice([0, 1, 200, 500, 900, 1200, 3000, 65535]),
                         rng.randint(0, 65535)) for i in range(n)])
    return streams


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "..", "..", "open_oura", "oura.db")
    os.makedirs(OUT_DIR, exist_ok=True)
    for stale in os.listdir(OUT_DIR):
        if stale.endswith(".json"):
            os.remove(os.path.join(OUT_DIR, stale))
    count = 0
    for i, stream in enumerate(real_streams(db)):
        out = correct_ibis(stream)
        with open(os.path.join(OUT_DIR, f"real_{i}.json"), "w") as fh:
            json.dump({"input": [[t, b, a] for t, b, a in stream],
                       "output": [[t, b, a, v] for t, b, a, v in out]}, fh,
                      separators=(",", ":"))
        count += 1
        print(f"real_{i}: {len(stream)} beats -> {len(out)} corrected")
    for i, stream in enumerate(synth_streams()):
        out = correct_ibis(stream)
        with open(os.path.join(OUT_DIR, f"synth_{i:02d}.json"), "w") as fh:
            json.dump({"input": [[t, b, a] for t, b, a in stream],
                       "output": [[t, b, a, v] for t, b, a, v in out]}, fh,
                      separators=(",", ":"))
        count += 1
        print(f"synth_{i:02d}: {len(stream)} beats -> {len(out)} corrected")
    print(f"wrote {count} vectors to {OUT_DIR}")


if __name__ == "__main__":
    main()
