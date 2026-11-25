#!/usr/bin/env python3
"""Golden vectors for the run_sleep_model.py plumbing that crates/nssa-plumbing ports.

An oracle harness, not a reimplementation: it imports the real script (with a
controlled argv, so its module-level DB read + RingClock build run exactly as the
dashboard runs them) and wraps its real functions to capture inputs, outputs and
wall time. Nothing here is used by the dashboard.

  .venv/Scripts/python.exe tools/nssa_plumbing_golden.py \
      --db <path-to>/oura.db --serial <ring-serial> \
      --out crates/nssa-plumbing/tests/vectors

Written into --out:
  manifest.json    parameters, versions, per-night sha256, phase timings
  clock.json       the exact epochs JSON the script feeds epoch_time
  windows.json     per summary window: is_dated results, ring_bedtimes() output,
                   and the full score_window() result (or its error string)
  night_N.json     per ring bedtime: the raw event rows it consumed (input) and
                   acm/ibi/temp/breath (output), plus the model outputs the
                   plumbing feeds (stage epochs, BDI apnea)
Monotone integer series (event timestamps) are stored as [first, delta, delta…]
to keep vectors small; the Rust test rebuilds them with a running sum.
"""
import argparse
import bisect
import hashlib
import importlib.util
import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def f(v):
    """JSON-safe float: repr round-trips every bit."""
    return float(v)


def deltas(vals):
    """[first, v1-v0, v2-v1, …] for a monotone integer series."""
    if not vals:
        return []
    out = [int(vals[0])]
    for prev, cur in zip(vals, vals[1:]):
        out.append(int(cur) - int(prev))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--serial", required=True)
    ap.add_argument("--pairs", default=str(HERE.parent / "pairs.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    # Import the real runner: argv decides DB + serial; --batch with empty stdin
    # lets its module body finish without staging anything.
    argv, stdin = sys.argv, sys.stdin
    sys.argv = ["run_sleep_model.py", "--batch", "--serial", args.serial, str(args.db)]
    sys.stdin = io.StringIO("[]")
    t0 = time.perf_counter()
    # exec_module (not import … as R): the module body raises SystemExit under
    # --batch, which would both skip the binding and drop it from sys.modules.
    spec = importlib.util.spec_from_file_location(
        "run_sleep_model", str(HERE / "run_sleep_model.py"))
    R = importlib.util.module_from_spec(spec)
    sys.modules["run_sleep_model"] = R
    try:
        with redirect_stdout(io.StringIO()):
            spec.loader.exec_module(R)
    except SystemExit:
        pass
    finally:
        sys.argv, sys.stdin = argv, stdin
    timings["import_read_rows_and_clock"] = round(time.perf_counter() - t0, 3)
    R._cache_path = out / "nssa_cache.json"  # never touch the real DB dir

    (out / "clock.json").write_text(
        json.dumps(R._epochs, separators=(",", ":")), encoding="utf-8")

    nights, windows_rec = [], []
    cur = {"i": -1}
    cur_win = {"is_dated": [], "bedtimes": [], "bedtimes_called": False}

    # ---- wrap the real functions (the names the script resolves at call time) --
    _collect = R.collect_nssa_inputs

    def collect_wrap(rows, s, e, cu, unix_ms, corr):
        t = time.perf_counter()
        res = _collect(rows, s, e, cu, unix_ms, corr)
        cur["t_collect"] = time.perf_counter() - t
        cur["bedtime_ds"] = [int(s), int(e)]
        cur["captured_unix"] = int(cu)
        cur["bedtime_ms"] = [int(res["bedtime"][0]), int(res["bedtime"][1])]
        cur["acm_t"] = [a[0] for a in res["acm"]]
        cur["acm_v"] = [[f(x) for x in a[1]] for a in res["acm"]]
        cur["temp_t"] = [a[0] for a in res["temp"]]
        cur["temp_v"] = [f(a[1]) for a in res["temp"]]
        return res

    _correct = R.correct_ibis

    def correct_wrap(beats):
        t = time.perf_counter()
        res = _correct(beats)
        cur["t_correct_ibis"] = time.perf_counter() - t
        cur["beats_pre"] = [[int(b[0]), int(b[1]), int(b[2])] for b in beats]
        cur["ibi"] = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in res]
        return res

    _breath = R.breath_series

    def breath_wrap(inputs):
        t = time.perf_counter()
        res = _breath(inputs)
        cur["t_breath"] = time.perf_counter() - t
        cur["breath"] = [[int(a[0]), f(a[1])] for a in res]
        return res

    _stage = R.stage_bedtime

    def stage_wrap(b_start, b_end, cu, demo):
        nonlocal cur
        cur = {"i": len(nights)}
        nights.append(cur)
        t = time.perf_counter()
        res = _stage(b_start, b_end, cu, demo)
        cur["t_stage_model"] = time.perf_counter() - t
        if res:
            cur["stage"] = {"t": [int(x) for x in res[0]],
                            "codes": [int(x) for x in res[1]]}
            cur["apnea"] = ([[int(a[0]), f(a[1])] for a in res[2]]
                            if res[2] is not None else None)
        else:
            cur["stage"], cur["apnea"] = None, None
        return res

    _ring = R.ring_bedtimes

    def ring_wrap(s, e, cu):
        res = _ring(s, e, cu)
        cur_win["bedtimes_called"] = True
        cur_win["bedtimes"] = [[int(a[0]), int(a[1]), int(a[2])] for a in res]
        return res

    _dated = R.is_dated

    def dated_wrap(epochs, ds, cu):
        res = _dated(epochs, ds, cu)
        cur_win["is_dated"].append([int(ds), int(cu), bool(res)])
        return res

    R.collect_nssa_inputs, R.correct_ibis = collect_wrap, correct_wrap
    R.breath_series, R.stage_bedtime = breath_wrap, stage_wrap
    R.ring_bedtimes, R.is_dated = ring_wrap, dated_wrap

    # ---- the real pass ---------------------------------------------------------
    pairs = json.loads(Path(args.pairs).read_text())
    if args.limit:
        pairs = pairs[:args.limit]
    t0 = time.perf_counter()
    for w, pair in enumerate(pairs):
        cur_win = {"i": w, "window": [int(pair[0]), int(pair[1])],
                   "captured_unix_arg": int(pair[2]) if len(pair) > 2 else None,
                   "is_dated": [], "bedtimes": [], "bedtimes_called": False}
        res, err, _ = R.score_window(int(pair[0]), int(pair[1]),
                                     int(pair[2]) if len(pair) > 2 else None)
        cur_win["result"] = res if isinstance(res, dict) else None
        cur_win["error"] = None if isinstance(res, dict) else res
        windows_rec.append(cur_win)
        print(f"window {w} {pair[0]}..{pair[1]} bedtimes={len(cur_win['bedtimes'])} "
              f"{'ok' if cur_win['result'] else err}", flush=True)
    timings["full_score_window_pass"] = round(time.perf_counter() - t0, 3)

    # ---- raw input rows per night (the plumbing's own input) -------------------
    # Same filter collect_nssa_inputs applies: tag in (0x60,0x72,0x75) and
    # start_ds-6000 <= ds <= end_ds+6000, kept in DB order (captured_unix, id).
    decoded = [(ds, tag, js, cu, i) for i, (ds, tag, js, cu, _b) in enumerate(R.rows)
               if tag in (0x60, 0x72, 0x75)]
    order = sorted(range(len(decoded)), key=lambda i: decoded[i][0])
    ds_sorted = [decoded[i][0] for i in order]
    bodies = {i: b for i, (ds, tag, js, cu, b) in enumerate(R.rows) if tag in (0x72, 0x75)}

    def raw_rows(s, e):
        lo = bisect.bisect_left(ds_sorted, s - 6000)
        hi = bisect.bisect_right(ds_sorted, e + 6000)
        picked = sorted((decoded[order[k]] for k in range(lo, hi)), key=lambda r: r[4])
        rows = []
        for ds, tag, js, cu, i in picked:
            v = json.loads(js)
            if tag == 0x60:
                if len(v.get("ibi_ms") or []) != 6:
                    continue
                rows.append([ds, tag, cu, v["ibi_ms"], v.get("amplitude") or []])
            else:
                body = bodies.get(i)
                rows.append([ds, tag, cu, None, body.hex() if body is not None else None])
        return rows

    # ---- the ring's own bedtime rows (ring_bedtimes' own input) ---------------
    # Every 0x76 in DB order; windows.json already carries what ring_bedtimes()
    # returned for each window, so the Rust port can be checked against both.
    bed_rows = []
    for ds, tag, js, cu, _b in R.rows:
        if tag != 0x76:
            continue
        v = json.loads(js)
        bed_rows.append([ds, tag, cu, v.get("bedtime_start_ds"), v.get("bedtime_end_ds")])
    (out / "bedtime_rows.json").write_text(
        json.dumps(bed_rows, separators=(",", ":")), encoding="utf-8")

    # ---- write vectors ---------------------------------------------------------
    manifest_nights = []
    for rec in nights:
        s, e = rec["bedtime_ds"]
        doc = {
            "bedtime_ds": [s, e],
            "captured_unix": rec["captured_unix"],
            "bedtime_ms": rec["bedtime_ms"],
            "rows": raw_rows(s, e),
            "acm": {"t": deltas(rec["acm_t"]), "v": rec["acm_v"]},
            "beats_pre": rec.get("beats_pre", []),
            "ibi": rec.get("ibi", []),
            "temp": {"t": deltas(rec["temp_t"]), "v": rec["temp_v"]},
            "breath": rec.get("breath", []),
            "stage": rec.get("stage"),
            "apnea": rec.get("apnea"),
            "seconds": {k[2:]: round(v, 4) for k, v in rec.items() if k.startswith("t_")},
        }
        p = out / f"night_{rec['i']}.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        manifest_nights.append({
            "i": rec["i"], "bedtime_ds": [s, e], "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:32],
            "seconds": doc["seconds"],
            "n_acm": len(rec["acm_t"]), "n_beats": len(doc["beats_pre"]),
            "n_temp": len(rec["temp_t"]), "n_breath": len(doc["breath"]),
        })
    (out / "windows.json").write_text(json.dumps(windows_rec), encoding="utf-8")

    PHASE_KEYS = {"collect_nssa_inputs": "collect", "correct_ibis": "correct_ibis",
                  "breath_series": "breath", "stage_model": "stage_model"}
    for name, key in PHASE_KEYS.items():
        timings[name] = round(sum(m["seconds"].get(key, 0.0) for m in manifest_nights), 3)
    timings["plumbing_total"] = round(
        timings["collect_nssa_inputs"] + timings["breath_series"], 3)
    manifest = {
        "db": str(args.db), "serial": args.serial, "pairs": str(args.pairs),
        "python": sys.version.split()[0], "cache_version": R.CACHE_VERSION,
        "counts": {"windows": len(windows_rec), "nights": len(nights),
                   "staged": sum(1 for rec in nights if rec.get("stage"))},
        "timings_s": timings,
        "nights": manifest_nights,
        "vector_bytes": sum(m["bytes"] for m in manifest_nights),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({"timings_s": timings, "counts": manifest["counts"],
                      "vector_bytes": manifest["vector_bytes"]}))


if __name__ == "__main__":
    main()
