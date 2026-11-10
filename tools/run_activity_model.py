#!/usr/bin/env python3
"""Run Oura's automatic_activity_detection model with Android-parity inputs.

The official Android app evaluates one local calendar day at a time and feeds
the 12-column step-motion series produced by steps_motion_decoder. Those details
are essential: omitting or misordering gait can turn a hike into cycling.

The model filename is never pinned — the asset set rotates with every app
version — so we resolve the newest `automatic_activity_detection_3_1_*.pt` and
detect its calling convention from the loaded schema (3.1.15 replaced the three
scalar settings tensors with an optional `global_settings`).
"""
import argparse
import calendar
import datetime
import hashlib
import json
import os
import re
import sqlite3
import sys
import warnings
from pathlib import Path

import torch

from _common import resolve_db, resolve_models_dir
from activity_features import decoder_to_aad, unpack_real_step
from epoch_time import build_epochs, is_dated, make_unix_s
from event_cache import EventSource, UnitCache, _pred

warnings.filterwarnings("ignore", message=".*searchsorted.*")

REPO = Path(__file__).resolve().parent.parent


def find_model():
    """Newest decrypted AAD model, like run_cva_model.find_model(): the asset
    set rotates every app version (3.1.11 is already gone), so match by prefix
    and take the highest version."""
    for d in (REPO / "notes" / "models", REPO.parent / "open_oura" / "notes" / "models"):
        cands = sorted(d.glob("automatic_activity_detection_3_1_*.pt"),
                       key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)])
        if cands:
            return cands[-1]
    sys.exit("error: no decrypted automatic_activity_detection_3_1_*.pt in notes/models "
             "— decrypt the assets with model_key.hex first")


MODEL = find_model()
MODEL_VERSION = ".".join(re.findall(r"\d+", MODEL.name))


def find_step_decoder():
    """Newest steps_motion_decoder in the same directory (also unpinned)."""
    cands = sorted(MODEL.parent.glob("steps_motion_decoder_*.pt"),
                   key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)])
    return cands[-1] if cands else None


STEP_MODEL = find_step_decoder()

BEHAVIOR = {
    -1: "nothing", 0: "<empty>", 1: "badminton", 2: "boxing", 3: "crossCountrySkiing",
    4: "crossTraining", 5: "cycling", 6: "dance", 7: "elliptical", 8: "strengthTraining",
    9: "hockey", 10: "pilates", 11: "rowing", 12: "running", 13: "swimming", 14: "walking",
    15: "yoga", 16: "golf", 17: "tennis", 18: "climbing", 19: "downhillSkiing",
    20: "snowboarding", 21: "hiking", 22: "horsebackRiding", 23: "volleyball", 24: "basketball",
    25: "americanFootball", 26: "soccer", 27: "baseball", 28: "coreExercise", 29: "cricket",
    30: "HIIT", 31: "diving", 32: "fitnessClass", 33: "floorball", 34: "gymnastics",
    35: "handball", 36: "houseWork", 37: "iceSkating", 38: "jumpingRope", 39: "martialArts",
    40: "flexibility", 41: "mountainBiking", 42: "nordicWalking", 48: "stairExercise",
    49: "stretching", 50: "surfing", 51: "waterFitness", 52: "yardwork", 53: "padel",
    69: "skateboarding", 65535: "other", 65536: "nap", 65537: "sleep", 65538: "pause",
    70937: "meditation", 71201: "eating", 71227: "relax", 71239: "transport",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Label activities with Oura's AAD model.")
    parser.add_argument("db", nargs="?", default=None)
    parser.add_argument("--tz", type=float, default=1.0, help="Fixed UTC offset in hours (default 1)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-duration", type=float, default=10.0,
                        help="Minimum segment minutes; Android default is 10")
    parser.add_argument("--date", help="Only process one local day (YYYY-MM-DD).")
    parser.add_argument("--full", action="store_true",
                        help="Ignore incremental state and recompute every day.")
    parser.add_argument("--no-stepmotion", action="store_true", help="Diagnostic fallback without gait")
    parser.add_argument("--serial", default=None, help="Only this ring's events (multi-ring DBs).")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args, extra = parser.parse_known_args()
    if extra:
        try:
            args.tz = float(extra[0])
        except (ValueError, IndexError):
            pass
    if args.db is not None and not Path(args.db).exists():
        try:
            args.tz = float(args.db)
            args.db = None
        except ValueError:
            pass
    # Retain the old debugging switch, but real gait is now the safe default.
    if os.environ.get("STEPMOTION") == "0":
        args.no_stepmotion = True
    return args


def tensor(rows, columns, window=None):
    """Rows as float32; `window=(lo, hi)` is the MET span of the day.

    A series with no sample inside that span is dropped entirely by the model's
    valid-time clipping, after which it indexes the empty tensor and throws. A NaN
    placeholder at the first MET minute keeps the channel present (it is treated as
    missing data). Mirror of ActivityModel.matrix on iOS.
    """
    if window is not None:
        lo, hi = window
        if not any(lo <= row[0] <= hi for row in rows):
            rows = sorted(rows + [[lo] + [float("nan")] * (columns - 1)], key=lambda row: row[0])
    if rows:
        return torch.tensor(rows, dtype=torch.float32)
    return torch.empty((0, columns), dtype=torch.float32)


NON_WEAR_MET_THRESHOLD = 0.2
MIN_VALID_METS = 10
ACCEPTABLE_LAST_HR_MISSING_MINUTES = 5


def model_would_reject(met, motion, temperature, heart_rate, step_rows):
    """Mirror of the model's `get_last_valid_time` (AAD 3.1.11 preprocessor).

    With fewer than MIN_VALID_METS worn MET minutes before the last motion sample the
    valid window collapses to minute 0 and the graph indexes an empty MET tensor.
    Such a day has no evaluable data: skip it instead of catching the exception.
    Kept identical to ActivityModel.isDegenerate on iOS.
    """
    import math
    last = lambda rows: rows[-1][0] if rows else 0.0
    last_motion = math.ceil(last(motion))
    worn = [row[0] for row in met if row[1] > NON_WEAR_MET_THRESHOLD and row[0] <= last_motion]
    if len(worn) < MIN_VALID_METS:
        return met[0][0] > 0
    last_valid = min(worn[-1], last_motion, math.ceil(last(heart_rate)) + ACCEPTABLE_LAST_HR_MISSING_MINUTES,
                     math.ceil(last(step_rows)), math.ceil(last(temperature)))
    return met[0][0] > last_valid


STEP_MODEL_MOBILE = MODEL.parent / "mobile" / (
    STEP_MODEL.name.replace(".pt", ".ptl") if STEP_MODEL else "steps_motion_decoder_2_0_0.ptl")


def load_step_decoder():
    """The gait decoder: the full TorchScript when installed, else the mobile export
    (same weights, lite-interpreter bytecode) that ships in the iOS app. Without it
    the model runs on MET/motion alone and finds far fewer sessions than the app."""
    if STEP_MODEL and STEP_MODEL.exists():
        return torch.jit.load(str(STEP_MODEL), map_location="cpu").eval()
    if STEP_MODEL_MOBILE.exists():
        from torch.jit.mobile import _load_for_lite_interpreter
        return _load_for_lite_interpreter(str(STEP_MODEL_MOBILE))
    return None


def decode_stepmotion(db, unix_seconds, enabled, epochs, serial=None):
    if not enabled:
        return []
    decoder = load_step_decoder()
    if decoder is None:
        print(f"warning: step decoder not found ({STEP_MODEL} or {STEP_MODEL_MOBILE}); "
              "running without gait features, expect fewer and differently labelled sessions",
              file=sys.stderr)
        return []
    packets = db.execute(
        "SELECT ring_timestamp, tag, body, captured_unix FROM events "
        "WHERE tag IN (126,127) AND body IS NOT NULL"
        + (" AND serial = ?" if serial else "")
        + " ORDER BY captured_unix,id",
        (serial,) if serial else (),
    ).fetchall()
    return decode_step_packets(packets, unix_seconds, epochs, decoder)


def decode_step_packets(packets, unix_seconds, epochs, decoder):
    seconds = {}
    for ds, tag, body, captured in packets:
        if tag == 0x7F and len(body) == 14 and is_dated(epochs, ds, captured):
            wall_ds = round(unix_seconds(ds, captured) * 10)
            seconds[wall_ds] = body
    raw, timestamps = [], []
    for ds, tag, body, captured in packets:
        if tag != 0x7E or len(body) != 14 or not is_dated(epochs, ds, captured):
            continue
        wall_ds = round(unix_seconds(ds, captured) * 10)
        second = seconds.get(wall_ds + 1)
        if second is None:
            continue
        raw.append(unpack_real_step(body, second))
        timestamps.append(wall_ds * 100)
    if not raw:
        return []
    # The decoder windows consecutive packets: feed them in time order (the DB is in
    # capture order, which interleaves boots after a re-anchor), as the iOS app does.
    order = sorted(range(len(raw)), key=lambda i: timestamps[i])
    raw, timestamps = [raw[i] for i in order], [timestamps[i] for i in order]
    with torch.no_grad():
        out_ts, out_data = decoder(torch.tensor(timestamps, dtype=torch.int64),
                                   torch.tensor(raw, dtype=torch.float32))
    # Keep all three 10-second sub-windows, as Android's TimeseriesDbStepMotion does.
    return sorted((ts / 60_000.0, decoder_to_aad(values))
                  for ts, values in zip(out_ts.flatten().tolist(), out_data.tolist()))


def model_settings(model, args):
    """Trailing settings args for the model call, adapted to the loaded schema.

    AAD <=3.1.12 takes three scalar tensors positionally (probability_threshold,
    minimum_duration_minutes, allow_non_wear). 3.1.15 replaced them with one
    optional `global_settings` row read by column — 3=allow_non_wear,
    12=is_workout_proba_threshold, 26=min_workout_duration_minutes — where NaN
    falls back to the model's baked-in default (verified against the serialized
    `_global_setting` helper). We only override the three we know; the rest stay NaN.
    """
    names = [a.name for a in model.forward.schema.arguments]
    if "global_settings" in names:
        row = [float("nan")] * 72
        row[3] = 0.0  # allow_non_wear=0 -> non-wear segments filtered, as before
        row[12] = args.threshold
        row[26] = args.min_duration
        return [torch.tensor(row, dtype=torch.float32).reshape(1, -1)]
    return [torch.tensor(args.threshold), torch.tensor(args.min_duration), torch.tensor(0.0)]


def run_day(model, day, day_start_min, signals, steps, args):
    day_end_min = day_start_min + 24 * 60

    def select(name):
        return sorted(([t - day_start_min] + values for t, values in signals[name]
                       if day_start_min <= t < day_end_min), key=lambda row: row[0])

    # The ring may resend a MET minute; Android's Realm series has one value per timestamp.
    met_by_minute = {round(row[0]): row for row in select("met")}
    met = sorted(met_by_minute.values(), key=lambda row: row[0])
    if not met:
        return []
    motion, temperature, heart_rate = select("motion"), select("temperature"), select("heart_rate")
    step_rows = [[t - day_start_min] + values for t, values in steps
                 if day_start_min <= t < day_end_min]
    if model_would_reject(met, motion, temperature, heart_rate, step_rows):
        return []
    lo, hi = met[0][0], met[-1][0]
    step_rows = [[lo] + [float("nan")] * 11] + step_rows + [[hi] + [float("nan")] * 11]

    context = torch.tensor([day.year, day.month, day.day, day.weekday()], dtype=torch.float32)
    user = torch.tensor(args.user_vector + [float("nan")] * 10, dtype=torch.float32)
    with torch.no_grad():
        workouts, _, _ = model(
            context, user, tensor(met, 2), tensor(step_rows, 12), tensor(motion, 9, (lo, hi)),
            tensor(temperature, 2, (lo, hi)), tensor(heart_rate, 2, (lo, hi)), None, None,
            *model_settings(model, args),
        )

    sessions = []
    local_midnight = datetime.datetime.combine(day, datetime.time())
    for row in workouts.tolist():
        start = local_midnight + datetime.timedelta(minutes=row[0])
        end = local_midnight + datetime.timedelta(minutes=row[1])
        top = [(BEHAVIOR.get(int(row[3 + 2 * i]), str(int(row[3 + 2 * i]))),
                round(row[4 + 2 * i], 3)) for i in range(3)]
        sessions.append({
            "start": start.strftime("%Y-%m-%d %H:%M"), "end": end.strftime("%H:%M"),
            "duration_min": round(row[1] - row[0]), "is_workout": round(row[2], 3),
            "label": top[0][0], "label_confidence": top[0][1], "top3": top,
        })
    return sessions


WHERE = "decoded_json IS NOT NULL"
SIGNAL_TAGS = (0x50, 0x47, 0x46, 0x80)
STEP_TAGS = (126, 127)
STATE_VERSION = "activity-inc-2"


def build_signals(rows, epochs, unix_seconds):
    """Date every row through the clock and bucket it into the per-minute
    signal series the Android TimeseriesDb tables would have held."""
    signals = {name: [] for name in ("met", "motion", "temperature", "heart_rate")}
    scale = float(os.environ.get("ACM_SCALE", "1"))
    for ds, tag, payload, captured in rows:
        try:
            value = json.loads(payload)
        except Exception:
            continue
        # Undated data (an untrustworthy boot clock) belongs to no calendar day.
        if not is_dated(epochs, ds, captured):
            continue
        # Ring summaries are minute buckets. Epoch reconstruction can leave them a
        # few seconds off the boundary; Android stores the bucket timestamp itself.
        minute = round(unix_seconds(ds, captured) / 60)
        if tag == 0x50 and isinstance(value.get("met"), list):
            signals["met"].extend((minute + i, [float(v)]) for i, v in enumerate(value["met"]))
        elif tag == 0x47:
            signals["motion"].append((minute, [
                float(value.get("orientation", 0)), float(value.get("motion_seconds", 0)),
                float(value.get("avg_x", 0)) * scale, float(value.get("avg_y", 0)) * scale,
                float(value.get("avg_z", 0)) * scale, float("nan"),
                float(value.get("low_intensity", 0)), float(value.get("high_intensity", 0)),
            ]))
        elif tag == 0x46 and value.get("temps_c"):
            signals["temperature"].append((minute, [float(value["temps_c"][0])]))
        elif tag == 0x80 and value.get("hr_bpm"):
            bpm = value["hr_bpm"]
            signals["heart_rate"].append((minute, [sum(bpm) / len(bpm)]))
    return signals


def profile_vector(db_path):
    """Demographics from profile.json next to the DB (what the iOS app and the CVA
    runner use); the model takes [age, sex(M=1), height_m, weight_kg]."""
    try:
        profile = json.loads((Path(db_path).parent / "profile.json").read_text())
    except Exception:
        profile = {}
    return [float(profile.get("age") or 30),
            1.0 if str(profile.get("sex", "M")).upper() == "M" else 0.0,
            float(profile.get("height_m") or 1.78), float(profile.get("weight_kg") or 75)]


def local_day(unix, tz):
    return datetime.datetime.utcfromtimestamp(unix + tz * 3600).date().isoformat()


def step_decoder_path():
    if STEP_MODEL and STEP_MODEL.exists():
        return str(STEP_MODEL)
    if STEP_MODEL_MOBILE.exists():
        return str(STEP_MODEL_MOBILE)
    return None


def run_one_day(args, model, day, signals, steps):
    utc_midnight = (datetime.datetime.combine(day, datetime.time())
                    - datetime.timedelta(minutes=args.tz * 60))
    day_start_min = utc_midnight.replace(tzinfo=datetime.timezone.utc).timestamp() / 60
    try:
        return run_day(model, day, day_start_min, signals, steps, args)
    except (torch.jit.Error, RuntimeError) as error:
        if args.verbose:
            print(f"{day}: model rejected incomplete day ({error})", file=sys.stderr)
        return []


def run_full(args, db_path, model, units, src):
    """Cold path: read everything, run every day, commit the watermark. This is
    the reference the incremental path must match (test_event_cache.py)."""
    db = sqlite3.connect(str(db_path))
    rows = db.execute(
        "SELECT ring_timestamp,tag,decoded_json,captured_unix FROM events "
        "WHERE " + WHERE
        + (" AND serial = ?" if args.serial else "")
        + " ORDER BY captured_unix,id",
        (args.serial,) if args.serial else (),
    ).fetchall()
    if not rows:
        sys.exit(f"error: no decoded events in {db_path}")
    epochs = build_epochs(rows)
    unix_seconds = make_unix_s(epochs)
    signals = build_signals(rows, epochs, unix_seconds)
    if not signals["met"]:
        sys.exit("no MET events in DB — cannot run activity model")
    steps = decode_stepmotion(db, unix_seconds, not args.no_stepmotion, epochs, args.serial)
    dates = sorted({datetime.datetime.utcfromtimestamp(t * 60 + args.tz * 3600).date()
                    for t, _ in signals["met"]})
    units.clear()
    sessions = []
    for day in dates:
        day_sessions = run_one_day(args, model, day, signals, steps)
        units.set(day.isoformat(), day_sessions)
        sessions.extend(day_sessions)
    units.save()
    src.note_epochs(epochs)
    src.commit(epochs, src.max_id(WHERE))
    return sessions


def run_incremental(args, db_path, model, units, src, epochs, tail, force_day=None):
    """Warm path: only days whose inputs changed (new rows, or a boot that gained
    clock anchors since the last commit) are recomputed; every other day comes
    from the unit cache. Output equals run_full() on the same data."""
    unix_seconds = make_unix_s(epochs)
    dirty = set()
    # Signal rows and step packets are separate row sets (step packets carry no
    # decoded_json); either can dirty a day.
    for where in (WHERE, "tag IN (126,127) AND body IS NOT NULL"):
        for ds, tag, _payload, captured in src.new_rows(where):
            if is_dated(epochs, ds, captured):
                dirty.add(local_day(unix_seconds(ds, captured), args.tz))
    for i, lo, hi in tail:
        u0, u1 = sorted((_pred(epochs[i], lo), _pred(epochs[i], hi)))
        day = datetime.date.fromisoformat(local_day(u0, args.tz))
        last = local_day(u1, args.tz)
        while day.isoformat() <= last:
            dirty.add(day.isoformat())
            day += datetime.timedelta(days=1)
    if force_day:
        dirty.add(force_day)
    decoder = None if args.no_stepmotion else load_step_decoder()
    sessions = []
    for key in sorted(set(units.days()) | dirty):
        if key not in dirty:
            sessions.extend(units.get(key))
            continue
        day = datetime.date.fromisoformat(key)
        u0 = calendar.timegm(day.timetuple()) - int(args.tz * 3600)
        windows = src.ds_windows(u0, u0 + 86400, epochs)
        rows = src.rows_in_windows(windows, SIGNAL_TAGS, where_extra=WHERE)
        signals = build_signals(rows, epochs, unix_seconds)
        steps = []
        if decoder is not None:
            packets = src.rows_in_windows(
                windows, STEP_TAGS, columns="id,ring_timestamp,tag,body,captured_unix",
                where_extra="body IS NOT NULL")
            steps = decode_step_packets(packets, unix_seconds, epochs, decoder)
        day_sessions = run_one_day(args, model, day, signals, steps)
        units.set(key, day_sessions)
        sessions.extend(day_sessions)
    units.save()
    if not force_day:
        src.commit(epochs, src.max_id(WHERE))
    return sessions


def main():
    args = parse_args()
    db_path = resolve_db(args.db, REPO)
    if not MODEL.exists():
        sys.exit(f"error: model not found: {MODEL}")
    args.user_vector = profile_vector(db_path)
    fp = hashlib.sha256(json.dumps([
        STATE_VERSION, MODEL.name, MODEL_VERSION, args.threshold, args.min_duration,
        args.user_vector, args.tz, args.no_stepmotion, os.environ.get("ACM_SCALE", "1"),
        None if args.no_stepmotion else step_decoder_path(),
    ]).encode()).hexdigest()
    src = EventSource(db_path, args.serial, "activity", STATE_VERSION)
    units = UnitCache(db_path, "activity", args.serial, fp)
    model = torch.jit.load(str(MODEL), map_location="cpu").eval()
    if args.full or not src.state["epochs"] or not units.units:
        sessions = run_full(args, db_path, model, units, src)
    else:
        epochs, tail = src.clock(WHERE)
        sessions = run_incremental(args, db_path, model, units, src, epochs, tail,
                                  force_day=args.date)
    if args.date:
        sessions = [s for s in sessions if s["start"].startswith(args.date)]
    result = {"model": MODEL_VERSION, "pipeline": "android-parity", "sessions": sessions}
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"Activity sessions — Oura AAD v{MODEL_VERSION}, Android-parity inputs\n")
    if not sessions:
        print("  No activity segments detected.")
        return
    print(f"  {'date':<10} {'time':<13} {'dur':>4}  {'workout':>7}  activity (model confidence)")
    for session in sessions:
        date, start = session["start"].split(" ")
        alternatives = "   ".join(f"{name} {prob:.2f}" for name, prob in session["top3"][1:])
        mark = "✓" if session["is_workout"] >= args.threshold else " "
        print(f"  {date:<10} {start}-{session['end']:<8} {session['duration_min']:>3}m  "
              f"{session['is_workout']:.2f} {mark}  {session['label']} "
              f"{session['label_confidence']:.2f}   ·   {alternatives}")


if __name__ == "__main__":
    main()
