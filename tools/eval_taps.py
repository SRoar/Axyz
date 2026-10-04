"""Replay recorded tap tests (data/taps/*.txt from tools/verify_taps.py) through the tap detector.

    python tools/eval_taps.py                       # newest recording, current imu_params.h
    python tools/eval_taps.py --grid                # also try nearby settings and show how sensitive the result is

The detector runs together with the motion classifier (taps are accepted unless the pen is WRITING or was within
TAP_NOT_WRITING_MS, and the classifier is frozen during tap activity), exactly as in the firmware. The recording only
holds the ~40 Hz S stream, the firmware sees faster internal samples, so this is a close proxy, not an exact replay.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from itertools import product
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imu_logic as L  # noqa: E402


def load(path: str):
    marks, samples = [], []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("# "):
            p = line[2:].split(",")
            if p[0] in ("single", "double", "active", "still"):
                marks.append((p[0], int(p[1]), float(p[2]), float(p[3])))
            continue
        p = line.split(",")
        if len(p) >= 6 and p[1] == "S":
            samples.append((float(p[0]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))  # wall_s, dev_ms, ax, ay, az
    return marks, samples


def replay(samples, params) -> List[Tuple[float, int]]:
    """Returns [(wall_time_s, taps)] for every gesture the detector reports."""
    clf = L.MotionClassifier(params)
    det = L.TapDetector(params)
    events, last_writing = [], -1e9
    for wall, ms, ax, ay, az in samples:
        if clf.state == L.WRITING:
            last_writing = ms
        allowed = clf.state != L.WRITING and (ms - last_writing) >= params["TAP_NOT_WRITING_MS"]
        n = det.update(ms, ax, ay, az, allowed)
        if n:
            events.append((wall, n))
        clf.update(ax, ay, az, det.frozen(ms))
    return events


def score(marks, events):
    res = {"single": [0, 0, 0, 0], "double": [0, 0, 0, 0]}  # correct, missed, wrong count, extra events
    false_active = false_still = 0
    for kind, expected, t0, t1 in marks:
        got = [n for t, n in events if t0 <= t < t1 + (1.0 if expected == 0 else 0.0)]
        if kind in res:
            r = res[kind]
            if got == [expected]:
                r[0] += 1
            elif not got:
                r[1] += 1
            elif len(got) == 1:
                r[2] += 1
            else:
                r[3] += 1
        elif kind == "active":
            false_active += len(got)
        elif kind == "still":
            false_still += len(got)
    return res, false_active, false_still


def show(name, marks, events):
    res, fa, fs = score(marks, events)
    s, d = res["single"], res["double"]
    total = s[0] + d[0]
    print(f"{name}: single {s[0]}/10 (missed {s[1]}, wrong count {s[2]}, extra {s[3]})   "
          f"double {d[0]}/10 (missed {d[1]}, wrong count {d[2]}, extra {d[3]})   "
          f"=> {total}/20 correct;  false taps: writing/moving {fa}, still {fs}")
    return total, fa + fs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?")
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = args.file or sorted(glob.glob(os.path.join(here, "data", "taps", "taps_*.txt")))[-1]
    marks, samples = load(path)
    print(f"{os.path.basename(path)}: {len(samples)} samples, {len(marks)} marks")
    params = L.load_params()
    ev = replay(samples, params)
    show("current settings", marks, ev)
    print("  gestures reported:", [(round(t, 1), n) for t, n in ev])
    if args.grid:
        print("\nsensitivity (each row changes one or two settings):")
        for key, vals in (("TAP_ONE_MAX_MS", (150, 200, 250, 300, 350)), ("TAP_PEAK_G", (0.25, 0.30, 0.35, 0.40)),
                          ("TAP_ACT_G", (0.20, 0.25, 0.30)), ("TAP_PRE_QUIET_MS", (200, 400, 600)),
                          ("TAP_BURST_GAP_MS", (100, 150, 200, 300)), ("TAP_NOT_WRITING_MS", (500, 1000, 1500))):
            for v in vals:
                show(f"  {key}={v}", marks, replay(samples, dict(params, **{key: v})))


if __name__ == "__main__":
    main()
