"""Tune the motion-classifier thresholds from recorded clips ("train on one person").

    python tools/tune_thresholds.py                 # fit + report, change nothing
    python tools/tune_thresholds.py --write         # also rewrite T_ACTIVE / T_WRITE in imu_params.h
    python tools/tune_thresholds.py --write --no-writing   # plan section 8 kill switch: never report WRITING

Clips come from data/imu/*.csv (tools/imu_record.py). LIFTED clips are scored as STILL, because the
firmware only reports STILL / MOVING / WRITING (LIFTED cannot be told from STILL by motion alone).
The k-th clip of each label counts as "run k", so with two or more runs the script also reports a
leave-one-run-out check: thresholds fitted on the other runs, scored on the held-out one. With a single
person that is the only honest estimate of how fragile the thresholds are.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imu_logic as L  # noqa: E402

EXPECTED = {"STILL": L.STILL, "LIFTED": L.STILL, "MOVING": L.MOVING, "WRITING": L.WRITING}


def load_clips(folder: str) -> Dict[int, Dict[str, np.ndarray]]:
    """{run_index: {label: (n,3) array}}; the k-th file (by name/time) of each label is run k."""
    per_label: Dict[str, List[str]] = defaultdict(list)
    for fp in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        label = os.path.basename(fp).split("_")[0]
        if label in EXPECTED:
            per_label[label].append(fp)
    runs: Dict[int, Dict[str, np.ndarray]] = defaultdict(dict)
    for label, files in per_label.items():
        for k, fp in enumerate(files):
            with open(fp) as f:
                rows = list(csv.DictReader(f))
            runs[k][label] = np.array([[float(r["ax"]), float(r["ay"]), float(r["az"])] for r in rows])
    return runs


def scores(a: np.ndarray, window: int) -> np.ndarray:
    """Window spread after every sample once the window is full (what the firmware computes)."""
    return np.array([L.window_spread([tuple(x) for x in a[i - window + 1: i + 1]]) for i in range(window - 1, len(a))])


def fit(sc: Dict[str, np.ndarray]) -> Tuple[float, float, float, float]:
    """Returns (t_active, t_write, acc_still_vs_active, acc_moving_vs_writing) for the given scores."""
    still = np.concatenate([sc[l] for l in ("STILL", "LIFTED") if l in sc])
    mov, wri = sc.get("MOVING"), sc.get("WRITING")
    grid = np.linspace(0.02, 1.5, 600)

    def bal_active(t: float) -> float:
        parts = [np.mean(mov >= t)] if mov is not None else []
        if wri is not None:
            parts.append(np.mean(wri >= t))
        return 0.5 * np.mean(still < t) + 0.5 * float(np.mean(parts))

    accs = np.array([bal_active(t) for t in grid])
    best = accs.max()
    t_active = float(np.median(grid[accs >= best - 0.005]))  # middle of the plateau, not its edge
    if mov is None or wri is None:
        return t_active, 0.4, float(best), float("nan")
    accs2 = np.array([0.5 * np.mean(mov < t) + 0.5 * np.mean(wri >= t) for t in grid])
    best2 = accs2.max()
    t_write = float(np.median(grid[accs2 >= best2 - 0.005]))
    return t_active, t_write, float(best), float(best2)


def replay(clip: np.ndarray, expected: str, params: dict) -> Dict[str, float]:
    """Run the real state machine over a clip and report time spent in each state (after warm-up)."""
    clf = L.MotionClassifier(params)
    out = []
    warm = int(params["CLS_WINDOW"]) + 30  # window fill + about 1 s settle
    for i, (ax, ay, az) in enumerate(clip):
        clf.update(ax, ay, az)
        if i >= warm:
            out.append(clf.state)
    n = max(1, len(out))
    return {s: out.count(s) / n for s in (L.STILL, L.MOVING, L.WRITING)}


def write_params(path: str, t_active: float, t_write: float, emit_writing: bool) -> None:
    text = open(path, newline="").read()

    def sub(name: str, value: str, text: str) -> str:
        pat = re.compile(rf"(constexpr\s+\w+\s+{name}\s*=\s*)[^;]+(;)")
        assert pat.search(text), f"{name} not found in {path}"
        return pat.sub(rf"\g<1>{value}\g<2>", text, count=1)

    text = sub("T_ACTIVE", f"{t_active:.3f}f", text)
    text = sub("T_WRITE", f"{t_write:.3f}f", text)
    text = sub("EMIT_WRITING", "true" if emit_writing else "false", text)
    open(path, "w", newline="").write(text)


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(here, "data", "imu"))
    ap.add_argument("--write", action="store_true", help="write the fitted thresholds into imu_params.h")
    ap.add_argument("--no-writing", action="store_true", help="set EMIT_WRITING = false (never report WRITING)")
    args = ap.parse_args()

    params = L.load_params()
    window = int(params["CLS_WINDOW"])
    runs = load_clips(args.dir)
    if not runs:
        raise SystemExit(f"no clips in {args.dir}")
    print(f"{len(runs)} run(s) of clips, window {window} samples\n")

    sc_by_run = {k: {lab: scores(a, window) for lab, a in clips.items()} for k, clips in runs.items()}
    pooled = {lab: np.concatenate([sc_by_run[k][lab] for k in sc_by_run if lab in sc_by_run[k]])
              for lab in EXPECTED if any(lab in sc_by_run[k] for k in sc_by_run)}
    print("median window spread per state:  " + "   ".join(f"{lab} {np.median(v):.3f}" for lab, v in pooled.items()))
    t_act, t_wr, acc_a, acc_w = fit(pooled)
    print(f"\nfit on ALL runs:  T_ACTIVE = {t_act:.3f} (STILL vs active accuracy {acc_a:.2f})   "
          f"T_WRITE = {t_wr:.3f} (MOVING vs WRITING accuracy {acc_w:.2f})")

    if len(sc_by_run) >= 2:
        print("\nleave-one-run-out (fit on the other runs, score on the held-out run):")
        accs_w: List[float] = []
        for k in sorted(sc_by_run):
            rest = {lab: np.concatenate([sc_by_run[j][lab] for j in sc_by_run if j != k and lab in sc_by_run[j]])
                    for lab in EXPECTED if any(lab in sc_by_run[j] for j in sc_by_run if j != k)}
            ta, tw, _, _ = fit(rest)
            held = sc_by_run[k]
            still = np.concatenate([held[l] for l in ("STILL", "LIFTED") if l in held])
            a_act = 0.5 * np.mean(still < ta) + 0.5 * np.mean([np.mean(held[l] >= ta) for l in ("MOVING", "WRITING") if l in held])
            a_wr = 0.5 * np.mean(held["MOVING"] < tw) + 0.5 * np.mean(held["WRITING"] >= tw) if "MOVING" in held and "WRITING" in held else float("nan")
            accs_w.append(a_wr)
            print(f"  hold out run {k}: thresholds ({ta:.3f}, {tw:.3f})  STILL-vs-active {a_act:.2f}   MOVING-vs-WRITING {a_wr:.2f}")
        worst = np.nanmin(accs_w)
        if worst < 0.75:
            print(f"\n  WARNING: MOVING vs WRITING falls to {worst:.2f} on a held-out run. A fixed threshold is fragile here:\n"
                  f"  re-tune right before the demo with a fresh session, or use --no-writing (tap 2 = next question ends the answer, plan section 8).")
    else:
        print("\n(only one run of clips: record another session for a leave-one-run-out check)")

    emit_writing = not args.no_writing
    trial = dict(params, T_ACTIVE=t_act, T_WRITE=t_wr, EMIT_WRITING=emit_writing)
    print("\nfull state-machine replay with the fitted thresholds (share of time in each state, after warm-up):")
    print(f"  {'clip':26} {'expected':9} {'STILL':>6} {'MOVING':>7} {'WRITING':>8}")
    for k in sorted(runs):
        for lab in ("STILL", "LIFTED", "MOVING", "WRITING"):
            if lab in runs[k]:
                r = replay(runs[k][lab], EXPECTED[lab], trial)
                hit = r[EXPECTED[lab]]
                print(f"  run {k} {lab:19} {EXPECTED[lab]:9} {r[L.STILL]:6.2f} {r[L.MOVING]:7.2f} {r[L.WRITING]:8.2f}   -> correct {hit:.0%}")

    if args.write:
        write_params(L.PARAMS_H, t_act, t_wr, emit_writing)
        print(f"\nwrote T_ACTIVE={t_act:.3f} T_WRITE={t_wr:.3f} EMIT_WRITING={str(emit_writing).lower()} into {L.PARAMS_H}")
        print("re-upload the firmware for the new thresholds to take effect")
    else:
        print("\n(dry run; add --write to update imu_params.h)")


if __name__ == "__main__":
    main()
