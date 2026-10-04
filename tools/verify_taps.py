"""Guided tap test against the REAL board: how many taps does it catch, miscount, or invent?

    python tools/verify_taps.py --start-delay 20

Phases (cues are beeps, so you can keep your eyes on the pen):
  A  10 single-tap trials   one beep            -> tap ONCE with the pen tip on the paper / desk
  B  10 double-tap trials   two quick beeps     -> tap TWICE, quickly
  C  30 s of writing and moving (no taps)       -> the board must report NO taps
  D  15 s of staying still (no taps)            -> the board must report NO taps
Each trial lasts 4 s; the board reports a gesture ~0.5 s after the last tap. Raw lines are saved to
data/taps/ so thresholds can be tuned afterwards.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imu_record import beep, find_port, reopen  # noqa: E402

TRIAL_S = 4.0


def run(port: str, start_delay: float, n_trials: int, out_path: str):
    import serial

    lines = []   # (t, raw line)
    marks = []   # (kind, expected, t_start, t_end)
    with serial.Serial(port, 115200, timeout=0.05) as ser:
        ser.write(b"PING\n")
        time.sleep(start_delay)
        ser.reset_input_buffer()
        t_zero = time.time()
        last_ping = [0.0]

        def pump(until):
            while time.time() < until:
                if time.time() - last_ping[0] >= 1.0:
                    ser.write(b"PING\n")
                    last_ping[0] = time.time()
                raw = ser.readline().decode("ascii", errors="ignore").strip()
                if raw:
                    lines.append((time.time() - t_zero, raw))

        def countdown():
            for _ in range(3):
                beep(700, 150)
                pump(time.time() + 0.85)
            beep(1200, 500)
            pump(time.time() + 0.5)

        import serial as _serial

        def survive(fn, what):
            """Run fn(); if the USB link drops, wait for the board, announce it, and run fn() again."""
            while True:
                try:
                    return fn()
                except (_serial.SerialException, OSError) as e:
                    print(f"  USB dropped during {what} ({type(e).__name__}); waiting for the board, that attempt is redone")
                    drops.append((time.time() - t_zero, what))
                    reopen(ser)
                    ser.write(b"PING\n")
                    pump(time.time() + 1.0)
                    for _ in range(2):
                        beep(500, 400)

        drops = []

        def trial_single():
            t0 = time.time()
            beep(1000, 300)
            pump(t0 + TRIAL_S)
            return ("single", 1, t0 - t_zero, t0 - t_zero + TRIAL_S)

        def trial_double():
            t0 = time.time()
            beep(1500, 150)
            time.sleep(0.12)
            beep(1500, 150)
            pump(t0 + TRIAL_S)
            return ("double", 2, t0 - t_zero, t0 - t_zero + TRIAL_S)

        def phase_timed(kind, seconds):
            def f():
                countdown()
                t0 = time.time()
                pump(t0 + seconds)
                return (kind, 0, t0 - t_zero, t0 - t_zero + seconds)
            return f

        print("Phase A: single taps (one beep = tap once)")
        survive(countdown, "countdown")
        for i in range(n_trials):
            marks.append(survive(trial_single, f"single trial {i + 1}"))

        print("Phase B: double taps (two quick beeps = tap twice)")
        beep(500, 150); beep(500, 150)
        survive(lambda: pump(time.time() + 1.5), "pause")
        survive(countdown, "countdown")
        for i in range(n_trials):
            marks.append(survive(trial_double, f"double trial {i + 1}"))

        print("Phase C: write and move, NO taps, 30 s")
        beep(500, 150); beep(500, 150)
        survive(lambda: pump(time.time() + 1.5), "pause")
        marks.append(survive(phase_timed("active", 30), "write/move phase"))
        beep(500, 150); beep(500, 150)

        print("Phase D: stay still, NO taps, 15 s")
        survive(lambda: pump(time.time() + 1.5), "pause")
        marks.append(survive(phase_timed("still", 15), "still phase"))
        beep(500, 150); beep(500, 150)
        if drops:
            print(f"USB drops during this test: {len(drops)} -> {[(round(t, 1), w) for t, w in drops]}")

    with open(out_path, "w") as f:
        f.write("# marks: kind,expected_taps,t_start,t_end\n")
        for m in marks:
            f.write("# " + ",".join(str(x) for x in m) + "\n")
        for t, raw in lines:
            f.write(f"{t:.3f},{raw}\n")
    return lines, marks


def analyze(lines, marks):
    taps = []
    samples = []
    for t, raw in lines:
        p = raw.split(",")
        try:
            if p[0] == "T" and len(p) >= 2:
                taps.append((t, int(p[1])))
            elif p[0] == "S" and len(p) >= 5:
                samples.append((t, math.sqrt(float(p[2]) ** 2 + float(p[3]) ** 2 + float(p[4]) ** 2)))
        except ValueError:
            pass

    def peak(t0, t1):
        seg = [m for t, m in samples if t0 <= t < t1]
        if len(seg) < 5:
            return float("nan")
        base = sorted(seg)[len(seg) // 2]
        return max(abs(m - base) for m in seg)

    print(f"\nboard sent {len(taps)} tap events in total: {[(round(t,1), n) for t, n in taps]}")
    for kind, expected in (("single", 1), ("double", 2)):
        trials = [m for m in marks if m[0] == kind]
        ok = miss = wrong = extra = 0
        peaks = []
        wrong_detail = []
        for _, _, t0, t1 in trials:
            got = [n for t, n in taps if t0 <= t < t1]
            peaks.append(peak(t0, t1))
            if got == [expected]:
                ok += 1
            elif not got:
                miss += 1
            elif len(got) == 1:
                wrong += 1
                wrong_detail.append(got[0])
            else:
                extra += 1
                wrong_detail.append(got)
        n = len(trials)
        valid_peaks = [p for p in peaks if p == p]
        print(f"  {kind:6} trials: {ok}/{n} correct   missed {miss}   wrong count {wrong} {wrong_detail if wrong_detail else ''}   "
              f"two+ events {extra}   (median peak |a| jump seen in the 38 Hz stream: "
              f"{sorted(valid_peaks)[len(valid_peaks)//2]:.2f} g)" if valid_peaks else f"  {kind}: no data")
    for kind in ("active", "still"):
        for _, _, t0, t1 in [m for m in marks if m[0] == kind]:
            got = [(round(t, 1), n) for t, n in taps if t0 <= t < t1 + 1.0]
            print(f"  {kind:6} phase ({t1 - t0:.0f} s): {'no false taps' if not got else 'FALSE TAPS ' + str(got)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--start-delay", type=float, default=20.0)
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(here, "data", "taps"), exist_ok=True)
    out = os.path.join(here, "data", "taps", f"taps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    port = find_port(args.port)
    print(f"Using {port}; starting in {args.start_delay:.0f} s")
    lines, marks = run(port, args.start_delay, args.trials, out)
    print(f"raw lines saved to {out}")
    analyze(lines, marks)


if __name__ == "__main__":
    main()
