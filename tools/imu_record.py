"""Record labeled IMU data from the pen probe (task 1.2).

    python tools/imu_record.py --label STILL --seconds 30          # one labeled recording
    python tools/imu_record.py --session --seconds 10              # guided STILL/MOVING/WRITING/LIFTED, with beeps
    python tools/imu_record.py --summarize                         # table of features per label

Each recording is a CSV in data/imu/ named <LABEL>_<timestamp>.csv with columns
    wall_s, ms, ax, ay, az, light, label
Mount the probe on the pen BEFORE recording and do not move it afterwards: the classifier thresholds
are only valid for that mounting.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
import time
from datetime import datetime

LABELS = ["STILL", "MOVING", "WRITING", "LIFTED"]
INSTRUCTIONS = {
    "STILL": "pen resting on the paper, hand off or just holding it, not moving",
    "MOVING": "slide the pen across the page the way you would travel to a box (no writing)",
    "WRITING": "write continuously on the paper (a sentence, then another)",
    "LIFTED": "pick the pen up and hold it in the air, hovering",
}
BAUD = 115200
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "imu")


def find_port(preferred: str | None) -> str:
    import serial.tools.list_ports as lp

    if preferred:
        return preferred
    ports = list(lp.comports())
    for p in ports:
        text = f"{p.description} {p.manufacturer or ''}".lower()
        if "arduino" in text or (p.vid in (0x2341, 0x8087)):
            return p.device
    if len(ports) == 1:
        return ports[0].device
    raise SystemExit(f"Could not pick a serial port automatically. Found: {[p.device for p in ports]}. Use --port.")


def beep(freq: int, ms: int) -> None:
    try:
        import winsound

        winsound.Beep(freq, ms)
    except Exception:
        print("\a", end="", flush=True)


def parse_s(line: str):
    """'S,<ms>,<ax>,<ay>,<az>,<light>' -> (ms, ax, ay, az, light) or None."""
    p = line.strip().split(",")
    if len(p) < 5 or p[0] != "S":
        return None
    try:
        light = int(float(p[5])) if len(p) > 5 else 0
        return float(p[1]), float(p[2]), float(p[3]), float(p[4]), light
    except ValueError:
        return None


def record(ser, label: str, seconds: float, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    ser.reset_input_buffer()  # drop any backlog so wall time and device time line up
    t0 = time.time()
    n = 0
    last_ms = None
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wall_s", "ms", "ax", "ay", "az", "light", "label"])
        while time.time() - t0 < seconds:
            raw = ser.readline().decode("ascii", errors="ignore")
            rec = parse_s(raw)
            if rec is None:
                continue
            if last_ms is not None and rec[0] < last_ms:  # device rebooted mid-recording
                print("  warning: device clock went backwards (board restarted?)")
            last_ms = rec[0]
            w.writerow([f"{time.time() - t0:.3f}", *rec, label])
            n += 1
    print(f"  {label}: {n} samples in {seconds:.0f} s ({n / seconds:.1f} Hz) -> {path}")
    return path


def features(rows):
    """Cheap features that separate the four states. rows = list of (ms, ax, ay, az, light)."""
    mags = [math.sqrt(r[1] ** 2 + r[2] ** 2 + r[3] ** 2) for r in rows]
    n = len(rows)
    mean_mag = sum(mags) / n
    std_mag = math.sqrt(sum((m - mean_mag) ** 2 for m in mags) / n)
    # mean absolute successive difference of the magnitude: high for jitter/writing, tiny for stillness
    masd = sum(abs(b - a) for a, b in zip(mags, mags[1:])) / max(1, n - 1)
    span = (rows[-1][0] - rows[0][0]) / 1000.0 if n > 1 else 0.0
    hz = (n - 1) / span if span > 0 else 0.0
    means = [sum(r[i] for r in rows) / n for i in (1, 2, 3)]
    max_dev = max(abs(m - mean_mag) for m in mags)
    return dict(n=n, hz=hz, mean_mag=mean_mag, std_mag=std_mag, masd=masd, max_dev=max_dev, ax=means[0], ay=means[1], az=means[2])


def summarize(out_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(out_dir, "*.csv")))
    if not files:
        print(f"No recordings in {out_dir}")
        return
    print(f"{'file':34} {'label':8} {'n':>5} {'Hz':>5} {'|a|':>6} {'std|a|':>7} {'masd':>7} {'maxdev':>7}  mean(ax,ay,az)")
    for fp in files:
        with open(fp) as f:
            rd = csv.DictReader(f)
            rows = [(float(r["ms"]), float(r["ax"]), float(r["ay"]), float(r["az"]), int(float(r["light"]))) for r in rd]
            label = os.path.basename(fp).split("_")[0]
        if len(rows) < 5:
            print(f"{os.path.basename(fp):34} {label:8} too few samples ({len(rows)})")
            continue
        ft = features(rows)
        print(
            f"{os.path.basename(fp):34} {label:8} {ft['n']:5d} {ft['hz']:5.1f} {ft['mean_mag']:6.3f} {ft['std_mag']:7.3f} "
            f"{ft['masd']:7.3f} {ft['max_dev']:7.3f}  ({ft['ax']:+.2f},{ft['ay']:+.2f},{ft['az']:+.2f})"
        )


def run_session(ser, seconds: float, out_dir: str, countdown: int) -> None:
    print("Guided session. Beeps: 3 short = get ready, 1 long = START, 2 short = STOP.\n")
    for label in LABELS:
        print(f"Next: {label} for {seconds:.0f} s - {INSTRUCTIONS[label]}")
        for _ in range(countdown):
            beep(700, 150)
            time.sleep(0.85)
        beep(1200, 500)
        record(ser, label, seconds, out_dir)
        beep(500, 150)
        beep(500, 150)
        time.sleep(1.5)
    print("\nDone. Run with --summarize to see the features.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port (default: auto-detect)")
    ap.add_argument("--label", choices=LABELS, help="record one labeled clip")
    ap.add_argument("--seconds", type=float, default=30.0, help="seconds per clip (default 30)")
    ap.add_argument("--session", action="store_true", help="guided recording of all four states with beeps")
    ap.add_argument("--countdown", type=int, default=3, help="beeps before each guided clip")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output folder")
    ap.add_argument("--summarize", action="store_true", help="print per-file features and exit")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.out)
        return
    if not args.label and not args.session:
        ap.error("pick --label, --session or --summarize")

    import serial

    port = find_port(args.port)
    print(f"Using {port} @ {BAUD}")
    with serial.Serial(port, BAUD, timeout=1.0) as ser:
        time.sleep(0.5)
        if args.session:
            run_session(ser, args.seconds, args.out, args.countdown)
        else:
            record(ser, args.label, args.seconds, args.out)


if __name__ == "__main__":
    sys.exit(main())
