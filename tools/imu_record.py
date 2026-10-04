"""Record labeled IMU data from the pen probe (task 1.2).

    python tools/imu_record.py --label STILL --seconds 30          # one labeled recording
    python tools/imu_record.py --session --seconds 10              # guided STILL/MOVING/WRITING/LIFTED, with beeps
    python tools/imu_record.py --summarize                         # table of features per label

Each recording is a CSV in data/imu/ named <LABEL>_<timestamp>.csv with columns
    wall_s, ms, ax, ay, az, label
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
    """'S,<ms>,<ax>,<ay>,<az>' -> (ms, ax, ay, az) or None. (An old 6th light field is ignored.)"""
    p = line.strip().split(",")
    if len(p) < 5 or p[0] != "S":
        return None
    try:
        return float(p[1]), float(p[2]), float(p[3]), float(p[4])
    except ValueError:
        return None


def drain(ser, max_s: float = 120.0) -> None:
    """Throw away data buffered while nobody was reading, so recorded data is live.

    The backlog is NOT only in the PC's port buffer: the board side keeps buffering too (seen: a
    clip that held 14 minutes of old data, 27000 lines, and 5000 lines after ~4 min). So keep
    reading until lines arrive at the live rate (~33 per second), not just until the PC buffer is empty.
    """
    t0 = time.time()
    while time.time() - t0 < max_s:
        window_start = time.time()
        lines = 0
        while time.time() - window_start < 0.5:
            if ser.readline():
                lines += 1
        if lines <= 24:  # live rate is ~16 lines per 0.5 s; a backlog delivers hundreds
            return
    print("  warning: could not catch up with the live stream; this clip may include old data")


def record(ser, label: str, seconds: float, out_dir: str, tag: str | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    name = f"{label}_{tag}_" if tag else f"{label}_"  # e.g. MOVING_slow_20261004_...csv
    path = os.path.join(out_dir, f"{name}{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    ser.write(b"PING\n")  # the board only streams while it keeps hearing from the PC (keepalive, ~1 per second)
    drain(ser)
    t0 = time.time()
    last_ping = t0
    n = 0
    first_ms = None
    last_ms = None
    try:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["wall_s", "ms", "ax", "ay", "az", "label"])
            while time.time() - t0 < seconds:
                if time.time() - last_ping >= 1.0:
                    ser.write(b"PING\n")
                    last_ping = time.time()
                raw = ser.readline().decode("ascii", errors="ignore")
                rec = parse_s(raw)
                if rec is None:
                    continue
                if last_ms is not None and rec[0] < last_ms:  # device rebooted mid-recording
                    print("  warning: device clock went backwards (board restarted?)")
                if first_ms is None:
                    first_ms = rec[0]
                last_ms = rec[0]
                w.writerow([f"{time.time() - t0:.3f}", *rec, label])
                n += 1
    except Exception:
        if os.path.exists(path):
            os.remove(path)  # never leave a half clip behind (the port can drop mid-read)
        raise
    msg = f"  {label}: {n} samples in {seconds:.0f} s ({n / seconds:.1f} Hz) -> {path}"
    if first_ms is not None and last_ms is not None and last_ms > first_ms:
        ratio = (last_ms - first_ms) / 1000.0 / seconds  # device time / wall time, should be ~1.0
        if abs(ratio - 1.0) > 0.1:
            msg += f"\n  WARNING: device time / wall time = {ratio:.2f}; backlog was not fully drained, re-record this clip"
    print(msg)
    return path


def reopen(ser, wait_s: float = 90.0) -> None:
    """Close and reopen the port after the board dropped off USB (it re-enumerates on its own)."""
    try:
        ser.close()
    except Exception:
        pass
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            ser.open()
            time.sleep(1.0)
            return
        except Exception:
            time.sleep(1.0)
    raise SystemExit(f"The board did not come back within {wait_s:.0f} s - replug it and run again.")


def record_with_retry(ser, label: str, seconds: float, out_dir: str, tag: str | None = None, tries: int = 3) -> str:
    """record(), but if the port drops mid-clip, wait for the board, re-countdown, and redo the clip."""
    import serial

    for attempt in range(1, tries + 1):
        try:
            return record(ser, label, seconds, out_dir, tag)
        except (serial.SerialException, OSError) as e:
            print(f"  port dropped ({type(e).__name__}) during {label}; waiting for the board (try {attempt}/{tries})")
            if attempt == tries:
                raise
            reopen(ser)
            print(f"  board is back; redoing {label}: get ready")
            for _ in range(3):
                beep(700, 150)
                time.sleep(0.85)
            beep(1200, 500)
    raise RuntimeError("unreachable")


def features(rows):
    """Cheap features that separate the four states. rows = list of (ms, ax, ay, az)."""
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
            rows = [(float(r["ms"]), float(r["ax"]), float(r["ay"]), float(r["az"])) for r in rd]
            label = os.path.basename(fp).split("_")[0]
        if len(rows) < 5:
            print(f"{os.path.basename(fp):34} {label:8} too few samples ({len(rows)})")
            continue
        ft = features(rows)
        print(
            f"{os.path.basename(fp):34} {label:8} {ft['n']:5d} {ft['hz']:5.1f} {ft['mean_mag']:6.3f} {ft['std_mag']:7.3f} "
            f"{ft['masd']:7.3f} {ft['max_dev']:7.3f}  ({ft['ax']:+.2f},{ft['ay']:+.2f},{ft['az']:+.2f})"
        )


BASIC_PLAN = [(label, None, INSTRUCTIONS[label]) for label in LABELS]
SPEEDS_PLAN = [
    ("MOVING", "slow", "slide the pen across the page SLOWLY, feeling your way (a page width in ~5-6 s), no writing, no pauses, pen on the paper"),
    ("MOVING", "normal", "slide the pen at a NORMAL relaxed speed (a page width in ~2-3 s), no writing, no pauses, pen on the paper"),
    ("MOVING", "quick", "slide the pen QUICKLY and confidently (a page width in ~1 s), no writing, no pauses, pen on the paper"),
    ("STILL", None, INSTRUCTIONS["STILL"]),
    ("LIFTED", None, INSTRUCTIONS["LIFTED"]),
]


def run_session(ser, seconds: float, out_dir: str, countdown: int, plan=BASIC_PLAN) -> None:
    print("Guided session. Beeps: 3 short = get ready, 1 long = START, 2 short = STOP.\n")
    for label, tag, text in plan:
        print(f"Next: {label}{' (' + tag + ')' if tag else ''} for {seconds:.0f} s - {text}")
        for _ in range(countdown):
            beep(700, 150)
            time.sleep(0.85)
        beep(1200, 500)
        record_with_retry(ser, label, seconds, out_dir, tag)
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
    ap.add_argument("--speeds", action="store_true", help="guided pass: MOVING slow/normal/quick, then STILL and LIFTED")
    ap.add_argument("--tag", help="optional tag put in the file name of a single --label clip, e.g. slow")
    ap.add_argument("--countdown", type=int, default=3, help="beeps before each guided clip")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output folder")
    ap.add_argument("--summarize", action="store_true", help="print per-file features and exit")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.out)
        return
    if not args.label and not args.session and not args.speeds:
        ap.error("pick --label, --session, --speeds or --summarize")

    import serial

    port = find_port(args.port)
    print(f"Using {port} @ {BAUD}")
    with serial.Serial(port, BAUD, timeout=1.0) as ser:
        time.sleep(0.5)
        if args.speeds:
            run_session(ser, args.seconds, args.out, args.countdown, SPEEDS_PLAN)
        elif args.session:
            run_session(ser, args.seconds, args.out, args.countdown)
        else:
            print(f"{args.label} for {args.seconds:.0f} s - {INSTRUCTIONS[args.label]}")
            for _ in range(args.countdown):
                beep(700, 150)
                time.sleep(0.85)
            beep(1200, 500)
            record_with_retry(ser, args.label, args.seconds, args.out, args.tag)
            beep(500, 150)
            beep(500, 150)


if __name__ == "__main__":
    sys.exit(main())
