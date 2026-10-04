"""Check the REAL firmware against the Python mirror on a live routine (no C++ compiler needed).

    python tools/verify_on_board.py                 # 60 s guided routine: STILL, MOVING, WRITING, LIFTED (15 s each)
    python tools/verify_on_board.py --seconds 10    # shorter stages

It records every line the board sends (S samples and the board's own M state changes, T taps), then feeds
the SAME S samples through tools/imu_logic.py and compares: for each sample, is the board's state the same
as the mirror's? Disagreements around a state change (a sample or two) are rounding: the board classifies
float samples, the S line carries 3 decimals. Anything bigger means the C++ and the Python differ.
Also reports how well the board's states match what you were asked to do in each stage.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imu_logic as L  # noqa: E402
from imu_record import beep, find_port  # noqa: E402

STAGES = [
    ("STILL", "pen resting on the paper, not moving"),
    ("MOVING", "slide the pen across the page, no writing, no pauses"),
    ("WRITING", "write continuously"),
    ("LIFTED", "hold the pen in the air (the firmware reports this as STILL)"),
]
EXPECT = {"STILL": "STILL", "MOVING": "MOVING", "WRITING": "WRITING", "LIFTED": "STILL"}


def capture(port: str, seconds: float, countdown: int = 3):
    import serial

    samples, events = [], []   # samples: (stage, ax, ay, az); events: (sample_index, kind, value)
    stage_marks = []
    with serial.Serial(port, 115200, timeout=0.05) as ser:
        ser.write(b"PING\n")
        time.sleep(1.0)
        ser.reset_input_buffer()
        last_ping = time.time()
        for stage, text in STAGES:
            print(f"Next: {stage} for {seconds:.0f} s - {text}")
            for _ in range(countdown):
                beep(700, 150)
                ser.write(b"PING\n")
                time.sleep(0.85)
            ser.reset_input_buffer()
            beep(1200, 500)
            t0 = time.time()
            stage_marks.append((stage, len(samples)))
            while time.time() - t0 < seconds:
                if time.time() - last_ping >= 1.0:
                    ser.write(b"PING\n")
                    last_ping = time.time()
                line = ser.readline().decode("ascii", errors="ignore").strip()
                p = line.split(",")
                try:
                    if p[0] == "S" and len(p) >= 5:
                        samples.append((stage, float(p[2]), float(p[3]), float(p[4])))
                    elif p[0] == "M" and len(p) >= 2:
                        events.append((len(samples) - 1, "M", p[1]))
                    elif p[0] == "T" and len(p) >= 2:
                        events.append((len(samples) - 1, "T", p[1]))
                except ValueError:
                    pass
            beep(500, 150)
            beep(500, 150)
            time.sleep(1.0)
    return samples, events


def board_state_per_sample(n, events, start="STILL"):
    """The board prints M right after the S line that caused the change (the PING reply also repeats the state)."""
    state, out, ev = start, [], sorted((i, v) for i, k, v in events if k == "M")
    j = 0
    for i in range(n):
        while j < len(ev) and ev[j][0] <= i:
            state = ev[j][1]
            j += 1
        out.append(state)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--start-delay", type=float, default=0.0, help="seconds to wait before the first stage")
    ap.add_argument("--dwell", type=int, help="compare against this CLS_DWELL instead of imu_params.h (use when the board runs an older upload)")
    args = ap.parse_args()

    port = find_port(args.port)
    print(f"Using {port}")
    if args.start_delay:
        time.sleep(args.start_delay)
    samples, events = capture(port, args.seconds)
    n = len(samples)
    print(f"\n{n} samples, {sum(1 for e in events if e[1]=='M')} M lines, {sum(1 for e in events if e[1]=='T')} T lines")

    # ---- mirror replay on the same samples
    params = L.load_params()
    if args.dwell:
        params["CLS_DWELL"] = args.dwell
    clf = L.MotionClassifier(params)
    mirror = []
    for _, ax, ay, az in samples:
        clf.update(ax, ay, az)
        mirror.append(clf.state)
    board = board_state_per_sample(n, events)

    agree = sum(1 for a, b in zip(board, mirror) if a == b)
    print(f"\nboard vs Python mirror: same state on {agree}/{n} samples ({agree / n:.1%})")
    # disagreements as runs, so a one-sample rounding shift shows up as a run of length 1
    runs, i = [], 0
    while i < n:
        if board[i] != mirror[i]:
            j = i
            while j < n and board[j] != mirror[j]:
                j += 1
            runs.append((i, j - i, board[i], mirror[i]))
            i = j
        else:
            i += 1
    print(f"disagreement runs: {len(runs)}  (lengths in samples: {[r[1] for r in runs][:15]})")
    for i, ln, b, m in runs[:8]:
        print(f"   at sample {i}: board={b} mirror={m} for {ln} samples")

    # ---- how well the board's states match the stage instructions
    print("\nboard state vs what you were doing (share of each stage, after the first 1.5 s):")
    print(f"  {'stage':8} {'expected':8} {'STILL':>6} {'MOVING':>7} {'WRITING':>8}")
    for stage, _ in STAGES:
        idx = [i for i, s in enumerate(samples) if s[0] == stage][50:]
        if not idx:
            continue
        st = [board[i] for i in idx]
        sh = {s: st.count(s) / len(st) for s in ("STILL", "MOVING", "WRITING")}
        print(f"  {stage:8} {EXPECT[stage]:8} {sh['STILL']:6.2f} {sh['MOVING']:7.2f} {sh['WRITING']:8.2f}   -> {sh[EXPECT[stage]]:.0%} as expected")
    taps = [e for e in events if e[1] == "T"]
    print(f"\ntap events during the routine (should be none): {taps}")


if __name__ == "__main__":
    main()
