"""
Offline grid-search for TAP_ACT_G / TAP_PEAK_G against every real recording in data/taps/,
replaying through the exact same TapDetector logic the firmware runs (tools/imu_logic.py).
Picks the pair with the fewest missed/wrong taps that still has ZERO false positives during
the recorded active (writing/moving) and still phases - that's the real safety constraint.
"""
import glob
import re
import sys

sys.path.insert(0, "tools")
from imu_logic import TapDetector, load_params  # noqa: E402


def parse_log(path):
    marks = []  # (kind, expected, t_start, t_end)
    samples = []  # (t_ms, ax, ay, az)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("# ") and "," in line and not line.startswith("# marks"):
                kind, expected, t0, t1 = line[2:].split(",")
                marks.append((kind, int(expected), float(t0) * 1000, float(t1) * 1000))
            elif line.startswith("0.") or re.match(r"^\d+\.\d+,S,", line):
                parts = line.split(",")
                if len(parts) >= 6 and parts[1] == "S":
                    t, ax, ay, az = float(parts[0]) * 1000, float(parts[3]), float(parts[4]), float(parts[5])
                    samples.append((t, ax, ay, az))
    return marks, samples


def replay(marks, samples, params):
    det = TapDetector(params)
    events = []  # (t_ms, count)
    for t, ax, ay, az in samples:
        n = det.update(t, ax, ay, az, allowed=True)
        if n:
            events.append((t, n))

    results = {"single": [0, 0], "double": [0, 0], "false_in_active_or_still": 0}  # [correct, total]
    used = [False] * len(events)
    for kind, expected, t0, t1 in marks:
        if kind in ("single", "double"):
            results[kind][1] += 1
            # a tap reported up to ~1s after the window end still counts (TAP_GROUP_MS delay)
            for i, (t, n) in enumerate(events):
                if not used[i] and t0 - 50 <= t <= t1 + 1000 and n == expected:
                    used[i] = True
                    results[kind][0] += 1
                    break
        elif kind in ("active", "still"):
            for i, (t, n) in enumerate(events):
                if not used[i] and t0 <= t <= t1:
                    results["false_in_active_or_still"] += 1
                    used[i] = True
    extra = sum(1 for u in used if not u)
    return results, extra


def main():
    files = sorted(glob.glob("data/taps/*.txt"))
    print(f"Replaying {len(files)} recorded sessions: {files}\n")
    parsed = [parse_log(f) for f in files]
    base = load_params()

    act_candidates = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
    peak_candidates = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]

    best = None
    print(f"{'ACT':>5} {'PEAK':>5} {'single':>8} {'double':>8} {'false+':>7} {'extra':>6}")
    for act in act_candidates:
        for peak in peak_candidates:
            if peak < act:
                continue
            params = dict(base, TAP_ACT_G=act, TAP_PEAK_G=peak)
            tot = {"single": [0, 0], "double": [0, 0], "false_in_active_or_still": 0}
            extra_total = 0
            for marks, samples in parsed:
                r, extra = replay(marks, samples, params)
                for k in ("single", "double"):
                    tot[k][0] += r[k][0]
                    tot[k][1] += r[k][1]
                tot["false_in_active_or_still"] += r["false_in_active_or_still"]
                extra_total += extra
            score = tot["single"][0] + tot["double"][0]
            safe = tot["false_in_active_or_still"] == 0
            line = (f"{act:5.2f} {peak:5.2f} {tot['single'][0]:3}/{tot['single'][1]:<4} "
                   f"{tot['double'][0]:3}/{tot['double'][1]:<4} {tot['false_in_active_or_still']:7} {extra_total:6}")
            print(line + ("  <-- unsafe (false tap during writing/still)" if not safe else ""))
            if safe and (best is None or score > best[0]):
                best = (score, act, peak, tot, extra_total)

    print()
    if best is None:
        print("No candidate avoided false positives during active/still - none are safe to ship as-is.")
    else:
        score, act, peak, tot, extra = best
        print(f"BEST SAFE CANDIDATE: TAP_ACT_G={act}  TAP_PEAK_G={peak}")
        print(f"  single: {tot['single'][0]}/{tot['single'][1]}   double: {tot['double'][0]}/{tot['double'][1]}"
              f"   false positives: {tot['false_in_active_or_still']}   extra/unmatched events: {extra}")


if __name__ == "__main__":
    main()
