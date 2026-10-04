"""
Offline tests for hand_depth.HandTracker using synthetic depth frames (no phone needed).

  python test_hand_depth.py
"""
import os
import sys

import numpy as np

from hand_depth import HandTracker

PASSED, FAILED = [], []
H, W, FPS = 256, 192, 60


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def scene(hand_z=None, noise=0.003, rng=np.random.default_rng(0)):
    """Wall at 1.0 m with sensor noise, plus an optional 40x40 px hand at hand_z."""
    d = (1.0 + rng.normal(0, noise, (H, W))).astype(np.float32)
    if hand_z is not None:
        d[100:140, 70:110] = hand_z + rng.normal(0, noise, (40, 40))
    return d


def main():
    tr = HandTracker()
    t = 0.0

    print("\nCalibration")
    for i in range(tr.bg_frames):
        r = tr.update(scene(), t); t += 1 / FPS
    check("background learned after bg_frames", tr.background is not None)

    print("\nCalibration with a hand briefly in view (should not be baked into the background)")
    tr2 = HandTracker()
    t2 = 0.0
    for i in range(tr2.bg_frames):
        # A hand sits in the frame for the first third of calibration, then leaves -
        # as if someone was still getting out of the way when calibration started.
        hand_z = 0.5 if i < tr2.bg_frames // 3 else None
        tr2.update(scene(hand_z), t2)
        t2 += 1 / FPS
    patch = tr2.background[100:140, 70:110]
    check("background recovers the true wall, not the transient hand",
          np.all(np.abs(patch - 1.0) < 0.02), f"patch mean {patch.mean():.3f} m (wall is 1.0 m)")
    r = tr2.update(scene(), t2)
    check("table itself isn't flagged as a hand afterward", not r["present"], f"area {r['area']}")
    r = tr2.update(scene(0.6), t2 + 1 / FPS)
    check("a real hand is still detected afterward", r["present"] and abs(r["depth"] - 0.6) < 0.01)

    print("\nEmpty scene")
    r = tr.update(scene(), t); t += 1 / FPS
    check("no hand on background only", not r["present"] and r["state"] == "no hand")

    print("\nSpeckle noise")
    d = scene()
    d[np.random.default_rng(1).random((H, W)) < 0.002] = 0.4  # scattered single-pixel spikes
    r = tr.update(d, t); t += 1 / FPS
    check("isolated noisy pixels ignored", not r["present"], f"area {r['area']}")

    print("\nHand enters at 0.60 m")
    r = tr.update(scene(0.60), t); t += 1 / FPS
    check("entered event", r["event"] == "entered")
    check("hand depth measured", abs(r["depth"] - 0.60) < 0.01, f"{r['depth']:.3f} m")
    check("hand area ~ 40x40 px", abs(r["area"] - 1600) < 200, f"{r['area']} px")

    print("\nHand moves closer at 0.5 m/s")
    z = 0.60
    for _ in range(30):
        z -= 0.5 / FPS
        r = tr.update(scene(z), t); t += 1 / FPS
    check("state = moving closer", r["state"] == "moving closer", r["state"])
    check("speed ~ -0.5 m/s", abs(r["velocity"] + 0.5) < 0.05, f"{r['velocity']:+.3f} m/s")
    check("frame change > 0", r["frame_change"] > 0, f"{r['frame_change'] * 1000:.2f} mm")

    print("\nHand holds still")
    for _ in range(30):
        r = tr.update(scene(z), t); t += 1 / FPS
    check("state = holding still", r["state"] == "holding still", f"{r['velocity']:+.3f} m/s")

    print("\nHand moves away at 0.3 m/s")
    for _ in range(30):
        z += 0.3 / FPS
        r = tr.update(scene(z), t); t += 1 / FPS
    check("state = moving away", r["state"] == "moving away", r["state"])
    check("speed ~ +0.3 m/s", abs(r["velocity"] - 0.3) < 0.05, f"{r['velocity']:+.3f} m/s")

    print("\nHand leaves")
    r = tr.update(scene(), t); t += 1 / FPS
    check("left event", r["event"] == "left")

    print("\nStored vectors")
    times, depths = tr.depth_series()
    check("depth series same length as times", len(times) == len(depths), f"{len(depths)} samples")
    check("NaN where no hand, values where hand", np.isnan(depths[0]) and np.isfinite(depths[-2]))
    fm = tr.frame_matrix()
    check("frame matrix is (N, H*W)", fm.shape == (len(tr.frames), H * W), str(fm.shape))

    print("\nRe-calibration")
    tr.reset_background()
    r = tr.update(scene(), t)
    check("reset returns to calibrating", r["state"] == "calibrating")

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("Failed: " + ", ".join(FAILED))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
