"""Replay a clip recorded by tools/record_frames.py through the pen-tip detector, offline.

    python tools/tip_replay.py                     # data/tip_rec -> data/tip_debug/sheet_*.jpg
    python tools/tip_replay.py data/rec2 --every 3

Prints one line per frame (background learned? tip x, y, source) and writes contact sheets:
camera frame with tip (red), arm entry (blue) and the pen axis (green) next to the foreground
mask the detector saw (pen pixels highlighted).
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hand_tip import TipDetector  # noqa: E402
from src.pen_profile import PROFILE_PATH, PenProfile  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default="data/tip_rec")
    ap.add_argument("--out", default="data/tip_debug")
    ap.add_argument("--every", type=int, default=6, help="put every Nth frame on the contact sheet")
    ap.add_argument("--scale", type=float, default=0.25, help="contact sheet tile scale")
    ap.add_argument("--no-profile", action="store_true", help=f"ignore {PROFILE_PATH} (generic pen model)")
    ap.add_argument("--no-depth", action="store_true", help="ignore the recorded *_depth.npy frames")
    ap.add_argument("--left-handed", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.folder, "*.jpg")))
    if not files:
        print(f"no frames in {a.folder}: record some with tools/record_frames.py")
        return 1
    os.makedirs(a.out, exist_ok=True)
    for old in glob.glob(os.path.join(a.out, "sheet_*.jpg")):
        os.remove(old)
    profile = None if a.no_profile else PenProfile.load()
    print(f"pen model: {'ours (' + PROFILE_PATH + ')' if profile else 'generic'}")
    det = TipDetector(right_handed=not a.left_handed, profile=profile)
    tiles = []
    for i, fn in enumerate(files):
        t = int(os.path.basename(fn)[:6]) / 1000.0
        img = cv2.imread(fn)
        depth_fn = fn[:-4] + "_depth.npy"
        depth = np.load(depth_fn).astype(np.float32) if not a.no_depth and os.path.exists(depth_fn) else None
        d = det.detect(img, t, depth)
        moved = det.scene_changed
        det.scene_changed = False
        if i % a.every == 0:
            v = img.copy()
            if d is not None:
                cv2.circle(v, (int(d.entry[0]), int(d.entry[1])), 14, (255, 0, 0), 3)
                if getattr(d, "pen_axis", None) is not None:
                    (x0, y0), (x1, y1) = d.pen_axis
                    cv2.line(v, (int(x0), int(y0)), (int(x1), int(y1)), (0, 220, 0), 3)
                cv2.circle(v, (int(d.x), int(d.y)), 10, (0, 0, 255), 3)
            label = f"{t:.1f}s {'' if d is None else getattr(d, 'source', '')}"
            cv2.putText(v, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            m = det.mask if det.mask is not None else np.zeros((10, 10), np.uint8)
            m = cv2.cvtColor(cv2.resize(m, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            pen = getattr(det, "pen_mask", None)
            if pen is not None:
                pm = cv2.resize(pen, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                m[pm] = (0, 220, 0)
            tiles.append(cv2.resize(np.hstack([v, m]), None, fx=a.scale, fy=a.scale))
        tip = "-" if d is None else f"({d.x:.0f}, {d.y:.0f}) {d.source} {d.end_cue}"
        print(f"{t:6.2f}s bg={det.has_background} tip={tip}{'  CAMERA MOVED' if moved else ''}")
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack(tiles[k:k + 4] + [blank] * (4 - len(tiles[k:k + 4]))) for k in range(0, len(tiles), 4)]
    for j in range(0, len(rows), 2):
        cv2.imwrite(os.path.join(a.out, f"sheet_{j // 2}.jpg"), np.vstack(rows[j:j + 2]))
    print(f"contact sheets in {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
