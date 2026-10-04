"""Teach the tracker what OUR pen looks like (run once per pen / lighting setup).

    python tools/capture_pen.py                               # live: overhead camera
    python tools/capture_pen.py --image data/pen_reference.jpg --tip 262,522   # from a photo

Live: lay the pen on the page (or hold it), wait for the green outline around the pen, then
CLICK THE WRITING TIP end. r = retry, ESC = quit. With --image and --tip it runs headless.
Saves data/pen_profile.json (colours of the pen + of its writing end and back end). The tracker
loads it automatically.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pen_profile import PROFILE_PATH, PenSegment, build_profile, segment_pen  # noqa: E402


def orient(seg: PenSegment, click) -> PenSegment:
    """Make seg.tip the end nearest the click (the writing tip)."""
    da = np.hypot(seg.tip[0] - click[0], seg.tip[1] - click[1])
    db = np.hypot(seg.back[0] - click[0], seg.back[1] - click[1])
    return seg if da <= db else PenSegment(seg.mask, seg.back, seg.tip, seg.width)


def preview(frame: np.ndarray, seg, profile=None) -> np.ndarray:
    v = frame.copy()
    if seg is not None:
        cnts, _ = cv2.findContours(seg.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(v, cnts, -1, (0, 255, 0), 2)
        for p in (seg.tip, seg.back):
            cv2.circle(v, (int(p[0]), int(p[1])), 12, (0, 200, 255), 2)
    if profile is not None:
        cv2.circle(v, (int(seg.tip[0]), int(seg.tip[1])), 14, (0, 0, 255), 3)
        prob = profile.probability(frame)
        v = np.hstack([v, cv2.cvtColor(prob, cv2.COLOR_GRAY2BGR)])
    return v


def save(frame, seg, out):
    profile = build_profile(frame, seg)
    profile.save(out)
    print(f"saved {out}: pen {profile.length_px:.0f} px long, {profile.width_px:.0f} px wide; "
          f"writing-end Lab {np.round(profile.tip_lab, 0)}, back-end Lab {np.round(profile.back_lab, 0)}")
    return profile


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="build the profile from a photo instead of the live camera")
    ap.add_argument("--tip", help="x,y of the writing tip in the photo (pixels)")
    ap.add_argument("--out", default=PROFILE_PATH)
    ap.add_argument("--no-preview", action="store_true")
    a, rest = ap.parse_known_args()

    if a.image:
        frame = cv2.imread(a.image)
        if frame is None:
            print(f"cannot read {a.image}")
            return 1
        seg = segment_pen(frame)
        if seg is None:
            print("no pen found in the photo (needs a long, narrow, non-skin object on the paper)")
            return 1
        if not a.tip:
            print(f"pen ends at {np.round(seg.tip)} and {np.round(seg.back)}: pass --tip x,y for the writing end")
            return 1
        seg = orient(seg, tuple(float(v) for v in a.tip.split(",")))
        profile = save(frame, seg, a.out)
        cv2.imwrite(os.path.splitext(a.out)[0] + "_check.jpg", preview(frame, seg, profile))
        return 0

    from src.webcam import add_camera_args, config_from_args, make_camera
    cp = argparse.ArgumentParser()
    add_camera_args(cp)
    cam = make_camera(config_from_args(cp.parse_args(rest)))
    cam.start()
    clicked = []
    win = "capture pen: click the WRITING TIP"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda ev, x, y, *_: clicked.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    seg, frame, last_id, n = None, None, 0, 0
    try:
        while True:
            f = cam.wait(last_id, timeout=1.0)
            if f is None:
                continue
            last_id, n = f.id, n + 1
            frame = f.image
            if n % 5 == 0:
                seg = segment_pen(frame)
            v = preview(frame, seg)
            msg = "click the WRITING TIP end" if seg is not None else "lay the pen on the page..."
            cv2.putText(v, msg, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow(win, v)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return 1
            if clicked and seg is not None:
                seg = orient(seg, clicked.pop())
                profile = save(frame, seg, a.out)
                if not a.no_preview:
                    cv2.imshow(win, preview(frame, seg, profile))
                    cv2.waitKey(1)
                    time.sleep(2.0)
                return 0
            clicked.clear()
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
