"""Record a short clip of raw camera frames for offline tuning of the pen-tip detector.

    python tools/record_frames.py                  # 3 s empty page, then 15 s with the pen
    python tools/record_frames.py --seconds 20 --out data/rec2

Script for the person at the desk (shown on screen too):
    0-3 s   hands OUT of the view, whole page visible, don't move
    3+ s    reach in with the pen: write a bit, point at the 4 corners, slide off the left and
            right edges, take the hand out and back in once
Frames are saved as JPEGs (t in ms in the file name) at --save-fps.
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.webcam import add_camera_args, config_from_args, make_camera  # noqa: E402

EMPTY_S = 3.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_camera_args(ap)
    ap.add_argument("--seconds", type=float, default=15.0, help="pen part, after the empty part")
    ap.add_argument("--save-fps", type=float, default=10.0)
    ap.add_argument("--out", default="data/tip_rec")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    for name in os.listdir(a.out):
        if name.endswith(".jpg"):
            os.remove(os.path.join(a.out, name))
    cam = make_camera(config_from_args(a))
    cam.start()
    t0 = time.monotonic()
    next_save, last_id, saved = 0.0, 0, 0
    try:
        while True:
            f = cam.wait(last_id, timeout=1.0)
            if f is None:
                continue
            last_id = f.id
            el = time.monotonic() - t0
            if el > EMPTY_S + a.seconds:
                break
            if el >= next_save:
                cv2.imwrite(os.path.join(a.out, f"{int(el * 1000):06d}.jpg"), f.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1
                next_save += 1.0 / a.save_fps
            view = f.image.copy()
            if el < EMPTY_S:
                msg, col = f"HANDS OUT, hold still  {EMPTY_S - el:.1f}", (0, 0, 255)
            else:
                msg, col = f"PEN IN: write, corners, off left/right  {EMPTY_S + a.seconds - el:.1f}", (0, 160, 0)
            cv2.putText(view, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            cv2.imshow("record", view)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    print(f"saved {saved} frames to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
