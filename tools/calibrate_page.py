"""Step 1: tell the tracker where the page is (4 corners -> homography) -> data/page_calibration.json.

Usage:
    python tools/calibrate_page.py             # hands-free: finds the page, saves once it holds still 1.5 s
    python tools/calibrate_page.py --confirm   # same, but you press ENTER to accept and again on the preview
    python tools/calibrate_page.py --auto      # no window at all: detect once and save

Optional: if there is no calibration file, PenTracker (python -m src.tracker) calibrates itself the
same way as soon as it sees the whole page holding still.

Hands-free needs the whole sheet in view on a darker desk. Clicking a corner in the window
switches to manual: click TL, TR, BR, BL then press ENTER. Re-run whenever the camera or the taped
page moves. The preview shows the rectified page with a 1-inch grid: squares must look square.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2  # noqa: E402

from src.page_calibration import CALIB_PATH, PageCalibration, click_corners, detect_page_corners  # noqa: E402
from src.webcam import CameraError, add_camera_args, config_from_args, make_camera  # noqa: E402

AUTO_ACCEPT_S = 1.5


def preview_rectified(calib, frame, px_per_cm=25.0, wait_ms=None):
    """Show the rectified page with a 1-inch grid. wait_ms=None: wait for ENTER/ESC; else auto-close."""
    page = calib.rectify(frame, px_per_cm)
    h, w = page.shape[:2]
    step = 2.54 * px_per_cm
    for i in range(1, int(w / step) + 1):
        cv2.line(page, (int(i * step), 0), (int(i * step), h), (255, 160, 0), 1)
    for j in range(1, int(h / step) + 1):
        cv2.line(page, (0, int(j * step)), (w, int(j * step)), (255, 160, 0), 1)
    msg = "1-inch grid. ENTER = save  ESC = discard" if wait_ms is None else "saved - 1-inch grid"
    cv2.putText(page, msg, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("rectified page", page)
    if wait_ms is not None:
        cv2.waitKey(wait_ms)
        cv2.destroyWindow("rectified page")
        return True
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10, 27):
            cv2.destroyWindow("rectified page")
            return key != 27


def main():
    ap = argparse.ArgumentParser()
    add_camera_args(ap)
    ap.add_argument("--auto", action="store_true", help="detect the page once and save, no window")
    ap.add_argument("--confirm", action="store_true", help="press ENTER to accept instead of auto-saving")
    ap.add_argument("--out", default=CALIB_PATH)
    args = ap.parse_args()

    cam = make_camera(config_from_args(args))
    try:
        cam.start()
    except CameraError as e:
        print(f"{e}\nRun tools/camera_check.py --list.")
        return 1
    try:
        time.sleep(0.5)
        frame = cam.latest().image
        h, w = frame.shape[:2]
        if args.auto:
            corners = detect_page_corners(frame)
            if corners is None:
                print("auto-detect failed; run without --auto")
                return 1
            PageCalibration(corners, (w, h)).save(args.out)
            print(f"saved {args.out}: {corners.round(1).tolist()}")
            return 0

        print("looking for the page: keep the whole sheet in view and still"
              + ("" if args.confirm else f"; it saves by itself after {AUTO_ACCEPT_S:.1f}s"))
        old = PageCalibration.load(args.out)
        initial = old.scaled_to(w, h).corners_px if old is not None else None
        while True:
            corners = click_corners(lambda: cam.latest().image, initial=initial,
                                    auto_accept_s=None if args.confirm else AUTO_ACCEPT_S)
            if corners is None:
                print("cancelled; nothing saved")
                return 1
            calib = PageCalibration(corners, (w, h))
            if args.confirm and not preview_rectified(calib, cam.latest().image):
                initial = corners
                continue
            calib.save(args.out)
            print(f"saved {args.out}  ({w}x{h}, corners {corners.round(1).tolist()})")
            if not args.confirm:
                preview_rectified(calib, cam.latest().image, wait_ms=2000)
            return 0
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
