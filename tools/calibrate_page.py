"""Step 2: tell the tracker where the page is (4 corners -> homography) -> data/page_calibration.json.

Usage:
    python tools/calibrate_page.py            # live view: click TL, TR, BR, BL (or press a = auto-detect)
    python tools/calibrate_page.py --auto     # auto-detect the white sheet and save without the UI

Run after tools/camera_check.py (same data/camera.json resolution) and again whenever the camera
or the taped page moves. The preview after clicking shows the rectified page with a 1-inch grid:
the grid squares must look square and the page edges straight.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.page_calibration import CALIB_PATH, PageCalibration, click_corners, detect_page_corners  # noqa: E402
from src.webcam import CameraError, add_camera_args, config_from_args, make_camera  # noqa: E402


def preview_rectified(calib, frame, px_per_cm=25.0):
    page = calib.rectify(frame, px_per_cm)
    h, w = page.shape[:2]
    step = 2.54 * px_per_cm
    for i in range(1, int(w / step) + 1):
        cv2.line(page, (int(i * step), 0), (int(i * step), h), (255, 160, 0), 1)
    for j in range(1, int(h / step) + 1):
        cv2.line(page, (0, int(j * step)), (w, int(j * step)), (255, 160, 0), 1)
    cv2.putText(page, "1-inch grid. ENTER = save  ESC = discard", (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("rectified page", page)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10, 27):
            cv2.destroyWindow("rectified page")
            return key != 27


def main():
    ap = argparse.ArgumentParser()
    add_camera_args(ap)
    ap.add_argument("--auto", action="store_true", help="auto-detect the page and save without the UI")
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
                print("auto-detect failed; run without --auto and click the corners")
                return 1
            PageCalibration(corners, (w, h)).save(args.out)
            print(f"saved {args.out}: {corners.round(1).tolist()}")
            return 0

        old = PageCalibration.load(args.out)
        initial = old.scaled_to(w, h).corners_px if old is not None else None
        while True:
            corners = click_corners(lambda: cam.latest().image, initial=initial)
            if corners is None:
                print("cancelled; nothing saved")
                return 1
            calib = PageCalibration(corners, (w, h))
            if preview_rectified(calib, cam.latest().image):
                calib.save(args.out)
                print(f"saved {args.out}  ({w}x{h}, corners {corners.round(1).tolist()})")
                return 0
            initial = corners
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
