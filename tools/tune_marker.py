"""Step 3: tune the HSV mask for the pen marker and save it (after camera_check + calibrate_page).

Usage:
    python tools/tune_marker.py                   # camera from data/camera.json
    python tools/tune_marker.py --index 1         # explicit camera index

Windows:
    "frame"  camera view with the detected marker; CLICK THE MARKER to auto-fill the sliders
    "mask"   what the mask keeps (the marker should be the only white blob)
    "hsv"    sliders for fine-tuning

Keys:  s = save to data/marker_hsv.json    p = toggle page-only search    d = defaults    ESC = quit

Tips:
- Pick a marker colour that appears nowhere else (page, desk, skin, sleeve). The status line
  warns when more than one blob passes the mask.
- Tune in the SAME lighting (and with the same data/camera.json exposure) you will demo in.
  Re-tune at the venue.
- Red wraps around hue 0/179. If your marker is red, set H low > H high
  (e.g. low 170, high 10) and the mask will use both ends. Clicking handles this automatically.
- Detection here uses src.marker.MarkerDetector, exactly what PenTracker runs, including the
  page-only search area from data/page_calibration.json.
"""
import argparse
import os
import sys
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.marker import (  # noqa: E402
    DEFAULTS, LIMITS, MARKER_PATH, MarkerDetector, find_blobs, load_marker_params,
    sample_hsv_range, save_marker_params,
)
from src.page_calibration import CALIB_PATH, PageCalibration  # noqa: E402
from src.tracker import ROI_MARGIN  # noqa: E402
from src.webcam import CameraError, add_camera_args, config_from_args, make_camera  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_camera_args(ap)
    args = ap.parse_args()

    cam = make_camera(config_from_args(args))
    try:
        cam.start()
    except CameraError as e:
        print(f"{e}\nRun tools/camera_check.py --list.")
        return

    vals = load_marker_params()
    detector = MarkerDetector(vals)
    calib_file = PageCalibration.load(CALIB_PATH)
    page_only = calib_file is not None
    calib = None
    if calib_file is None:
        print("no page calibration: searching the whole frame (run tools/calibrate_page.py first for best results)")

    cv2.namedWindow("hsv")
    cv2.namedWindow("frame")
    for name, mx in LIMITS.items():
        cv2.createTrackbar(name, "hsv", int(vals[name]), mx, lambda _x: None)

    latest = {"img": None}

    def on_click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and latest["img"] is not None:
            hsv = cv2.cvtColor(cv2.GaussianBlur(latest["img"], (5, 5), 0), cv2.COLOR_BGR2HSV)
            new = sample_hsv_range(hsv, x, y, min_area=cv2.getTrackbarPos("min_area", "hsv"))
            for k, v in new.items():
                cv2.setTrackbarPos(k, "hsv", int(v))
            print(f"sampled ({x},{y}) -> {new}")

    cv2.setMouseCallback("frame", on_click)
    hits = deque(maxlen=30)
    last_id = 0
    roi_on = False
    try:
        while True:
            f = cam.wait(last_id, timeout=1.0)
            if f is None:
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            last_id = f.id
            frame = f.image
            latest["img"] = frame
            for name in LIMITS:
                vals[name] = cv2.getTrackbarPos(name, "hsv")
            detector.params = dict(vals)
            if calib_file is not None and calib is None:
                calib = calib_file.scaled_to(frame.shape[1], frame.shape[0])
            want_roi = bool(page_only and calib)
            if want_roi != roi_on:
                detector.set_roi(calib.page_polygon_px(ROI_MARGIN) if want_roi else None)
                roi_on = want_roi

            t0 = time.perf_counter()
            det = detector.detect(frame)
            det_ms = (time.perf_counter() - t0) * 1000
            hits.append(det is not None)
            mask = detector.mask(frame)
            n_full = len(find_blobs(mask, vals["min_area"]))

            view = frame.copy()
            if calib is not None:
                cv2.polylines(view, [np.int32(calib.page_polygon_px())], True, (0, 255, 0), 1)
                if page_only:
                    cv2.polylines(view, [np.int32(calib.page_polygon_px(ROI_MARGIN))], True, (0, 120, 0), 1)
            if det is not None:
                cv2.circle(view, (int(det.x), int(det.y)), 10, (0, 0, 255), 2)
                text = f"marker ({det.x:.0f},{det.y:.0f}) area {det.area:.0f}"
                if calib is not None:
                    u, v = calib.to_page(det.x, det.y)
                    text += f"  page ({u:.3f},{v:.3f}) = ({u * calib.page_w_cm:.1f},{v * calib.page_h_cm:.1f}) cm"
                cv2.putText(view, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(view, "NO MARKER  (click the marker to sample its colour)", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            rate = 100.0 * sum(hits) / max(1, len(hits))
            status = (f"detect {rate:.0f}%  {det_ms:.1f} ms  camera {cam.fps:.1f} fps  "
                      f"search: {'page' if page_only and calib else 'whole frame'}")
            cv2.putText(view, status, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if n_full > 1:
                cv2.putText(view, f"WARNING: {n_full} blobs pass the mask - the colour appears elsewhere",
                            (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("frame", view)
            cv2.imshow("mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("s"):
                save_marker_params(vals, MARKER_PATH)
                print(f"saved {MARKER_PATH}: {vals}")
            elif key == ord("p") and calib_file is not None:
                page_only = not page_only
            elif key == ord("d"):
                for k, v in DEFAULTS.items():
                    cv2.setTrackbarPos(k, "hsv", v)
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
