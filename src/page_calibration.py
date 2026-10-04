"""
Page calibration: 4 page corners in camera pixels -> homography -> PAGE-NORMALIZED coords.  Owner: Dev 2.

Corners are always ordered as the student reads the page: top-left, top-right, bottom-right,
bottom-left. That fixes the page orientation no matter how the camera is rotated.
Saved to data/page_calibration.json and reloaded by PenTracker on start.
Run `python tools/calibrate_page.py` (or press `c` in `python -m src.tracker`).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.contracts import PAGE_H_CM, PAGE_W_CM

CALIB_PATH = "data/page_calibration.json"
UNIT_SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
CORNER_NAMES = ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT")


@dataclass
class PageCalibration:
    corners_px: np.ndarray                 # (4, 2) float32, TL TR BR BL
    image_size: Tuple[int, int]            # (w, h) of the frames the corners were clicked on
    page_w_cm: float = PAGE_W_CM
    page_h_cm: float = PAGE_H_CM

    def __post_init__(self) -> None:
        self.corners_px = np.asarray(self.corners_px, np.float32).reshape(4, 2)
        self.H = cv2.getPerspectiveTransform(self.corners_px, UNIT_SQUARE)       # px -> page
        self.H_inv = cv2.getPerspectiveTransform(UNIT_SQUARE, self.corners_px)   # page -> px

    # -- mapping
    def to_page(self, x: float, y: float) -> Tuple[float, float]:
        p = self.H @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def to_pixels(self, u: float, v: float) -> Tuple[float, float]:
        p = self.H_inv @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def page_polygon_px(self, margin: float = 0.0) -> np.ndarray:
        """Page outline in pixels, grown by `margin` (page units) on every side."""
        m = margin
        sq = np.float32([[-m, -m], [1 + m, -m], [1 + m, 1 + m], [-m, 1 + m]]).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(sq, self.H_inv).reshape(4, 2)

    def scaled_to(self, w: int, h: int) -> "PageCalibration":
        """Same calibration for a different capture resolution (same camera pose)."""
        sw, sh = w / float(self.image_size[0]), h / float(self.image_size[1])
        if abs(sw - 1) < 1e-6 and abs(sh - 1) < 1e-6:
            return self
        return PageCalibration(self.corners_px * np.float32([sw, sh]), (w, h), self.page_w_cm, self.page_h_cm)

    # -- rectification
    def rectified_size(self, px_per_cm: float) -> Tuple[int, int]:
        return int(round(self.page_w_cm * px_per_cm)), int(round(self.page_h_cm * px_per_cm))

    def rectify_matrix(self, px_per_cm: float) -> np.ndarray:
        w, h = self.rectified_size(px_per_cm)
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        return cv2.getPerspectiveTransform(self.corners_px, dst)

    def rectify(self, frame: np.ndarray, px_per_cm: float = 20.0,
                interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        return cv2.warpPerspective(frame, self.rectify_matrix(px_per_cm),
                                   self.rectified_size(px_per_cm), flags=interpolation)

    # -- persistence
    def save(self, path: str = CALIB_PATH) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "corners_px": self.corners_px.round(2).tolist(),
                "corner_order": list(CORNER_NAMES),
                "image_size": list(self.image_size),
                "page_w_cm": self.page_w_cm, "page_h_cm": self.page_h_cm,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)

    @staticmethod
    def load(path: str = CALIB_PATH) -> Optional["PageCalibration"]:
        if not os.path.exists(path):
            return None
        with open(path) as f:
            d = json.load(f)
        return PageCalibration(np.float32(d["corners_px"]), tuple(d["image_size"]),
                               d.get("page_w_cm", PAGE_W_CM), d.get("page_h_cm", PAGE_H_CM))


# --------------------------------------------------------------------------- corner helpers
def order_corners(pts: Sequence[Sequence[float]]) -> np.ndarray:
    """Order 4 points TL, TR, BR, BL assuming the page is roughly upright in the image."""
    p = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = p.sum(axis=1), np.diff(p, axis=1).ravel()
    return np.float32([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]])


def detect_page_corners(frame: np.ndarray, min_area_frac: float = 0.08) -> Optional[np.ndarray]:
    """Find the white sheet on a darker desk: largest bright 4-sided contour. None if not found."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < min_area_frac * w * h:
            break
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return order_corners(approx.reshape(4, 2))
    return None


# --------------------------------------------------------------------------- interactive UI
def click_corners(get_frame: Callable[[], Optional[np.ndarray]], window: str = "calibrate page",
                  initial: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Live view; click the 4 page corners in order TL, TR, BR, BL (drag a point to adjust).
    Keys: a = auto-detect   r = restart   ENTER = accept (needs 4 points)   ESC = cancel
    Returns (4, 2) float32 corners or None if cancelled.
    """
    pts: List[List[float]] = [] if initial is None else np.asarray(initial, np.float32).tolist()
    state = {"drag": None, "mouse": (0, 0)}

    def on_mouse(event, x, y, flags, _param):
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            near = [i for i, p in enumerate(pts) if (p[0] - x) ** 2 + (p[1] - y) ** 2 < 15 ** 2]
            if near:
                state["drag"] = near[0]
            elif len(pts) < 4:
                pts.append([float(x), float(y)])
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
            pts[state["drag"]] = [float(x), float(y)]
        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = None

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    result = None
    while True:
        frame = get_frame()
        if frame is None:
            if cv2.waitKey(10) & 0xFF == 27:
                break
            continue
        view = frame.copy()
        h, w = view.shape[:2]
        if len(pts) >= 2:
            cv2.polylines(view, [np.int32(pts)], len(pts) == 4, (0, 255, 0), 2)
        for i, p in enumerate(pts):
            cv2.circle(view, (int(p[0]), int(p[1])), 6, (0, 0, 255), 2)
            cv2.putText(view, CORNER_NAMES[i], (int(p[0]) + 8, int(p[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        _draw_loupe(view, frame, state["mouse"])
        msg = (f"click {CORNER_NAMES[len(pts)]} corner of the page" if len(pts) < 4
               else "ENTER = accept   drag to adjust")
        cv2.putText(view, msg + "   a=auto  r=restart  ESC=cancel", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(window, view)
        key = cv2.waitKey(15) & 0xFF
        if key == 27:
            break
        if key == ord("r"):
            pts.clear()
        elif key == ord("a"):
            found = detect_page_corners(frame)
            if found is None:
                print("[calibrate] auto-detect failed: click the corners (dark desk + white page helps)")
            else:
                pts[:] = found.tolist()
        elif key in (13, 10) and len(pts) == 4:
            result = np.float32(pts)
            break
    cv2.destroyWindow(window)
    return result


def _draw_loupe(view: np.ndarray, frame: np.ndarray, mouse: Tuple[int, int],
                r: int = 20, zoom: int = 4) -> None:
    h, w = frame.shape[:2]
    x, y = mouse
    x0, y0 = min(max(0, x - r), w - 2 * r), min(max(0, y - r), h - 2 * r)
    if x0 < 0 or y0 < 0:
        return
    patch = cv2.resize(frame[y0:y0 + 2 * r, x0:x0 + 2 * r], (2 * r * zoom, 2 * r * zoom),
                       interpolation=cv2.INTER_NEAREST)
    c = r * zoom
    cv2.line(patch, (c, 0), (c, 2 * c), (0, 255, 255), 1)
    cv2.line(patch, (0, c), (2 * c, c), (0, 255, 255), 1)
    s = patch.shape[0]
    if s + 10 < h and s + 10 < w:
        view[10:10 + s, w - 10 - s:w - 10] = patch
