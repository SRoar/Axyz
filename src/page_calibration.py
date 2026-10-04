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


def detect_page_corners(frame: np.ndarray, min_area_frac: float = 0.08,
                        min_contrast: float = 30.0, border_px: int = 3) -> Optional[np.ndarray]:
    """Find the white sheet on a darker desk: largest bright 4-sided contour. None if not found.
    Rejects quads touching the frame edge (page not fully in view / no page at all) and quads that
    are not clearly brighter than their surroundings."""
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
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            q = approx.reshape(4, 2)
            if (q[:, 0].min() < border_px or q[:, 1].min() < border_px
                    or q[:, 0].max() > w - 1 - border_px or q[:, 1].max() > h - 1 - border_px):
                break
            inside = np.zeros_like(gray)
            cv2.fillConvexPoly(inside, q.astype(np.int32), 255)
            if cv2.mean(gray, inside)[0] - cv2.mean(gray, 255 - inside)[0] < min_contrast:
                break
            return order_corners(q)
    return None


class StablePageDetector:
    """Hands-free calibration: accepts the auto-detected page once its corners have stayed
    within `tol_px` for `hold_s` seconds. Detected corners wobble a few pixels frame to frame
    (phone stabilisation / focus), and single frames can miss: misses shorter than `max_gap_s`
    are ignored; a longer gap (hand over a corner, page moved away) restarts the countdown."""

    def __init__(self, hold_s: float = 1.5, tol_px: Optional[float] = None, max_gap_s: float = 1.0,
                 tol_frac: float = 0.02) -> None:
        """tol_px: allowed corner wobble; default tol_frac x the larger frame side."""
        self.hold_s, self.tol_px_fixed, self.max_gap_s, self.tol_frac = hold_s, tol_px, max_gap_s, tol_frac
        self.tol_px = 10.0 if tol_px is None else tol_px
        self.reset()

    def reset(self) -> None:
        self._ref: Optional[np.ndarray] = None
        self._since = 0.0
        self._last_seen = -1e9
        self.corners: Optional[np.ndarray] = None
        self.progress = 0.0

    def update(self, frame: np.ndarray, t: float) -> float:
        """Feed one frame; returns progress 0..1 (1 = stable, `corners` is ready to use)."""
        if self.tol_px_fixed is None:
            self.tol_px = self.tol_frac * max(frame.shape[:2])
        found = detect_page_corners(frame)
        if found is None:
            if t - self._last_seen > self.max_gap_s:
                self.reset()
            return self.progress
        self._last_seen = t
        if self._ref is None or np.abs(found - self._ref).max() > self.tol_px:
            self._ref, self._since = found, t
        self.corners = 0.7 * self.corners + 0.3 * found if self.corners is not None and \
            np.abs(found - self.corners).max() <= self.tol_px else found
        self.progress = min(1.0, (t - self._since) / self.hold_s) if self.hold_s > 0 else 1.0
        return self.progress


# --------------------------------------------------------------------------- interactive UI
def click_corners(get_frame: Callable[[], Optional[np.ndarray]], window: str = "calibrate page",
                  initial: Optional[np.ndarray] = None, auto: bool = True,
                  auto_accept_s: Optional[float] = None) -> Optional[np.ndarray]:
    """
    Live view. AUTO mode (default) finds the page by itself and keeps the outline on it. With
    `auto_accept_s` it also ACCEPTS by itself once the page has held still that long (no keys);
    otherwise press ENTER. Clicking or dragging switches to manual: click the 4 page corners in
    order TL, TR, BR, BL (drag a point to adjust).
    Keys: a = back to auto   r = restart manual clicking   ENTER = accept (needs 4 points)   ESC = cancel
    Returns (4, 2) float32 corners or None if cancelled.
    """
    pts: List[List[float]] = [] if initial is None else np.asarray(initial, np.float32).tolist()
    state = {"drag": None, "mouse": (0, 0), "auto": auto, "found": False}
    stable = StablePageDetector(hold_s=auto_accept_s or 0.0)
    progress = 0.0
    n_frame = 0

    def on_mouse(event, x, y, flags, _param):
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["auto"] = False
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
        n_frame += 1
        if state["auto"] and n_frame % 3 == 0:
            progress = stable.update(frame, time.monotonic())
            state["found"] = stable.corners is not None
            if stable.corners is not None:
                pts[:] = stable.corners.tolist()
            if auto_accept_s and progress >= 1.0:
                result = np.float32(pts)
                break
        elif not state["auto"]:
            stable.reset()
            progress = 0.0
        view = frame.copy()
        h, w = view.shape[:2]
        auto_ok = state["auto"] and state["found"]
        col = (0, 255, 0) if (auto_ok or not state["auto"]) else (0, 165, 255)
        if len(pts) >= 2:
            cv2.polylines(view, [np.int32(pts)], len(pts) == 4, col, 2)
        for i, p in enumerate(pts):
            cv2.circle(view, (int(p[0]), int(p[1])), 6, (0, 0, 255), 2)
            cv2.putText(view, CORNER_NAMES[i], (int(p[0]) + 8, int(p[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        _draw_loupe(view, frame, state["mouse"])
        if state["auto"]:
            if auto_ok and auto_accept_s:
                msg = f"AUTO: page found - hold still... saving in {max(0.0, (1 - progress) * auto_accept_s):.1f}s"
                cv2.rectangle(view, (10, h - 70), (10 + int(progress * (w - 20)), h - 60), (0, 255, 0), -1)
            elif auto_ok:
                msg = "AUTO: page found - ENTER = accept, click a corner to adjust"
            else:
                msg = "AUTO: looking for the page (whole sheet in view, darker desk)... or click corners"
        else:
            msg = (f"click {CORNER_NAMES[len(pts)]} corner of the page" if len(pts) < 4
                   else "ENTER = accept   drag to adjust")
        cv2.putText(view, msg, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(view, "a=auto  r=restart manual  ESC=cancel", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow(window, view)
        key = cv2.waitKey(15) & 0xFF
        if key == 27:
            break
        if key == ord("r"):
            state["auto"] = False
            pts.clear()
        elif key == ord("a"):
            state["auto"] = True
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
