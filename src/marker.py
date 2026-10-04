"""
Pen-marker detection (HSV colour mask).  Owner: Dev 2.

`tools/tune_marker.py` writes data/marker_hsv.json with exactly these functions, and PenTracker
detects with the same functions, so what you see in the tuner is what the tracker sees.

MarkerDetector is the fast path used by the tracker: it searches only inside the calibrated page
(+ margin), on a downscaled copy, then refines the centroid at full resolution around the blob.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

MARKER_PATH = "data/marker_hsv.json"
DEFAULTS: Dict[str, int] = {"h_lo": 35, "h_hi": 85, "s_lo": 100, "s_hi": 255,
                            "v_lo": 80, "v_hi": 255, "min_area": 60}
LIMITS: Dict[str, int] = {"h_lo": 179, "h_hi": 179, "s_lo": 255, "s_hi": 255,
                          "v_lo": 255, "v_hi": 255, "min_area": 2000}
_KERNEL = np.ones((3, 3), np.uint8)


def load_marker_params(path: str = MARKER_PATH) -> Dict[str, int]:
    if os.path.exists(path):
        with open(path) as f:
            return {**DEFAULTS, **{k: int(v) for k, v in json.load(f).items() if k in DEFAULTS}}
    return dict(DEFAULTS)


def save_marker_params(params: Dict[str, int], path: str = MARKER_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: int(params[k]) for k in DEFAULTS}, f, indent=2)


def build_mask(hsv: np.ndarray, v: Dict[str, int]) -> np.ndarray:
    """Binary mask of marker-coloured pixels. Red wraps around hue 0: set h_lo > h_hi."""
    lo_sv, hi_sv = (v["s_lo"], v["v_lo"]), (v["s_hi"], v["v_hi"])
    if v["h_lo"] <= v["h_hi"]:
        mask = cv2.inRange(hsv, (v["h_lo"], *lo_sv), (v["h_hi"], *hi_sv))
    else:
        mask = cv2.bitwise_or(cv2.inRange(hsv, (v["h_lo"], *lo_sv), (179, *hi_sv)),
                              cv2.inRange(hsv, (0, *lo_sv), (v["h_hi"], *hi_sv)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)


def find_blobs(mask: np.ndarray, min_area: float) -> List[Tuple[float, float, float]]:
    """All blobs >= min_area as (cx, cy, area), largest first."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] > 0:
            out.append((m["m10"] / m["m00"], m["m01"] / m["m00"], area))
    return sorted(out, key=lambda b: -b[2])


def find_marker(mask: np.ndarray, min_area: float) -> Optional[Tuple[float, float, float]]:
    blobs = find_blobs(mask, min_area)
    return blobs[0] if blobs else None


def sample_hsv_range(hsv: np.ndarray, x: int, y: int, radius: int = 4,
                     min_area: int = DEFAULTS["min_area"]) -> Dict[str, int]:
    """Click-to-tune: an HSV range that covers the patch around (x, y) with some slack."""
    h, w = hsv.shape[:2]
    patch = hsv[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)]
    patch = patch.reshape(-1, 3).astype(np.int32)
    hue = patch[:, 0]
    # circular hue statistics so red (around 0/179) works
    ang = hue * (2 * np.pi / 180.0)
    mean_h = int(round(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * 180.0 / (2 * np.pi))) % 180
    spread = int(np.percentile(np.abs(((hue - mean_h + 90) % 180) - 90), 95))
    dh = max(8, spread + 6)
    return {
        "h_lo": (mean_h - dh) % 180, "h_hi": (mean_h + dh) % 180,
        "s_lo": int(max(0, np.percentile(patch[:, 1], 5) - 40)), "s_hi": 255,
        "v_lo": int(max(0, np.percentile(patch[:, 2], 5) - 50)), "v_hi": 255,
        "min_area": int(min_area),
    }


@dataclass
class Detection:
    x: float          # full-frame pixel coordinates of the marker centroid
    y: float
    area: float       # full-resolution pixels
    n_blobs: int      # candidate blobs seen (>1 means the colour appears elsewhere too)


class MarkerDetector:
    def __init__(self, params: Optional[Dict[str, int]] = None, max_work_width: int = 640) -> None:
        self.params = dict(params or load_marker_params())
        self.max_work_width = max_work_width
        self._roi_poly: Optional[np.ndarray] = None
        self._roi_cache: Dict[Tuple[int, int], tuple] = {}

    def set_roi(self, polygon_px: Optional[Sequence[Sequence[float]]]) -> None:
        """Only search inside this polygon (e.g. the calibrated page + margin). None = whole frame."""
        self._roi_poly = None if polygon_px is None else np.asarray(polygon_px, np.float32)
        self._roi_cache.clear()

    def _roi(self, shape: Tuple[int, ...]):
        key = shape[:2]
        if key in self._roi_cache:
            return self._roi_cache[key]
        h, w = key
        if self._roi_poly is None:
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            x0, y0 = np.floor(self._roi_poly.min(axis=0)).astype(int)
            x1, y1 = np.ceil(self._roi_poly.max(axis=0)).astype(int)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                x0, y0, x1, y1 = 0, 0, w, h
        s = min(1.0, self.max_work_width / float(max(1, x1 - x0)))
        size = (max(1, int(round((x1 - x0) * s))), max(1, int(round((y1 - y0) * s))))
        poly_mask = None
        if self._roi_poly is not None:
            poly_mask = np.zeros((size[1], size[0]), np.uint8)
            pts = ((self._roi_poly - (x0, y0)) * s).round().astype(np.int32)
            cv2.fillPoly(poly_mask, [pts], 255)
        self._roi_cache[key] = ((x0, y0, x1, y1), s, size, poly_mask)
        return self._roi_cache[key]

    def mask(self, frame: np.ndarray) -> np.ndarray:
        """Full-frame mask (for display in the tuner)."""
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        return build_mask(hsv, self.params)

    def detect(self, frame: np.ndarray, predict: Optional[Tuple[float, float]] = None,
               gate_px: float = 120.0) -> Optional[Detection]:
        (x0, y0, x1, y1), s, size, poly_mask = self._roi(frame.shape)
        roi = frame[y0:y1, x0:x1]
        small = roi if s >= 1.0 else cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = build_mask(hsv, self.params)
        if poly_mask is not None:
            mask = cv2.bitwise_and(mask, poly_mask)
        min_area = self.params["min_area"]
        blobs = find_blobs(mask, max(2.0, min_area * s * s))
        if not blobs:
            return None
        if predict is not None:
            px, py = (predict[0] - x0) * s, (predict[1] - y0) * s
            g2 = (gate_px * s) ** 2
            best = max(blobs, key=lambda b: b[2] / (1.0 + ((b[0] - px) ** 2 + (b[1] - py) ** 2) / g2))
        else:
            best = blobs[0]
        cx, cy, area = best[0] / s + x0, best[1] / s + y0, best[2] / (s * s)
        if s < 1.0:
            cx, cy, area = self._refine(frame, cx, cy, area)
        return Detection(cx, cy, area, len(blobs))

    def _refine(self, frame: np.ndarray, cx: float, cy: float, area: float) -> Tuple[float, float, float]:
        """Recompute the centroid at full resolution in a small window around the coarse blob."""
        r = int(max(12, 2.5 * np.sqrt(max(area, 1.0))))
        h, w = frame.shape[:2]
        x0, y0 = max(0, int(cx) - r), max(0, int(cy) - r)
        x1, y1 = min(w, int(cx) + r + 1), min(h, int(cy) + r + 1)
        win = frame[y0:y1, x0:x1]
        if win.size == 0:
            return cx, cy, area
        mask = build_mask(cv2.cvtColor(cv2.GaussianBlur(win, (5, 5), 0), cv2.COLOR_BGR2HSV), self.params)
        blobs = find_blobs(mask, 1.0)
        if not blobs:
            return cx, cy, area
        bx, by, ba = min(blobs, key=lambda b: (b[0] + x0 - cx) ** 2 + (b[1] + y0 - cy) ** 2)
        return bx + x0, by + y0, ba
