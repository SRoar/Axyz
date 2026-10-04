"""
Pen profile: what THIS pen looks like, learned once from the overhead camera.  Owner: Dev 2.

    python tools/capture_pen.py        # lay the pen on the page, hands out, click its writing tip

Saved to data/pen_profile.json:
  * a colour model, P(pen | HSV colour), learned against everything else in the view (paper,
    print, desk), so shadows and printed text don't look like the pen
  * the average colour of the writing end and of the back end, so the tip detector knows which
    end writes no matter which way the pen points
  * the pen's size in pixels at capture time
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

PROFILE_PATH = "data/pen_profile.json"
BINS = (18, 8, 8)                     # H, S, V
RANGES = [0, 180, 0, 256, 0, 256]
END_FRAC = 0.25                       # each end descriptor uses this fraction of the pen length

Point = Tuple[float, float]


@dataclass
class PenSegment:
    mask: np.ndarray        # uint8 0/255, full size
    tip: Point              # pen ends along its axis (orientation not known yet)
    back: Point
    width: float            # pixels


@dataclass
class PenProfile:
    hist: np.ndarray                       # float32 BINS, P(pen | colour) in 0..1
    tip_lab: Tuple[float, float, float]
    back_lab: Tuple[float, float, float]
    length_px: float
    width_px: float
    image_size: Tuple[int, int]

    # -------------------------------------------------------------- use
    def probability(self, bgr: np.ndarray) -> np.ndarray:
        """uint8 0..255 per pixel: how pen-like its colour is."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return cv2.calcBackProject([hsv], [0, 1, 2], (self.hist * 255).astype(np.float32), RANGES, 1).astype(np.uint8)

    def tip_likeness(self, lab_mean: np.ndarray) -> float:
        """> 0 when a pen end's mean Lab colour looks like the writing end, < 0 like the back end."""
        return float(np.linalg.norm(lab_mean - np.float32(self.back_lab))
                     - np.linalg.norm(lab_mean - np.float32(self.tip_lab)))

    # -------------------------------------------------------------- persistence
    def save(self, path: str = PROFILE_PATH) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"bins": list(BINS), "hist": np.round(self.hist, 4).ravel().tolist(),
                       "tip_lab": list(self.tip_lab), "back_lab": list(self.back_lab),
                       "length_px": self.length_px, "width_px": self.width_px,
                       "image_size": list(self.image_size)}, f)

    @staticmethod
    def load(path: str = PROFILE_PATH) -> Optional["PenProfile"]:
        if not os.path.exists(path):
            return None
        with open(path) as f:
            d = json.load(f)
        if tuple(d["bins"]) != BINS:
            return None
        return PenProfile(np.float32(d["hist"]).reshape(BINS), tuple(d["tip_lab"]), tuple(d["back_lab"]),
                          float(d["length_px"]), float(d["width_px"]), tuple(d["image_size"]))


# ------------------------------------------------------------------ capture
def segment_pen(frame: np.ndarray, region: Optional[np.ndarray] = None) -> Optional[PenSegment]:
    """Find the pen on the page: the largest long, narrow, non-skin object that differs from the
    paper (a hand holding it is fine). region: optional polygon (page corners, pixels)."""
    h, w = frame.shape[:2]
    inside = np.zeros((h, w), np.uint8)
    if region is not None:
        cv2.fillConvexPoly(inside, np.int32(region), 255)
    else:
        inside[5:-5, 5:-5] = 255
    lab = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    paper = np.median(lab[inside > 0], axis=0)
    diff = np.linalg.norm((lab - paper) * np.float32([0.6, 1.0, 1.0]), axis=2)
    mask = ((diff > 28) & (inside > 0)).astype(np.uint8) * 255
    k = max(3, int(round(w / 150)) | 1)                     # removes print strokes, keeps the pen
    skin = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb), (25, 138, 97), (255, 185, 128))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    mask[cv2.dilate(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, best_area, best_rect = 0, 0, None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 0.0005 * w * h:
            continue
        ys, xs = np.nonzero(labels == i)
        rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        short, long_ = sorted(rect[1])
        if long_ >= 0.08 * max(w, h) and long_ >= 4 * max(short, 1.0) and area > best_area:
            best, best_area, best_rect = i, area, rect
    if not best:
        return None
    pen = (labels == best).astype(np.uint8) * 255
    ys, xs = np.nonzero(pen)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    proj = (pts[:, 0] - cx) * vx + (pts[:, 1] - cy) * vy
    lo, hi = np.percentile(proj, 0.5), np.percentile(proj, 99.5)
    a = (float(cx + lo * vx), float(cy + lo * vy))
    b = (float(cx + hi * vx), float(cy + hi * vy))
    return PenSegment(pen, a, b, float(min(best_rect[1])))


def build_profile(frame: np.ndarray, seg: PenSegment) -> PenProfile:
    """seg.tip must be the writing end (swap seg.tip / seg.back after the user's click)."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grow = cv2.dilate(seg.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    rest = cv2.bitwise_not(grow)
    pen_h = cv2.calcHist([hsv], [0, 1, 2], seg.mask, list(BINS), RANGES)
    rest_h = cv2.calcHist([hsv], [0, 1, 2], rest, list(BINS), RANGES)
    pen_h = cv2.GaussianBlur(pen_h.reshape(BINS[0], -1), (3, 3), 0).reshape(BINS) / max(1.0, pen_h.sum())
    rest_h = rest_h / max(1.0, rest_h.sum())
    hist = pen_h / (pen_h + 4.0 * rest_h + 1e-9)            # rest of the view dominates: be strict
    hist[pen_h < 0.002] = 0.0

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    ys, xs = np.nonzero(seg.mask)
    t, b = np.float32(seg.tip), np.float32(seg.back)
    axis = b - t
    length = float(np.linalg.norm(axis))
    f = ((xs - t[0]) * axis[0] + (ys - t[1]) * axis[1]) / max(1e-6, length ** 2)
    tip_px, back_px = f <= END_FRAC, f >= 1 - END_FRAC
    tip_lab = lab[ys[tip_px], xs[tip_px]].mean(axis=0) if tip_px.any() else lab[ys, xs].mean(axis=0)
    back_lab = lab[ys[back_px], xs[back_px]].mean(axis=0) if back_px.any() else lab[ys, xs].mean(axis=0)
    return PenProfile(hist.astype(np.float32), tuple(float(v) for v in tip_lab), tuple(float(v) for v in back_lab),
                      length, seg.width, (w, h))
