"""
Marker-free pen-tip / fingertip detection.  Owner: Dev 2.

Idea: the overhead camera sees a static scene (page + desk). When calibration locks, that scene
is stored as the background. Anything that differs from it is foreground; the hand + pen always
reach in from the edge of the camera view, so the foreground blob touches the frame border where
the arm enters, and the POINTED END (pen tip, or the fingertip when pointing) is the blob pixel
farthest from that entry point. Works anywhere in view, including off the page.

Robustness:
  * global exposure changes (phone auto-exposure when the hand enters) -> background is rescaled
    by the frame/background brightness ratio before differencing
  * hand shadows on paper (darker, same colour) are not foreground
  * blobs that don't touch the frame border (fresh ink, "ghosts" of objects that moved) are
    absorbed into the background over a few seconds; the rest of the background adapts slowly
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class TipDetection:
    x: float                    # full-frame pixel coordinates of the tip
    y: float
    area: float                 # blob area, full-res pixels
    entry: Tuple[float, float]  # where the arm enters the view (full-res pixels)


class TipDetector:
    def __init__(self, work_width: int = 480, diff_thresh: float = 40.0,
                 min_area_frac: float = 0.004, adapt_rate: float = 0.03) -> None:
        self.work_width = work_width
        self.diff_thresh = diff_thresh
        self.min_area_frac = min_area_frac
        self.adapt_rate = adapt_rate
        self._bg: Optional[np.ndarray] = None       # float32, work size
        self._scale = 1.0
        self._gain = 1.0
        self.mask: Optional[np.ndarray] = None      # last foreground mask (work size), for display
        self._k3 = np.ones((3, 3), np.uint8)
        self._k5 = np.ones((5, 5), np.uint8)

    @property
    def has_background(self) -> bool:
        return self._bg is not None

    def _small(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._scale = min(1.0, self.work_width / float(w))
        size = (max(1, int(round(w * self._scale))), max(1, int(round(h * self._scale))))
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if self._scale < 1 else frame
        return cv2.GaussianBlur(small, (5, 5), 0)

    def set_background(self, frame: np.ndarray) -> None:
        """Call with a view of the empty page (no hand). Calibration does this when it locks."""
        self._bg = self._small(frame).astype(np.float32)

    def reset(self) -> None:
        self._bg = None

    def foreground(self, small: np.ndarray) -> np.ndarray:
        f = small.astype(np.float32)
        bg = self._bg
        gain = float(np.median(f.mean(axis=2))) / max(1.0, float(np.median(bg.mean(axis=2))))
        self._gain = float(np.clip(gain, 0.6, 1.6))
        bg = bg * self._gain
        diff = np.abs(f - bg).max(axis=2)
        ratio = f / (bg + 1.0)
        rmax, rmin = ratio.max(axis=2), ratio.min(axis=2)
        shadow = (rmax < 0.95) & (rmin > 0.40) & (rmax - rmin < 0.12)
        fg = ((diff > self.diff_thresh) & ~shadow).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._k3)
        return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._k5)

    def _refine(self, frame: np.ndarray, cx: float, cy: float,
                entry: Tuple[float, float]) -> Tuple[float, float]:
        """Re-find the tip at full resolution near the coarse estimate: the work-size blur and
        morphology round off a sharp pen tip by a few pixels."""
        s = self._scale
        if s >= 1.0:
            return cx, cy
        r = 10.0 / s
        bh, bw = self._bg.shape[:2]
        wx0, wy0 = max(0, int((cx - r) * s)), max(0, int((cy - r) * s))
        wx1, wy1 = min(bw, int(np.ceil((cx + r) * s)) + 1), min(bh, int(np.ceil((cy + r) * s)) + 1)
        fx0, fy0 = int(round(wx0 / s)), int(round(wy0 / s))
        fh, fw = frame.shape[:2]
        fx1, fy1 = min(fw, int(round(wx1 / s))), min(fh, int(round(wy1 / s)))
        if fx1 - fx0 < 3 or fy1 - fy0 < 3:
            return cx, cy
        bg = cv2.resize(self._bg[wy0:wy1, wx0:wx1], (fx1 - fx0, fy1 - fy0), interpolation=cv2.INTER_LINEAR)
        roi = cv2.GaussianBlur(frame[fy0:fy1, fx0:fx1], (3, 3), 0).astype(np.float32)
        fg = (np.abs(roi - bg * self._gain).max(axis=2) > self.diff_thresh).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._k3)
        n, labels = cv2.connectedComponents(fg, connectivity=8)
        k = labels[min(fy1 - fy0 - 1, max(0, int(cy) - fy0)), min(fx1 - fx0 - 1, max(0, int(cx) - fx0))]
        if k == 0:
            return cx, cy
        ys, xs = np.nonzero(labels == k)
        xs, ys = xs + fx0, ys + fy0
        d2 = (xs - entry[0]) ** 2 + (ys - entry[1]) ** 2
        top = min(len(xs), 3)
        idx = np.argpartition(d2, -top)[-top:]
        return float(xs[idx].mean()), float(ys[idx].mean())

    def detect(self, frame: np.ndarray) -> Optional[TipDetection]:
        small = self._small(frame)
        if self._bg is None or self._bg.shape[:2] != small.shape[:2]:
            self._bg = small.astype(np.float32)
            return None
        fg = self.foreground(small)
        self.mask = fg
        h, w = fg.shape
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        min_area = self.min_area_frac * w * h
        on_edge = set(np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])).tolist())
        best, best_area = 0, 0
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if k in on_edge and area >= min_area and area > best_area:
                best, best_area = k, area

        det = None
        keep = np.zeros_like(fg)
        if best:
            blob = labels == best
            keep[blob] = 255
            border = np.zeros_like(blob)
            border[0, :], border[-1, :], border[:, 0], border[:, -1] = blob[0, :], blob[-1, :], blob[:, 0], blob[:, -1]
            by, bx = np.nonzero(border)
            ys, xs = np.nonzero(blob)
            ex, ey = float(bx.mean()), float(by.mean())
            d2 = (xs - ex) ** 2 + (ys - ey) ** 2
            top = max(3, int(0.002 * len(xs)))
            idx = np.argpartition(d2, -top)[-top:]
            tx, ty = float(xs[idx].mean()), float(ys[idx].mean())
            s = self._scale
            entry = ((ex + 0.5) / s - 0.5, (ey + 0.5) / s - 0.5)
            fx, fy = self._refine(frame, (tx + 0.5) / s - 0.5, (ty + 0.5) / s - 0.5, entry)
            det = TipDetection(fx, fy, best_area / (s * s), entry)

        # adapt: everything except the hand blob (dilated) slowly joins the background, so ink,
        # moved objects and lighting drift disappear; the hand itself never does
        update = cv2.bitwise_not(cv2.dilate(keep, self._k5, iterations=2))
        cv2.accumulateWeighted(small.astype(np.float32), self._bg, self.adapt_rate, mask=update)
        return det
