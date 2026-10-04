"""
Marker-free pen-tip / fingertip detection.  Owner: Dev 2.

Idea: the overhead camera sees a static scene (page + desk). That scene is stored as the
background (when calibration locks, or after the view has been still for a moment). Anything
that differs from it is foreground; the hand + pen always reach in from the edge of the camera
view, so the foreground blob touches the frame border where the arm enters, and the POINTED END
(pen tip, or the fingertip when pointing) is the blob pixel farthest from that entry point.
Works anywhere in view, including off the page.

Robustness:
  * global exposure changes (phone auto-exposure when the hand enters) -> background is rescaled
    by the frame/background brightness ratio before differencing
  * hand shadows on paper (darker, same colour) are not foreground
  * blobs that don't touch the frame border (fresh ink, objects that moved) are absorbed into the
    background within a few seconds; the rest of the background adapts slowly
  * a border blob that doesn't move at all for STATIC_S (a "ghost" left by a bad background, a
    bag strap, ...) is absorbed too, so the tip can't get stuck on it
  * the camera must be FIXED (overhead mount). If it moves, most of the view changes at once:
    that is reported as a scene change (no tip), and the background is re-learned once still
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

STILL_S = 0.7          # view must be this still before the first background is learned
STILL_DIFF = 4.0       # mean frame-to-frame change (0..255) that still counts as "still"
STATIC_S = 6.0         # a blob whose tip + size don't change for this long is scenery
STATIC_TIP_PX = 2.5    # (work-size pixels)
STATIC_AREA = 0.05
MIN_BRIGHTNESS = 25.0  # never learn a background from a (near) black frame: stream not started
SCENE_CHANGE_FRAC = 0.35   # this much of the view "foreground" = the camera moved, not a hand


@dataclass
class TipDetection:
    x: float                    # full-frame pixel coordinates of the tip
    y: float
    area: float                 # blob area, full-res pixels
    entry: Tuple[float, float]  # where the arm enters the view (full-res pixels)


class TipDetector:
    def __init__(self, work_width: int = 360, diff_thresh: float = 40.0,
                 min_area_frac: float = 0.004, adapt_rate: float = 0.03) -> None:
        self.work_width = work_width
        self.diff_thresh = diff_thresh
        self.min_area_frac = min_area_frac
        self.adapt_rate = adapt_rate
        self._bg: Optional[np.ndarray] = None       # float32, work size
        self._scale = 1.0
        self._gain = 1.0
        self._n = 0
        self._prev: Optional[np.ndarray] = None
        self._still_since: Optional[float] = None
        self._static: Optional[Tuple[float, float, float, float]] = None   # tip x, y, area, since
        self.scene_changed = False                  # set when the camera moved; the caller clears it
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
        self._static = None

    def reset(self) -> None:
        """Forget the background; it is re-learned once the view is still (hands out)."""
        self._bg = None
        self._prev, self._still_since, self._static = None, None, None

    def _learn_when_still(self, small: np.ndarray, t: float) -> None:
        prev, self._prev = self._prev, small
        if (prev is None or prev.shape != small.shape or float(small.mean()) < MIN_BRIGHTNESS
                or cv2.norm(small, prev, cv2.NORM_L1) / small.size > STILL_DIFF):
            self._still_since = t
        elif self._still_since is not None and t - self._still_since >= STILL_S:
            self._bg = small.astype(np.float32)
            self._prev, self._still_since = None, None

    def foreground(self, small: np.ndarray) -> np.ndarray:
        g_now = float(np.median(cv2.cvtColor(np.ascontiguousarray(small[::4, ::4]), cv2.COLOR_BGR2GRAY)))
        g_bg = float(np.median(cv2.cvtColor(np.ascontiguousarray(self._bg[::4, ::4]), cv2.COLOR_BGR2GRAY)))
        self._gain = float(np.clip(g_now / max(1.0, g_bg), 0.6, 1.6))
        bg = cv2.convertScaleAbs(self._bg, alpha=self._gain)
        c = cv2.split(cv2.absdiff(small, bg))
        diff = cv2.max(cv2.max(c[0], c[1]), c[2])
        fg = (diff > self.diff_thresh).view(np.uint8) * np.uint8(255)
        fg[self._shadow(small, bg) > 0] = 0
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._k3)
        return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._k5)

    @staticmethod
    def _shadow(small: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """Darker but same colour as the background (all channels dimmed by a similar factor).
        Shadows are smooth, so this runs at half the work size."""
        h, w = small.shape[:2]
        half = (max(1, w // 2), max(1, h // 2))
        f = cv2.resize(small, half, interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(bg, half, interpolation=cv2.INTER_AREA).astype(np.float32)
        b += 1.0
        r = cv2.split(cv2.divide(f, b))
        rmax = cv2.max(cv2.max(r[0], r[1]), r[2])
        rmin = cv2.min(cv2.min(r[0], r[1]), r[2])
        shadow = ((rmax < 0.95) & (rmin > 0.40) & (rmax - rmin < 0.12)).view(np.uint8)
        return cv2.resize(shadow, (w, h), interpolation=cv2.INTER_NEAREST)

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
        roi = cv2.GaussianBlur(frame[fy0:fy1, fx0:fx1], (3, 3), 0)
        c = cv2.split(cv2.absdiff(roi, cv2.convertScaleAbs(bg, alpha=self._gain)))
        fg = (cv2.max(cv2.max(c[0], c[1]), c[2]) > self.diff_thresh).view(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._k3)
        _, labels = cv2.connectedComponents(fg, connectivity=8)
        k = labels[min(fy1 - fy0 - 1, max(0, int(cy) - fy0)), min(fx1 - fx0 - 1, max(0, int(cx) - fx0))]
        if k == 0:
            return cx, cy
        ys, xs = np.nonzero(labels == k)
        xs, ys = xs + fx0, ys + fy0
        d2 = (xs - entry[0]) ** 2 + (ys - entry[1]) ** 2
        top = min(len(xs), 3)
        idx = np.argpartition(d2, -top)[-top:]
        return float(xs[idx].mean()), float(ys[idx].mean())

    def _is_static(self, tx: float, ty: float, area: float, t: float) -> bool:
        s = self._static
        if s is None or np.hypot(tx - s[0], ty - s[1]) > STATIC_TIP_PX or abs(area - s[2]) > STATIC_AREA * s[2]:
            self._static = (tx, ty, area, t)
            return False
        return t - s[3] >= STATIC_S

    def detect(self, frame: np.ndarray, t: Optional[float] = None) -> Optional[TipDetection]:
        """t: frame timestamp in seconds (defaults to a 30 fps frame count)."""
        self._n += 1
        t = self._n / 30.0 if t is None else t
        small = self._small(frame)
        if self._bg is None or self._bg.shape[:2] != small.shape[:2]:
            self._bg = None
            self.mask = None
            self._learn_when_still(small, t)
            return None
        fg = self.foreground(small)
        self.mask = fg
        h, w = fg.shape
        if cv2.countNonZero(fg) > SCENE_CHANGE_FRAC * h * w:
            self.reset()
            self.scene_changed = True
            self._learn_when_still(small, t)
            return None
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        min_area = self.min_area_frac * w * h
        edges = [(labels[0, :], np.arange(w), np.zeros(w)), (labels[-1, :], np.arange(w), np.full(w, h - 1)),
                 (labels[:, 0], np.zeros(h), np.arange(h)), (labels[:, -1], np.full(h, w - 1), np.arange(h))]
        on_edge = set(np.unique(np.concatenate([e[0] for e in edges])).tolist())
        best, best_area = 0, 0
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if k in on_edge and area >= min_area and area > best_area:
                best, best_area = k, area

        det = None
        keep = np.zeros_like(fg)
        if best:
            bx0, by0, bw, bh = (int(v) for v in stats[best, :4])
            box = (slice(by0, by0 + bh), slice(bx0, bx0 + bw))
            blob = labels[box] == best
            ex = float(np.concatenate([e[1][e[0] == best] for e in edges]).mean())
            ey = float(np.concatenate([e[2][e[0] == best] for e in edges]).mean())
            ys, xs = np.nonzero(blob)
            xs, ys = xs + bx0, ys + by0
            d2 = (xs - ex) ** 2 + (ys - ey) ** 2
            top = max(3, int(0.002 * len(xs)))
            idx = np.argpartition(d2, -top)[-top:]
            tx, ty = float(xs[idx].mean()), float(ys[idx].mean())
            if self._is_static(tx, ty, best_area, t):
                self._bg[box][blob] = small[box][blob]     # it's scenery: absorb it
                self._static = None
            else:
                keep[box][blob] = 255
                s = self._scale
                entry = ((ex + 0.5) / s - 0.5, (ey + 0.5) / s - 0.5)
                fx, fy = self._refine(frame, (tx + 0.5) / s - 0.5, (ty + 0.5) / s - 0.5, entry)
                det = TipDetection(fx, fy, best_area / (s * s), entry)
        else:
            self._static = None

        # adapt: everything except the hand blob (dilated) slowly joins the background, so ink,
        # moved objects and lighting drift disappear; the hand itself never does
        update = cv2.bitwise_not(cv2.dilate(keep, self._k5, iterations=2))
        cv2.accumulateWeighted(small, self._bg, self.adapt_rate, mask=update)
        return det
