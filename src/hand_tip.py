"""
Marker-free pen-tip / fingertip detection.  Owner: Dev 2.

Works on a single frame (no background needed, so a bumped camera doesn't break it):

  1. HAND  = skin-coloured pixels (YCrCb), largest blob, preferably one entering from the frame
             edge. A wood desk can pass the skin test: its colour is learned from the frame
             border (which it mostly fills) and removed; skin is bluer (higher Cb) than wood.
  2. PEN   = pixels with the colours of OUR pen (data/pen_profile.json, learned once with
             tools/capture_pen.py); without a profile: non-skin pixels darker than the paper with
             a really dark or blue core. A piece is kept only if it TOUCHES the hand and is
             pen-shaped (long and narrow relative to the palm); collinear pieces are merged.
  3. TIP   = the writing end of the pen line, decided by, in order:
               a. LiDAR depth: the writing end is the LOWER end (the pen is tilted when writing)
               b. the pen profile: which end looks like the writing end
               c. how a hand holds a pen: writing end forward along the arm and toward the
                  thumb side (left for a right hand)
             then extended along the line over the (metal, not pen-coloured) nib.
             No pen in hand -> the fingertip itself (farthest hand pixel from where the arm enters).

LiDAR depth (Record3D), when available, also removes everything flat on the table from the hand
and pen masks: paper, print, ink and hand shadows have height ~0.

When the camera has been still, a background model is also learned and used to clean the masks.
If the camera moves, most of the view changes at once: that is flagged (scene_changed) and the
background is re-learned once still; detection keeps working on colour (and depth) meanwhile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.depth_plane import RAISED_M, TablePlane
from src.pen_profile import PenProfile

# colour model (YCrCb, OpenCV 8-bit ranges)
SKIN_LO = (25, 138, 97)
SKIN_HI = (255, 185, 128)
PEN_CR_MAX = 135            # generic pen: not reddish ...
PEN_DARKER_THAN_PAPER = 0.68    # ... and darker than this fraction of the paper brightness
PEN_CORE_DARK = 0.40        # a pen piece must contain some really dark (< this x paper) ...
PEN_CORE_CB = 138           # ... or clearly blue pixels; hand shadows on paper never do
PEN_CORE_FRAC = 0.05
PROFILE_MIN_P = 0.5         # with a pen profile: P(pen | colour) at least this
# a wood desk can pass the skin test; if skin-coloured pixels fill this much of the frame border
# (an arm only covers a little of it) their dominant colour is the desk's and is removed from skin
DESK_BORDER_FRAC = 0.35
DESK_BORDER_W = 0.03        # border strip width, fraction of the frame size
DESK_CR_R = 10.0            # removed: within this ellipse (Cr, Cb) around the desk colour
DESK_CB_R = 5.0
DESK_EVERY = 15             # frames between desk colour updates

# pen shape, relative to the palm radius (largest inscribed circle of the hand)
PEN_MAX_WIDTH = 0.8
PEN_MIN_LEN = 0.7
PEN_MIN_ELONGATION = 2.5
PEN_MERGE_DIST = 0.25       # other pieces whose pixels lie this close to the pen line join it

# deciding which end writes
DEPTH_END_DIFF_M = 0.015    # ends differing this much in height: the lower one writes
COLOUR_END_MARGIN = 8.0     # Lab distance margin for the profile's end colours to decide
COLOUR_ENDS_DISTINCT = 20.0  # ... used only if the pen's two ends really differ in colour
NIB_MAX_FRAC = 0.12         # the nib may stick out past the pen-coloured body by this x length
NIB_SIDE_FRAC = 0.06        # paper reference sampled this far (x length) beside the pen line
NIB_DARKER = 0.80           # nib pixels are darker than this x the paper beside them

STILL_S = 0.7               # view must be this still before a background is learned
STILL_DIFF = 4.0            # mean frame-to-frame change (0..255) that still counts as "still"
MIN_BRIGHTNESS = 25.0       # never learn a background from a (near) black frame
SCENE_CHANGE_FRAC = 0.35    # this much of the view different from the background = camera moved

Point = Tuple[float, float]


@dataclass
class TipDetection:
    x: float                    # full-frame pixel coordinates of the tip
    y: float
    area: float                 # hand blob area, full-res pixels
    entry: Point                # where the arm enters the view (full-res pixels)
    source: str = "finger"      # "pen" | "finger"
    pen_axis: Optional[Tuple[Point, Point]] = None   # (writing end, back end), full-res pixels
    end_cue: str = ""           # what decided the writing end: "depth" | "colour" | "hand"
    height_m: Optional[float] = None    # tip height above the table (LiDAR), if known


def _ellipse(d: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


class TipDetector:
    def __init__(self, work_width: int = 360, diff_thresh: float = 40.0,
                 min_area_frac: float = 0.004, adapt_rate: float = 0.03, right_handed: bool = True,
                 profile: Optional[PenProfile] = None) -> None:
        self.work_width = work_width
        self.right_handed = right_handed
        self.profile = profile
        self.diff_thresh = diff_thresh
        self.min_area_frac = min_area_frac
        self.adapt_rate = adapt_rate
        self.table = TablePlane()
        self.desk: Optional[Tuple[float, float]] = None     # (Cr, Cb) of a skin-coloured desk
        self._bg: Optional[np.ndarray] = None       # float32, work size
        self._scale = 1.0
        self._n = 0
        self._prev: Optional[np.ndarray] = None
        self._still_since: Optional[float] = None
        self.scene_changed = False                  # set when the camera moved; the caller clears it
        self.mask: Optional[np.ndarray] = None      # last hand mask (work size), for display
        self.pen_mask: Optional[np.ndarray] = None  # last pen pixels (work size), for display
        self._k3 = _ellipse(3)
        self._k5 = np.ones((5, 5), np.uint8)
        self._k7 = _ellipse(7)

    # ------------------------------------------------------------------ background (optional)
    @property
    def has_background(self) -> bool:
        return self._bg is not None

    def _small(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._scale = min(1.0, self.work_width / float(w))
        size = (max(1, int(round(w * self._scale))), max(1, int(round(h * self._scale))))
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if self._scale < 1 else frame
        return cv2.GaussianBlur(small, (3, 3), 0)

    def set_background(self, frame: np.ndarray) -> None:
        """Call with a view of the empty page (no hand). Calibration does this when it locks."""
        self._bg = self._small(frame).astype(np.float32)

    def reset(self) -> None:
        """Forget the background; it is re-learned once the view is still."""
        self._bg = None
        self._prev, self._still_since = None, None

    def _learn_when_still(self, small: np.ndarray, t: float) -> None:
        prev, self._prev = self._prev, small
        if (prev is None or prev.shape != small.shape or float(small.mean()) < MIN_BRIGHTNESS
                or cv2.norm(small, prev, cv2.NORM_L1) / small.size > STILL_DIFF):
            self._still_since = t
        elif self._still_since is not None and t - self._still_since >= STILL_S:
            self._bg = small.astype(np.float32)
            self._prev, self._still_since = None, None

    def foreground(self, small: np.ndarray) -> np.ndarray:
        """Pixels that differ from the background (exposure-normalised, shadows removed)."""
        g_now = float(np.median(cv2.cvtColor(np.ascontiguousarray(small[::4, ::4]), cv2.COLOR_BGR2GRAY)))
        g_bg = float(np.median(cv2.cvtColor(np.ascontiguousarray(self._bg[::4, ::4]), cv2.COLOR_BGR2GRAY)))
        gain = float(np.clip(g_now / max(1.0, g_bg), 0.6, 1.6))
        bg = cv2.convertScaleAbs(self._bg, alpha=gain)
        c = cv2.split(cv2.absdiff(small, bg))
        diff = cv2.max(cv2.max(c[0], c[1]), c[2])
        fg = (diff > self.diff_thresh).view(np.uint8) * np.uint8(255)
        fg[self._shadow(small, bg) > 0] = 0
        return cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._k3)

    @staticmethod
    def _shadow(small: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """Darker but same colour as the background. Shadows are smooth: half resolution."""
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

    # ------------------------------------------------------------------ detection
    def detect(self, frame: np.ndarray, t: Optional[float] = None, depth: Optional[np.ndarray] = None,
               confidence: Optional[np.ndarray] = None) -> Optional[TipDetection]:
        """t: frame timestamp in seconds (defaults to a 30 fps frame count).
        depth / confidence: Record3D LiDAR frames (metres), aligned to `frame`, any resolution."""
        self._n += 1
        t = self._n / 30.0 if t is None else t
        small = self._small(frame)
        h, w = small.shape[:2]
        if self._bg is not None and self._bg.shape[:2] != (h, w):
            self.reset()

        fg = None
        if self._bg is None:
            self._learn_when_still(small, t)
        else:
            fg = self.foreground(small)
            if cv2.countNonZero(fg) > SCENE_CHANGE_FRAC * h * w:
                self.reset()
                self.scene_changed = True
                self._learn_when_still(small, t)
                fg = None

        height = None                       # metres above the table, work size, NaN = unknown
        if depth is not None:
            hd = self.table.height(depth, confidence)
            if hd is not None:
                height = cv2.resize(hd, (w, h), interpolation=cv2.INTER_NEAREST)

        ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        skin_raw = cv2.inRange(ycc, SKIN_LO, SKIN_HI)
        if self._n % DESK_EVERY == 1 or DESK_EVERY <= 1:
            self.desk = self._desk_colour(ycc, skin_raw)
        if self.desk is not None:
            cr = ycc[..., 1].astype(np.float32) - self.desk[0]
            cb = ycc[..., 2].astype(np.float32) - self.desk[1]
            desk = ((cr / DESK_CR_R) ** 2 + (cb / DESK_CB_R) ** 2 <= 1.0).view(np.uint8) * np.uint8(255)
            skin_raw = cv2.bitwise_and(skin_raw, cv2.bitwise_not(desk))
        skin = skin_raw
        if self.profile is not None:
            pen = cv2.inRange(self.profile.probability(small), int(PROFILE_MIN_P * 255), 255)
            pen = cv2.bitwise_and(pen, cv2.bitwise_not(skin_raw))
            core = pen
        else:
            paper = float(np.percentile(ycc[::4, ::4, 0], 90))
            pen = cv2.inRange(ycc, (0, 0, 0), (int(PEN_DARKER_THAN_PAPER * paper), PEN_CR_MAX, 255))
            core = cv2.bitwise_or(cv2.inRange(ycc, (0, 0, 0), (int(PEN_CORE_DARK * paper), PEN_CR_MAX, 255)),
                                  cv2.inRange(ycc, (0, 0, PEN_CORE_CB), (int(PEN_DARKER_THAN_PAPER * paper), PEN_CR_MAX, 255)))
        if fg is not None:
            moving = cv2.dilate(fg, self._k5, iterations=2)
            skin = cv2.bitwise_and(skin, moving)
            pen = cv2.bitwise_and(pen, cv2.dilate(fg, self._k3))     # the pen itself must have changed
        if height is not None:
            off_table = ((height > RAISED_M) | np.isnan(height)).view(np.uint8) * np.uint8(255)
            skin = cv2.bitwise_and(skin, cv2.dilate(off_table, self._k5, iterations=2))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, self._k3)
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, self._k7)
        pen = cv2.morphologyEx(pen, cv2.MORPH_OPEN, self._k3)     # drops thin print and ruled lines

        hand, hand_area = self._hand(skin)
        self.mask, self.pen_mask = hand, None
        det = None
        if hand is not None:
            det = self._locate(small, hand, hand_area, pen, core, height)

        if self._bg is not None:
            keep = np.zeros((h, w), np.uint8) if hand is None else cv2.dilate(hand, self._k5, iterations=3)
            if self.pen_mask is not None:
                keep = cv2.bitwise_or(keep, cv2.dilate(self.pen_mask, self._k5, iterations=2))
            cv2.accumulateWeighted(small, self._bg, self.adapt_rate, mask=cv2.bitwise_not(keep))
        return det

    @staticmethod
    def _desk_colour(ycc: np.ndarray, skin: np.ndarray) -> Optional[Tuple[float, float]]:
        h, w = skin.shape
        bw = max(2, int(DESK_BORDER_W * min(h, w)))
        border = np.ones((h, w), bool)
        border[bw:h - bw, bw:w - bw] = False
        sel = border & (skin > 0)
        if np.count_nonzero(sel) < DESK_BORDER_FRAC * np.count_nonzero(border):
            return None
        cr = ycc[..., 1][sel].astype(np.int32)
        cb = ycc[..., 2][sel].astype(np.int32)
        hist = np.zeros((128, 128), np.float32)
        np.add.at(hist, (cr // 2, cb // 2), 1.0)
        hist = cv2.GaussianBlur(hist, (5, 5), 0)
        i, j = np.unravel_index(int(np.argmax(hist)), hist.shape)
        near = (np.abs(cr - (2 * i + 1)) <= DESK_CR_R) & (np.abs(cb - (2 * j + 1)) <= DESK_CB_R)
        return float(np.median(cr[near])), float(np.median(cb[near]))

    def _hand(self, skin: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        h, w = skin.shape
        n, labels, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
        if n <= 1:
            return None, 0
        edge = set(np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])).tolist())
        areas = stats[1:, cv2.CC_STAT_AREA]
        order = np.argsort(-areas) + 1
        min_area = self.min_area_frac * w * h
        best = next((int(k) for k in order if k in edge and stats[k, cv2.CC_STAT_AREA] >= min_area), 0)
        if not best and stats[order[0], cv2.CC_STAT_AREA] >= 3 * min_area:
            best = int(order[0])                      # hand fully inside the view
        if not best:
            return None, 0
        return (labels == best).view(np.uint8) * np.uint8(255), int(stats[best, cv2.CC_STAT_AREA])

    def _locate(self, small: np.ndarray, hand: np.ndarray, hand_area: int, pen: np.ndarray,
                core: np.ndarray, height: Optional[np.ndarray]) -> TipDetection:
        h, w = hand.shape
        s = self._scale
        ys, xs = np.nonzero(hand)
        on_border = (xs == 0) | (ys == 0) | (xs == w - 1) | (ys == h - 1)
        if on_border.any():
            ex, ey = float(xs[on_border].mean()), float(ys[on_border].mean())
        else:
            ex, ey = float(xs.mean()), float(ys.mean())
        d2 = (xs - ex) ** 2 + (ys - ey) ** 2
        top = max(3, int(0.002 * len(xs)))
        idx = np.argpartition(d2, -top)[-top:]
        fx, fy = float(xs[idx].mean()), float(ys[idx].mean())        # fingertip

        x0, y0, bw, bh = cv2.boundingRect(hand)
        palm = float(cv2.distanceTransform(hand[y0:y0 + bh, x0:x0 + bw], cv2.DIST_L2, 3).max())
        found = self._pen_axis(small, hand, pen, core, height, palm, (ex, ey), (fx, fy))

        def full(p: Point) -> Point:
            return (p[0] + 0.5) / s - 0.5, (p[1] + 0.5) / s - 0.5

        entry = full((ex, ey))
        area = hand_area / (s * s)
        if found is None:
            return TipDetection(*full((fx, fy)), area, entry, "finger",
                                height_m=self._height_at(height, (fx, fy), 3))
        tip, back, cue = found
        tip = self._extend_to_nib(small, hand, tip, back)
        return TipDetection(*full(tip), area, entry, "pen", (full(tip), full(back)), cue,
                            self._height_at(height, tip, 3))

    @staticmethod
    def _extend_to_nib(small: np.ndarray, hand: np.ndarray, tip: Point, back: Point) -> Point:
        """The nib (metal / ink cone) is not pen-coloured: follow the pen line past the tip
        while it stays clearly darker than the paper next to it (and is not the hand)."""
        dx, dy = tip[0] - back[0], tip[1] - back[1]
        length = float(np.hypot(dx, dy))
        if length < 10:
            return tip
        ux, uy = dx / length, dy / length
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        reach = int(NIB_MAX_FRAC * length)
        side = max(4.0, NIB_SIDE_FRAC * length)
        # paper brightness: beside the extension, on both sides
        ref = []
        for k in range(2, reach + 2):
            for sgn in (-1.0, 1.0):
                x = int(round(tip[0] + k * ux - sgn * side * uy))
                y = int(round(tip[1] + k * uy + sgn * side * ux))
                if 0 <= x < w and 0 <= y < h:
                    ref.append(gray[y, x])
        if not ref:
            return tip
        paper = float(np.percentile(ref, 75))
        last, gap = 0, 0
        for k in range(1, reach + 1):
            vals = [gray[y, x] for o in (-1, 0, 1)
                    for x, y in [(int(round(tip[0] + k * ux - o * uy)), int(round(tip[1] + k * uy + o * ux)))]
                    if 0 <= x < w and 0 <= y < h]
            cx, cy = int(round(tip[0] + k * ux)), int(round(tip[1] + k * uy))
            if not vals or not (0 <= cx < w and 0 <= cy < h) or hand[cy, cx]:
                break
            if min(vals) < NIB_DARKER * paper:
                last, gap = k, 0
            else:
                gap += 1
                if gap > 1:
                    break
        return tip[0] + last * ux, tip[1] + last * uy

    @staticmethod
    def _height_at(height: Optional[np.ndarray], p: Point, r: int) -> Optional[float]:
        if height is None:
            return None
        x, y = int(round(p[0])), int(round(p[1]))
        win = height[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        win = win[np.isfinite(win)]
        return float(np.median(win)) if win.size else None

    def _pen_axis(self, small: np.ndarray, hand: np.ndarray, pen: np.ndarray, core: np.ndarray,
                  height: Optional[np.ndarray], palm: float, entry: Point, finger: Point):
        if palm < 2:
            return None
        max_w = max(3.0, PEN_MAX_WIDTH * palm)
        min_len = max(6.0, PEN_MIN_LEN * palm)
        near = cv2.dilate(hand, _ellipse(2 * int(max(2, 0.15 * palm)) + 1))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(pen, connectivity=8)
        if n <= 1:
            return None
        touching = np.unique(labels[(near > 0) & (pen > 0)])
        best, best_len, best_pts = 0, 0.0, None
        for k in touching:
            if k == 0 or stats[k, cv2.CC_STAT_AREA] < 0.5 * min_len:
                continue
            bx, by, bw, bh = stats[k, :4]
            piece = labels[by:by + bh, bx:bx + bw] == k
            n_core = cv2.countNonZero(core[by:by + bh, bx:bx + bw] & piece.view(np.uint8) * np.uint8(255))
            if n_core < max(3, PEN_CORE_FRAC * stats[k, cv2.CC_STAT_AREA]):
                continue
            if height is not None:          # shadows / print are flat: some of the pen must stand up
                hp = height[by:by + bh, bx:bx + bw][piece]
                known = np.isfinite(hp)
                if known.sum() >= 0.5 * piece.sum() and np.count_nonzero(hp[known] > RAISED_M) < 0.1 * known.sum():
                    continue
            py, px = np.nonzero(piece)
            pts = np.column_stack([px + bx, py + by]).astype(np.float32)
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            short, long_ = min(rw, rh), max(rw, rh)
            if short <= max_w and long_ >= min_len and long_ >= PEN_MIN_ELONGATION * max(short, 1.0) and long_ > best_len:
                best, best_len, best_pts = k, long_, pts
        if not best:
            return None

        vx, vy, cx, cy = cv2.fitLine(best_pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        merge = max(2.0, PEN_MERGE_DIST * palm)
        pieces = [best_pts]
        for k in range(1, n):            # pieces of the same pen split by fingers / glare
            if k == best or stats[k, cv2.CC_STAT_AREA] < 4:
                continue
            mx, my = cents[k]
            if abs((mx - cx) * vy - (my - cy) * vx) > merge:
                continue
            if abs((mx - cx) * vx + (my - cy) * vy) > 4 * best_len:
                continue
            bx, by, bw, bh = stats[k, :4]
            py, px = np.nonzero(labels[by:by + bh, bx:bx + bw] == k)
            pts = np.column_stack([px + bx, py + by]).astype(np.float32)
            if np.percentile(np.abs((pts[:, 0] - cx) * vy - (pts[:, 1] - cy) * vx), 80) <= merge:
                pieces.append(pts)
        pts = np.vstack(pieces)
        proj = (pts[:, 0] - cx) * vx + (pts[:, 1] - cy) * vy
        perp = np.abs((pts[:, 0] - cx) * vy - (pts[:, 1] - cy) * vx)
        on_line = perp <= max_w
        lo, hi = float(np.percentile(proj[on_line], 1)), float(np.percentile(proj[on_line], 99))
        a = (cx + lo * vx, cy + lo * vy)
        b = (cx + hi * vx, cy + hi * vy)

        mask = np.zeros_like(pen)
        mask[pts[:, 1].astype(int), pts[:, 0].astype(int)] = 255
        self.pen_mask = mask

        length = hi - lo
        f = (proj - lo) / max(1e-6, length)
        end_sel = (f <= 0.25, f >= 0.75)
        # a. LiDAR: the writing end is the lower one (heights of the pen's own pixels near each end)
        if height is not None:
            hs = []
            for sel in end_sel:
                sp = pts[sel & on_line].astype(int)
                hv = height[sp[:, 1], sp[:, 0]] if len(sp) else np.empty(0, np.float32)
                hv = hv[np.isfinite(hv)]
                hs.append(float(np.median(hv)) if len(hv) >= 5 else None)
            ha, hb = hs
            if ha is not None and hb is not None and abs(ha - hb) >= DEPTH_END_DIFF_M:
                return (a, b, "depth") if ha < hb else (b, a, "depth")
        # b. the pen profile: which end looks like the writing end
        if self.profile is not None and np.linalg.norm(
                np.float32(self.profile.tip_lab) - np.float32(self.profile.back_lab)) >= COLOUR_ENDS_DISTINCT:
            lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
            ends = []
            for sel in end_sel:
                sp = pts[sel & on_line].astype(int)
                ends.append(lab[sp[:, 1], sp[:, 0]].mean(axis=0) if len(sp) else None)
            if ends[0] is not None and ends[1] is not None:
                diff = self.profile.tip_likeness(ends[0]) - self.profile.tip_likeness(ends[1])
                if abs(diff) >= COLOUR_END_MARGIN:
                    return (a, b, "colour") if diff > 0 else (b, a, "colour")
        # c. how a hand holds a pen: forward along the arm and toward the thumb side
        ux, uy = finger[0] - entry[0], finger[1] - entry[1]
        norm = max(1e-6, float(np.hypot(ux, uy)))
        ux, uy = ux / norm, uy / norm
        rx, ry = -uy, ux                          # the hand's right, in image coordinates
        side = -1.0 if self.right_handed else 1.0
        wx, wy = ux + side * rx, uy + side * ry
        return (a, b, "hand") if a[0] * wx + a[1] * wy >= b[0] * wx + b[1] * wy else (b, a, "hand")
