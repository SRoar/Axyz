"""
GuidanceEngine: (PenState, Box) -> Guidance.  Owner: Dev 2.
Contract: src.guidance.GuidanceEngine() with compute(pen, box) (see contracts.GuidanceEngine).

Geometry is done in centimetres (page units x PAGE_W_CM / PAGE_H_CM).

Hysteresis (the only state; reset whenever the active box changes):
  * in_box / write_status: the pen must be ENTER_INSET_CM inside the box to count as inside, and
    EXIT_MARGIN_CM outside it to count as outside again, so jitter at the edge can't flap WARN.
  * GUIDE_LEFT/RIGHT vs GUIDE_BOTH: switches at DEADZONE_CM +/- AXIS_HYST_CM, so the buzzer
    doesn't flicker between sides when the pen is almost aligned horizontally.

Off the page (x or y outside 0..1) the spoken cue starts with where the pen is:
"You are off the page, to the left. Move 3 inches Right."

dx_cm / dy_cm point from the pen to the target region (the box shrunk by TARGET_INSET_CM, which is
deeper than ENTER_INSET_CM), so following the cue always ends with in_box = True.
"""
from __future__ import annotations

import math
from typing import Optional

from src.contracts import PAGE_H_CM, PAGE_W_CM, Box, Guidance, HapticCmd, PenState, WriteStatus

CM_PER_INCH = 2.54
IN_BOX_SPEECH = "You are in the answer box."


def _amount(cm: float) -> str:
    inches = cm / CM_PER_INCH
    if inches < 0.75:
        return "a little"
    n = max(1, int(round(inches)))
    return f"{n} {'inch' if n == 1 else 'inches'}"


def phrase(dx_cm: float, dy_cm: float) -> str:
    """'Move 2 inches Down.' Adds the other axis when it is also big: 'Move 3 inches Down and 1 inch Left.'"""
    ax, ay = abs(dx_cm), abs(dy_cm)
    h_dir = "Right" if dx_cm > 0 else "Left"
    v_dir = "Down" if dy_cm > 0 else "Up"
    if ay >= ax:
        first, second = (ay, v_dir), (ax, h_dir)
    else:
        first, second = (ax, h_dir), (ay, v_dir)
    text = f"Move {_amount(first[0])} {first[1]}"
    if second[0] >= CM_PER_INCH * 0.75 and second[0] >= 0.4 * first[0]:
        text += f" and {_amount(second[0])} {second[1]}"
    return text + "."


def page_side(x: float, y: float) -> Optional[str]:
    """Where an off-page pen is relative to the page: 'left', 'right', 'above', 'below',
    'above left', ... or None when the pen is on the page."""
    v = "above" if y < 0 else ("below" if y > 1 else "")
    h = "left" if x < 0 else ("right" if x > 1 else "")
    side = f"{v} {h}".strip()
    return side or None


def off_page_phrase(side: str) -> str:
    """'left' -> 'You are off the page, to the left.'  'above left' -> '..., above and to the left.'"""
    parts = side.split()
    v = next((p for p in parts if p in ("above", "below")), None)
    h = next((p for p in parts if p in ("left", "right")), None)
    where = " and ".join(([v] if v else []) + ([f"to the {h}"] if h else []))
    return f"You are off the page, {where}."


def _axis_delta(p: float, lo: float, hi: float) -> float:
    """Signed distance from p to the interval [lo, hi] (0 inside)."""
    if p < lo:
        return lo - p
    if p > hi:
        return hi - p
    return 0.0


def box_offset_cm(x: float, y: float, box: Box, page_w_cm: float = PAGE_W_CM,
                  page_h_cm: float = PAGE_H_CM) -> tuple:
    """(dx_cm, dy_cm, dist_cm) from the page-normalized point (x, y) to the nearest point of the
    box (0, 0, 0 inside). dx > 0: the box is to the right; dy > 0: the box is below."""
    dx = _axis_delta(x * page_w_cm, box.xmin * page_w_cm, box.xmax * page_w_cm)
    dy = _axis_delta(y * page_h_cm, box.ymin * page_h_cm, box.ymax * page_h_cm)
    return dx, dy, math.hypot(dx, dy)


def direction_words(dx_cm: float, dy_cm: float, min_cm: float = 0.3) -> str:
    """'down 3.1 cm, left 1.0 cm' (axes below min_cm are left out); 'inside' at 0."""
    parts = []
    if abs(dy_cm) >= min_cm:
        parts.append(f"{'down' if dy_cm > 0 else 'up'} {abs(dy_cm):.1f} cm")
    if abs(dx_cm) >= min_cm:
        parts.append(f"{'right' if dx_cm > 0 else 'left'} {abs(dx_cm):.1f} cm")
    return ", ".join(parts) or "inside"


def box_status(inside_id: Optional[str], target: Optional[Box], offsets: dict) -> str:
    """One line for the display. offsets: box id -> box_offset_cm(...) for the current pen tip.
    'IN box2' when the tip is in the target box (or in any box when there is no target), else
    'box2: 4.2 cm away (down 3.1 cm, right 2.8 cm)' for the target (nearest box without one)."""
    if not offsets:
        return "pen not seen"
    if inside_id is not None and (target is None or target.id == inside_id):
        return f"IN {inside_id}"
    tid = target.id if target is not None and target.id in offsets else min(offsets, key=lambda k: offsets[k][2])
    dx, dy, dist = offsets[tid]
    text = f"{tid}: {dist:.1f} cm away ({direction_words(dx, dy)})"
    return text + (f"  [now in {inside_id}]" if inside_id is not None else "")


class GuidanceEngine:
    DEADZONE_CM = 1.0        # horizontal error below this -> GUIDE_BOTH (aligned, move vertically)
    AXIS_HYST_CM = 0.3
    ENTER_INSET_CM = 0.25    # must be this far inside the box to become INSIDE
    EXIT_MARGIN_CM = 0.40    # must be this far outside the box to become OUTSIDE again
    TARGET_INSET_CM = 0.75   # cues aim this far inside the box edge

    def __init__(self, page_w_cm: float = PAGE_W_CM, page_h_cm: float = PAGE_H_CM) -> None:
        self.page_w_cm, self.page_h_cm = page_w_cm, page_h_cm
        self.reset()

    def reset(self) -> None:
        self._box_id: Optional[str] = None
        self._inside: Optional[bool] = None
        self._horizontal: Optional[bool] = None

    def compute(self, pen: Optional[PenState], box: Optional[Box]) -> Guidance:
        if box is None:
            self.reset()
            return Guidance(write_status=WriteStatus.UNKNOWN, pen_visible=pen is not None)
        if box.id != self._box_id:
            self.reset()
            self._box_id = box.id
        if pen is None:
            return Guidance(write_status=WriteStatus.UNKNOWN, pen_visible=False)

        W, H = self.page_w_cm, self.page_h_cm
        x, y = pen.x * W, pen.y * H
        x0, x1, y0, y1 = box.xmin * W, box.xmax * W, box.ymin * H, box.ymax * H

        # signed depth inside the box (cm): > 0 inside, < 0 = distance outside
        dxo, dyo = _axis_delta(x, x0, x1), _axis_delta(y, y0, y1)
        if dxo == 0.0 and dyo == 0.0:
            depth = min(x - x0, x1 - x, y - y0, y1 - y)
        else:
            depth = -math.hypot(dxo, dyo)

        if self._inside is None:
            inside = depth >= 0.0
        elif self._inside:
            inside = depth > -self.EXIT_MARGIN_CM
        else:
            inside = depth >= min(self.ENTER_INSET_CM, self._max_inset(x0, x1, y0, y1))
        self._inside = inside
        if inside:
            self._horizontal = None
            return Guidance(None, IN_BOX_SPEECH, 0.0, 0.0, 0.0, True, WriteStatus.INSIDE, True)

        inset = self._max_inset(x0, x1, y0, y1, self.TARGET_INSET_CM)
        dx = _axis_delta(x, x0 + inset, x1 - inset)
        dy = _axis_delta(y, y0 + inset, y1 - inset)
        dist = math.hypot(dx, dy)

        threshold = self.DEADZONE_CM + (-self.AXIS_HYST_CM if self._horizontal else self.AXIS_HYST_CM)
        if self._horizontal is None:
            threshold = self.DEADZONE_CM
        self._horizontal = abs(dx) > threshold
        if self._horizontal:
            cmd = HapticCmd.GUIDE_RIGHT if dx > 0 else HapticCmd.GUIDE_LEFT
        else:
            cmd = HapticCmd.GUIDE_BOTH
        speech = phrase(dx, dy)
        side = page_side(pen.x, pen.y)
        if side:
            speech = f"{off_page_phrase(side)} {speech}"
        return Guidance(cmd, speech, dx, dy, dist, False, WriteStatus.OUTSIDE, True)

    @staticmethod
    def _max_inset(x0: float, x1: float, y0: float, y1: float, want: float = float("inf")) -> float:
        """Inset that still leaves a non-empty target, for very thin boxes."""
        return max(0.0, min(want, 0.45 * (x1 - x0), 0.45 * (y1 - y0)))
