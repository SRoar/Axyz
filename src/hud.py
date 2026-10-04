"""
HUD for judges (Dev 4, plan tasks 4.3 + 4.4).  Pure rendering: reads a System, returns a BGR image.

    hud = Hud(system, debug=debug_imu_or_None)
    each tick:    hud.observe()              # cheap: consumes new log lines / taps / audit result
    ~30 fps:      cv2.imshow("HUD", hud.render())

Layout (1600 x 900, readable from 2 m):
    +--------------------------------------------------------------------------------+
    | PHASE BANNER (READING / NAVIGATING / WRITING_LOCKED / OUT_OF_BOUNDS / AUDITING) |
    | question text                                              component health chips |
    +------------------+------------------------------+------------------------------+
    | CAMERA           | PEN PROBE (IMU)              | HAPTICS (buzzers + log)      |
    | where the pen is | what the pen is doing        |                              |
    | boxes, pen, 2 s  | state badge, waveform, taps  +------------------------------+
    | trail            +------------------------------+ VOICE log                    |
    |                  | GUIDANCE                     |                              |
    |                  | LAST AUDIT                   |                              |
    +------------------+------------------------------+------------------------------+
The pitch is on screen: when the camera loses the pen under the hand while the IMU still says
WRITING, the camera panel says so -- "the pen knows what it's doing; the camera knows where it is".
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from src.contracts import (
    PAGE_H_CM, PAGE_W_CM, HapticCmd, MotionState, Phase, Tap, WriteStatus,
)
from src.system import System

W, H = 1600, 900
BG, PANEL, PANEL_EDGE = (22, 20, 19), (38, 34, 32), (80, 74, 70)
WHITE, GREY, DIM = (240, 240, 240), (170, 170, 170), (110, 110, 110)
GREEN, RED, AMBER, BLUE, PURPLE, CYAN = (90, 200, 90), (70, 70, 235), (30, 170, 245), (235, 160, 60), (200, 100, 190), (230, 210, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX

MOTION_COLOR = {
    MotionState.STILL: (170, 150, 120), MotionState.MOVING: AMBER,
    MotionState.WRITING: GREEN, MotionState.LIFTED: PURPLE,
}
TRAIL_S = 2.0
TAP_FLASH_S = 1.4

# panel rectangles (x, y, w, h)
R_BANNER = (0, 0, W, 78)
R_QUESTION = (0, 78, W, 92)
PAGE_PH = 690
PAGE_PW = int(PAGE_PH * PAGE_W_CM / PAGE_H_CM)
R_CAM = (20, 186, PAGE_PW, PAGE_PH)
COL_A_X = R_CAM[0] + PAGE_PW + 20
COL_A_W = 500
COL_B_X = COL_A_X + COL_A_W + 20
COL_B_W = W - COL_B_X - 20
R_IMU = (COL_A_X, 186, COL_A_W, 330)
R_GUIDE = (COL_A_X, 532, COL_A_W, 150)
R_AUDIT = (COL_A_X, 698, COL_A_W, 178)
R_HAPTIC = (COL_B_X, 186, COL_B_W, 330)
R_VOICE = (COL_B_X, 532, COL_B_W, 344)
_BLANK = np.full((H, W, 3), BG, np.uint8)  # built once; render() copies it


# --------------------------------------------------------------------------- drawing helpers
def put(img, text: str, org: Tuple[int, int], scale: float = 0.7, color=WHITE, thick: int = 1) -> None:
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def text_w(text: str, scale: float, thick: int = 1) -> int:
    return cv2.getTextSize(text, FONT, scale, thick)[0][0]


def wrap(text: str, max_w: int, scale: float, thick: int = 1, max_lines: int = 99) -> List[str]:
    lines: List[str] = []
    cur = ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if text_w(trial, scale, thick) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(0, len(lines[-1]) - 3)] + "..."
    return lines


def panel(img, rect, title: str, accent=PANEL_EDGE) -> Tuple[int, int, int, int]:
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), accent, 2)
    put(img, title, (x + 12, y + 26), 0.62, GREY, 1)
    return x, y, w, h


def banner_for(phase: Phase, motion: MotionState, write_status: WriteStatus,
               in_box: bool) -> Tuple[str, Tuple[int, int, int]]:
    """Judges-facing phase name + colour (BGR)."""
    if phase == Phase.WRITING:
        if motion == MotionState.WRITING and write_status == WriteStatus.OUTSIDE:
            return "OUT_OF_BOUNDS", RED
        return "WRITING_LOCKED", GREEN
    return {
        Phase.IDLE: ("IDLE", (150, 130, 110)),
        Phase.READING: ("READING", BLUE),
        Phase.NAVIGATING: ("NAVIGATING", AMBER),
        Phase.AUDITING: ("AUDITING", PURPLE),
        Phase.COMPLETE: ("COMPLETE", GREEN),
    }[phase]


def off_page_side(x: float, y: float) -> Optional[str]:
    """'LEFT', 'ABOVE RIGHT', ... when the pen tip is off the paper, else None."""
    v = "ABOVE" if y < 0 else ("BELOW" if y > 1 else "")
    hz = "LEFT" if x < 0 else ("RIGHT" if x > 1 else "")
    return f"{v} {hz}".strip() or None


def _call(obj, name: str):
    """obj.name() if the component has that extra (real tracker), else None. Never raises."""
    fn = getattr(obj, name, None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _component_error(system: System, comp: str) -> str:
    return (getattr(system, "component_errors", {}) or {}).get(comp, "")


# --------------------------------------------------------------------------- the HUD
class Hud:
    def __init__(self, system: System, debug=None) -> None:
        self.s = system
        self.debug = debug                       # DebugImu or None (shows the override badge)
        self.trail: Deque[Tuple[float, float, float]] = deque()
        self.taps: Deque[Tuple[float, int]] = deque(maxlen=8)
        self.last_tap: Optional[Tuple[float, int, bool]] = None   # (t, n, typed on the keyboard)
        self.tap_counts = {1: 0, 2: 0}                            # taps the PEN reported
        self.key_taps = 0                                         # taps injected with --debug-keys
        self.link_rate = 0.0                                      # serial lines/s from the pen
        self._rate_t: Optional[float] = None
        self._rate_n = 0
        self.haptics: Deque[Tuple[float, str]] = deque(maxlen=60)
        self.voice: Deque[Tuple[float, str]] = deque(maxlen=30)
        self._log_i = 0
        self._t0: Optional[float] = None

    # ---------------------------------------------------------------- observe (every tick)
    def observe(self) -> None:
        s = self.s
        now = s.clock.now()
        if self._t0 is None:
            self._t0 = now
        inp = s.inputs
        if inp is not None:
            for e in inp.imu_events:
                if e.tap != Tap.NONE:
                    n, typed = int(e.tap), bool(getattr(e, "injected", False))
                    self.taps.append((now, n))
                    self.last_tap = (now, n, typed)
                    if typed:
                        self.key_taps += 1
                    else:
                        self.tap_counts[n] = self.tap_counts.get(n, 0) + 1
            if inp.pen is not None:
                self.trail.append((now, inp.pen.x, inp.pen.y))
        while self.trail and now - self.trail[0][0] > TRAIL_S:
            self.trail.popleft()
        seen = getattr(s.c.imu, "lines_seen", None)
        if seen is not None:
            if self._rate_t is None:
                self._rate_t, self._rate_n = now, seen
            elif now - self._rate_t >= 1.0:
                self.link_rate = (seen - self._rate_n) / (now - self._rate_t)
                self._rate_t, self._rate_n = now, seen
        while self._log_i < len(s.log):
            t, kind, text = s.log[self._log_i]
            self._log_i += 1
            if kind == "HAPTIC":
                self.haptics.append((t, text))
            elif kind == "SPEAK":
                self.voice.append((t, text))
            elif kind == "SILENCE":
                self.voice.append((t, "(silence)"))

    # ---------------------------------------------------------------- render
    def render(self) -> np.ndarray:
        img = _BLANK.copy()                      # np.full with a colour tuple costs ~20 ms per frame on a slow CPU
        s, inp = self.s, self.s.inputs
        now = s.clock.now()
        phase = s.c.brain.phase
        motion = inp.motion if inp else MotionState.STILL
        gd = inp.guidance if inp else None
        status = gd.write_status if gd else WriteStatus.UNKNOWN
        label, color = banner_for(phase, motion, status, bool(gd and gd.in_box))
        self._banner(img, label, color, phase, motion, gd)
        self._question(img, phase)
        self._camera(img, now, motion)
        self._imu(img, now, motion)
        self._guidance(img, gd)
        self._audit(img, phase, now)
        self._haptic_panel(img, now)
        self._voice_panel(img, now)
        if self.debug is not None:
            self._debug_badge(img)
        return img

    # ---------------------------------------------------------------- pieces
    def _banner(self, img, label, color, phase, motion, gd) -> None:
        cv2.rectangle(img, (0, 0), (W, R_BANNER[3]), color, -1)
        put(img, label, (24, 58), 1.9, (15, 15, 15), 4)
        sub = {
            "IDLE": "Hold the pen still to hear the next question",
            "READING": "Reading the question aloud",
            "NAVIGATING": (gd.speech if gd and gd.speech else "Guiding the pen to the answer box"),
            "WRITING_LOCKED": "Voice silent - margin guard armed",
            "OUT_OF_BOUNDS": "Writing outside the box - WARN buzzer",
            "AUDITING": "Checking the answer with Gemini",
            "COMPLETE": "All questions done",
        }[label]
        put(img, sub, (W - 24 - text_w(sub, 0.95, 2), 50), 0.95, (15, 15, 15), 2)

    def _question(self, img, phase: Phase) -> None:
        s = self.s
        cv2.rectangle(img, (0, R_QUESTION[1]), (W, R_QUESTION[1] + R_QUESTION[3]), (30, 27, 25), -1)
        q = s.c.brain.question
        total = len(s.question_order)
        if q is None or phase == Phase.COMPLETE:
            put(img, "All questions complete.", (24, R_QUESTION[1] + 56), 1.1, GREEN, 2)
        else:
            idx = s.c.brain.index + 1
            put(img, f"Q{idx}/{total}", (24, R_QUESTION[1] + 38), 0.8, CYAN, 2)
            for i, ln in enumerate(wrap(q.text, W - 790, 0.95, 2, max_lines=2)):   # width stops short of the health chips
                put(img, ln, (140, R_QUESTION[1] + 36 + i * 36), 0.95, WHITE, 2)
        # component health chips (right side)
        x = W - 24
        for name in ("auditor", "voice", "guidance", "tracker", "imu", "brain"):
            n = s.errors.get(name, 0)
            txt = f"{name[:3].upper()} {'ok' if n == 0 else 'ERR x%d' % n}"
            if name == "imu" and getattr(s.c.imu, "connected", None) is False:
                txt, n = "IMU no link", max(n, 1)              # real pen unplugged / not found
            if name == "tracker" and getattr(s.c.tracker, "camera_error", ""):
                txt, n = "TRA no cam", max(n, 1)               # iPhone not streaming yet (being retried)
            tw = text_w(txt, 0.5) + 16
            x -= tw + 6
            cv2.rectangle(img, (x, R_QUESTION[1] + 12), (x + tw, R_QUESTION[1] + 38), (50, 90, 50) if n == 0 else (40, 40, 160), -1)
            put(img, txt, (x + 8, R_QUESTION[1] + 31), 0.5, WHITE)
        if s.last_error:
            put(img, s.last_error[:70], (W - 24 - text_w(s.last_error[:70], 0.45), R_QUESTION[1] + 62), 0.45, (120, 120, 235))

    def _camera(self, img, now: float, motion: MotionState) -> None:
        s = self.s
        occl = s.inputs is not None and (s.inputs.pen is None or s.inputs.pen.confidence < 0.5)
        pitch = occl and motion == MotionState.WRITING
        x, y, w, h = panel(img, R_CAM, "CAMERA - where the pen is", AMBER if pitch else PANEL_EDGE)
        iy, ih = y + 36, h - 44
        iw = w - 16
        ix = x + 8
        tr = s.c.tracker
        pen = s.inputs.pen if s.inputs else None
        # real tracker extras (fakes have none of these: their frame is always the page)
        status = _call(tr, "calibration_status")              # (searching, progress, corners px) or None
        page_known = status is None or getattr(tr, "calibration", None) is not None
        stats = _call(tr, "stats") or {}
        info = []
        if pen is not None and page_known:
            info.append(f"tip {pen.x * PAGE_W_CM:.1f}, {pen.y * PAGE_H_CM:.1f} cm")
        if stats.get("camera_fps"):
            info.append(f"{stats['camera_fps']:.0f} fps")
        if info:
            txt = "   ".join(info)
            put(img, txt, (x + w - 12 - text_w(txt, 0.5), y + 26), 0.5, GREY)

        frame = s.reading.frame if (s.reading is not None and s.reading.frame is not None) else None
        if not page_known:
            img[iy:iy + ih, ix:ix + iw] = self._page_search_view(frame, status, iw, ih)
            return
        if frame is not None:
            page = cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_AREA)
        else:
            page = np.full((ih, iw, 3), 235, np.uint8)
            put(page, "no camera frame", (14, 30), 0.6, (90, 90, 90))
            err = getattr(tr, "camera_error", "") or _component_error(s, "tracker")
            if err:
                for i, ln in enumerate(wrap(err, iw - 28, 0.5, 1, max_lines=4)):
                    put(page, ln, (14, 58 + i * 22), 0.5, (40, 40, 170))
        # answer boxes
        for b in s.layout.values():
            active = s.current_box is not None and b.id == s.current_box.id
            p0, p1 = (int(b.xmin * iw), int(b.ymin * ih)), (int(b.xmax * iw), int(b.ymax * ih))
            cv2.rectangle(page, p0, p1, (60, 170, 60) if active else (150, 150, 150), 4 if active else 1)
            put(page, b.id, (p0[0] + 6, p0[1] + 20), 0.55, (40, 140, 40) if active else (130, 130, 130), 1)
        # 2 s trail
        pts = [(t, int(px * iw), int(py * ih)) for t, px, py in self.trail]
        for (t0, ax, ay), (t1, bx, by) in zip(pts, pts[1:]):
            age = min(1.0, (now - t1) / TRAIL_S)
            c = (int(40 + 180 * age), int(60 + 160 * age), int(230 - 20 * age))
            cv2.line(page, (ax, ay), (bx, by), c, max(1, int(5 * (1 - age)) + 1), cv2.LINE_AA)
        # pen (off the paper: pinned to the edge with an arrow pointing to where it is)
        if pen is not None:
            side = off_page_side(pen.x, pen.y)
            px, py = int(np.clip(pen.x, 0.02, 0.98) * iw), int(np.clip(pen.y, 0.02, 0.98) * ih)
            c = (0, 0, 230) if pen.confidence >= 0.5 else (0, 140, 255)
            cv2.circle(page, (px, py), 10, c, -1 if pen.confidence >= 0.5 else 2, cv2.LINE_AA)
            cv2.circle(page, (px, py), 12, (255, 255, 255), 1, cv2.LINE_AA)
            if side:
                dx = -1 if "LEFT" in side else (1 if "RIGHT" in side else 0)
                dy = -1 if "ABOVE" in side else (1 if "BELOW" in side else 0)
                cv2.arrowedLine(page, (px - 40 * dx, py - 40 * dy), (px + 14 * dx, py + 14 * dy),
                                (0, 0, 200), 4, cv2.LINE_AA, tipLength=0.4)
                cv2.rectangle(page, (0, 0), (iw, 40), (30, 30, 150), -1)
                put(page, f"PEN TIP OFF THE PAGE: {side}", (12, 28), 0.7, WHITE, 2)
        if status is not None and status[0]:                  # page known, but re-checking where it is
            cv2.rectangle(page, (0, ih - 30), (iw, ih), (40, 90, 140), -1)
            put(page, f"checking the page position... {int(100 * status[1])}%", (10, ih - 9), 0.55, WHITE)
        img[iy:iy + ih, ix:ix + iw] = page
        if pen is None and s.inputs is not None:
            cv2.rectangle(img, (ix, iy + ih - 92), (ix + iw, iy + ih), (30, 30, 150), -1)
            put(img, "CAMERA LOST THE PEN", (ix + 12, iy + ih - 58), 0.8, WHITE, 2)
            put(img, f"but the IMU still says: {motion.value}", (ix + 12, iy + ih - 20),
                0.75, AMBER if pitch else GREY, 2)

    def _page_search_view(self, frame, status, iw: int, ih: int) -> np.ndarray:
        """No page calibration yet: the raw camera view (aspect kept), the tracker's corner guess and
        what to do. Boxes and the pen are NOT drawn here: without the page there are no page coordinates."""
        out = np.full((ih, iw, 3), 40, np.uint8)
        if frame is not None:
            fh, fw = frame.shape[:2]
            k = min(iw / fw, ih / fh)
            nw, nh = int(fw * k), int(fh * k)
            ox, oy = (iw - nw) // 2, (ih - nh) // 2
            out[oy:oy + nh, ox:ox + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            corners = status[2] if status is not None else None
            if corners is not None:
                pts = (np.asarray(corners, np.float32) * k + (ox, oy)).astype(np.int32)
                cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (0, 165, 255), 3, cv2.LINE_AA)
            # a nearly black / featureless view looks like "no camera feed": say what is wrong instead
            small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            if small.mean() < 30:
                hint = "CAMERA IMAGE IS DARK: lens covered or facing the desk?"
            elif small.std() < 8:
                hint = "CAMERA SEES A BLANK BLUR: too close to a surface?"
            else:
                hint = ""
            if hint:
                cv2.rectangle(out, (0, ih // 2 - 46), (iw, ih // 2 + 40), (30, 30, 150), -1)
                put(out, hint, (12, ih // 2 - 14), 0.55, WHITE, 2)
                put(out, "Point the iPhone's back camera straight down at the sheet.", (12, ih // 2 + 16), 0.5, WHITE)
        else:
            put(out, "waiting for the first camera frame...", (14, 130), 0.6, GREY)
            err = getattr(self.s.c.tracker, "camera_error", "")
            for i, ln in enumerate(wrap(err, iw - 28, 0.5, 1, max_lines=5)):
                put(out, ln, (14, 160 + i * 22), 0.5, (120, 120, 235))
        progress = status[1] if status is not None else 0.0
        cv2.rectangle(out, (0, 0), (iw, 92), (30, 60, 110), -1)
        put(out, "LOOKING FOR THE PAGE", (12, 30), 0.8, WHITE, 2)
        put(out, "whole sheet in view, hands out, hold still", (12, 58), 0.55, WHITE)
        cv2.rectangle(out, (12, 70), (iw - 12, 82), (80, 80, 80), -1)
        cv2.rectangle(out, (12, 70), (12 + int((iw - 24) * min(1.0, progress)), 82), (90, 200, 90), -1)
        return out

    def _imu(self, img, now: float, motion: MotionState) -> None:
        s = self.s
        x, y, w, h = panel(img, R_IMU, "PEN PROBE - what the pen is doing", MOTION_COLOR[motion])
        # state badge
        cv2.rectangle(img, (x + 12, y + 40), (x + 250, y + 100), MOTION_COLOR[motion], -1)
        put(img, motion.value, (x + 24, y + 83), 1.25, (15, 15, 15), 3)
        # taps: flash for TAP_FLASH_S, then keep showing the last one (bring-up: "did it see my tap?")
        tx, tw = x + 270, w - 282
        words = {1: ("SINGLE TAP", "read / repeat"), 2: ("DOUBLE TAP", "next question")}
        lt = self.last_tap
        if lt is not None and now - lt[0] < TAP_FLASH_S:
            k = 1.0 - (now - lt[0]) / TAP_FLASH_S
            cv2.rectangle(img, (tx, y + 40), (tx + tw, y + 100), (int(40 + 50 * k), int(120 + 100 * k), int(220 * k + 20)), -1)
            big, small = words.get(lt[1], (f"TAP x{lt[1]}", ""))
            put(img, big, (tx + 12, y + 72), 0.85, (15, 15, 15), 2)
            put(img, small + ("  (keyboard)" if lt[2] else ""), (tx + 12, y + 93), 0.5, (15, 15, 15), 1)
        else:
            cv2.rectangle(img, (tx, y + 40), (tx + tw, y + 100), (52, 48, 46), 1)
            if lt is None:
                put(img, "no tap yet", (tx + 12, y + 66), 0.6, GREY)
                put(img, "1 = read   2 = next", (tx + 12, y + 90), 0.5, DIM)
            else:
                big, small = words.get(lt[1], (f"TAP x{lt[1]}", ""))
                put(img, f"last: {big}", (tx + 12, y + 66), 0.55, WHITE)
                put(img, f"{now - lt[0]:.0f} s ago" + ("  (keyboard)" if lt[2] else f"  {small}"), (tx + 12, y + 90), 0.5, GREY)
        counts = f"pen taps seen:  single {self.tap_counts.get(1, 0)}   double {self.tap_counts.get(2, 0)}"
        if self.key_taps:
            counts += f"   (keyboard {self.key_taps})"
        put(img, counts, (x + 12, y + 124), 0.55, WHITE)
        # waveform: ax, ay, az-1 (g), auto-scaled
        gx, gy, gw, gh = x + 12, y + 138, w - 24, 118
        cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (26, 24, 23), -1)
        cv2.line(img, (gx, gy + gh // 2), (gx + gw, gy + gh // 2), (60, 56, 52), 1)
        samples = self._samples(200)
        scale_g = 0.2
        if samples:
            data = np.array([[sm.ax, sm.ay, sm.az - 1.0] for sm in samples], np.float32)
            scale_g = max(0.2, float(np.percentile(np.abs(data), 97)) * 1.3)
            n = len(data)
            xs = (np.arange(n) * (gw - 1) / max(1, n - 1)).astype(np.int32) + gx
            for ch, col in zip(range(3), (RED, GREEN, BLUE)):
                ys = (gy + gh / 2 - np.clip(data[:, ch] / scale_g, -1, 1) * (gh / 2 - 4)).astype(np.int32)
                cv2.polylines(img, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
        else:
            put(img, "no IMU samples", (gx + 12, gy + 80), 0.6, DIM)
        put(img, f"+/-{scale_g:.2f} g   ax ay az", (gx + 6, gy + gh + 22), 0.5, GREY)
        link, col = self._link_status()
        put(img, link, (x + 12, y + h - 14), 0.55, col, 1)

    def _link_status(self) -> Tuple[str, Tuple[int, int, int]]:
        imu = self.s.c.imu
        connected = getattr(imu, "connected", None)
        if connected is None:
            return "fake pen (scripted world)", DIM
        if connected:
            port = getattr(imu, "port_name", "") or "pen"
            ok = self.link_rate >= 10
            return f"{port} connected   {self.link_rate:.0f} lines/s", GREEN if ok else AMBER
        err = getattr(imu, "last_error", "") or "connecting..."
        return f"PEN NOT CONNECTED: {err}"[:60], RED

    def _samples(self, n: int):
        try:
            return self.s.c.imu.recent_samples(n)
        except Exception:  # noqa: BLE001
            return []

    def _guidance(self, img, gd) -> None:
        x, y, w, h = panel(img, R_GUIDE, "GUIDANCE - pen vs answer box")
        if gd is None or not gd.pen_visible and gd.speech is None:
            put(img, "no pen / no active box", (x + 14, y + 74), 0.8, DIM, 1)
            return
        speech = gd.speech or "-"
        if text_w(speech, 0.95, 2) <= w - 28:
            put(img, speech, (x + 14, y + 66), 0.95, WHITE, 2)
        else:                                                   # long cue (e.g. off the page): two smaller lines
            for i, ln in enumerate(wrap(speech, w - 28, 0.62, 1, max_lines=2)):
                put(img, ln, (x + 14, y + 52 + i * 22), 0.62, WHITE, 1)
        put(img, f"dx {gd.dx_cm:+.1f} cm   dy {gd.dy_cm:+.1f} cm   dist {gd.dist_cm:.1f} cm", (x + 14, y + 98), 0.62, GREY)
        put(img, f"cmd {gd.cmd.value if gd.cmd else '-'}", (x + 14, y + 128), 0.62, AMBER)
        col = {WriteStatus.INSIDE: GREEN, WriteStatus.OUTSIDE: RED, WriteStatus.UNKNOWN: DIM}[gd.write_status]
        put(img, f"{'IN BOX' if gd.in_box else 'outside'} / {gd.write_status.value}", (x + 230, y + 128), 0.62, col, 2)

    def _audit(self, img, phase: Phase, now: float) -> None:
        x, y, w, h = panel(img, R_AUDIT, "LAST AUDIT")
        if phase == Phase.AUDITING:
            dots = "." * (int(now * 3) % 4)
            put(img, "checking answer" + dots, (x + 14, y + 70), 0.95, PURPLE, 2)
        la = self.s.last_audit
        if la is None:
            if phase != Phase.AUDITING:
                put(img, "nothing audited yet", (x + 14, y + 70), 0.7, DIM)
            return
        _, r = la
        ok = r.ink_present
        put(img, f"{r.question_id}: " + ("INK IN BOX" if ok else "NO INK IN BOX"),
            (x + 14, y + (110 if phase == Phase.AUDITING else 66)), 0.95, GREEN if ok else RED, 2)
        yy = y + (136 if phase == Phase.AUDITING else 98)
        put(img, f"outside: {'yes' if r.ink_outside else 'no'}   confidence {r.confidence:.2f}", (x + 14, yy), 0.6, GREY)
        for i, ln in enumerate(wrap(r.note, w - 28, 0.5, 1, max_lines=2)):
            put(img, ln, (x + 14, yy + 26 + i * 20), 0.5, DIM)

    def _haptic_panel(self, img, now: float) -> None:
        x, y, w, h = panel(img, R_HAPTIC, "HAPTICS - buzzers + command log")
        cmd, age = self._active_haptic(now)
        left = cmd in ("GUIDE_LEFT", "GUIDE_BOTH", "LOCK", "COMPLETE") or (cmd == "WARN" and int(now * 10) % 2 == 0)
        right = cmd in ("GUIDE_RIGHT", "GUIDE_BOTH", "LOCK", "COMPLETE") or (cmd == "WARN" and int(now * 10) % 2 == 1)
        col = RED if cmd == "WARN" else (GREEN if cmd in ("LOCK", "COMPLETE") else AMBER)
        for cx, lit, name in ((x + 70, left, "L  D3"), (x + 190, right, "R  D4")):
            cv2.circle(img, (cx, y + 78), 30, col if lit else (60, 56, 52), -1, cv2.LINE_AA)
            cv2.circle(img, (cx, y + 78), 30, GREY, 2, cv2.LINE_AA)
            put(img, name, (cx - 32, y + 132), 0.55, GREY)
        put(img, cmd if cmd != "OFF" else "silent", (x + 270, y + 90), 1.0, col if cmd != "OFF" else DIM, 2)
        for i, (t, text) in enumerate(list(self.haptics)[-8:][::-1]):
            c = {"WARN": RED, "LOCK": GREEN, "COMPLETE": GREEN, "OFF": DIM}.get(text, AMBER)
            put(img, f"{t - (self._t0 or 0):6.1f}s  {text}", (x + 14, y + 168 + i * 19), 0.5, c if i == 0 else GREY)

    def _active_haptic(self, now: float) -> Tuple[str, float]:
        """What the buzzers are doing right now, mimicking the firmware (dead-man + one-shots)."""
        if not self.haptics:
            return "OFF", 0.0
        t, cmd = self.haptics[-1]
        age = now - t
        if cmd in ("LOCK", "COMPLETE"):
            return (cmd, age) if age < 0.5 else ("OFF", age)
        if cmd == "OFF":
            return "OFF", age
        return (cmd, age) if age < 1.5 else ("OFF", age)       # GUIDE_*/WARN auto-expire after 1.5 s

    def _voice_panel(self, img, now: float) -> None:
        s = self.s
        speaking = s.inputs.speaking if s.inputs else False
        x, y, w, h = panel(img, R_VOICE, "VOICE", GREEN if speaking else PANEL_EDGE)
        put(img, "SPEAKING" if speaking else "quiet", (x + w - 150, y + 26), 0.62, GREEN if speaking else DIM, 2)
        yy = y + 56
        for t, text in list(self.voice)[-9:][::-1]:
            lines = wrap(text, w - 110, 0.58, 1, max_lines=2)
            put(img, f"{t - (self._t0 or 0):6.1f}s", (x + 12, yy), 0.5, DIM)
            for ln in lines:
                put(img, ln, (x + 92, yy), 0.58, WHITE if yy == y + 56 else GREY, 1)
                yy += 24
            yy += 6
            if yy > y + h - 10:
                break

    def _debug_badge(self, img) -> None:
        d = self.debug
        ov = getattr(d, "override", None)
        txt = f"DEBUG KEYS ON  override={ov.value if ov else 'none'}   1 still 2 moving 3 writing 4 lifted | r tap1 t/n tap2 | 0 release"
        cv2.rectangle(img, (0, H - 26), (W, H), (60, 40, 20), -1)
        put(img, txt, (12, H - 8), 0.55, AMBER if ov else GREY)
