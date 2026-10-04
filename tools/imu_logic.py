"""Python mirror of the pen firmware logic (firmware/illumin_pen/*.h).

The firmware cannot be run on this PC (no host C++ compiler), so the same algorithms live here
and are tested against the recorded clips and synthetic signals. Keep this file and the headers
in sync: motion_classifier.h, tap_detector.h and haptics.h implement exactly what is below, and
every constant is read from firmware/illumin_pen/imu_params.h (never hard-coded here).
"""
from __future__ import annotations

import math
import os
import re
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_H = os.path.join(REPO, "firmware", "illumin_pen", "imu_params.h")

STILL, MOVING, WRITING = "STILL", "MOVING", "WRITING"


# --------------------------------------------------------------------------- parameters
def load_params(path: str = PARAMS_H) -> Dict[str, float]:
    """Parse `constexpr <type> NAME = value;` lines from imu_params.h."""
    params: Dict[str, float] = {}
    pat = re.compile(r"constexpr\s+(float|int|bool)\s+(\w+)\s*=\s*([^;/]+?)\s*;")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            kind, name, raw = m.groups()
            raw = raw.strip().rstrip("f")
            if kind == "bool":
                params[name] = raw == "true"
            elif kind == "int":
                params[name] = int(raw)
            else:
                params[name] = float(raw)
    return params


# --------------------------------------------------------------------------- motion classifier
def window_spread(window: List[Tuple[float, float, float]]) -> float:
    """Sum over the three axes of the population standard deviation. The feature that
    separated STILL from active motion best in the recordings."""
    n = len(window)
    total = 0.0
    for k in range(3):
        mean = sum(s[k] for s in window) / n
        var = sum((s[k] - mean) ** 2 for s in window) / n
        total += math.sqrt(var)
    return total


class MotionClassifier:
    """STILL / MOVING / WRITING from a sliding window of accelerometer samples.

    Feed every emitted sample (the ~33 Hz stream, the same data the thresholds were tuned on).
    A new state is reported only after it has held for CLS_DWELL samples, and leaving a state
    needs the score to cross a slightly lower (hysteresis) threshold than entering it.
    """

    def __init__(self, params: Optional[Dict[str, float]] = None) -> None:
        self.p = params or load_params()
        self.window: Deque[Tuple[float, float, float]] = deque(maxlen=int(self.p["CLS_WINDOW"]))
        self.state = STILL
        self._cand = STILL
        self._dwell = 0
        self.score = 0.0

    def _target(self, score: float) -> str:
        t_act, t_wr, h = self.p["T_ACTIVE"], self.p["T_WRITE"], self.p["CLS_HYST"]
        if self.state == STILL:
            if score < t_act:
                tgt = STILL
            elif score >= t_wr:
                tgt = WRITING
            else:
                tgt = MOVING
        elif self.state == MOVING:
            if score < t_act * h:
                tgt = STILL
            elif score >= t_wr:
                tgt = WRITING
            else:
                tgt = MOVING
        else:  # WRITING
            if score < t_act * h:
                tgt = STILL
            elif score < t_wr * h:
                tgt = MOVING
            else:
                tgt = WRITING
        if tgt == WRITING and not self.p["EMIT_WRITING"]:
            tgt = MOVING
        return tgt

    def update(self, ax: float, ay: float, az: float, frozen: bool = False) -> Optional[str]:
        """Returns the new state if it changed on this sample, else None."""
        self.window.append((ax, ay, az))
        if len(self.window) < self.window.maxlen:
            return None
        self.score = window_spread(list(self.window))
        if frozen:  # a tap just happened: its energy is not motion, hold the current state
            self._cand, self._dwell = self.state, 0
            return None
        tgt = self._target(self.score)
        if tgt == self.state:
            self._cand, self._dwell = tgt, 0
            return None
        if tgt == self._cand:
            self._dwell += 1
        else:
            self._cand, self._dwell = tgt, 1
        if self._dwell >= int(self.p["CLS_DWELL"]):
            self.state, self._cand, self._dwell = tgt, tgt, 0
            return self.state
        return None


# --------------------------------------------------------------------------- tap detector
class TapDetector:
    """1- or 2-tap gestures from BURSTS of |a| activity. Feed EVERY internal sample.

    A pen tap rings for 0.1-0.5 s, so one tap is a short burst and a double tap a longer one (or two bursts).
    A burst starts at the first sample whose |a| deviates TAP_ACT_G from its slow baseline, provided the pen has
    been resting (`allowed`) and there was no activity for TAP_PRE_QUIET_MS; it ends after TAP_BURST_GAP_MS without
    activity. Span <= TAP_ONE_MAX_MS counts as one tap, a longer span up to TAP_BURST_MAX_MS as two; anything longer
    is motion and cancels the whole gesture. Taps are reported TAP_GROUP_MS after the last burst; more than
    TAP_MAX in total cancels. Tuned on 20 recorded taps from one person (tools/verify_taps.py).
    """

    IDLE, IN_BURST, WAIT = 0, 1, 2

    def __init__(self, params: Optional[Dict[str, float]] = None) -> None:
        self.p = params or load_params()
        self.ema: Optional[float] = None
        self.state = self.IDLE
        self._last_act = -1e9
        self._start = self._last = 0.0
        self._peak = 0.0
        self._total = 0
        self._freeze_until = -1.0

    def frozen(self, t_ms: float) -> bool:
        return t_ms < self._freeze_until

    def _cancel(self) -> None:
        self.state, self._total, self._freeze_until = self.IDLE, 0, -1.0

    def update(self, t_ms: float, ax: float, ay: float, az: float, allowed: bool = True) -> int:
        """Returns 1 or 2 when a tap gesture is complete on this sample, else 0."""
        p = self.p
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        if self.ema is None:
            self.ema = mag
        dev = abs(mag - self.ema)
        act = dev >= p["TAP_ACT_G"]
        if not act:
            self.ema += p["TAP_EMA_ALPHA"] * (mag - self.ema)  # the baseline follows quiet only, never a tap
        out = 0

        if self.state == self.IDLE:
            if act and allowed and t_ms - self._last_act >= p["TAP_PRE_QUIET_MS"]:
                self.state, self._start, self._last, self._peak, self._total = self.IN_BURST, t_ms, t_ms, dev, 0
                self._freeze_until = t_ms + p["TAP_FREEZE_MS"]
        elif self.state == self.IN_BURST:
            if act:
                self._last, self._peak = t_ms, max(self._peak, dev)
                self._freeze_until = t_ms + p["TAP_FREEZE_MS"]
                if self._last - self._start > p["TAP_BURST_MAX_MS"]:
                    self._cancel()  # too long: this is motion
            elif t_ms - self._last >= p["TAP_BURST_GAP_MS"]:
                span = self._last - self._start
                if self._peak < p["TAP_PEAK_G"] or span > p["TAP_BURST_MAX_MS"]:
                    self._cancel()  # a small bump, not a tap
                else:
                    self._total += 1 if span <= p["TAP_ONE_MAX_MS"] else 2
                    if self._total > p["TAP_MAX"]:
                        self._cancel()
                    else:
                        self.state = self.WAIT
        elif self.state == self.WAIT:
            if act:  # a second burst: its own span decides whether it adds one tap or two
                self.state, self._start, self._last, self._peak = self.IN_BURST, t_ms, t_ms, dev
                self._freeze_until = t_ms + p["TAP_FREEZE_MS"]
            elif t_ms - self._last >= p["TAP_GROUP_MS"]:
                out, self.state, self._total = self._total, self.IDLE, 0

        if act:
            self._last_act = t_ms
        return out


# --------------------------------------------------------------------------- haptic engine
# Each step is (duration_ms, left_hz, right_hz). D3 = left buzzer, D4 = right buzzer. 0 Hz = silent.
ONE_SHOT: Dict[str, List[Tuple[int, int, int]]] = {
    "LOCK": [(80, 1000, 1000), (40, 0, 0), (80, 1500, 1500)],
    "COMPLETE": [(120, 600, 600), (120, 900, 900), (120, 1200, 1200)],
}
REPEATING: Dict[str, List[Tuple[int, int, int]]] = {
    "GUIDE_LEFT": [(100, 1000, 0), (150, 0, 0)],
    "GUIDE_RIGHT": [(100, 0, 1000), (150, 0, 0)],
    "GUIDE_BOTH": [(100, 800, 800), (400, 0, 0)],
    "WARN": [(100, 2500, 0), (100, 0, 2500)],
}


class HapticEngine:
    """Pattern player: command() sets what to play, tick(now_ms) returns (left_hz, right_hz).

    No hardware calls here, so it is testable and the buzzer wiring can be added at integration.
    LOCK/COMPLETE are one-shot and play to the end. GUIDE_* and WARN repeat until replaced and
    stop on their own HAPTIC_DEADMAN_MS after the last refresh (the dead-man switch). OFF stops all.
    """

    def __init__(self, params: Optional[Dict[str, float]] = None) -> None:
        self.p = params or load_params()
        self.cmd: Optional[str] = None
        self._steps: List[Tuple[int, int, int]] = []
        self._idx = 0
        self._step_start = 0.0
        self._oneshot = False
        self._last_refresh = 0.0
        self._pending: Optional[Tuple[str, float]] = None

    def _start(self, cmd: str, now: float) -> None:
        self.cmd = cmd
        self._oneshot = cmd in ONE_SHOT
        self._steps = ONE_SHOT[cmd] if self._oneshot else REPEATING[cmd]
        self._idx, self._step_start, self._last_refresh = 0, now, now

    def _stop(self) -> None:
        self.cmd, self._steps, self._idx, self._oneshot = None, [], 0, False

    def command(self, cmd: str, now_ms: float) -> None:
        if cmd == "OFF":
            self._stop()
            self._pending = None
        elif cmd in ONE_SHOT:
            if not (self._oneshot and self.cmd == cmd):  # re-sent while already playing: let it finish
                self._start(cmd, now_ms)
        elif cmd in REPEATING:
            if self._oneshot:
                self._pending = (cmd, now_ms)  # start right after the one-shot ends
            elif self.cmd == cmd:
                self._last_refresh = now_ms
            else:
                self._start(cmd, now_ms)

    def tick(self, now_ms: float) -> Tuple[int, int]:
        if self.cmd is None:
            return (0, 0)
        if not self._oneshot and now_ms - self._last_refresh > self.p["HAPTIC_DEADMAN_MS"]:
            self._stop()
            return (0, 0)
        while self.cmd is not None and now_ms - self._step_start >= self._steps[self._idx][0]:
            self._step_start += self._steps[self._idx][0]
            self._idx += 1
            if self._idx >= len(self._steps):
                if self._oneshot:
                    self._stop()
                    pend, self._pending = self._pending, None
                    if pend and now_ms - pend[1] <= self.p["HAPTIC_DEADMAN_MS"]:
                        self._start(pend[0], self._step_start)
                        self._last_refresh = pend[1]  # the dead-man clock runs from the last real refresh
                else:
                    self._idx = 0
        if self.cmd is None:
            return (0, 0)
        _, left, right = self._steps[self._idx]
        return (left, right)
