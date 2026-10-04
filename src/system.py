"""
The glue loop. ONE tick = read sensors -> build Inputs -> brain.update -> execute Actions.
Owner: Dev 4 (extend it; keep step() as the single place where components meet).
The headless sim and the real main.py both call System.step().

Resilience (stage rule: ONE flaky component must never kill the demo):
every call into a component goes through _safe().  A failure is counted in
`System.errors`, remembered in `System.last_error`, logged as an "ERROR" entry
(throttled) and replaced by a harmless default.  The headless sim FAILS if any
ERROR was logged, so bugs are still caught before merge -- they are just not fatal live.
"""
from __future__ import annotations

import statistics
import sys
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple, TypeVar

from src.contracts import (
    Auditor, AuditResult, Box, Clock, Guidance, GuidanceEngine, Haptic, ImuLink, Inputs,
    MotionState, Phase, PenState, Question, SetBox, Silence, Snapshot, Speak, StateMachine,
    Tracker, TrackerReading, Voice,
)

T = TypeVar("T")

TIP_SMOOTH_WINDOW_S = 0.2   # median over this much recent history before guidance sees it
TIP_SMOOTH_MIN_POINTS = 3   # fewer than this: not enough to median, use the raw reading


@dataclass
class Components:
    imu: ImuLink
    tracker: Tracker
    voice: Voice
    guidance: GuidanceEngine
    auditor: Auditor
    brain: StateMachine


class System:
    def __init__(self, c: Components, clock: Clock, questions: List[Question],
                 layout: Dict[str, Box]) -> None:
        self.c, self.clock = c, clock
        self.questions = {q.id: q for q in questions}
        self.question_order: List[str] = [q.id for q in questions]
        self.layout = layout
        self.current_box: Optional[Box] = None
        self._tip_window: Deque[PenState] = deque()     # recent raw tip reads, for median smoothing
        self._logged_waiting = False     # one-shot: logged once while waiting for the page to lock
        self.log: List[Tuple[float, str, str]] = []     # (t, kind, text) for HUD + sim asserts
        self.inputs: Optional[Inputs] = None            # last tick, for the HUD
        self.reading: Optional[TrackerReading] = None
        self.last_audit: Optional[Tuple[float, AuditResult]] = None   # (t, result) for the HUD
        self.errors: Dict[str, int] = {}                # component -> failure count
        self.last_error: str = ""
        self.component_errors: Dict[str, str] = {}      # component -> its last error message
        self._last_phase: Optional[Phase] = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        for name, comp in (("imu", self.c.imu), ("tracker", self.c.tracker)):
            self._safe(name + ".start", comp.start, None, comp=name)

    def stop(self) -> None:
        for name, comp in (("imu", self.c.imu), ("tracker", self.c.tracker), ("voice", self.c.voice)):
            self._safe(name + ".stop", comp.stop, None, comp=name)

    # ------------------------------------------------------------------ helpers
    def _log(self, kind: str, text: str) -> None:
        self.log.append((self.clock.now(), kind, text))

    def _safe(self, what: str, fn: Callable[[], T], default: T, comp: Optional[str] = None) -> T:
        """Run one component call; on failure count it, log it (throttled) and return `default`."""
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 -- deliberately broad: nothing may kill the loop
            comp = comp or what.split(".")[0]
            n = self.errors[comp] = self.errors.get(comp, 0) + 1
            self.last_error = f"{what}: {type(e).__name__}: {e}"
            self.component_errors[comp] = self.last_error
            if n == 1:
                print(f"[system] {self.last_error}", file=sys.stderr)
                traceback.print_exc()
            if n == 1 or n % 100 == 0:
                self._log("ERROR", f"{self.last_error} (x{n})")
            return default

    def _smoothed_pen(self, pen: Optional[PenState], now: float) -> Optional[PenState]:
        """Median of recent tip reads over TIP_SMOOTH_WINDOW_S, so a single bad frame (wrong pen
        end, a stray detection - tip detection is not perfectly accurate frame to frame) can't by
        itself flip an inside/outside verdict or jerk a guidance cue. Falls straight through (no
        smoothing, no stale carry-over) once the pen is lost, so "pen gone" is reported immediately."""
        if pen is None:
            self._tip_window.clear()
            return None
        self._tip_window.append(pen)
        while self._tip_window and self._tip_window[0].t < now - TIP_SMOOTH_WINDOW_S:
            self._tip_window.popleft()
        if len(self._tip_window) < TIP_SMOOTH_MIN_POINTS:
            return pen
        mx = statistics.median(p.x for p in self._tip_window)
        my = statistics.median(p.y for p in self._tip_window)
        return PenState(t=pen.t, x=mx, y=my, lift_cm=pen.lift_cm, confidence=pen.confidence)

    def _tracker_ready(self) -> bool:
        """True once a real camera tracker has actually locked the page (boxes are now known in
        pixel space). Always True for fakes/sim, which have no such concept (no calibration_status).
        Gates the brain (and so voice/guidance/the rest of the algorithm) so nothing runs - no
        question is read, no haptic fires - before the boxes are detected."""
        status_fn = getattr(self.c.tracker, "calibration_status", None)
        if status_fn is None:
            return True
        searching, _progress, _corners = self._safe(
            "tracker.calibration_status", status_fn, (True, 0.0, None), comp="tracker")
        return not searching and getattr(self.c.tracker, "calibration", None) is not None

    # ------------------------------------------------------------------ the tick
    def step(self) -> Inputs:
        c, now = self.c, self.clock.now()
        self.reading = self._safe("tracker.read", c.tracker.read, None, comp="tracker")
        pen = self.reading.pen if self.reading else None
        pen = self._smoothed_pen(pen, now)
        events = self._safe("imu.poll", c.imu.poll, [], comp="imu") or []
        motion = self._safe("imu.motion", lambda: c.imu.motion, MotionState.STILL, comp="imu")
        guidance = self._safe("guidance.compute", lambda: c.guidance.compute(pen, self.current_box),
                              Guidance(), comp="guidance")
        audit = self._safe("auditor.poll", c.auditor.poll, None, comp="auditor")
        inp = Inputs(
            t=now,
            pen=pen,
            guidance=guidance,
            motion=motion,
            imu_events=tuple(events),
            speaking=self._safe("voice.is_speaking", c.voice.is_speaking, False, comp="voice"),
            voice_cmd=self._safe("voice.poll_command", c.voice.poll_command, None, comp="voice"),
            audit_result=audit,
        )
        if audit is not None:
            self.last_audit = (now, audit)
            self._log("AUDIT", f"{audit.question_id}: ink={audit.ink_present} "
                               f"outside={audit.ink_outside} conf={audit.confidence:.2f} {audit.note}")
        if self._tracker_ready():
            self._logged_waiting = False
            actions = self._safe("brain.update", lambda: c.brain.update(inp), [], comp="brain") or []
            for a in actions:
                self._execute(a)
            if c.brain.phase != self._last_phase:
                self._last_phase = c.brain.phase
                self._log("PHASE", c.brain.phase.value)
        elif not self._logged_waiting:
            self._logged_waiting = True
            self._log("WAITING", "page not locked yet - holding before question 1 (voice, guidance idle)")
        self.inputs = inp
        return inp

    def _execute(self, a) -> None:
        c = self.c
        if isinstance(a, Speak):
            self._safe("voice.speak", lambda: c.voice.speak(a.text, a.interrupt), None, comp="voice")
            self._log("SPEAK", a.text)
        elif isinstance(a, Silence):
            self._safe("voice.silence", c.voice.silence, None, comp="voice")
            self._log("SILENCE", "")
        elif isinstance(a, Haptic):
            self._safe("imu.send", lambda: c.imu.send(a.cmd), None, comp="imu")
            self._log("HAPTIC", a.cmd.value)
        elif isinstance(a, SetBox):
            self.current_box = a.box
            self._log("BOX", a.box.id)
        elif isinstance(a, Snapshot):
            def go() -> None:
                q = self.questions[a.question_id]
                c.auditor.submit(c.tracker.snapshot(), self.layout[q.box_id], q)
            self._safe("auditor.submit", go, None, comp="auditor")
            self._log("SNAPSHOT", a.question_id)
        else:
            self._safe("action", lambda: (_ for _ in ()).throw(TypeError(f"unknown action {a!r}")),
                       None, comp="brain")
