"""
Fake implementations of EVERY contract interface, driven by one shared scripted timeline
(FakeWorld) so the fakes agree with each other: when the world says the pen is WRITING,
the fake IMU reports WRITING *and* the fake tracker moves the pen inside the box.

Use these to develop your piece with the other three pieces faked:
    python -m src.sim                    # everything fake (must always pass)
    python -m src.sim --real brain       # YOUR state machine against fake everything else
    python -m src.sim --real imu,voice   # real hardware pieces (runs in real time)

ReferenceStateMachine is a HAPPY-PATH reference of the transition table in PLAN.md.
Dev 3 replaces it with src/state_machine.py (same interface, plus the error paths).
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from src.contracts import (
    PAGE_H_CM, PAGE_W_CM, TIMING, Action, AuditResult, Auditor, Box, Clock, Guidance,
    GuidanceEngine, Haptic, HapticCmd, ImuEvent, ImuLink, ImuSample, Inputs, MotionState,
    Phase, Question, SetBox, Silence, Snapshot, Speak, Tap, Timing, Tracker, TrackerReading,
    Voice, WriteStatus, PenState, sample_layout, sample_questions,
)

S, M, W, L = MotionState.STILL, MotionState.MOVING, MotionState.WRITING, MotionState.LIFTED
P0 = (0.50, 0.10)

# (t_start, t_end, motion, pen_from, pen_to, jitter) -- script time in seconds.
# Three questions. q1 includes a deliberate drift OUT of box1 to test WARN, and a short pen
# occlusion (OCCLUSIONS). Every answer ends the only way the plan allows: writing stops for
# ANSWER_IDLE_S (there is no third tap). q3 is started with tap 1 ("read the question").
SEGMENTS: List[Tuple[float, float, MotionState, Tuple[float, float], Tuple[float, float], float]] = [
    # Q1: IDLE -> MOVING -> STILL in box1 -> WRITING (with drift) -> IDLE
    (0.0, 5.0, S, P0, P0, 0.0),
    (5.0, 8.0, M, P0, (0.50, 0.20), 0.0),
    (8.0, 9.0, S, (0.50, 0.20), (0.50, 0.20), 0.0),
    (9.0, 11.0, W, (0.50, 0.20), (0.80, 0.20), 0.006),
    (11.0, 12.0, W, (0.80, 0.20), (0.80, 0.05), 0.006),   # leaves box1 (ymin 0.10)
    (12.0, 12.8, W, (0.80, 0.05), (0.70, 0.20), 0.006),   # re-enters
    (12.8, 16.0, W, (0.70, 0.20), (0.20, 0.25), 0.006),
    (16.0, 23.5, S, (0.20, 0.25), (0.20, 0.25), 0.0),     # Answer 1 done (idle timeout)
    # Transition to Q2
    (23.5, 24.0, L, (0.20, 0.25), (0.20, 0.25), 0.0),
    (24.0, 27.0, M, (0.20, 0.25), (0.50, 0.50), 0.0),     # move to box2
    (27.0, 28.0, S, (0.50, 0.50), (0.50, 0.50), 0.0),
    (28.0, 33.0, W, (0.50, 0.50), (0.30, 0.55), 0.006),
    (33.0, 38.0, S, (0.30, 0.55), (0.30, 0.55), 0.0),     # Answer 2 done (idle timeout)
    # Transition to Q3
    (38.0, 39.0, L, (0.30, 0.55), (0.30, 0.55), 0.0),
    (39.0, 42.0, M, (0.30, 0.55), (0.50, 0.75), 0.0),     # move to box3
    (42.0, 43.0, S, (0.50, 0.75), (0.50, 0.75), 0.0),     # lock in box3
    (43.0, 48.0, W, (0.50, 0.75), (0.40, 0.80), 0.006),   # writing q3
    (48.0, 1e9, S, (0.40, 0.80), (0.40, 0.80), 0.0),      # Answer 3 done (idle timeout)
]
OCCLUSIONS = [(6.0, 6.3)]
TAPS = [
    (36.3, Tap.SINGLE),  # after answer 2 is recorded: "read the next question"
]


class FakeWorld:
    """One scripted timeline shared by the fake tracker and the fake IMU."""

    def __init__(self, clock: Clock, speed: float = 1.0) -> None:
        self.clock = clock
        self.speed = speed
        self._t0: Optional[float] = None

    def script_t(self) -> float:
        if self._t0 is None:
            self._t0 = self.clock.now()
        return (self.clock.now() - self._t0) * self.speed

    def _seg(self, ts: float):
        for seg in SEGMENTS:
            if seg[0] <= ts < seg[1]:
                return seg
        return SEGMENTS[-1]

    def motion_at(self, ts: float) -> MotionState:
        return self._seg(ts)[2]

    def pen_at(self, ts: float) -> Optional[Tuple[float, float]]:
        if any(a <= ts < b for a, b in OCCLUSIONS):
            return None
        t0, t1, _, p0, p1, jit = self._seg(ts)
        u = 0.0 if t1 - t0 > 1e6 else (ts - t0) / (t1 - t0)
        x = p0[0] + (p1[0] - p0[0]) * u + jit * math.sin(ts * 40)
        y = p0[1] + (p1[1] - p0[1]) * u + jit * math.cos(ts * 53)
        return (x, y)


# --------------------------------------------------------------------------- fakes
class FakeTracker:
    def __init__(self, world: FakeWorld, layout: Optional[Dict[str, Box]] = None,
                 render: bool = True) -> None:
        self.world = world
        self.layout = layout or sample_layout()
        self.render = render
        self.clock = world.clock

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def read(self) -> Optional[TrackerReading]:
        xy = self.world.pen_at(self.world.script_t())
        t = self.clock.now()
        pen = PenState(t, xy[0], xy[1], 0.0, 1.0) if xy else None
        return TrackerReading(t, pen, self._frame(pen) if self.render else None)

    def snapshot(self) -> Any:
        return self._frame(None) if self.render else None

    def _frame(self, pen: Optional[PenState]) -> np.ndarray:
        h, w = 240, 320
        img = np.full((h, w, 3), 255, np.uint8)
        for b in self.layout.values():
            x0, x1 = int(b.xmin * w), int(b.xmax * w) - 1
            y0, y1 = int(b.ymin * h), int(b.ymax * h) - 1
            img[y0:y0 + 2, x0:x1] = 0
            img[y1 - 1:y1 + 1, x0:x1] = 0
            img[y0:y1, x0:x0 + 2] = 0
            img[y0:y1, x1 - 1:x1 + 1] = 0
        if pen:
            px = min(w - 5, max(4, int(pen.x * w)))
            py = min(h - 5, max(4, int(pen.y * h)))
            img[py - 3:py + 4, px - 3:px + 4] = (0, 0, 255)
        return img


class FakeImuLink:
    SAMPLE_HZ = 50

    def __init__(self, world: FakeWorld, seed: int = 7) -> None:
        self.world = world
        self.clock = world.clock
        self.rng = random.Random(seed)
        self.motion: MotionState = S
        self._samples: Deque[ImuSample] = deque(maxlen=1000)
        self._next_sample: Optional[float] = None
        self._taps_done = 0
        self.haptic_log: List[Tuple[float, HapticCmd]] = []

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def poll(self) -> List[ImuEvent]:
        now, ts = self.clock.now(), self.world.script_t()
        events: List[ImuEvent] = []
        m = self.world.motion_at(ts)
        if m != self.motion:
            self.motion = m
            events.append(ImuEvent(now, motion=m))
        while self._taps_done < len(TAPS) and TAPS[self._taps_done][0] <= ts:
            events.append(ImuEvent(now, tap=TAPS[self._taps_done][1]))
            self._taps_done += 1
        if self._next_sample is None:
            self._next_sample = now   # RealClock is seconds since boot; don't backfill from 0
        while self._next_sample <= now:
            self._samples.append(self._sample(self._next_sample, self.motion))
            self._next_sample += 1.0 / self.SAMPLE_HZ
        return events

    def _sample(self, t: float, m: MotionState) -> ImuSample:
        g = self.rng.gauss
        if m == S:
            ax, ay, az = g(0, .003), g(0, .003), 1 + g(0, .003)
        elif m == M:
            ax, ay, az = .15 * math.sin(t * 3) + g(0, .01), .10 * math.cos(t * 2) + g(0, .01), 1 + g(0, .01)
        elif m == W:
            ax, ay, az = .05 * math.sin(t * 60) + g(0, .01), .05 * math.cos(t * 75) + g(0, .01), 1 + .03 * math.sin(t * 50)
        else:
            ax, ay, az = g(0, .05), g(0, .05), 1.8 + g(0, .1)
        return ImuSample(t, ax, ay, az, 700 if m != W else 400)

    def send(self, cmd: HapticCmd) -> None:
        self.haptic_log.append((self.clock.now(), cmd))

    def recent_samples(self, n: int = 200) -> List[ImuSample]:
        return list(self._samples)[-n:]


class FakeVoice:
    SECONDS_PER_CHAR = 0.045

    def __init__(self, clock: Clock, scripted: Optional[List[Tuple[float, str]]] = None) -> None:
        self.clock = clock
        self._free_at = 0.0
        self.log: List[Tuple[float, str]] = []
        self._scripted = sorted(scripted or [])

    def speak(self, text: str, interrupt: bool = False) -> None:
        now = self.clock.now()
        start = now if interrupt else max(now, self._free_at)
        self._free_at = start + 0.3 + self.SECONDS_PER_CHAR * len(text)
        self.log.append((now, text))

    def silence(self) -> None:
        self._free_at = self.clock.now()

    def is_speaking(self) -> bool:
        return self.clock.now() < self._free_at

    def poll_command(self) -> Optional[str]:
        if self._scripted and self._scripted[0][0] <= self.clock.now():
            return self._scripted.pop(0)[1]
        return None

    def stop(self) -> None: ...


class FakeAuditor:
    def __init__(self, clock: Clock, latency: float = 1.0, ink_present: bool = True) -> None:
        self.clock, self.latency, self.ink_present = clock, latency, ink_present
        self._pending: List[Tuple[float, AuditResult]] = []

    def submit(self, snapshot: Any, box: Box, question: Question) -> None:
        r = AuditResult(question.id, self.ink_present, False, 0.9, "fake")
        self._pending.append((self.clock.now() + self.latency, r))

    def poll(self) -> Optional[AuditResult]:
        if self._pending and self._pending[0][0] <= self.clock.now():
            return self._pending.pop(0)[1]
        return None


class FakeGuidanceEngine:
    """Basic geometry. Dev 2's real one adds smoothing, hysteresis and better phrasing."""
    DEADZONE_CM = 1.0

    def compute(self, pen: Optional[PenState], box: Optional[Box]) -> Guidance:
        if pen is None or box is None:
            return Guidance(write_status=WriteStatus.UNKNOWN, pen_visible=pen is not None)
        dx = box.xmin - pen.x if pen.x < box.xmin else (box.xmax - pen.x if pen.x > box.xmax else 0.0)
        dy = box.ymin - pen.y if pen.y < box.ymin else (box.ymax - pen.y if pen.y > box.ymax else 0.0)
        dx_cm, dy_cm = dx * PAGE_W_CM, dy * PAGE_H_CM
        dist = math.hypot(dx_cm, dy_cm)
        in_box = box.contains(pen.x, pen.y)
        if in_box:
            return Guidance(None, "You are in the answer box.", 0, 0, 0, True, WriteStatus.INSIDE, True)
        if abs(dx_cm) > self.DEADZONE_CM:
            cmd = HapticCmd.GUIDE_RIGHT if dx_cm > 0 else HapticCmd.GUIDE_LEFT
        else:
            cmd = HapticCmd.GUIDE_BOTH
        if abs(dy_cm) >= abs(dx_cm):
            d, n = ("Down" if dy_cm > 0 else "Up"), abs(dy_cm)
        else:
            d, n = ("Right" if dx_cm > 0 else "Left"), abs(dx_cm)
        inches = max(1, int(round(n / 2.54)))
        speech = f"Move {inches} {'inch' if inches == 1 else 'inches'} {d}."
        return Guidance(cmd, speech, dx_cm, dy_cm, dist, False, WriteStatus.OUTSIDE, True)


# --------------------------------------------------------------------------- reference brain
def _voice_to_gestures(cmd: Optional[str]) -> List[Tap]:
    if not cmd:
        return []
    if any(k in cmd for k in ("repeat", "again")):
        return [Tap.SINGLE]
    if any(k in cmd for k in ("skip", "next")):
        return [Tap.DOUBLE]
    if any(k in cmd for k in ("done", "finished")):
        return [Tap.TRIPLE]
    return []


class ReferenceStateMachine:
    """
    Happy-path reference of the transition table (see PLAN.md section 6, Dev 3).
    IMU motion state drives EVERY transition; the camera only supplies Guidance.

      IDLE ──(STILL>=HOVER_S | tap1)──► READING ──(speech finished)──► NAVIGATING
      NAVIGATING ──(motion==WRITING)──► WRITING ──(not writing >= ANSWER_IDLE_S | tap3)──► AUDITING
      AUDITING ──(ink ok)──► IDLE (next q) or COMPLETE      AUDITING ──(no ink)──► NAVIGATING
      tap2 (or "skip") in IDLE/READING/NAVIGATING skips the question
    """

    def __init__(self, questions: Optional[List[Question]] = None,
                 layout: Optional[Dict[str, Box]] = None, timing: Timing = TIMING) -> None:
        self.tm = timing
        self.layout = layout or sample_layout()
        self.reset(questions or sample_questions())

    # -- lifecycle
    def reset(self, questions: List[Question]) -> None:
        self.questions = list(questions)
        self.index = 0
        self.phase = Phase.IDLE
        self._phase_t: Optional[float] = None
        self._still_since: Optional[float] = None
        self._nonwriting_since: Optional[float] = None
        self._last_cmd = HapticCmd.OFF
        self._last_cmd_t = -1e9
        self._last_nav_speech = -1e9
        self._locked = False
        self._heard_speaking = False

    @property
    def question(self) -> Optional[Question]:
        return self.questions[self.index] if self.index < len(self.questions) else None

    # -- helpers
    def _enter(self, phase: Phase, now: float) -> None:
        self.phase, self._phase_t = phase, now
        self._heard_speaking = False
        self._nonwriting_since = None
        self._locked = False

    def _still_for(self, now: float) -> float:
        return 0.0 if self._still_since is None else now - self._still_since

    def _haptic(self, cmd: HapticCmd, now: float, acts: List[Action]) -> None:
        refresh = cmd in (HapticCmd.GUIDE_LEFT, HapticCmd.GUIDE_RIGHT, HapticCmd.GUIDE_BOTH,
                          HapticCmd.WARN) and now - self._last_cmd_t >= self.tm.HAPTIC_REFRESH_S
        if cmd != self._last_cmd or refresh:
            acts.append(Haptic(cmd))
            self._last_cmd, self._last_cmd_t = cmd, now

    def _advance(self, now: float, acts: List[Action]) -> None:
        self.index += 1
        if self.question is None:
            acts.append(Speak("That was the last question. Test complete."))
            self.phase = Phase.COMPLETE
        else:
            self._enter(Phase.IDLE, now)

    # -- main entry
    def update(self, inp: Inputs) -> List[Action]:
        now, acts = inp.t, []
        if self._phase_t is None:
            self._phase_t = now
        if inp.motion == MotionState.STILL:
            if self._still_since is None:
                self._still_since = now
        else:
            self._still_since = None
        gestures = [e.tap for e in inp.imu_events if e.tap != Tap.NONE] + _voice_to_gestures(inp.voice_cmd)
        q = self.question
        if q is None or self.phase == Phase.COMPLETE:
            return acts
        getattr(self, "_on_" + self.phase.value.lower())(inp, q, gestures, acts)
        return acts

    # -- phase handlers
    def _on_idle(self, inp: Inputs, q: Question, g: List[Tap], acts: List[Action]) -> None:
        now = inp.t
        if Tap.DOUBLE in g:
            acts.append(Speak("Skipping question.", interrupt=True))
            return self._advance(now, acts)
        hovered = self._still_for(now) >= self.tm.HOVER_S and now - self._phase_t >= self.tm.HOVER_S
        if Tap.SINGLE in g or hovered:
            acts += [SetBox(self.layout[q.box_id]),
                     Speak(f"Question {self.index + 1}. {q.text}", interrupt=True)]
            self._enter(Phase.READING, now)

    def _on_reading(self, inp: Inputs, q: Question, g: List[Tap], acts: List[Action]) -> None:
        now = inp.t
        if Tap.DOUBLE in g:
            acts.append(Speak("Skipping question.", interrupt=True))
            return self._advance(now, acts)
        if Tap.SINGLE in g and now - self._phase_t > 0.5:
            acts.append(Speak(f"Question {self.index + 1}. {q.text}", interrupt=True))
            self._phase_t, self._heard_speaking = now, False
            return
        if inp.speaking:
            self._heard_speaking = True
        elif self._heard_speaking or now - self._phase_t > self.tm.READ_GRACE_S:
            self._enter(Phase.NAVIGATING, now)

    def _on_navigating(self, inp: Inputs, q: Question, g: List[Tap], acts: List[Action]) -> None:
        now, gd = inp.t, inp.guidance
        if Tap.DOUBLE in g:
            acts.append(Speak("Skipping question.", interrupt=True))
            return self._advance(now, acts)
        if Tap.SINGLE in g:
            acts.append(Speak(f"Question {self.index + 1}. {q.text}", interrupt=True))
        if inp.motion == MotionState.WRITING:
            acts.append(Silence())
            self._enter(Phase.WRITING, now)
            return
        if inp.motion == MotionState.MOVING:
            self._locked = False
            cmd = gd.cmd if (gd and gd.cmd and not gd.in_box) else HapticCmd.OFF
            self._haptic(cmd, now, acts)
            if (gd and not gd.in_box and gd.speech and not inp.speaking
                    and now - self._last_nav_speech >= self.tm.NAV_SPEECH_EVERY_S):
                acts.append(Speak(gd.speech))
                self._last_nav_speech = now
        else:  # STILL or LIFTED: no guidance buzzing
            self._haptic(HapticCmd.OFF, now, acts)
            if (inp.motion == MotionState.STILL and gd and gd.in_box and not self._locked
                    and self._still_for(now) >= self.tm.LOCK_STILL_S):
                self._locked = True
                acts += [Haptic(HapticCmd.LOCK), Speak("In the answer box. Start writing.")]
                self._last_cmd = HapticCmd.OFF   # LOCK is one-shot

    def _on_writing(self, inp: Inputs, q: Question, g: List[Tap], acts: List[Action]) -> None:
        now, gd = inp.t, inp.guidance
        if inp.motion == MotionState.WRITING:
            self._nonwriting_since = None
        elif self._nonwriting_since is None:
            self._nonwriting_since = now
        done = Tap.TRIPLE in g or (self._nonwriting_since is not None
                                   and now - self._nonwriting_since >= self.tm.ANSWER_IDLE_S)
        if done:
            self._haptic(HapticCmd.OFF, now, acts)
            acts.append(Snapshot(q.id))
            self._enter(Phase.AUDITING, now)
            return
        status = gd.write_status if gd else WriteStatus.UNKNOWN
        # IMU gating: the margin buzzer only fires while the pen is actually WRITING.
        if inp.motion == MotionState.WRITING and status == WriteStatus.OUTSIDE:
            self._haptic(HapticCmd.WARN, now, acts)
        else:
            self._haptic(HapticCmd.OFF, now, acts)

    def _on_auditing(self, inp: Inputs, q: Question, g: List[Tap], acts: List[Action]) -> None:
        now, r = inp.t, inp.audit_result
        if r is not None and r.question_id == q.id:
            if not r.ink_present:
                acts.append(Speak("I did not find writing in the box. Move to the box and try again."))
                return self._enter(Phase.NAVIGATING, now)
            msg = "Answer recorded." + (" Some writing went outside the box." if r.ink_outside else "")
        elif now - self._phase_t > self.tm.AUDIT_TIMEOUT_S:
            msg = "I could not check your answer. Moving on."
        else:
            return
        acts += [Speak(msg), Haptic(HapticCmd.COMPLETE)]
        self._last_cmd = HapticCmd.OFF
        self._advance(now, acts)


# static interface check (cheap safety net: fails at import if a fake drifts from the contract)
def _check() -> None:
    c = type("C", (), {"now": lambda self: 0.0})()
    w = FakeWorld(c)
    assert isinstance(FakeImuLink(w), ImuLink)
    assert isinstance(FakeTracker(w), Tracker)
    assert isinstance(FakeVoice(c), Voice)
    assert isinstance(FakeAuditor(c), Auditor)
    assert isinstance(FakeGuidanceEngine(), GuidanceEngine)


_check()
