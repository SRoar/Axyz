"""State machine implementation for Dev 3.

Tap-only demo mode (ILLUMIN_TAP_MODE=1, in the shell or .env):
    tap 1 reads the question, the student answers in silence (buzzer guidance and the
    out-of-box warning still run), tap 2 announces the next question, tap 1 reads it.
    No hover auto-read, no spoken direction cues, no "In the answer box", no audit.
"""
from __future__ import annotations
import os
import src.config  # noqa: F401  (loads .env so ILLUMIN_TAP_MODE can live there)
from src import speech
from typing import Dict, List, Optional
from src.contracts import (
    Action, Box, Haptic, HapticCmd, Inputs, MotionState,
    Phase, Question, SetBox, Silence, Snapshot, Speak,
    Tap, Timing, TIMING, WriteStatus
)

TAP_DEBOUNCE_S = 0.4      # ignore a gesture this soon after the previous accepted one
READ_TAP_IGNORE_S = 0.5   # ignore taps right after a question starts being read
READ_MAX_S = 30.0         # leave READING even if is_speaking() never goes False
PEN_LOST_HOLD_S = 1.0     # keep the last guide cue this long after the camera loses the pen
TAP_MODE_ENV = "ILLUMIN_TAP_MODE"

GUIDE_CMDS = (HapticCmd.GUIDE_LEFT, HapticCmd.GUIDE_RIGHT, HapticCmd.GUIDE_BOTH)


class StateMachine:
    MAX_AUDIT_RETRIES = 2

    def __init__(self, questions: Optional[List[Question]] = None,
                 layout: Optional[Dict[str, Box]] = None,
                 timing: Timing = TIMING, tap_mode: Optional[bool] = None) -> None:
        self.tm = timing
        self.layout = layout or {}
        self.tap_mode = os.getenv(TAP_MODE_ENV, "") == "1" if tap_mode is None else tap_mode
        self.reset(questions or [])

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
        self._last_gesture_t = -1e9
        self._last_guide: Optional[HapticCmd] = None
        self._pen_seen_t = -1e9
        self._locked = False
        self._heard_speaking = False
        self._audit_retries = 0
        self._wrote = False
        self.answered = 0
        self.skipped = 0

    @property
    def question(self) -> Optional[Question]:
        return self.questions[self.index] if self.index < len(self.questions) else None

    def _enter(self, phase: Phase, now: float) -> None:
        self.phase, self._phase_t = phase, now
        self._heard_speaking = False
        self._nonwriting_since = None
        self._locked = False

    def _still_for(self, now: float) -> float:
        return 0.0 if self._still_since is None else now - self._still_since

    def _haptic(self, cmd: HapticCmd, now: float, acts: List[Action]) -> None:
        refresh = (
            cmd in (*GUIDE_CMDS, HapticCmd.WARN)
            and (now - self._last_cmd_t >= self.tm.HAPTIC_REFRESH_S)
        )
        if cmd != self._last_cmd or refresh:
            acts.append(Haptic(cmd))
            self._last_cmd, self._last_cmd_t = cmd, now

    def _gestures(self, inp: Inputs) -> List[Tap]:
        taps = [e.tap for e in inp.imu_events if e.tap != Tap.NONE]   # taps are the only commands (no voice input)
        if not taps or inp.t - self._last_gesture_t < TAP_DEBOUNCE_S:
            return []
        self._last_gesture_t = inp.t
        return [taps[-1]]

    def _read_question(self, q: Question, now: float, acts: List[Action]) -> None:
        acts.append(Speak(speech.question_intro(self.index + 1, q.text), interrupt=True))
        self._enter(Phase.READING, now)

    def _skip(self, now: float, acts: List[Action]) -> None:
        # Tap 2 (double-tap / 't' / 'n') always counts the question as answered, regardless of
        # whether writing was actually detected - the operator's deliberate "next question" gesture
        # is taken as confirmation, not a skip.
        self.answered += 1
        if self.index + 1 < len(self.questions):
            acts.append(Speak(speech.moving_to_question(self.index + 2), interrupt=True))
        self._haptic(HapticCmd.OFF, now, acts)
        self._advance(now, acts)

    def _advance(self, now: float, acts: List[Action]) -> None:
        self.index += 1
        self._audit_retries = 0
        self._wrote = False
        if self.question is None:
            acts.append(Speak(speech.test_complete(self.answered, self.skipped)))
            self.phase = Phase.COMPLETE
        else:
            self._enter(Phase.IDLE, now)

    def update(self, inp: Inputs) -> List[Action]:
        now, acts = inp.t, []
        if self._phase_t is None:
            self._phase_t = now

        if inp.motion == MotionState.STILL:
            if self._still_since is None:
                self._still_since = now
        else:
            self._still_since = None

        q = self.question
        if q is None or self.phase == Phase.COMPLETE:
            return acts

        gestures = self._gestures(inp)
        getattr(self, "_on_" + self.phase.value.lower())(inp, q, gestures, acts)
        return acts

    def _on_idle(self, inp: Inputs, q: Question, g: List[Tap], acts: List[Action]) -> None:
        now = inp.t
        if Tap.DOUBLE in g:
            return self._skip(now, acts)

        hovered = (not self.tap_mode
                   and self._still_for(now) >= self.tm.HOVER_S
                   and now - self._phase_t >= self.tm.HOVER_S
                   and not inp.speaking)
        if Tap.SINGLE in g or hovered:
            if q.box_id in self.layout:
                acts.append(SetBox(self.layout[q.box_id]))
            self._read_question(q, now, acts)

    def _on_reading(self, inp: Inputs, q: Question, g: List[Tap], acts: List[Action]) -> None:
        now = inp.t
        if now - self._phase_t < READ_TAP_IGNORE_S:
            g = []
        if Tap.DOUBLE in g:
            return self._skip(now, acts)
        if Tap.SINGLE in g:
            return self._read_question(q, now, acts)

        if inp.speaking:
            self._heard_speaking = True
            if now - self._phase_t > READ_MAX_S:
                self._enter(Phase.NAVIGATING, now)
        elif self._heard_speaking or (now - self._phase_t > self.tm.READ_GRACE_S):
            self._enter(Phase.NAVIGATING, now)

    def _guide_cmd(self, now: float, gd) -> HapticCmd:
        if gd and gd.pen_visible:
            self._pen_seen_t = now
            self._last_guide = gd.cmd if (gd.cmd in GUIDE_CMDS and not gd.in_box) else None
        elif now - self._pen_seen_t > PEN_LOST_HOLD_S:
            self._last_guide = None
        return self._last_guide or HapticCmd.OFF

    def _on_navigating(self, inp: Inputs, q: Question, g: List[Tap], acts: List[Action]) -> None:
        now, gd = inp.t, inp.guidance
        if Tap.DOUBLE in g:
            return self._skip(now, acts)
        if Tap.SINGLE in g:
            acts.append(Speak(speech.question_intro(self.index + 1, q.text), interrupt=True))

        if inp.motion == MotionState.WRITING:
            self._haptic(HapticCmd.OFF, now, acts)
            acts.append(Silence())
            self._wrote = True
            self._enter(Phase.WRITING, now)
            return

        cmd = self._guide_cmd(now, gd)
        if inp.motion == MotionState.MOVING:
            self._locked = False
            self._haptic(cmd, now, acts)

            if (not self.tap_mode and gd and gd.pen_visible and not gd.in_box and gd.speech
                    and not inp.speaking
                    and (now - self._last_nav_speech >= self.tm.NAV_SPEECH_EVERY_S)):
                acts.append(Speak(gd.speech))
                self._last_nav_speech = now
        else:
            self._haptic(HapticCmd.OFF, now, acts)
            if (inp.motion == MotionState.STILL and gd and gd.in_box and not self._locked
                    and self._still_for(now) >= self.tm.LOCK_STILL_S):
                self._locked = True
                acts.append(Haptic(HapticCmd.LOCK))
                if not self.tap_mode:
                    acts.append(Speak(speech.IN_ANSWER_BOX))
                self._last_cmd = HapticCmd.OFF

    def _on_writing(self, inp: Inputs, q: Question, g: List[Tap], acts: List[Action]) -> None:
        now = inp.t
        if inp.motion == MotionState.WRITING:
            self._nonwriting_since = None
        elif self._nonwriting_since is None:
            self._nonwriting_since = now

        if self.tap_mode:
            return self._on_writing_tap_mode(inp, g, acts)

        # The answer ends when the writing stops for ANSWER_IDLE_S, or early with tap 2 ("next question").
        done = (
            Tap.DOUBLE in g
            or (self._nonwriting_since is not None and (now - self._nonwriting_since >= self.tm.ANSWER_IDLE_S))
        )

        if done:
            self._haptic(HapticCmd.OFF, now, acts)
            acts.append(Snapshot(q.id))
            self._enter(Phase.AUDITING, now)
            return

        self._margin_haptic(inp, acts)

    def _on_writing_tap_mode(self, inp: Inputs, g: List[Tap], acts: List[Action]) -> None:
        now = inp.t
        if Tap.DOUBLE in g:
            return self._skip(now, acts)
        if self._nonwriting_since is not None and now - self._nonwriting_since >= self.tm.ANSWER_IDLE_S:
            # Paused writing: back to NAVIGATING so tap 1 can repeat the question. Already in
            # the box, so don't fire LOCK again.
            self._haptic(HapticCmd.OFF, now, acts)
            self._enter(Phase.NAVIGATING, now)
            self._locked = True
            return
        self._margin_haptic(inp, acts)

    def _margin_haptic(self, inp: Inputs, acts: List[Action]) -> None:
        status = inp.guidance.write_status if inp.guidance else WriteStatus.UNKNOWN
        if inp.motion == MotionState.WRITING and status == WriteStatus.OUTSIDE:
            self._haptic(HapticCmd.WARN, inp.t, acts)
        else:
            self._haptic(HapticCmd.OFF, inp.t, acts)

    def _on_auditing(self, inp: Inputs, q: Question, g: List[Tap], acts: List[Action]) -> None:
        now, r = inp.t, inp.audit_result
        if r is not None and r.question_id == q.id:
            if not r.ink_present:
                self._audit_retries += 1
                if self._audit_retries <= self.MAX_AUDIT_RETRIES:
                    acts.append(Speak(speech.NO_INK_FOUND))
                    return self._enter(Phase.NAVIGATING, now)
            msg = speech.ANSWER_RECORDED_OUTSIDE if r.ink_outside else speech.ANSWER_RECORDED
        elif now - self._phase_t > self.tm.AUDIT_TIMEOUT_S:
            msg = speech.AUDIT_TIMEOUT
        else:
            return

        self.answered += 1
        acts += [Speak(msg), Haptic(HapticCmd.COMPLETE)]
        self._last_cmd = HapticCmd.OFF
        self._advance(now, acts)
