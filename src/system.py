"""
The glue loop. ONE tick = read sensors -> build Inputs -> brain.update -> execute Actions.
Owner: Dev 4 (extend it; keep step() as the single place where components meet).
The headless sim and the real main.py both call System.step().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.contracts import (
    Auditor, Box, Clock, GuidanceEngine, Haptic, ImuLink, Inputs, Phase, Question, SetBox,
    Silence, Snapshot, Speak, StateMachine, Tracker, TrackerReading, Voice,
)


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
        self.layout = layout
        self.current_box: Optional[Box] = None
        self.log: List[Tuple[float, str, str]] = []     # (t, kind, text) for HUD + sim asserts
        self.inputs: Optional[Inputs] = None            # last tick, for the HUD
        self.reading: Optional[TrackerReading] = None
        self._last_phase: Optional[Phase] = None

    def start(self) -> None:
        self.c.imu.start()
        self.c.tracker.start()

    def stop(self) -> None:
        self.c.imu.stop()
        self.c.tracker.stop()
        self.c.voice.stop()

    def _log(self, kind: str, text: str) -> None:
        self.log.append((self.clock.now(), kind, text))

    def step(self) -> Inputs:
        c, now = self.c, self.clock.now()
        self.reading = c.tracker.read()
        pen = self.reading.pen if self.reading else None
        events = c.imu.poll()
        inp = Inputs(
            t=now,
            pen=pen,
            guidance=c.guidance.compute(pen, self.current_box),
            motion=c.imu.motion,
            imu_events=tuple(events),
            speaking=c.voice.is_speaking(),
            voice_cmd=c.voice.poll_command(),
            audit_result=c.auditor.poll(),
        )
        for a in c.brain.update(inp):
            self._execute(a)
        if c.brain.phase != self._last_phase:
            self._last_phase = c.brain.phase
            self._log("PHASE", c.brain.phase.value)
        self.inputs = inp
        return inp

    def _execute(self, a) -> None:
        c = self.c
        if isinstance(a, Speak):
            c.voice.speak(a.text, a.interrupt)
            self._log("SPEAK", a.text)
        elif isinstance(a, Silence):
            c.voice.silence()
            self._log("SILENCE", "")
        elif isinstance(a, Haptic):
            c.imu.send(a.cmd)
            self._log("HAPTIC", a.cmd.value)
        elif isinstance(a, SetBox):
            self.current_box = a.box
            self._log("BOX", a.box.id)
        elif isinstance(a, Snapshot):
            q = self.questions[a.question_id]
            c.auditor.submit(c.tracker.snapshot(), self.layout[q.box_id], q)
            self._log("SNAPSHOT", q.id)
        else:
            raise TypeError(f"unknown action {a!r}")
