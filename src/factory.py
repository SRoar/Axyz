"""
Builds a Components bundle where each piece is either the REAL class or the FAKE one.
Real class locations + constructor signatures are part of the contract:

    imu       src.arduino_link.ArduinoImuLink(clock)        Dev 1
    tracker   src.tracker.PenTracker(clock)                 Dev 2
    guidance  src.guidance.GuidanceEngine()                 Dev 2
    voice     src.voice.VoiceEngine(clock)                  Dev 3
    brain     src.state_machine.StateMachine(questions, layout)   Dev 3
    auditor   src.audit.GeminiAuditor(clock)                Dev 4
"""
from __future__ import annotations

import importlib
from typing import Dict, Iterable, List, Set

from src.contracts import Box, Clock, Question
from src.fakes import (
    FakeAuditor, FakeGuidanceEngine, FakeImuLink, FakeTracker, FakeVoice, FakeWorld,
    ReferenceStateMachine,
)
from src.system import Components

REAL = {
    "imu": ("src.arduino_link", "ArduinoImuLink"),
    "tracker": ("src.tracker", "PenTracker"),
    "guidance": ("src.guidance", "GuidanceEngine"),
    "voice": ("src.voice", "VoiceEngine"),
    "brain": ("src.state_machine", "StateMachine"),
    "auditor": ("src.audit", "GeminiAuditor"),
}
# These need wall-clock time (hardware / network / audio). Everything else can run on SimClock.
REALTIME_PARTS = {"imu", "tracker", "voice", "auditor"}


def needs_realtime(real: Iterable[str]) -> bool:
    return bool(set(real) & REALTIME_PARTS)


def _load(name: str):
    mod, cls = REAL[name]
    return getattr(importlib.import_module(mod), cls)


def build(real: Iterable[str], clock: Clock, questions: List[Question],
          layout: Dict[str, Box], render: bool = True) -> Components:
    real: Set[str] = set(real)
    unknown = real - set(REAL)
    if unknown:
        raise SystemExit(f"unknown component(s): {sorted(unknown)}; choose from {sorted(REAL)}")
    world = FakeWorld(clock)
    return Components(
        imu=_load("imu")(clock) if "imu" in real else FakeImuLink(world),
        tracker=_load("tracker")(clock) if "tracker" in real else FakeTracker(world, layout, render),
        voice=_load("voice")(clock) if "voice" in real else FakeVoice(clock),
        guidance=_load("guidance")() if "guidance" in real else FakeGuidanceEngine(),
        auditor=_load("auditor")(clock) if "auditor" in real else FakeAuditor(clock),
        brain=_load("brain")(questions, layout) if "brain" in real else ReferenceStateMachine(questions, layout),
    )
