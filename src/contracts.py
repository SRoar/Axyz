"""
TactileReader / Illumin -- THE CONTRACT.  (CONTRACT_VERSION 1.0)

Every module talks to every other module ONLY through the types in this file.
Nobody edits this file without telling Dev 4 (the contract owner) first.
If you change it, bump CONTRACT_VERSION and announce it in the team chat.

Division of truth (this is the whole design):
    CAMERA  -> WHERE the pen is on the page                (PenState)
    IMU     -> WHAT the pen is doing (still/moving/writing/lifted) + tap commands
    BRAIN   -> decides everything from those two + voice   (StateMachine)

Coordinate space: PAGE-NORMALIZED.  (0,0) = top-left of the paper, (1,1) = bottom-right.
x grows right, y grows DOWN (OpenCV convention).  The tracker converts camera pixels
to page space (via 4-corner homography) BEFORE anything leaves the tracker.
Real-world size of a page unit: PAGE_W_CM x PAGE_H_CM.

Units: seconds (float, from Clock.now()), centimetres, g for accelerometer.

Serial protocol (115200 baud, newline-terminated ASCII, Arduino 101):
    Arduino -> PC
        READY                         sent once at boot
        S,<ms>,<ax>,<ay>,<az>,<light> raw sample, ~50 Hz  (ax..az in g, light = analogRead(A0))
        M,<STILL|MOVING|WRITING|LIFTED>   sent ONLY when the motion state changes
        T,<1|2|3>                     tap gesture (1=single, 2=double, 3=triple)
    PC -> Arduino
        H,<LOCK|WARN|COMPLETE|GUIDE_LEFT|GUIDE_RIGHT|GUIDE_BOTH|OFF>
        PING                          (Arduino answers READY)
    Haptic rules (firmware): LOCK and COMPLETE are one-shot patterns.  GUIDE_* and WARN
    repeat until replaced, and AUTO-EXPIRE after 1500 ms without a refresh (dead-man
    switch: if the PC crashes the buzzers go quiet).  OFF stops everything.
    D3 = LEFT buzzer, D4 = RIGHT buzzer.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

CONTRACT_VERSION = "1.0"

DEVICE_BAUD = 115200
PAGE_W_CM = 21.59   # US Letter
PAGE_H_CM = 27.94


# --------------------------------------------------------------------------- enums
class MotionState(str, Enum):
    STILL = "STILL"        # pen resting / hovering motionless
    MOVING = "MOVING"      # smooth, large motion (travelling across the page)
    WRITING = "WRITING"    # sustained high-frequency jitter (pen is on paper, writing)
    LIFTED = "LIFTED"      # picked up / put down (gravity vector changed, z spike)


class Tap(int, Enum):
    NONE = 0
    SINGLE = 1   # "repeat the question"
    DOUBLE = 2   # "skip this question"
    TRIPLE = 3   # "I'm done, audit my answer"


class HapticCmd(str, Enum):
    LOCK = "LOCK"                # one-shot: short dual-frequency pulse, pen entered the box
    WARN = "WARN"                # continuous harsh tone: writing outside the box
    COMPLETE = "COMPLETE"        # one-shot: ascending tones, question finished
    GUIDE_LEFT = "GUIDE_LEFT"    # pulse left buzzer: target is to the left
    GUIDE_RIGHT = "GUIDE_RIGHT"  # pulse right buzzer: target is to the right
    GUIDE_BOTH = "GUIDE_BOTH"    # both pulse slowly: aligned horizontally, move vertically
    OFF = "OFF"


class Phase(str, Enum):
    IDLE = "IDLE"              # waiting for hover-pause / tap to read the next question
    READING = "READING"        # voice is reading the question
    NAVIGATING = "NAVIGATING"  # guiding the pen to the answer box
    WRITING = "WRITING"        # voice silent, margin-lock armed
    AUDITING = "AUDITING"      # waiting for the ink-in-box check
    COMPLETE = "COMPLETE"      # all questions done


class WriteStatus(str, Enum):
    INSIDE = "INSIDE"
    OUTSIDE = "OUTSIDE"
    UNKNOWN = "UNKNOWN"        # pen not visible / no active box


# --------------------------------------------------------------------------- timing knobs
@dataclass(frozen=True)
class Timing:
    HOVER_S: float = 0.8            # STILL this long (in IDLE) -> read the question
    LOCK_STILL_S: float = 0.3       # STILL inside the box this long -> LOCK cue
    ANSWER_IDLE_S: float = 2.0      # not WRITING this long (in WRITING phase) -> answer done
    NAV_SPEECH_EVERY_S: float = 3.5 # spoken direction cue interval while MOVING
    HAPTIC_REFRESH_S: float = 0.5   # re-send GUIDE_*/WARN this often (firmware dead-man = 1.5 s)
    AUDIT_TIMEOUT_S: float = 8.0
    READ_GRACE_S: float = 1.5       # how long to wait for speech to start before moving on


TIMING = Timing()


# --------------------------------------------------------------------------- clocks
class Clock(Protocol):
    def now(self) -> float: ...


class RealClock:
    def now(self) -> float:
        return time.monotonic()


class SimClock:
    """Manually advanced clock for deterministic headless simulation."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# --------------------------------------------------------------------------- data types
@dataclass(frozen=True)
class Box:
    """An answer box in PAGE-NORMALIZED coordinates."""
    id: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.xmin - margin <= x <= self.xmax + margin
                and self.ymin - margin <= y <= self.ymax + margin)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Box":
        return Box(d["id"], float(d["xmin"]), float(d["ymin"]), float(d["xmax"]), float(d["ymax"]))


@dataclass
class PenState:
    """Pen-tip position in PAGE-NORMALIZED coordinates. Tracker returns None when pen is not seen."""
    t: float
    x: float
    y: float
    lift_cm: float = 0.0       # height above the page (0 if unknown)
    confidence: float = 1.0    # 0..1 (lower while hand-occluded / extrapolated)


@dataclass
class ImuSample:
    t: float
    ax: float
    ay: float
    az: float
    light: int = 0


@dataclass
class ImuEvent:
    """Either a motion-state change (motion set) or a tap (tap set). Never both."""
    t: float
    motion: Optional[MotionState] = None
    tap: Tap = Tap.NONE


@dataclass
class Question:
    id: str
    text: str       # exactly what the voice reads (after "Question N.")
    box_id: str     # key into the layout (answer boxes)


@dataclass
class TrackerReading:
    t: float
    pen: Optional[PenState]
    frame: Any = None        # BGR numpy image for the HUD (page-rectified or raw); may be None


@dataclass
class Guidance:
    """Pure geometry: pen vs active box. Produced by GuidanceEngine, consumed by the brain."""
    cmd: Optional[HapticCmd] = None     # GUIDE_LEFT/RIGHT/BOTH, or None when already in the box
    speech: Optional[str] = None        # e.g. "Move 2 inches Down."
    dx_cm: float = 0.0                  # +right / -left to reach the box
    dy_cm: float = 0.0                  # +down / -up to reach the box
    dist_cm: float = 0.0
    in_box: bool = False
    write_status: WriteStatus = WriteStatus.UNKNOWN
    pen_visible: bool = False


@dataclass
class AuditResult:
    question_id: str
    ink_present: bool              # is there handwriting inside the target box?
    ink_outside: bool = False      # is there handwriting just outside the box?
    confidence: float = 1.0
    note: str = ""


@dataclass
class Inputs:
    """Everything the brain sees on one tick."""
    t: float
    pen: Optional[PenState]
    guidance: Optional[Guidance]
    motion: MotionState                      # current IMU motion state (level, not edge)
    imu_events: Tuple[ImuEvent, ...] = ()    # edges since last tick (motion changes + taps)
    speaking: bool = False                   # voice.is_speaking()
    voice_cmd: Optional[str] = None          # lowercased phrase from the mic, if any
    audit_result: Optional[AuditResult] = None


# ---- actions the brain returns; the System executes them -----------------------
@dataclass
class Speak:
    text: str
    interrupt: bool = False


@dataclass
class Silence:
    """Stop speaking NOW and clear the speech queue (used when writing starts)."""


@dataclass
class Haptic:
    cmd: HapticCmd


@dataclass
class SetBox:
    box: Box


@dataclass
class Snapshot:
    """Take a clean page image now and submit it to the auditor for this question."""
    question_id: str


Action = Union[Speak, Silence, Haptic, SetBox, Snapshot]


# --------------------------------------------------------------------------- interfaces
# Real classes live at the import path shown and take the constructor args shown.
@runtime_checkable
class ImuLink(Protocol):
    """src.arduino_link.ArduinoImuLink(clock)   -- owner: Dev 1"""
    motion: MotionState                              # current state (property is fine)
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def poll(self) -> List[ImuEvent]: ...            # events since last call, oldest first
    def send(self, cmd: HapticCmd) -> None: ...      # non-blocking
    def recent_samples(self, n: int = 200) -> List[ImuSample]: ...   # for HUD waveform


@runtime_checkable
class Tracker(Protocol):
    """src.tracker.PenTracker(clock)   -- owner: Dev 2"""
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self) -> Optional[TrackerReading]: ...  # non-blocking, latest frame; None if no frame yet
    def snapshot(self) -> Any: ...                   # clean page image (numpy BGR) for the auditor


@runtime_checkable
class GuidanceEngine(Protocol):
    """src.guidance.GuidanceEngine()   -- owner: Dev 2   (pure function of pen + box)"""
    def compute(self, pen: Optional[PenState], box: Optional[Box]) -> Guidance: ...


@runtime_checkable
class Voice(Protocol):
    """src.voice.VoiceEngine(clock)   -- owner: Dev 3
    is_speaking() must be True from the moment speak() is called until the audio has
    FULLY finished, including anything still queued (no gaps between queued phrases)."""
    def speak(self, text: str, interrupt: bool = False) -> None: ...
    def silence(self) -> None: ...
    def is_speaking(self) -> bool: ...
    def poll_command(self) -> Optional[str]: ...     # lowercased phrase or None
    def stop(self) -> None: ...


@runtime_checkable
class Auditor(Protocol):
    """src.audit.GeminiAuditor(clock)   -- owner: Dev 4   (NON-blocking)"""
    def submit(self, snapshot: Any, box: Box, question: Question) -> None: ...
    def poll(self) -> Optional[AuditResult]: ...     # returns a finished result once, else None


@runtime_checkable
class StateMachine(Protocol):
    """src.state_machine.StateMachine(questions, layout)   -- owner: Dev 3"""
    phase: Phase
    index: int
    @property
    def question(self) -> Optional[Question]: ...
    def reset(self, questions: List[Question]) -> None: ...
    def update(self, inp: Inputs) -> List[Action]: ...


# --------------------------------------------------------------------------- serial helpers
def format_haptic_line(cmd: HapticCmd) -> str:
    return f"H,{cmd.value}\n"


def parse_device_line(line: str, t: float) -> Optional[Union[ImuSample, ImuEvent]]:
    """Parse one Arduino->PC line. Returns None for READY / blanks / garbage."""
    line = line.strip()
    if not line:
        return None
    p = line.split(",")
    try:
        if p[0] == "S" and len(p) >= 5:
            light = int(float(p[5])) if len(p) > 5 else 0
            return ImuSample(t, float(p[2]), float(p[3]), float(p[4]), light)
        if p[0] == "M" and len(p) >= 2:
            return ImuEvent(t, motion=MotionState(p[1].strip()))
        if p[0] == "T" and len(p) >= 2:
            return ImuEvent(t, tap=Tap(int(p[1])))
    except (ValueError, KeyError):
        return None
    return None


# --------------------------------------------------------------------------- data files
def load_questions(path: str) -> List[Question]:
    with open(path) as f:
        return [Question(**q) for q in json.load(f)]


def save_questions(path: str, questions: List[Question]) -> None:
    with open(path, "w") as f:
        json.dump([asdict(q) for q in questions], f, indent=2)


def load_layout(path: str) -> Dict[str, Box]:
    with open(path) as f:
        return {b["id"]: Box.from_dict(b) for b in json.load(f)}


def save_layout(path: str, boxes: Dict[str, Box]) -> None:
    with open(path, "w") as f:
        json.dump([b.to_dict() for b in boxes.values()], f, indent=2)


def sample_questions() -> List[Question]:
    return [
        Question("q1", "Explain the first law of thermodynamics.", "box1"),
        Question("q2", "Describe how the second law of thermodynamics relates to entropy.", "box2"),
    ]


def sample_layout() -> Dict[str, Box]:
    return {
        "box1": Box("box1", 0.10, 0.25, 0.90, 0.45),
        "box2": Box("box2", 0.10, 0.55, 0.90, 0.80),
    }
