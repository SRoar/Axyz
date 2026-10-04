"""Hardware bring-up rehearsal without the hardware.

The REAL PenTracker (fed synthetic camera frames instead of the iPhone), the REAL ArduinoImuLink (fed
contract serial lines from a fake port instead of the UNO Q), the real brain + guidance, through
System + HUD, exactly as `python -m src.main --real imu,tracker,guidance,brain` wires them.
Checks what the demo UI must show: the page search, the pen tip on the page (and off it), and every
single / double tap the pen reports.

    ILLUMIN_HUD_SHOTS=captures/bringup pytest tests/test_bringup.py   # also save the HUD frames
"""
import os
import time

import cv2
import numpy as np
import pytest

from src.contracts import MotionState, Phase, SimClock, Tap, sample_questions
from src.fakes import FakeAuditor, FakeVoice
from src.guidance import GuidanceEngine
from src.hud import Hud, off_page_side
from src.main import DebugImu
from src.page_calibration import PageCalibration
from src.state_machine import StateMachine
from src.system import Components, System
from src.tracker import PenTracker
from test_link import FakePort, make_link, wait_for
from test_tracker import CORNERS, H, W, ScriptedCamera, hand_with_pen, page_image

def _layout():
    from src.contracts import Box
    return {
        "box1": Box("box1", 0.30, 0.50, 0.70, 0.65),
        "box2": Box("box2", 0.30, 0.70, 0.70, 0.85),
        "box3": Box("box3", 0.30, 0.15, 0.70, 0.30),
    }


class Rig:
    def __init__(self, tmp_path):
        self.clock = SimClock(0.0)
        self.cam = ScriptedCamera()
        self.tracker = PenTracker(self.clock, calib_path=str(tmp_path / "calib.json"),
                                  marker_path=str(tmp_path / "none.json"),
                                  profile_path=str(tmp_path / "none.json"), camera=self.cam)
        self.port = FakePort()
        self.link = make_link([self.port])
        self.layout = _layout()
        qs = sample_questions()
        self.system = System(Components(
            imu=DebugImu(self.link, self.clock), tracker=self.tracker, voice=FakeVoice(self.clock),
            guidance=GuidanceEngine(), auditor=FakeAuditor(self.clock),
            brain=StateMachine(qs, self.layout, tap_mode=False)), self.clock, qs, self.layout)
        self.hud = Hud(self.system, self.system.c.imu)
        self.calib = PageCalibration(CORNERS, (W, H))
        self.shots = os.environ.get("ILLUMIN_HUD_SHOTS")
        self._ms = 0

    def frame(self, img, n=1, serial=()):
        """Push `n` camera frames (30 fps); the pen streams S lines meanwhile; one System tick per frame."""
        for _ in range(n):
            self.clock.advance(1 / 30)
            self._ms += 33
            self.port.feed(f"S,{self._ms},0.01,-0.02,0.99", *serial)
            serial = ()
            fid = self.cam.push(img, self.clock.now())
            assert wait_for(lambda: self.tracker._frame is not None and self.tracker._frame.id == fid, 10.0)
            self.system.step()
            self.hud.observe()

    def pen_frame(self, uv):
        tip = np.float32(self.calib.to_pixels(*uv))
        back = tip + (240, 120)
        palm = (int(tip[0] + 160), int(tip[1] + 170))
        return hand_with_pen(tuple(map(int, tip)), tuple(map(int, back)), palm, entry_px=(palm[0], H + 40))

    def until(self, cond, img, max_frames=60, serial=()):
        self.frame(img, serial=serial)
        for _ in range(max_frames):
            if cond():
                return True
            self.frame(img)
        return cond()

    def shot(self, name):
        img = self.hud.render()
        if self.shots:
            os.makedirs(self.shots, exist_ok=True)
            cv2.imwrite(os.path.join(self.shots, f"{name}.png"), img)
        return img


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    r.system.start()
    assert r.link.wait_connected(3.0)
    yield r
    r.system.stop()


def test_bringup_page_pen_and_taps_reach_the_hud(rig):
    # 1. camera on, page not found yet: raw view + "looking for the page", no page coordinates
    rig.frame(page_image(), 3)
    assert rig.tracker.calibration is None
    assert rig.hud._link_status()[0].startswith("pen connected") or "connected" in rig.hud._link_status()[0]
    rig.shot("1_looking_for_page")

    # 2. still, empty page -> locks by itself (and learns the empty scene)
    assert rig.until(lambda: rig.tracker.calibration is not None, page_image(), 120)
    rig.frame(page_image(), 30)                              # background settles
    rig.shot("2_page_locked")

    # 3. pen in the hand: the tip shows up in page coordinates
    target = (0.42, 0.40)
    assert rig.until(lambda: rig.system.inputs.pen is not None, rig.pen_frame(target), 30)
    rig.frame(rig.pen_frame(target), 10)
    pen = rig.system.inputs.pen
    assert abs(pen.x - target[0]) < 0.03 and abs(pen.y - target[1]) < 0.03, pen
    rig.shot("3_pen_on_page")

    # 4. single tap from the pen -> HUD shows it (and the brain reads question 1)
    assert rig.until(lambda: rig.hud.tap_counts[1] == 1, rig.pen_frame(target), 30, serial=["T,1"])
    assert rig.hud.last_tap[1] == 1 and not rig.hud.last_tap[2]
    assert rig.system.c.brain.phase == Phase.READING
    rig.shot("4_single_tap")

    # 5. double tap -> counted separately, brain moves to the next question
    rig.frame(rig.pen_frame(target), 20)
    assert rig.until(lambda: rig.hud.tap_counts[2] == 1, rig.pen_frame(target), 30, serial=["T,2"])
    assert rig.hud.last_tap[1] == 2
    assert rig.system.c.brain.index == 1
    rig.frame(rig.pen_frame(target), 60)                     # 2 s later the tap is still on screen
    assert rig.hud.last_tap[1] == 2
    rig.shot("5_double_tap_2s_later")

    # 6. a keyboard tap (--debug-keys) is labelled, not counted as the pen's
    rig.system.c.imu.handle_key(ord("r"))
    rig.frame(rig.pen_frame(target), 2)
    assert rig.hud.key_taps == 1 and rig.hud.tap_counts == {1: 1, 2: 1} and rig.hud.last_tap[2]

    # 7. pen tip off the left edge of the paper
    off = (-0.12, 0.45)
    assert rig.until(lambda: rig.system.inputs.pen is not None and rig.system.inputs.pen.x < 0,
                     rig.pen_frame(off), 40)
    assert off_page_side(rig.system.inputs.pen.x, rig.system.inputs.pen.y) == "LEFT"
    rig.shot("7_off_page_left")

    # 8. motion state from the pen reaches the HUD badge
    assert rig.until(lambda: rig.system.inputs.motion == MotionState.WRITING, rig.pen_frame(target), 30,
                     serial=["M,WRITING"])

    # 9. USB unplugged: the HUD says so instead of silently showing nothing
    rig.port.close()
    assert wait_for(lambda: not rig.link.connected, 3.0)
    rig.frame(rig.pen_frame(target), 2)
    text, _ = rig.hud._link_status()
    assert text.startswith("PEN NOT CONNECTED"), text
    rig.shot("9_pen_unplugged")


def test_camera_that_is_not_ready_at_startup_is_retried_and_shown(tmp_path, monkeypatch):
    import src.tracker as tracker_mod
    monkeypatch.setattr(tracker_mod, "CAMERA_RETRY_S", 0.05)

    class LateCamera(ScriptedCamera):
        """Like an iPhone whose Record3D app starts streaming a little after the system starts."""
        fails = 3

        def start(self):
            if LateCamera.fails > 0:
                LateCamera.fails -= 1
                raise RuntimeError("no Record3D device: plug in the iPhone")
            super().start()

    from src.fakes import FakeImuLink, FakeWorld
    clock = SimClock(0.0)
    cam = LateCamera()
    tr = PenTracker(clock, calib_path=str(tmp_path / "c.json"), marker_path=str(tmp_path / "m.json"), camera=cam)
    qs, layout = sample_questions(), _layout()
    s = System(Components(imu=FakeImuLink(FakeWorld(clock)), tracker=tr, voice=FakeVoice(clock),
                          guidance=GuidanceEngine(), auditor=FakeAuditor(clock),
                          brain=StateMachine(qs, layout, tap_mode=False)), clock, qs, layout)
    s.start()                                                # must not raise, must not count as a crash
    try:
        assert "Record3D" in tr.camera_error and not s.errors
        hud = Hud(s)
        s.step()
        hud.observe()
        hud.render()                                         # page-search view with the reason
        assert wait_for(lambda: tr.camera_error == "", 3.0)  # retried until the camera came up
        clock.advance(1 / 30)
        fid = cam.push(page_image(), clock.now())
        assert wait_for(lambda: tr._frame is not None and tr._frame.id == fid, 5.0)
        s.step()
        assert s.reading is not None and s.reading.frame is not None
    finally:
        s.stop()
