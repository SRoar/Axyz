"""Dev 4 glue tests: System resilience, debug-key injector, HUD smoke, main CLI."""
import numpy as np
import pytest

from src.contracts import (
    Haptic, HapticCmd, ImuLink, MotionState, Phase, SimClock, Snapshot, Speak, Tap,
    sample_layout, sample_questions,
)
from src.factory import build, parse_real
from src.fakes import FakeImuLink, FakeWorld
from src.hud import H, W, Hud, banner_for, wrap
from src.contracts import WriteStatus
from src.main import DebugImu, main, make_system, parse_args
from src.system import System

Q, L = sample_questions(), sample_layout()


def make(real=(), render=True):
    clock = SimClock()
    return System(build(list(real), clock, Q, L, render=render), clock, Q, L), clock


def run(system, clock, seconds, hud=None):
    for _ in range(int(seconds / 0.02)):
        system.step()
        if hud:
            hud.observe()
        clock.advance(0.02)


# ------------------------------------------------------------------ System resilience
def test_a_crashing_component_does_not_kill_the_loop(capsys):
    s, clock = make()
    s.start()

    def boom(*a, **k):
        raise RuntimeError("camera unplugged")
    s.c.tracker.read = boom
    run(s, clock, 2.0)                                   # 100 ticks, must not raise
    assert s.errors["tracker"] == 100
    assert "camera unplugged" in s.last_error
    assert sum(1 for _, k, _ in s.log if k == "ERROR") <= 2     # log is throttled, not spammed


def test_action_failures_are_isolated():
    s, clock = make()
    s.c.voice.speak = lambda *a, **k: (_ for _ in ()).throw(OSError("no audio device"))
    s._execute(Speak("hello"))
    s._execute(Haptic(HapticCmd.LOCK))                   # next action still runs
    assert s.errors == {"voice": 1}
    assert ("HAPTIC", "LOCK") in [(k, v) for _, k, v in s.log]


def test_snapshot_for_unknown_question_is_contained():
    s, _ = make()
    s._execute(Snapshot("nope"))
    assert "auditor" in s.errors


def test_sim_check_flags_component_errors():
    from src.sim import check
    errs = check([(0.0, "ERROR", "tracker.read: boom")], Phase.COMPLETE)
    assert any("component error" in e for e in errs)


def test_full_fake_flow_has_no_errors_and_records_audits():
    s, clock = make(render=False)
    s.start()
    while s.c.brain.phase != Phase.COMPLETE and clock.now() < 90:
        s.step()
        clock.advance(0.02)
    assert s.errors == {} and s.last_audit is not None


# ------------------------------------------------------------------ debug keys
def test_debug_imu_overrides_motion_and_injects_taps():
    clock = SimClock()
    inner = FakeImuLink(FakeWorld(clock))
    d = DebugImu(inner, clock)
    assert isinstance(d, ImuLink)
    assert d.motion == MotionState.STILL
    assert d.handle_key(ord("3")) and d.motion == MotionState.WRITING
    ev = d.poll()
    assert [e.motion for e in ev] == [MotionState.WRITING]
    assert d.poll() == []                                # edges delivered once
    assert d.handle_key(ord("t"))
    assert [e.tap for e in d.poll()] == [Tap.DOUBLE]
    assert d.handle_key(ord("r")) and d.handle_key(ord("n"))
    assert [e.tap for e in d.poll()] == [Tap.SINGLE, Tap.DOUBLE]
    assert not d.handle_key(ord("x")) and not d.handle_key(255)
    d.release()
    assert d.motion == inner.motion


def test_debug_override_drops_contradicting_hardware_motion_but_keeps_taps():
    clock = SimClock()
    world = FakeWorld(clock)
    d = DebugImu(FakeImuLink(world), clock)
    d.handle_key(ord("1"))                               # force STILL
    d.poll()
    clock.advance(6.0)                                   # script is now MOVING (t=5..8)
    ev = d.poll()
    assert d.motion == MotionState.STILL and all(e.motion is None for e in ev)


def test_debug_keys_can_drive_the_whole_flow_with_a_dead_sensor():
    """Nobody touches the pen; a teammate types keys: still -> (read) -> writing -> still -> audit."""
    clock = SimClock()
    q, l = sample_questions(), sample_layout()
    comps = build([], clock, q, l, render=False)
    comps.imu = DebugImu(_DeadImu(), clock)
    s = System(comps, clock, q, l)
    s.start()
    script = {0.5: "1", 6.0: "3", 9.0: "1"}
    t = 0.0
    seen = set()
    while t < 14:
        for at, key in list(script.items()):
            if t >= at:
                comps.imu.handle_key(ord(key))
                script.pop(at)
        s.step()
        seen.add(s.c.brain.phase)
        clock.advance(0.02)
        t += 0.02
    assert {Phase.READING, Phase.WRITING, Phase.AUDITING} <= seen


class _DeadImu:
    motion = MotionState.STILL
    def start(self): ...
    def stop(self): ...
    def poll(self): return []
    def send(self, cmd): ...
    def recent_samples(self, n=200): return []


# ------------------------------------------------------------------ HUD
def test_hud_renders_every_phase_without_error_and_isnt_blank():
    s, clock = make()
    hud = Hud(s)
    s.start()
    phases = set()
    t = 0.0
    while t < 55:
        s.step()
        hud.observe()
        phases.add(s.c.brain.phase)
        if int(t * 50) % 25 == 0:                         # render ~2x per sim second
            img = hud.render()
            assert img.shape == (H, W, 3) and img.dtype == np.uint8
            assert img.std() > 10
        clock.advance(0.02)
        t += 0.02
    assert {Phase.READING, Phase.NAVIGATING, Phase.WRITING, Phase.AUDITING, Phase.COMPLETE} <= phases


def test_hud_tracks_taps_trail_voice_and_haptics():
    s, clock = make()
    hud = Hud(s)
    s.start()
    run(s, clock, 36.6, hud)                              # scripted tap 1 at t=36.3
    assert any(n == 1 for _, n in hud.taps)
    assert len(hud.haptics) > 3 and len(hud.voice) > 3
    assert len(hud.trail) == 0 or hud.trail[0][0] >= clock.now() - 2.1


def test_banner_names_match_the_plan():
    P, M = Phase, MotionState
    names = {banner_for(P.READING, M.STILL, WriteStatus.INSIDE, False)[0],
             banner_for(P.NAVIGATING, M.MOVING, WriteStatus.OUTSIDE, False)[0],
             banner_for(P.WRITING, M.WRITING, WriteStatus.INSIDE, True)[0],
             banner_for(P.WRITING, M.WRITING, WriteStatus.OUTSIDE, False)[0],
             banner_for(P.AUDITING, M.STILL, WriteStatus.INSIDE, True)[0]}
    assert names == {"READING", "NAVIGATING", "WRITING_LOCKED", "OUT_OF_BOUNDS", "AUDITING"}
    # margin alarm needs the IMU to say WRITING (design rule)
    assert banner_for(P.WRITING, M.STILL, WriteStatus.OUTSIDE, False)[0] == "WRITING_LOCKED"


def test_wrap_respects_width_and_line_cap():
    lines = wrap("one two three four five six seven eight nine ten " * 3, 300, 0.7, 1, max_lines=2)
    assert len(lines) == 2 and lines[-1].endswith("...")


# ------------------------------------------------------------------ main / CLI
def test_cli_fast_headless_run_completes_and_exits_zero(capsys):
    assert main(["--headless", "--fast", "--questions", "x.json", "--layout", "y.json"]) == 0
    assert "COMPLETE" in capsys.readouterr().out


def test_cli_records_a_video(tmp_path):
    out = tmp_path / "r.mp4"
    assert main(["--headless", "--fast", "--max-seconds", "3", "--record", str(out),
                 "--questions", "x.json", "--layout", "y.json"]) == 0
    assert out.exists() and out.stat().st_size > 1000


def test_cli_unknown_component_is_a_clear_error():
    with pytest.raises(SystemExit) as e:
        make_system(parse_args(["--real", "banana"]))
    assert "banana" in str(e.value)


def test_cli_real_component_not_merged_yet_names_the_owner(monkeypatch):
    import src.factory
    monkeypatch.setitem(src.factory.REAL, "brain", ("src.not_merged_yet", "StateMachine"))
    with pytest.raises(SystemExit) as e:
        make_system(parse_args(["--real", "brain", "--questions", "x", "--layout", "y"]))
    assert "Dev 3" in str(e.value)


def test_fast_with_real_hardware_is_rejected():
    with pytest.raises(SystemExit):
        make_system(parse_args(["--real", "imu", "--fast"]))


def test_parse_real_all():
    assert set(parse_real("all")) == {"imu", "tracker", "guidance", "voice", "brain", "auditor"}
    assert parse_real("imu, brain") == ["imu", "brain"] and parse_real("") == []


# ------------------------------------------------------------------ real auditor inside the real glue
def test_real_auditor_closes_the_loop_through_system():
    """FakeTracker.snapshot() returns a photo WITH handwriting in both boxes; the real GeminiAuditor
    (offline -> pixel path) must deliver results the brain accepts: every question ends 'recorded'."""
    import time
    from demo.make_audit_samples import H as IH, W as IW, _scribble, blank_sheet
    from src.audit import GeminiAuditor

    layout = sample_layout()
    rng = np.random.default_rng(1)
    photo = blank_sheet(layout, rng)
    for b in layout.values():
        for k in range(3):
            _scribble(photo, int(b.xmin * IW) + 30, int(b.xmax * IW) - 30,
                      int((b.ymin + b.ymax) / 2 * IH) - 30 + k * 30, rng)

    clock = SimClock()
    comps = build([], clock, Q, layout, render=False)
    comps.tracker.snapshot = lambda: photo
    comps.auditor = GeminiAuditor(clock, api_key="")      # real class, offline
    s = System(comps, clock, Q, layout)
    s.start()
    t = 0.0
    while s.c.brain.phase != Phase.COMPLETE and t < 120:
        s.step()
        clock.advance(0.02)
        t += 0.02
        time.sleep(0.0005)                                # let the worker thread run in "sim time"
    audits = [v for _, k, v in s.log if k == "AUDIT"]
    spoken = [v for _, k, v in s.log if k == "SPEAK"]
    assert s.c.brain.phase == Phase.COMPLETE and s.errors == {}
    assert len(audits) == len(Q) and all("ink=True" in a for a in audits), audits
    assert sum("Answer recorded" in x for x in spoken) == len(Q)
