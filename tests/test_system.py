"""
_smoothed_pen(): median-of-recent-tip-points smoothing before guidance ever sees a raw read.
_tracker_ready(): the brain (voice, guidance, the whole algorithm) must not run a single tick
before a real tracker has actually locked the page - fakes/sim (no calibration_status) always ready.
"""
from types import SimpleNamespace

from src.contracts import PenState
from src.system import System, TIP_SMOOTH_MIN_POINTS, TIP_SMOOTH_WINDOW_S


def new_system(tracker=None) -> System:
    """A System with no real Components - only the method(s) under test are exercised."""
    s = System.__new__(System)
    s._tip_window = __import__("collections").deque()
    s.errors, s.component_errors, s.log = {}, {}, []
    s.last_error = None
    s.clock = SimpleNamespace(now=lambda: 0.0)
    if tracker is not None:
        s.c = SimpleNamespace(tracker=tracker)
    return s


class StubRealTracker:
    """Mimics PenTracker's calibration_status()/calibration surface, nothing else."""

    def __init__(self, searching=True, locked_calibration=None):
        self.searching = searching
        self.calibration = locked_calibration

    def calibration_status(self):
        return self.searching, 0.0, None


def pen(t, x, y):
    return PenState(t=t, x=x, y=y, lift_cm=0.0, confidence=1.0)


def test_passes_through_before_enough_points():
    s = new_system()
    for i in range(TIP_SMOOTH_MIN_POINTS - 1):
        t = i * 0.01
        out = s._smoothed_pen(pen(t, 0.5 + i * 0.1, 0.5), t)
        assert out.x == 0.5 + i * 0.1, "should be the raw, unsmoothed reading"


def test_one_bad_frame_does_not_move_the_median():
    s = new_system()
    t = 0.0
    for _ in range(8):
        out = s._smoothed_pen(pen(t, 0.50, 0.50), t)
        t += 0.02
    assert abs(out.x - 0.50) < 1e-9 and abs(out.y - 0.50) < 1e-9

    # one wild outlier frame (wrong pen end / stray detection)
    out = s._smoothed_pen(pen(t, 0.95, 0.05), t)
    assert abs(out.x - 0.50) < 0.02 and abs(out.y - 0.50) < 0.02, (
        f"a single outlier swayed the smoothed point to ({out.x}, {out.y})")


def test_a_real_sustained_move_does_show_up():
    s = new_system()
    t = 0.0
    for _ in range(8):
        s._smoothed_pen(pen(t, 0.30, 0.30), t)
        t += 0.02
    # genuinely moves and stays there for longer than the window
    for _ in range(int(TIP_SMOOTH_WINDOW_S / 0.02) + 5):
        out = s._smoothed_pen(pen(t, 0.70, 0.70), t)
        t += 0.02
    assert abs(out.x - 0.70) < 0.02 and abs(out.y - 0.70) < 0.02, "a sustained move should fully show up"


def test_old_points_fall_out_of_the_window():
    s = new_system()
    s._smoothed_pen(pen(0.0, 0.10, 0.10), 0.0)
    s._smoothed_pen(pen(0.01, 0.10, 0.10), 0.01)
    # a long gap - old points must not still be averaged in
    t = 5.0
    out = None
    for i in range(TIP_SMOOTH_MIN_POINTS):
        out = s._smoothed_pen(pen(t + i * 0.01, 0.90, 0.90), t + i * 0.01)
    assert abs(out.x - 0.90) < 1e-6, f"stale points leaked into the median: x={out.x}"


def test_pen_lost_clears_the_window_immediately():
    s = new_system()
    t = 0.0
    for _ in range(8):
        s._smoothed_pen(pen(t, 0.5, 0.5), t)
        t += 0.02
    assert s._smoothed_pen(None, t) is None
    assert len(s._tip_window) == 0, "losing the pen should drop smoothing history, not carry it over"


def test_other_pen_fields_are_preserved():
    s = new_system()
    t = 0.0
    for i in range(TIP_SMOOTH_MIN_POINTS):
        p = PenState(t=t, x=0.5, y=0.5, lift_cm=1.5 + i, confidence=0.8)
        out = s._smoothed_pen(p, t)
        t += 0.02
    assert out.t == p.t and out.lift_cm == p.lift_cm and out.confidence == p.confidence


def test_fake_tracker_is_always_ready():
    """No calibration_status at all (FakeTracker, sim/tests): never gated."""
    s = new_system(tracker=SimpleNamespace())
    assert s._tracker_ready() is True


def test_real_tracker_not_ready_while_searching():
    s = new_system(tracker=StubRealTracker(searching=True, locked_calibration=None))
    assert s._tracker_ready() is False


def test_real_tracker_not_ready_if_locked_flag_is_false_but_no_calibration_yet():
    # defensive: calibration_status() could say "not searching" a tick before .calibration is set
    s = new_system(tracker=StubRealTracker(searching=False, locked_calibration=None))
    assert s._tracker_ready() is False


def test_real_tracker_ready_once_actually_locked():
    s = new_system(tracker=StubRealTracker(searching=False, locked_calibration=object()))
    assert s._tracker_ready() is True


def test_calibration_status_error_fails_safe_to_not_ready():
    class Broken:
        calibration = object()

        def calibration_status(self):
            raise RuntimeError("camera dropped")

    s = new_system(tracker=Broken())
    assert s._tracker_ready() is False   # never run the algorithm on an error, not even by accident
