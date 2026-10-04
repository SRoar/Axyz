import pytest

from src.contracts import PAGE_H_CM, PAGE_W_CM, Box, GuidanceEngine as GuidanceProtocol, HapticCmd, PenState, WriteStatus
from src.guidance import IN_BOX_SPEECH, GuidanceEngine, phrase

BOX = Box("box1", 0.10, 0.25, 0.90, 0.45)     # same as sample_layout()


def pen(x, y):
    return PenState(0.0, x, y)


def pen_cm(x_cm, y_cm):
    return pen(x_cm / PAGE_W_CM, y_cm / PAGE_H_CM)


def test_satisfies_contract():
    assert isinstance(GuidanceEngine(), GuidanceProtocol)


def test_no_pen_or_no_box_is_unknown():
    g = GuidanceEngine()
    r = g.compute(None, BOX)
    assert r.write_status == WriteStatus.UNKNOWN and not r.pen_visible and not r.in_box and r.cmd is None
    r = g.compute(pen(0.5, 0.3), None)
    assert r.write_status == WriteStatus.UNKNOWN and r.pen_visible and not r.in_box


def test_inside_box():
    r = GuidanceEngine().compute(pen(0.5, 0.35), BOX)
    assert r.in_box and r.cmd is None and r.write_status == WriteStatus.INSIDE
    assert r.speech == IN_BOX_SPEECH and r.dist_cm == 0.0 and r.pen_visible


def test_above_box_horizontally_aligned_guides_both_and_says_down():
    r = GuidanceEngine().compute(pen(0.5, 0.10), BOX)
    assert not r.in_box and r.write_status == WriteStatus.OUTSIDE
    assert r.cmd == HapticCmd.GUIDE_BOTH
    assert r.dx_cm == 0.0 and r.dy_cm > 0
    # 0.15 page * 27.94 = 4.19 cm (+0.75 cm target inset) -> 1.94 in -> "2 inches"
    assert r.speech == "Move 2 inches Down."
    assert r.dist_cm == pytest.approx(r.dy_cm)


def test_left_and_right_of_box():
    r = GuidanceEngine().compute(pen(0.02, 0.35), BOX)
    assert r.cmd == HapticCmd.GUIDE_RIGHT and r.dx_cm > 0 and r.dy_cm == 0.0
    assert r.speech.endswith("Right.")
    r = GuidanceEngine().compute(pen(0.99, 0.35), BOX)
    assert r.cmd == HapticCmd.GUIDE_LEFT and r.dx_cm < 0
    assert r.speech.endswith("Left.")


def test_below_box_says_up():
    r = GuidanceEngine().compute(pen(0.5, 0.60), BOX)
    assert r.dy_cm < 0 and r.speech.startswith("Move") and r.speech.endswith("Up.")


def test_write_status_hysteresis_at_the_edge():
    g = GuidanceEngine()
    top = BOX.ymin * PAGE_H_CM
    x = 10.0
    assert g.compute(pen_cm(x, top + 1.0), BOX).write_status == WriteStatus.INSIDE
    # jitter 0.2 cm outside the edge: still INSIDE (exit margin 0.4 cm)
    assert g.compute(pen_cm(x, top - 0.2), BOX).write_status == WriteStatus.INSIDE
    # 0.6 cm outside: OUTSIDE
    assert g.compute(pen_cm(x, top - 0.6), BOX).write_status == WriteStatus.OUTSIDE
    # back just 0.1 cm inside: still OUTSIDE (enter inset 0.25 cm)
    r = g.compute(pen_cm(x, top + 0.1), BOX)
    assert r.write_status == WriteStatus.OUTSIDE and r.speech == "Move a little Down."
    # 0.3 cm inside: INSIDE
    assert g.compute(pen_cm(x, top + 0.3), BOX).write_status == WriteStatus.INSIDE


def test_edge_jitter_does_not_flap():
    g = GuidanceEngine()
    top = BOX.ymin * PAGE_H_CM
    g.compute(pen_cm(10.0, top + 1.0), BOX)
    flips, last = 0, None
    for i in range(100):
        y = top + (0.15 if i % 2 else -0.15)        # +/- 1.5 mm jitter across the edge
        s = g.compute(pen_cm(10.0, y), BOX).write_status
        flips += last is not None and s != last
        last = s
    assert flips == 0 and last == WriteStatus.INSIDE


def test_horizontal_cmd_hysteresis():
    g = GuidanceEngine()
    left = BOX.xmin * PAGE_W_CM + g.TARGET_INSET_CM
    y = 3.0   # above the box

    def cmd(dx_cm):
        return g.compute(pen_cm(left - dx_cm, y), BOX).cmd

    assert cmd(1.1) == HapticCmd.GUIDE_RIGHT     # first decision at 1.0 cm
    assert cmd(0.8) == HapticCmd.GUIDE_RIGHT     # stays until < 0.7
    assert cmd(0.6) == HapticCmd.GUIDE_BOTH
    assert cmd(1.2) == HapticCmd.GUIDE_BOTH      # stays until > 1.3
    assert cmd(1.4) == HapticCmd.GUIDE_RIGHT


def test_changing_box_resets_state():
    g = GuidanceEngine()
    top = BOX.ymin * PAGE_H_CM
    g.compute(pen_cm(10.0, top + 1.0), BOX)
    assert g.compute(pen_cm(10.0, top - 0.2), BOX).in_box
    other = Box("box2", 0.10, 0.25, 0.90, 0.45)
    assert not g.compute(pen_cm(10.0, top - 0.2), other).in_box


def test_following_the_cues_reaches_the_box():
    g = GuidanceEngine()
    box = Box("box2", 0.10, 0.55, 0.90, 0.80)
    x, y = 0.02, 0.05
    for _ in range(50):
        r = g.compute(pen(x, y), box)
        if r.in_box:
            break
        x += 0.5 * r.dx_cm / PAGE_W_CM
        y += 0.5 * r.dy_cm / PAGE_H_CM
    assert r.in_box and box.contains(x, y)


def test_sim_drift_out_of_box_is_outside():
    g = GuidanceEngine()
    assert g.compute(pen(0.80, 0.35), BOX).write_status == WriteStatus.INSIDE
    assert g.compute(pen(0.80, 0.20), BOX).write_status == WriteStatus.OUTSIDE


@pytest.mark.parametrize("dx,dy,expected", [
    (0.0, 5.08, "Move 2 inches Down."),
    (0.0, -2.54, "Move 1 inch Up."),
    (-1.0, 0.0, "Move a little Left."),
    (12.7, 0.0, "Move 5 inches Right."),
    (-3.2, 7.6, "Move 3 inches Down and 1 inch Left."),    # second axis >= 40% of the first: mention it
    (-2.6, 7.6, "Move 3 inches Down."),                    # second axis too small to mention
])
def test_phrase(dx, dy, expected):
    assert phrase(dx, dy) == expected
