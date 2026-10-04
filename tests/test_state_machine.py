"""Scripted Inputs for every row of the PLAN.md section 4 table plus the error paths.

    python -m pytest tests/test_state_machine.py
"""
from src import speech
from src.contracts import (
    AuditResult, Guidance, Haptic, HapticCmd, ImuEvent, Inputs, MotionState, Phase,
    Question, SetBox, Silence, Snapshot, Speak, Tap, WriteStatus, TIMING, sample_layout,
)
from src.state_machine import PEN_LOST_HOLD_S, READ_MAX_S, StateMachine

S, M, W, L = MotionState.STILL, MotionState.MOVING, MotionState.WRITING, MotionState.LIFTED

QUESTIONS = [
    Question("q1", "First question?", "box1"),
    Question("q2", "Second question?", "box2"),
    Question("q3", "Third question?", "box3"),
]

GUIDE_LEFT = Guidance(HapticCmd.GUIDE_LEFT, "Move 1 inch Left.", -3.0, 0.0, 3.0, False,
                      WriteStatus.OUTSIDE, True)
IN_BOX = Guidance(None, "You are in the answer box.", 0, 0, 0, True, WriteStatus.INSIDE, True)
OUTSIDE = Guidance(HapticCmd.GUIDE_LEFT, None, -1.0, 0, 1.0, False, WriteStatus.OUTSIDE, True)
PEN_LOST = Guidance(write_status=WriteStatus.UNKNOWN, pen_visible=False)


def inp(t, motion=S, taps=(), speaking=False, guidance=None, audit=None):
    return Inputs(t=t, pen=None, guidance=guidance, motion=motion,
                  imu_events=tuple(ImuEvent(t, tap=x) for x in taps),
                  speaking=speaking, audit_result=audit)


def spoken(acts):
    return [a.text for a in acts if isinstance(a, Speak)]


def haptics(acts):
    return [a.cmd for a in acts if isinstance(a, Haptic)]


def brain():
    return StateMachine(QUESTIONS, sample_layout(), tap_mode=False)


def tap_brain():
    return StateMachine(QUESTIONS, sample_layout(), tap_mode=True)


def to_reading(sm, t=0.0):
    sm.update(inp(t, motion=M, taps=[Tap.SINGLE]))
    assert sm.phase == Phase.READING
    return t


def to_navigating(sm, t=0.0):
    to_reading(sm, t)
    sm.update(inp(t + 0.6, motion=M, speaking=True))
    sm.update(inp(t + 1.0, motion=M, speaking=False))
    assert sm.phase == Phase.NAVIGATING
    return t + 1.0


def to_writing(sm, t=0.0):
    t = to_navigating(sm, t) + 0.1
    sm.update(inp(t, motion=W, guidance=IN_BOX))
    assert sm.phase == Phase.WRITING
    return t


def to_auditing(sm, t=0.0):
    t = to_writing(sm, t) + 1.0
    sm.update(inp(t, motion=W, taps=[Tap.TRIPLE], guidance=IN_BOX))
    assert sm.phase == Phase.AUDITING
    return t


# --------------------------------------------------------------------------- IDLE
def test_idle_hover_reads_question():
    sm = brain()
    assert sm.update(inp(0.0)) == []
    assert sm.update(inp(0.5)) == []
    acts = sm.update(inp(0.9))
    assert any(isinstance(a, SetBox) and a.box.id == "box1" for a in acts)
    assert spoken(acts) == [speech.question_intro(1, "First question?")]
    assert sm.phase == Phase.READING


def test_idle_tap1_reads_question():
    sm = brain()
    acts = sm.update(inp(0.0, motion=M, taps=[Tap.SINGLE]))
    assert spoken(acts) == [speech.question_intro(1, "First question?")]
    assert sm.phase == Phase.READING


def test_idle_hover_waits_until_voice_is_quiet():
    sm = brain()
    sm.update(inp(0.0, speaking=True))
    assert sm.update(inp(1.0, speaking=True)) == []
    assert sm.phase == Phase.IDLE
    sm.update(inp(1.1, speaking=False))
    assert sm.phase == Phase.READING


def test_idle_moving_does_not_read():
    sm = brain()
    for i in range(50):
        sm.update(inp(i * 0.1, motion=M))
    assert sm.phase == Phase.IDLE


# --------------------------------------------------------------------------- skip (tap 2)
def test_skip_from_idle():
    sm = brain()
    acts = sm.update(inp(0.0, motion=M, taps=[Tap.DOUBLE]))
    assert spoken(acts) == [speech.moving_to_question(2)]
    assert sm.index == 1 and sm.phase == Phase.IDLE


def test_skip_from_reading():
    sm = brain()
    to_reading(sm)
    acts = sm.update(inp(1.0, motion=M, taps=[Tap.DOUBLE]))
    assert spoken(acts) == [speech.moving_to_question(2)]
    assert sm.index == 1 and sm.phase == Phase.IDLE


def test_skip_from_navigating_turns_buzzers_off():
    sm = brain()
    t = to_navigating(sm)
    sm.update(inp(t + 0.1, motion=M, guidance=GUIDE_LEFT))
    acts = sm.update(inp(t + 1.0, motion=M, taps=[Tap.DOUBLE], guidance=GUIDE_LEFT))
    assert HapticCmd.OFF in haptics(acts)
    assert sm.index == 1 and sm.phase == Phase.IDLE


def test_skip_last_question_completes_with_summary():
    sm = brain()
    for i in range(3):
        sm.update(inp(i * 1.0, motion=M, taps=[Tap.DOUBLE]))
    assert sm.phase == Phase.COMPLETE
    assert sm.skipped == 3


# --------------------------------------------------------------------------- READING
def test_reading_tap1_repeats_question():
    sm = brain()
    to_reading(sm)
    acts = sm.update(inp(1.0, motion=M, taps=[Tap.SINGLE], speaking=True))
    assert spoken(acts) == [speech.question_intro(1, "First question?")]
    assert acts[0].interrupt
    assert sm.phase == Phase.READING


def test_reading_ignores_taps_right_after_start():
    sm = brain()
    to_reading(sm)
    acts = sm.update(inp(0.45, motion=M, taps=[Tap.DOUBLE], speaking=True))
    assert acts == [] and sm.index == 0 and sm.phase == Phase.READING


def test_reading_to_navigating_when_speech_finishes():
    sm = brain()
    to_reading(sm)
    sm.update(inp(0.2, motion=M, speaking=True))
    sm.update(inp(0.3, motion=M, speaking=True))
    assert sm.phase == Phase.READING
    sm.update(inp(0.4, motion=M, speaking=False))
    assert sm.phase == Phase.NAVIGATING


def test_reading_grace_when_speech_never_starts():
    sm = brain()
    to_reading(sm)
    sm.update(inp(1.0, motion=M))
    assert sm.phase == Phase.READING
    sm.update(inp(TIMING.READ_GRACE_S + 0.1, motion=M))
    assert sm.phase == Phase.NAVIGATING


def test_reading_timeout_if_voice_stuck():
    sm = brain()
    to_reading(sm)
    sm.update(inp(1.0, motion=M, speaking=True))
    assert sm.phase == Phase.READING
    sm.update(inp(READ_MAX_S + 1.0, motion=M, speaking=True))
    assert sm.phase == Phase.NAVIGATING


# --------------------------------------------------------------------------- NAVIGATING
def test_navigating_moving_guides_and_refreshes():
    sm = brain()
    t = to_navigating(sm)
    acts = sm.update(inp(t + 0.1, motion=M, guidance=GUIDE_LEFT))
    assert haptics(acts) == [HapticCmd.GUIDE_LEFT]
    assert spoken(acts) == ["Move 1 inch Left."]
    assert haptics(sm.update(inp(t + 0.3, motion=M, guidance=GUIDE_LEFT))) == []
    assert haptics(sm.update(inp(t + 0.7, motion=M, guidance=GUIDE_LEFT))) == [HapticCmd.GUIDE_LEFT]


def test_navigating_spoken_cue_rate_limited_and_not_while_speaking():
    sm = brain()
    t = to_navigating(sm)
    sm.update(inp(t + 0.1, motion=M, guidance=GUIDE_LEFT))
    assert spoken(sm.update(inp(t + 2.0, motion=M, guidance=GUIDE_LEFT))) == []
    late = t + 0.1 + TIMING.NAV_SPEECH_EVERY_S + 0.1
    assert spoken(sm.update(inp(late, motion=M, guidance=GUIDE_LEFT, speaking=True))) == []
    assert spoken(sm.update(inp(late + 0.1, motion=M, guidance=GUIDE_LEFT))) == ["Move 1 inch Left."]


def test_navigating_still_or_lifted_turns_buzzers_off():
    sm = brain()
    t = to_navigating(sm)
    sm.update(inp(t + 0.1, motion=M, guidance=GUIDE_LEFT))
    assert haptics(sm.update(inp(t + 0.2, motion=S, guidance=GUIDE_LEFT))) == [HapticCmd.OFF]
    sm.update(inp(t + 0.3, motion=M, guidance=GUIDE_LEFT))
    assert haptics(sm.update(inp(t + 0.4, motion=L, guidance=GUIDE_LEFT))) == [HapticCmd.OFF]


def test_navigating_lock_once_when_still_in_box():
    sm = brain()
    t = to_navigating(sm)
    sm.update(inp(t + 0.1, motion=S, guidance=IN_BOX))
    assert sm.update(inp(t + 0.2, motion=S, guidance=IN_BOX)) == []
    acts = sm.update(inp(t + 0.5, motion=S, guidance=IN_BOX))
    assert haptics(acts) == [HapticCmd.LOCK]
    assert spoken(acts) == [speech.IN_ANSWER_BOX]
    assert sm.update(inp(t + 1.5, motion=S, guidance=IN_BOX)) == []


def test_navigating_tap1_repeats_question():
    sm = brain()
    t = to_navigating(sm)
    acts = sm.update(inp(t + 0.5, motion=S, taps=[Tap.SINGLE]))
    assert spoken(acts) == [speech.question_intro(1, "First question?")]
    assert sm.phase == Phase.NAVIGATING


def test_navigating_writing_silences_and_enters_writing():
    sm = brain()
    t = to_navigating(sm)
    sm.update(inp(t + 0.1, motion=M, guidance=GUIDE_LEFT))
    acts = sm.update(inp(t + 0.2, motion=W, guidance=IN_BOX))
    assert any(isinstance(a, Silence) for a in acts)
    assert HapticCmd.OFF in haptics(acts)
    assert sm.phase == Phase.WRITING


def test_pen_lost_keeps_last_cue_then_off():
    sm = brain()
    t = to_navigating(sm)
    sm.update(inp(t + 0.1, motion=M, guidance=GUIDE_LEFT))
    held = sm.update(inp(t + 0.7, motion=M, guidance=PEN_LOST))
    assert haptics(held) == [HapticCmd.GUIDE_LEFT]
    gone = sm.update(inp(t + 0.1 + PEN_LOST_HOLD_S + 0.1, motion=M, guidance=PEN_LOST))
    assert haptics(gone) == [HapticCmd.OFF]
    assert spoken(gone) == []


# --------------------------------------------------------------------------- WRITING
def test_writing_outside_warns_inside_is_quiet():
    sm = brain()
    t = to_writing(sm)
    assert haptics(sm.update(inp(t + 0.1, motion=W, guidance=OUTSIDE))) == [HapticCmd.WARN]
    assert haptics(sm.update(inp(t + 0.2, motion=W, guidance=IN_BOX))) == [HapticCmd.OFF]


def test_writing_never_warns_unless_imu_says_writing():
    sm = brain()
    t = to_writing(sm)
    for i in range(10):
        acts = sm.update(inp(t + 0.1 * (i + 1), motion=S, guidance=OUTSIDE))
        assert HapticCmd.WARN not in haptics(acts)


def test_writing_never_speaks():
    sm = brain()
    t = to_writing(sm)
    for i in range(30):
        acts = sm.update(inp(t + 0.1 * (i + 1), motion=W, taps=[Tap.SINGLE], guidance=OUTSIDE))
        assert spoken(acts) == []


def test_writing_idle_timeout_takes_snapshot():
    sm = brain()
    t = to_writing(sm)
    sm.update(inp(t + 0.1, motion=S, guidance=IN_BOX))
    assert sm.phase == Phase.WRITING
    acts = sm.update(inp(t + 0.1 + TIMING.ANSWER_IDLE_S, motion=S, guidance=IN_BOX))
    assert any(isinstance(a, Snapshot) and a.question_id == "q1" for a in acts)
    assert sm.phase == Phase.AUDITING


def test_writing_resumes_reset_idle_timer():
    sm = brain()
    t = to_writing(sm)
    sm.update(inp(t + 0.1, motion=S, guidance=IN_BOX))
    sm.update(inp(t + 1.5, motion=W, guidance=IN_BOX))
    sm.update(inp(t + 1.6, motion=S, guidance=IN_BOX))
    sm.update(inp(t + 2.5, motion=S, guidance=IN_BOX))
    assert sm.phase == Phase.WRITING


def test_writing_tap3_takes_snapshot():
    sm = brain()
    t = to_writing(sm)
    acts = sm.update(inp(t + 1.0, motion=W, taps=[Tap.TRIPLE], guidance=IN_BOX))
    assert any(isinstance(a, Snapshot) for a in acts)
    assert sm.phase == Phase.AUDITING


# --------------------------------------------------------------------------- AUDITING
def test_audit_ink_present_records_and_advances():
    sm = brain()
    t = to_auditing(sm)
    acts = sm.update(inp(t + 1.0, audit=AuditResult("q1", True)))
    assert spoken(acts) == [speech.ANSWER_RECORDED]
    assert haptics(acts) == [HapticCmd.COMPLETE]
    assert sm.index == 1 and sm.phase == Phase.IDLE


def test_audit_ink_outside_mentions_it():
    sm = brain()
    t = to_auditing(sm)
    acts = sm.update(inp(t + 1.0, audit=AuditResult("q1", True, ink_outside=True)))
    assert spoken(acts) == [speech.ANSWER_RECORDED_OUTSIDE]


def test_audit_no_ink_retries_then_accepts():
    sm = brain()
    t = to_auditing(sm)
    for retry in range(StateMachine.MAX_AUDIT_RETRIES):
        acts = sm.update(inp(t + 1.0, audit=AuditResult("q1", False)))
        assert spoken(acts) == [speech.NO_INK_FOUND]
        assert sm.phase == Phase.NAVIGATING
        t += 2.0
        sm.update(inp(t, motion=W, guidance=IN_BOX))
        sm.update(inp(t + 1.0, motion=W, taps=[Tap.TRIPLE], guidance=IN_BOX))
        assert sm.phase == Phase.AUDITING
        t += 1.0
    acts = sm.update(inp(t + 1.0, audit=AuditResult("q1", False)))
    assert spoken(acts) == [speech.ANSWER_RECORDED]
    assert sm.index == 1


def test_audit_timeout_moves_on():
    sm = brain()
    t = to_auditing(sm)
    assert sm.update(inp(t + 1.0)) == []
    acts = sm.update(inp(t + TIMING.AUDIT_TIMEOUT_S + 0.1))
    assert spoken(acts) == [speech.AUDIT_TIMEOUT]
    assert haptics(acts) == [HapticCmd.COMPLETE]
    assert sm.index == 1


def test_audit_result_for_other_question_ignored():
    sm = brain()
    t = to_auditing(sm)
    assert sm.update(inp(t + 1.0, audit=AuditResult("q9", True))) == []
    assert sm.phase == Phase.AUDITING


# --------------------------------------------------------------------------- error paths
def test_tap_debounce():
    sm = brain()
    t = to_navigating(sm)
    first = sm.update(inp(t + 0.5, motion=S, taps=[Tap.SINGLE]))
    second = sm.update(inp(t + 0.6, motion=S, taps=[Tap.SINGLE]))
    assert len(spoken(first)) == 1 and spoken(second) == []


def test_writing_seen_in_idle_does_not_read_or_crash():
    sm = brain()
    for i in range(30):
        assert sm.update(inp(i * 0.1, motion=W)) == []
    assert sm.phase == Phase.IDLE


def test_writing_seen_in_reading_is_handled_after_speech():
    sm = brain()
    to_reading(sm)
    sm.update(inp(0.6, motion=W, speaking=True))
    assert sm.phase == Phase.READING
    sm.update(inp(1.0, motion=W, speaking=False))
    assert sm.phase == Phase.NAVIGATING
    acts = sm.update(inp(1.1, motion=W, guidance=IN_BOX))
    assert any(isinstance(a, Silence) for a in acts)
    assert sm.phase == Phase.WRITING


def test_complete_summary_and_no_actions_after():
    sm = brain()
    t = to_auditing(sm)
    sm.update(inp(t + 1.0, audit=AuditResult("q1", True)))
    sm.update(inp(t + 2.0, motion=M, taps=[Tap.DOUBLE]))
    acts = sm.update(inp(t + 3.0, motion=M, taps=[Tap.DOUBLE]))
    assert spoken(acts) == [speech.test_complete(1, 2)]
    assert sm.phase == Phase.COMPLETE
    assert sm.update(inp(t + 4.0, motion=M, taps=[Tap.SINGLE])) == []


def test_summary_phrasing():
    assert speech.test_complete(1, 0).endswith("You answered 1 question.")
    assert speech.test_complete(2, 1).endswith("You answered 2 questions and skipped 1.")


def test_empty_question_list_is_safe():
    sm = StateMachine([], {})
    assert sm.update(inp(0.0, taps=[Tap.SINGLE])) == []


# --------------------------------------------------------------------------- tap-only demo mode
def test_tap_mode_from_env(monkeypatch):
    monkeypatch.setenv("ILLUMIN_TAP_MODE", "1")
    assert StateMachine(QUESTIONS, sample_layout()).tap_mode
    monkeypatch.setenv("ILLUMIN_TAP_MODE", "0")
    assert not StateMachine(QUESTIONS, sample_layout()).tap_mode


def test_tap_mode_never_reads_without_a_tap():
    sm = tap_brain()
    for i in range(100):
        assert sm.update(inp(i * 0.1)) == []
    assert sm.phase == Phase.IDLE


def test_tap_mode_tap1_reads_and_sets_box():
    sm = tap_brain()
    acts = sm.update(inp(0.0, taps=[Tap.SINGLE]))
    assert any(isinstance(a, SetBox) for a in acts)
    assert spoken(acts) == [speech.question_intro(1, "First question?")]


def test_tap_mode_guidance_is_buzz_only():
    sm = tap_brain()
    t = to_navigating(sm)
    acts = sm.update(inp(t + 0.1, motion=M, guidance=GUIDE_LEFT))
    assert haptics(acts) == [HapticCmd.GUIDE_LEFT]
    assert spoken(acts) == []
    sm.update(inp(t + 0.2, motion=S, guidance=IN_BOX))
    acts = sm.update(inp(t + 0.6, motion=S, guidance=IN_BOX))
    assert haptics(acts) == [HapticCmd.LOCK]
    assert spoken(acts) == []


def test_tap_mode_writing_warns_outside_and_never_audits():
    sm = tap_brain()
    t = to_writing(sm)
    assert haptics(sm.update(inp(t + 0.1, motion=W, guidance=OUTSIDE))) == [HapticCmd.WARN]
    for i in range(50):
        acts = sm.update(inp(t + 0.2 + i * 0.1, motion=S, guidance=IN_BOX))
        assert not any(isinstance(a, Snapshot) for a in acts)
        assert spoken(acts) == []
    assert sm.phase == Phase.NAVIGATING


def test_tap_mode_pause_then_tap1_repeats_without_relock():
    sm = tap_brain()
    t = to_writing(sm)
    sm.update(inp(t + 0.1, motion=S, guidance=IN_BOX))
    sm.update(inp(t + 0.1 + TIMING.ANSWER_IDLE_S, motion=S, guidance=IN_BOX))
    assert sm.phase == Phase.NAVIGATING
    acts = sm.update(inp(t + 3.0, motion=S, taps=[Tap.SINGLE], guidance=IN_BOX))
    assert spoken(acts) == [speech.question_intro(1, "First question?")]
    assert HapticCmd.LOCK not in haptics(acts)


def test_tap_mode_full_flow():
    sm = tap_brain()
    t = 0.0
    for n in range(1, 4):
        acts = sm.update(inp(t, taps=[Tap.SINGLE]))
        assert spoken(acts) == [speech.question_intro(n, QUESTIONS[n - 1].text)]
        sm.update(inp(t + 2.0))
        assert sm.phase == Phase.NAVIGATING
        sm.update(inp(t + 3.0, motion=W, guidance=IN_BOX))
        assert sm.phase == Phase.WRITING
        acts = sm.update(inp(t + 6.0, motion=S, taps=[Tap.DOUBLE], guidance=IN_BOX))
        if n < 3:
            assert spoken(acts) == [speech.moving_to_question(n + 1)]
            assert sm.phase == Phase.IDLE and sm.index == n
        t += 10.0
    assert sm.phase == Phase.COMPLETE
    assert spoken(acts) == [speech.test_complete(3, 0)]


def test_tap_mode_counts_unanswered_as_skipped():
    sm = tap_brain()
    t = to_writing(sm)
    sm.update(inp(t + 1.0, motion=S, taps=[Tap.DOUBLE]))
    sm.update(inp(t + 2.0, taps=[Tap.DOUBLE]))
    acts = sm.update(inp(t + 3.0, taps=[Tap.DOUBLE]))
    assert spoken(acts) == [speech.test_complete(1, 2)]
