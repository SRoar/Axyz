"""Motion classifier / tap detector / haptic engine (the Python mirror of the firmware headers)."""
import glob
import math
import os
import random

import numpy as np
import pytest

import imu_logic as L
from imu_logic import HapticEngine, MotionClassifier, TapDetector, load_params

P = load_params()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_classifier(samples, params=None, frozen=False):
    clf = MotionClassifier(params or P)
    changes = []
    for i, (ax, ay, az) in enumerate(samples):
        new = clf.update(ax, ay, az, frozen)
        if new:
            changes.append((i, new))
    return clf, changes


def alternating(amp, n, axis=0):
    """Signal whose window spread is exactly `amp` (one axis flips +/-amp, the others sit still)."""
    out = []
    for i in range(n):
        v = [0.0, 1.0, 0.0]
        v[axis] = amp if i % 2 == 0 else -amp
        out.append(tuple(v))
    return out


# --------------------------------------------------------------------------- parameters
def test_params_file_parses():
    for k in ("T_ACTIVE", "T_WRITE", "EMIT_WRITING", "CLS_WINDOW", "CLS_DWELL", "CLS_HYST", "TAP_ACT_G", "TAP_PEAK_G",
              "TAP_PRE_QUIET_MS", "TAP_NOT_WRITING_MS", "TAP_BURST_GAP_MS", "TAP_ONE_MAX_MS", "TAP_BURST_MAX_MS", "TAP_MAX",
              "TAP_GROUP_MS", "TAP_FREEZE_MS", "TAP_EMA_ALPHA", "HAPTIC_DEADMAN_MS"):
        assert k in P, k
    assert 0 < P["T_ACTIVE"] < P["T_WRITE"]
    assert isinstance(P["EMIT_WRITING"], bool)


# --------------------------------------------------------------------------- classifier
def test_resting_noise_stays_still_and_emits_nothing():
    rng = random.Random(1)
    sig = [(rng.gauss(0, .02), 0.95 + rng.gauss(0, .02), rng.gauss(0, .02)) for _ in range(300)]
    clf, changes = run_classifier(sig)
    assert clf.state == L.STILL and changes == []


def test_moves_then_writes_then_stops():
    t_act, t_wr = P["T_ACTIVE"], P["T_WRITE"]
    mov, wri = alternating(t_act * 1.6, 80), alternating(t_wr * 1.6, 80)
    _, changes = run_classifier(alternating(0.0, 40) + mov + wri + alternating(0.0, 80))
    assert [s for _, s in changes] == [L.MOVING, L.WRITING, L.MOVING, L.STILL] or [s for _, s in changes] == [L.MOVING, L.WRITING, L.STILL]


def test_new_state_needs_to_hold_for_the_dwell_time():
    t_act = P["T_ACTIVE"]
    still = alternating(0.0, 40)
    # one single sample of strong motion in an otherwise still window would raise the spread for 16 samples,
    # so check the dwell on the classifier directly: the first CLS_DWELL-1 over-threshold samples report nothing
    clf = MotionClassifier(P)
    for s in still:
        clf.update(*s)
    reported = []
    for s in alternating(t_act * 4, 12):
        reported.append(clf.update(*s))
    first = next(i for i, r in enumerate(reported) if r)
    assert first >= int(P["CLS_DWELL"]) - 1


def test_hysteresis_keeps_moving_until_score_drops_well_below_threshold():
    t_act, h = P["T_ACTIVE"], P["CLS_HYST"]
    clf = MotionClassifier(P)
    for s in alternating(t_act * 1.5, 60):
        clf.update(*s)
    assert clf.state == L.MOVING
    for s in alternating(t_act * (h + 1.0) / 2, 60):  # between the exit and the enter threshold: stay
        clf.update(*s)
    assert clf.state == L.MOVING
    for s in alternating(t_act * h * 0.6, 60):
        clf.update(*s)
    assert clf.state == L.STILL


def test_kill_switch_never_reports_writing():
    params = dict(P, EMIT_WRITING=False)
    clf, changes = run_classifier(alternating(P["T_WRITE"] * 3, 120), params)
    assert clf.state == L.MOVING and L.WRITING not in [s for _, s in changes]


def test_frozen_holds_the_state():
    clf = MotionClassifier(P)
    for s in alternating(0.0, 40):
        clf.update(*s)
    for s in alternating(P["T_WRITE"] * 3, 40):
        assert clf.update(*s, frozen=True) is None
    assert clf.state == L.STILL


def _clips():
    runs = {}
    for fp in sorted(glob.glob(os.path.join(ROOT, "data", "imu", "*.csv"))):
        label = os.path.basename(fp).split("_")[0]
        if label in ("STILL", "LIFTED", "MOVING", "WRITING"):
            import csv
            rows = list(csv.DictReader(open(fp)))
            arr = np.array([[float(r["ax"]), float(r["ay"]), float(r["az"])] for r in rows])
            runs.setdefault(label, []).append(arr)
    return runs


def _share(clip, state):
    clf = MotionClassifier(P)
    seen = []
    for i, (ax, ay, az) in enumerate(clip):
        clf.update(ax, ay, az)
        if i >= int(P["CLS_WINDOW"]) + 30:
            seen.append(clf.state)
    return seen.count(state) / len(seen)


@pytest.mark.skipif(not glob.glob(os.path.join(ROOT, "data", "imu", "STILL_*.csv")), reason="no recorded clips")
def test_recorded_clips_are_classified_as_labeled():
    """Regression guard on the recorded data (this is the person the thresholds were tuned on)."""
    clips = _clips()
    for clip in clips["STILL"] + clips.get("LIFTED", []):   # LIFTED is reported as STILL
        assert _share(clip, L.STILL) >= 0.80
    for clip in clips["MOVING"]:
        assert 1 - _share(clip, L.STILL) >= 0.85            # at least "not still"
        assert _share(clip, L.MOVING) >= 0.60
    for clip in clips["WRITING"]:
        assert 1 - _share(clip, L.STILL) >= 0.95
        assert _share(clip, L.WRITING) >= 0.60


# --------------------------------------------------------------------------- taps
# Real pen taps ring: a single tap is a ~0.1-0.2 s burst of |a| swinging 0.4-0.8 g, a double tap a ~0.3-0.55 s burst
# (or two). These tests build signals with that shape: baseline 1.0 g plus small noise, bursts of alternating spikes.
DT = 15  # ms between internal samples (~66 Hz)


def burst(start_ms, span_ms, amp=0.5, step_ms=45):
    """Spikes of alternating sign from start_ms to start_ms + span_ms."""
    out, t, sign = [], start_ms, 1
    while t <= start_ms + span_ms:
        out.append((t, sign * amp))
        t += step_ms
        sign = -sign
    return out


def run_taps(spikes, total_ms=5000, allowed=lambda t: True, noise=0.005, seed=3):
    rng = random.Random(seed)
    det = TapDetector(P)
    by_idx = {int(t // DT): a for t, a in spikes}
    out, frozen_log = [], []
    for k in range(int(total_ms / DT)):
        t = k * DT
        mag = 1.0 + by_idx.get(k, rng.gauss(0, noise))
        n = det.update(t, 0.0, mag, 0.0, allowed(t))
        if n:
            out.append((t, n))
        frozen_log.append((t, det.frozen(t)))
    return out, frozen_log


def test_a_short_burst_is_one_tap_reported_after_the_group_wait():
    spikes = burst(500, 90)
    out, _ = run_taps(spikes)
    assert [n for _, n in out] == [1]
    assert out[0][0] >= spikes[-1][0] - DT + P["TAP_GROUP_MS"]  # samples are on a 15 ms grid


def test_a_long_burst_is_two_taps_even_when_the_two_taps_run_together():
    out, _ = run_taps(burst(500, 400))
    assert [n for _, n in out] == [2]


def test_two_separate_bursts_are_two_taps():
    out, _ = run_taps(burst(500, 90) + burst(850, 90))
    assert [n for _, n in out] == [2]


@pytest.mark.parametrize("spikes", [
    burst(500, 90) + burst(1000, 90) + burst(1500, 90),        # three taps is not a gesture we have
    burst(500, 90) + burst(850, 400),                           # one tap plus a double = three
    burst(500, 90) + burst(900, 90) + burst(1300, 90) + burst(1700, 90),
])
def test_three_or_more_taps_report_nothing(spikes):
    out, _ = run_taps(spikes, total_ms=6000)
    assert out == []


def test_a_normal_tap_still_works_after_a_cancelled_burst():
    out, _ = run_taps(burst(500, 90) + burst(1000, 90) + burst(1500, 90) + burst(4500, 90), total_ms=7000)
    assert [n for _, n in out] == [1]


def test_a_long_burst_is_motion_not_a_tap():
    out, _ = run_taps(burst(500, 1500, amp=0.4))     # a second and a half of swinging: writing or moving
    assert out == []


def test_a_small_bump_is_not_a_tap():
    out, _ = run_taps(burst(500, 90, amp=0.27))
    assert out == []


def test_taps_are_ignored_when_not_allowed():
    out, _ = run_taps(burst(500, 90), allowed=lambda t: False)
    assert out == []


def test_a_burst_right_after_other_activity_is_not_a_tap():
    """Activity (here at 300-400 ms, while taps were not allowed) must be followed by a quiet moment first."""
    spikes = burst(300, 90) + burst(600, 90)
    out, _ = run_taps(spikes, allowed=lambda t: t >= 550)
    assert out == []


def test_slow_drift_and_noise_make_no_taps():
    rng = random.Random(5)
    det = TapDetector(P)
    for k in range(2000):
        mag = 1.0 + 0.2 * math.sin(k / 200.0) + rng.gauss(0, 0.01)
        assert det.update(k * DT, 0.0, mag, 0.0, True) == 0


def test_motion_state_is_frozen_during_a_tap_then_released():
    spikes = burst(500, 90)
    last = spikes[-1][0]
    _, frozen = run_taps(spikes)
    f = dict(frozen)
    assert not f[450] and f[600]
    assert f[last + int(P["TAP_FREEZE_MS"]) - 2 * DT]
    assert not f[last + int(P["TAP_FREEZE_MS"]) + 2 * DT]


def test_a_cancelled_gesture_releases_the_motion_freeze_at_once():
    _, frozen = run_taps(burst(500, 1500, amp=0.4))
    assert not dict(frozen)[(500 + int(P["TAP_BURST_MAX_MS"]) + 100) // DT * DT]


def test_a_tap_that_rings_for_a_few_samples_is_one_tap():
    out, _ = run_taps([(1500, 0.6), (1515, -0.3), (1530, 0.28), (1545, -0.15)])
    assert [n for _, n in out] == [1]


def test_a_tap_does_not_make_the_classifier_report_motion():
    """The point of the freeze: tap energy must not show up as MOVING/WRITING."""
    rng = random.Random(2)
    det, clf = TapDetector(P), MotionClassifier(P)
    by_idx = {int(t // 10): a for t, a in burst(1000, 400)}
    states = []
    for k in range(500):                       # internal samples every 10 ms; the classifier sees every 2nd one
        t = k * 10
        mag = 1.0 + by_idx.get(k, rng.gauss(0, 0.005))
        det.update(t, 0.0, mag, 0.0, True)
        if k % 2 == 1:
            clf.update(0.0, mag, 0.0, det.frozen(t))
            states.append(clf.state)
    assert set(states) == {L.STILL}


# --------------------------------------------------------------------------- taps vs. writing (the dangerous confusion)
def test_start_of_writing_never_becomes_a_tap_gesture():
    """A still pen, then writing-like bumps start. 'Next question' must never come out of that."""
    for seed in range(30):
        rng = random.Random(seed)
        det = TapDetector(P)
        got = []
        for k in range(600):
            if k < 100:
                mag = 1.0 + rng.gauss(0, 0.005)                    # resting
            else:                                                  # writing-like: bumps of 0.2-0.7 g, many per second
                mag = 1.0 + rng.choice([-1, 1]) * rng.uniform(0.2, 0.7) * (1 if rng.random() < 0.6 else 0.2)
            got.append(det.update(k * 15, 0.0, mag, 0.0, True))
        assert not any(got), f"seed {seed}: writing produced taps {[g for g in got if g]}"


# --------------------------------------------------------------------------- haptics
def freqs(engine, times):
    return [engine.tick(t) for t in times]


def test_lock_plays_two_pulses_then_stops():
    h = HapticEngine(P)
    h.command("LOCK", 0)
    assert freqs(h, [0, 79, 80, 119, 120, 199, 200, 500]) == [
        (1000, 1000), (1000, 1000), (0, 0), (0, 0), (1500, 1500), (1500, 1500), (0, 0), (0, 0)]
    assert h.cmd is None


def test_complete_is_an_ascending_three_note_one_shot():
    h = HapticEngine(P)
    h.command("COMPLETE", 100)
    assert freqs(h, [100, 220, 340, 460]) == [(600, 600), (900, 900), (1200, 1200), (0, 0)]


def test_warn_alternates_buzzers_and_loops_while_refreshed():
    h = HapticEngine(P)
    h.command("WARN", 0)
    seen = []
    for t in range(0, 1400, 100):
        if t % 500 == 0:
            h.command("WARN", t)  # the brain refreshes every 0.5 s
        seen.append(h.tick(t + 1))
    assert seen[:4] == [(2500, 0), (0, 2500), (2500, 0), (0, 2500)]
    assert all(s != (0, 0) for s in seen)


@pytest.mark.parametrize("cmd,on", [("GUIDE_LEFT", (1000, 0)), ("GUIDE_RIGHT", (0, 1000)), ("GUIDE_BOTH", (800, 800))])
def test_guide_patterns_pulse_on_the_right_buzzers(cmd, on):
    h = HapticEngine(P)
    h.command(cmd, 0)
    assert h.tick(10) == on
    assert h.tick(120) == (0, 0)  # the off part of the pulse


def test_dead_man_switch_silences_a_repeating_pattern():
    h = HapticEngine(P)
    h.command("GUIDE_LEFT", 0)
    assert h.tick(50) == (1000, 0)
    assert h.tick(P["HAPTIC_DEADMAN_MS"] + 60) == (0, 0)
    assert h.cmd is None
    h2 = HapticEngine(P)
    h2.command("WARN", 0)
    h2.command("WARN", 1400)                      # a refresh just before expiry keeps it alive
    assert h2.tick(2000) != (0, 0) or h2.cmd == "WARN"
    assert h2.tick(1400 + P["HAPTIC_DEADMAN_MS"] + 10) == (0, 0)


def test_changing_the_command_switches_immediately_and_off_stops_everything():
    h = HapticEngine(P)
    h.command("GUIDE_LEFT", 0)
    h.command("GUIDE_RIGHT", 40)
    assert h.tick(50) == (0, 1000)
    h.command("OFF", 60)
    assert h.tick(70) == (0, 0) and h.cmd is None


def test_one_shot_is_not_restarted_when_resent_and_is_not_cut_by_guidance():
    h = HapticEngine(P)
    h.command("LOCK", 0)
    h.command("LOCK", 100)                         # re-sent mid-play: must not restart
    assert h.tick(130) == (1500, 1500)
    h.command("GUIDE_BOTH", 150)                   # arrives during the one-shot: starts after it
    assert h.tick(190) == (1500, 1500)
    assert h.tick(210) == (800, 800)
    assert h.cmd == "GUIDE_BOTH"


def test_unknown_commands_are_ignored():
    h = HapticEngine(P)
    h.command("BOGUS", 0)
    assert h.tick(10) == (0, 0) and h.cmd is None


