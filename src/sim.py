"""
Headless full-system simulation.  RUN THIS BEFORE EVERY MERGE.

    python -m src.sim                      # all fakes, deterministic, instant, asserts the whole flow
    python -m src.sim --real brain         # your real state machine vs fake everything else (still instant)
    python -m src.sim --real imu           # real Arduino link, everything else fake (real time, no asserts)
    python -m src.sim --real imu,tracker,voice,guidance,auditor,brain   # the full real system
    python -m src.sim --quiet              # only print PASS/FAIL
"""
from __future__ import annotations

import argparse
import sys
import time

from src.contracts import Phase, RealClock, SimClock, load_layout, load_questions, sample_layout, sample_questions
from src.factory import build, needs_realtime
from src.system import System

DT = 0.02          # 50 Hz loop
MAX_SIM_S = 90.0


def check(log, final_phase) -> list:
    """The behaviour every implementation of the full flow must show with the scripted world."""
    errs = []
    kinds = [(k, v) for _, k, v in log]
    phases = [v for k, v in kinds if k == "PHASE"]
    haptics = [v for k, v in kinds if k == "HAPTIC"]

    if final_phase != Phase.COMPLETE:
        errs.append(f"did not finish: ended in {final_phase.value}")
    for p in ("READING", "NAVIGATING", "WRITING", "AUDITING"):
        if phases.count(p) < 2:
            errs.append(f"phase {p} seen {phases.count(p)}x, expected >=2 (one per question)")
    if haptics.count("LOCK") < 2:
        errs.append("expected a LOCK haptic for each question")
    if haptics.count("COMPLETE") < 2:
        errs.append("expected a COMPLETE haptic for each question")
    if "WARN" not in haptics:
        errs.append("expected a WARN haptic when the pen drifted out of the box while writing")
    if "GUIDE_BOTH" not in haptics and "GUIDE_RIGHT" not in haptics and "GUIDE_LEFT" not in haptics:
        errs.append("expected GUIDE_* haptics while moving toward the box")

    # voice must be silent while the phase is WRITING
    phase = None
    for _, k, v in log:
        if k == "PHASE":
            phase = v
        elif k == "SPEAK" and phase == "WRITING":
            errs.append(f"brain spoke during WRITING: {v!r}")
    # WARN must never be sent outside the WRITING phase (IMU gating)
    phase = None
    for _, k, v in log:
        if k == "PHASE":
            phase = v
        elif k == "HAPTIC" and v == "WARN" and phase != "WRITING":
            errs.append(f"WARN sent during {phase}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="", help="comma list: imu,tracker,guidance,voice,brain,auditor")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--questions", default="data/questions.json")
    ap.add_argument("--layout", default="",
                    help="layout file; default = sample_layout(), which the fake pen's scripted path targets "
                         "(data/layout.json is the real sheet)")
    a = ap.parse_args()
    real = [x for x in a.real.split(",") if x]

    try:
        questions = load_questions(a.questions)
        layout = load_layout(a.layout) if a.layout else sample_layout()
    except FileNotFoundError:
        questions, layout = sample_questions(), sample_layout()

    realtime = needs_realtime(real)
    clock = RealClock() if realtime else SimClock()
    system = System(build(real, clock, questions, layout, render=False), clock, questions, layout)
    system.start()
    t_start = clock.now()
    try:
        while clock.now() - t_start < MAX_SIM_S and system.c.brain.phase != Phase.COMPLETE:
            system.step()
            if realtime:
                time.sleep(DT)
            else:
                clock.advance(DT)
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()

    if not a.quiet:
        t0 = system.log[0][0] if system.log else 0.0
        for t, kind, text in system.log:
            print(f"{t - t0:7.2f}s  {kind:<8} {text}")

    if realtime:
        print(f"\nfinal phase: {system.c.brain.phase.value}  (real-time run: scripted asserts skipped)")
        return 0
    errs = check(system.log, system.c.brain.phase)
    if errs:
        print("\nSIM FAIL")
        for e in errs:
            print("  -", e)
        return 1
    print(f"\nSIM PASS  (finished in {clock.now() - t_start:.1f}s sim time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
