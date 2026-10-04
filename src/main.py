"""
TactileReader / Illumin -- the one command that runs the whole system.   Owner: Dev 4.

    python -m src.main                                  # HUD on ALL FAKES (scripted world, real time, ~50 s)
    python -m src.main --real imu,brain,voice           # IMU-driven core (Dev 1 + Dev 3 pair)
    python -m src.main --real tracker,guidance,auditor  # camera pair (Dev 2 + Dev 4)
    python -m src.main --real all --debug-keys          # the full demo, keyboard rescue enabled

    --debug-keys   keys 1-4 inject STILL/MOVING/WRITING/LIFTED, t or n = tap 2 (next question),
                   r = tap 1 (read / repeat), 0 = release the override.  There is no third tap:
                   an answer ends when the writing stops for 2 s (key 1 after key 3).
                   A dead sensor can't kill the demo.  (Works with or without a real IMU.)
    --web [PORT]   also serve the browser UI + live data at http://localhost:8765 (see ui/README.md).
                   With --headless it keeps serving until Ctrl-C (add --exit-on-complete to stop sooner).
    --record F.mp4 screen-record the HUD (the backup video).      --headless  no window (CI / soak)
    --fast         fakes only: simulated clock, no sleeping (instant run, deterministic)
    q / ESC / close the window = quit.   s = save a HUD screenshot to captures/.

This replaces the old main.py (it called an undefined draw_hud).  The only place components meet
is System.step(); this file just owns the clock, the loop, the window and shutdown.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from typing import List, Optional

import cv2

from src.contracts import (
    Clock, HapticCmd, ImuEvent, ImuSample, MotionState, Phase, RealClock, SimClock, Tap,
    load_layout, load_questions, sample_layout, sample_questions,
)
from src.factory import REAL, build, needs_realtime, parse_real
from src.system import System

DT = 0.02                   # 50 Hz control loop
DRAW_DT = 1.0 / 30.0        # HUD at 30 fps
WINDOW = "Illumin - TactileReader"
OWNER = {
    "imu": "Dev 1 (src/arduino_link.py)", "tracker": "Dev 2 (src/tracker.py)",
    "guidance": "Dev 2 (src/guidance.py)", "voice": "Dev 3 (src/voice.py)",
    "brain": "Dev 3 (src/state_machine.py)", "auditor": "Dev 4 (src/audit.py)",
}


# =========================================================================== debug key injector
class DebugImu:
    """
    Wraps ANY ImuLink (fake or real).  Keyboard overrides motion + injects taps, so a dead or
    flaky accelerometer cannot kill the demo.  Satisfies the ImuLink protocol.

    While an override is set the wrapped link's own motion edges are dropped (the brain would
    otherwise see two contradicting sources); '0' releases it.  Taps from the real device and
    haptic output pass straight through.
    """

    def __init__(self, inner, clock: Clock) -> None:
        self.inner, self.clock = inner, clock
        self.override: Optional[MotionState] = None
        self._pending: List[ImuEvent] = []

    # ---- ImuLink protocol
    @property
    def motion(self) -> MotionState:
        return self.override if self.override is not None else self.inner.motion

    def start(self) -> None:
        self.inner.start()

    def stop(self) -> None:
        self.inner.stop()

    def poll(self) -> List[ImuEvent]:
        real = self.inner.poll()
        if self.override is not None:
            real = [e for e in real if e.motion is None]    # drop real motion edges, keep taps
        out = real + self._pending
        self._pending = []
        return out

    def send(self, cmd: HapticCmd) -> None:
        self.inner.send(cmd)

    def recent_samples(self, n: int = 200) -> List[ImuSample]:
        return self.inner.recent_samples(n)

    def __getattr__(self, name):          # anything else (haptic_log, ...) comes from the wrapped link
        return getattr(self.inner, name)

    # ---- keyboard
    def inject_motion(self, m: MotionState) -> None:
        if m != self.motion:
            self._pending.append(ImuEvent(self.clock.now(), motion=m))
        self.override = m

    def inject_tap(self, tap: Tap) -> None:
        ev = ImuEvent(self.clock.now(), tap=tap)
        ev.injected = True                  # the HUD labels keyboard taps so they are never mistaken for the pen's
        self._pending.append(ev)

    def release(self) -> None:
        self.override = None

    def handle_key(self, key: int) -> bool:
        """Returns True if the key was a debug key."""
        k = chr(key) if 0 <= key < 256 else ""
        motions = {"1": MotionState.STILL, "2": MotionState.MOVING,
                   "3": MotionState.WRITING, "4": MotionState.LIFTED}
        if k in motions:
            self.inject_motion(motions[k])
        elif k == "t":
            self.inject_tap(Tap.DOUBLE)
        elif k == "r":
            self.inject_tap(Tap.SINGLE)
        elif k == "n":
            self.inject_tap(Tap.DOUBLE)
        elif k == "0":
            self.release()
        else:
            return False
        return True


# =========================================================================== setup
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TactileReader full system + HUD")
    ap.add_argument("--real", default="", help="comma list: imu,tracker,guidance,voice,brain,auditor (or: all)")
    ap.add_argument("--debug-keys", action="store_true", help="keyboard injection of motion states / taps")
    ap.add_argument("--freeze-page", action="store_true",
                    help="camera is physically fixed (taped down, top-down): lock the page/box "
                         "positions after the initial scan instead of continuously re-evaluating "
                         "them, which is sensitive to corner-detector jitter. With --debug-keys, "
                         "'c' forces a fresh re-scan (e.g. the rig got bumped).")
    ap.add_argument("--questions", default=None,
                    help="default: data/questions.json with a real tracker, else the built-in sample questions")
    ap.add_argument("--layout", default=None,
                    help="default: data/layout.json (the real sheet) with a real tracker, else sample_layout(), "
                         "which the fake pen's scripted path targets")
    ap.add_argument("--record", default=None, metavar="FILE.mp4", help="record the HUD to a video file")
    ap.add_argument("--headless", action="store_true", help="no window; exit when the test completes")
    ap.add_argument("--fast", action="store_true", help="fakes only: simulated clock, no sleeping")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="stop after N seconds (0 = no limit)")
    ap.add_argument("--exit-on-complete", action="store_true", help="close the window ~4 s after COMPLETE")
    ap.add_argument("--web", nargs="?", const=8765, default=None, type=int, metavar="PORT",
                    help="serve the browser UI + live data (default port 8765)")
    ap.add_argument("--web-host", default="127.0.0.1",
                    help="bind address; use 0.0.0.0 to open the UI from a phone/tablet on the same Wi-Fi")
    ap.add_argument("--open-ui", action="store_true", help="open the UI in the default browser (needs --web)")
    return ap.parse_args(argv)


def load_data(qpath: str, lpath: str):
    try:
        return load_questions(qpath), load_layout(lpath)
    except FileNotFoundError:
        print(f"[main] {qpath} / {lpath} not found -> using the built-in sample questions + layout")
        return sample_questions(), sample_layout()


def make_system(args: argparse.Namespace):
    """-> (system, clock, debug_imu_or_None).  Raises SystemExit with a human message on bad setup."""
    real = parse_real(args.real)
    unknown = set(real) - set(REAL)
    if unknown:
        raise SystemExit(f"unknown component(s) {sorted(unknown)}; choose from {sorted(REAL)} or 'all'")
    realtime = needs_realtime(real)
    if args.fast and realtime:
        raise SystemExit("--fast only works with fakes (real hardware needs the wall clock)")
    if args.fast and args.web is not None:
        raise SystemExit("--web streams in real time; drop --fast")
    clock: Clock = SimClock() if args.fast else RealClock()
    if args.questions or args.layout or "tracker" in real:
        questions, layout = load_data(args.questions or "data/questions.json", args.layout or "data/layout.json")
    else:
        questions, layout = sample_questions(), sample_layout()
    try:
        comps = build(real, clock, questions, layout,
                      render=not args.headless or bool(args.record) or args.web is not None)
    except ImportError as e:
        missing = [n for n in real if REAL[n][0] == getattr(e, "name", None)]
        who = ", ".join(OWNER[n] for n in missing) or "the owner of that module"
        raise SystemExit(f"cannot import a real component: {e}\n -> is {who} merged into this branch yet?")
    if args.freeze_page and hasattr(comps.tracker, "freeze_after_lock"):
        comps.tracker.freeze_after_lock = True
    debug = None
    if args.debug_keys:
        debug = DebugImu(comps.imu, clock)
        comps.imu = debug
    return System(comps, clock, questions, layout), clock, debug


# =========================================================================== loop
def run(args: argparse.Namespace) -> int:
    from src.hud import H, W, Hud

    system, clock, debug = make_system(args)
    hud = Hud(system, debug)
    bridge = None
    if args.web is not None:
        from src.web import WebBridge
        try:
            bridge = WebBridge(system, debug, host=args.web_host, port=args.web)
            bridge.start()
        except OSError as e:
            raise SystemExit(f"cannot start the web UI on {args.web_host}:{args.web}: {e}\n"
                             f" -> is another run still going? try --web {args.web + 1}")
        print(f"[main] UI: {bridge.url}   (live data{'; debug keys enabled' if debug else ''})")
        if args.open_ui:
            import webbrowser
            webbrowser.open(bridge.url)
    show = not args.headless
    realtime = not args.fast
    writer = None
    running = True

    def _stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if args.record:
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
        if not writer.isOpened():
            print(f"[main] cannot open {args.record} for writing; recording disabled", file=sys.stderr)
            writer = None
    if show:
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1280, 720)
        except cv2.error:
            # opencv-python-headless, or WSL/SSH without a display: no cv2 window is possible.
            # Carry on exactly as if --headless was given (the web UI keeps serving).
            show = False
            args.headless = True
            hint = f"open {bridge.url}" if bridge is not None else "add --web to get a browser UI"
            print(f"[main] no display for the cv2 HUD window; running headless ({hint})", file=sys.stderr)

    print(f"[main] real={sorted(parse_real(args.real)) or 'none (all fakes)'}  debug_keys={args.debug_keys}  "
          f"{'fast/sim clock' if args.fast else 'real time'}.  q / ESC to quit.")
    system.start()
    t_begin = clock.now()
    next_loop = time.monotonic()
    next_draw = clock.now()
    complete_at: Optional[float] = None
    hud_ok = True
    try:
        while running:
            system.step()
            hud.observe()
            if bridge is not None:
                bridge.observe()
            now = clock.now()

            if now >= next_draw:
                next_draw = now + DRAW_DT
                if hud_ok and (show or writer):
                    try:
                        frame = hud.render()
                        if writer is not None:
                            writer.write(frame)
                        if show:
                            cv2.imshow(WINDOW, frame)
                    except Exception as e:  # noqa: BLE001 -- a HUD bug must never stop the demo
                        import traceback
                        traceback.print_exc()
                        print(f"[main] HUD disabled after error: {e}", file=sys.stderr)
                        hud_ok = False
                if show:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if debug is not None and key != 255:
                        debug.handle_key(key)
                    if key == ord("s") and hud_ok:
                        os.makedirs("captures", exist_ok=True)
                        p = f"captures/hud_{int(time.time())}.png"
                        cv2.imwrite(p, hud.render())
                        print(f"[main] saved {p}")
                    if key == ord("c") and hasattr(system.c.tracker, "recalibrate"):
                        system.c.tracker.recalibrate()
                        print("[main] forcing a fresh page re-scan (hands out of view)")
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break

            if system.c.brain.phase == Phase.COMPLETE:
                complete_at = complete_at if complete_at is not None else now
                serve_on = bridge is not None and args.headless and not args.exit_on_complete
                if (args.headless or args.exit_on_complete) and not serve_on \
                        and now - complete_at > (0.5 if args.headless else 4.0):
                    break
            if args.max_seconds and now - t_begin >= args.max_seconds:
                break

            if realtime:
                next_loop += DT
                lag = next_loop - time.monotonic()
                if lag > 0:
                    time.sleep(lag)
                elif lag < -0.5:                  # fell far behind (debugger, slow HUD): don't spiral
                    next_loop = time.monotonic()
            else:
                clock.advance(DT)
    finally:
        if bridge is not None:
            bridge.stop()
        system.stop()
        if writer is not None:
            writer.release()
            print(f"[main] recording saved to {args.record}")
        if show:
            cv2.destroyAllWindows()

    print(f"[main] finished in phase {system.c.brain.phase.value}; component errors: {system.errors or 'none'}")
    return 0 if not system.errors else 1


def main(argv: Optional[List[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
