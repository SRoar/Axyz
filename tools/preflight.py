"""Pre-demo hardware check: is everything plugged in and talking, BEFORE the full system starts?

    python tools/preflight.py                 # environment, data files, keys, pen, camera, voice
    python tools/preflight.py --taps          # + asks you to tap once, then twice, and checks both arrive
    python tools/preflight.py --only pen      # just one part: env | files | keys | pen | camera | voice
    python tools/preflight.py --no-voice      # don't speak the test phrase

Nothing waits for keyboard input: every step that needs you (tapping, keeping your hands off the page)
prints what to do and then watches for a few seconds. Camera images are saved to captures/ so you
can check the framing. Exit code 0 = no FAIL (WARNs are fallbacks that still let the demo run).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from typing import Callable, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
RESULTS: List[Tuple[str, str, str]] = []
PARTS = ("env", "files", "keys", "pen", "camera", "voice")


def report(status: str, what: str, detail: str = "") -> None:
    RESULTS.append((status, what, detail))
    print(f"  [{status}] {what}" + (f": {detail}" if detail else ""), flush=True)


def step(title: str) -> None:
    print(f"\n== {title}", flush=True)


def wait_until(cond: Callable[[], bool], timeout: float, poll: float = 0.05) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(poll)
    return cond()


# --------------------------------------------------------------------------- checks
def check_env() -> None:
    step("Python + packages")
    v = sys.version_info
    if v >= (3, 14):
        report(WARN, f"Python {v.major}.{v.minor}", "record3d and pyaudio have no wheels for 3.14 yet: use Python 3.13")
    else:
        report(PASS, f"Python {v.major}.{v.minor}.{v.micro}")
    needs = [("cv2", FAIL, "pip install opencv-python"), ("numpy", FAIL, "pip install numpy"),
             ("serial", FAIL, "pip install pyserial (the pen link)"),
             ("record3d", FAIL, "pip install record3d (the iPhone camera)"),
             ("pyaudio", WARN, "pip install pyaudio (streamed ElevenLabs audio; otherwise the offline voice)"),
             ("elevenlabs", WARN, "pip install elevenlabs (otherwise the offline voice)"),
             ("google.genai", WARN, "pip install google-genai (otherwise the pixel answer check)")]
    for mod, sev, hint in needs:
        try:
            importlib.import_module(mod)
            report(PASS, f"import {mod}")
        except Exception as e:  # noqa: BLE001
            report(sev, f"import {mod}", f"{type(e).__name__}: {e} -> {hint}")


def check_files() -> None:
    step("Data files (the real sheet)")
    from src.contracts import load_layout, load_questions
    try:
        qs = load_questions("data/questions.json")
        layout = load_layout("data/layout.json")
    except Exception as e:  # noqa: BLE001
        report(FAIL, "data/questions.json + data/layout.json", f"{e} -> run: python -m src.prescan")
        return
    missing = [q.box_id for q in qs if q.box_id not in layout]
    if missing:
        report(FAIL, "questions -> boxes", f"no box for {missing} in data/layout.json")
    else:
        report(PASS, "questions -> boxes", f"{len(qs)} questions, {len(layout)} boxes")
    for q in qs:
        print(f"         {q.id} -> {q.box_id}: {q.text}")
    if os.path.exists("data/pen_profile.json"):
        report(PASS, "data/pen_profile.json", "our pen's colours (tip detection)")
    else:
        report(WARN, "data/pen_profile.json", "missing: generic pen model (python tools/capture_pen.py)")
    try:
        with open("data/camera.json") as f:
            src = json.load(f).get("source", "webcam")
        report(PASS if src == "record3d" else WARN, "data/camera.json source", src)
    except Exception as e:  # noqa: BLE001
        report(WARN, "data/camera.json", str(e))


def check_keys() -> None:
    step("API keys (.env)")
    import src.config  # noqa: F401  (loads .env)
    if not os.path.exists(".env"):
        report(WARN, ".env", "missing: copy .env.example to .env and fill in the keys")
    for key, fallback in (("ELEVENLABS_API_KEY", "offline Windows/macOS voice"),
                          ("GEMINI_API_KEY", "pixel-based answer check")):
        val = os.getenv(key, "")
        if val and not val.startswith("your_"):
            report(PASS, key, "set")
        else:
            report(WARN, key, f"not set -> {fallback}")


def check_pen(taps: bool) -> None:
    step("Pen probe (Arduino UNO Q over USB serial)")
    import serial.tools.list_ports as lp
    import src.config  # noqa: F401  (ARDUINO_PORT from .env)
    from src.arduino_link import ArduinoImuLink, find_port
    from src.contracts import HapticCmd, RealClock, Tap

    ports = list(lp.comports())
    for p in ports:
        print(f"         {p.device}: {p.description} (vid={hex(p.vid) if p.vid else '-'})")
    port = find_port()
    if port is None:
        report(FAIL, "serial port", "no Arduino port found. Plug the pen in; close the Arduino IDE Serial Monitor; "
                                     "or set ARDUINO_PORT=COMx in .env" + ("" if ports else " (no serial ports at all)"))
        return
    report(PASS, "serial port", port)

    link = ArduinoImuLink(RealClock(), port=port)
    link.start()
    try:
        print("         connecting (draining old buffered data can take a few seconds)...", flush=True)
        if not link.wait_connected(20.0):
            report(FAIL, "pen link", f"could not open {port}: {link.last_error or 'timeout'} "
                                     "(another program using the port? unplug, wait 60 s, replug)")
            return
        n0, t0 = link.lines_seen, time.time()
        time.sleep(3.0)
        rate = (link.lines_seen - n0) / (time.time() - t0)
        samples = link.recent_samples(100)
        if not samples:
            report(FAIL, "pen data", f"{rate:.0f} lines/s but no S,... samples: is the illumin_pen firmware flashed?")
            return
        report(PASS if rate >= 20 else WARN, "pen stream", f"{rate:.0f} lines/s (expect ~33)")
        s = samples[-1]
        g = (s.ax ** 2 + s.ay ** 2 + s.az ** 2) ** 0.5
        report(PASS if 0.6 < g < 1.4 else WARN, "accelerometer", f"a=({s.ax:+.2f},{s.ay:+.2f},{s.az:+.2f}) g, |a|={g:.2f} "
                                                                 "(about 1 g at rest)")
        report(PASS, "motion state", link.motion.value)
        link.poll()
        link.send(HapticCmd.LOCK)
        report(PASS, "haptic LOCK sent", "you should hear two short beeps (1000 Hz then 1500 Hz)")
        time.sleep(0.8)
        if taps:
            for want, how in ((Tap.SINGLE, "Tap the pen tip on the paper ONCE"), (Tap.DOUBLE, "Now tap TWICE, quickly")):
                link.poll()
                print(f"\n  >>> {how} (watching for 10 s)", flush=True)
                got: List[int] = []
                end = time.time() + 10.0
                while time.time() < end and not got:
                    got = [int(e.tap) for e in link.poll() if e.tap != Tap.NONE]
                    time.sleep(0.05)
                if not got:
                    report(FAIL, f"tap {int(want)}", "nothing arrived (taps are ignored while WRITING and for 1 s after)")
                elif got[0] == int(want):
                    report(PASS, f"tap {int(want)}", f"T,{got[0]} received")
                else:
                    report(WARN, f"tap {int(want)}", f"expected T,{int(want)} but got T,{got[0]}")
                time.sleep(1.5)                 # let the burst detector settle before the next gesture
    finally:
        link.stop()


def check_camera(seconds: float) -> None:
    step("Overhead camera (iPhone via Record3D) + page lock")
    import cv2
    from src.contracts import RealClock
    from src.webcam import load_camera_config, make_camera
    from src.tracker import PenTracker

    cfg = load_camera_config()
    if cfg.source == "record3d":
        try:
            from record3d import Record3DStream
            devs = Record3DStream.get_connected_devices()
        except Exception as e:  # noqa: BLE001
            report(FAIL, "record3d", f"{type(e).__name__}: {e}")
            return
        if not devs:
            report(FAIL, "iPhone", "no Record3D device. Windows: install Apple's driver (the 'Apple Devices' app "
                                   "from the Microsoft Store, or iTunes), plug the iPhone in, unlock it, tap 'Trust', "
                                   "open Record3D, turn on USB Streaming and press the red button")
            return
        report(PASS, "iPhone", f"{len(devs)} Record3D device(s)")

    clock = RealClock()
    cam = make_camera(cfg, time_fn=clock.now)
    tr = PenTracker(clock, camera=cam)          # the tracker starts the camera (exactly once: twice crashes record3d)
    try:
        tr.start()
        if tr.camera_error:
            report(FAIL, "camera stream", tr.camera_error)
            return
        time.sleep(2.0)
        f = cam.latest()
        if f is None:
            report(FAIL, "camera frames", "stream opened but no frame arrived")
            return
        h, w = f.image.shape[:2]
        depth = getattr(f, "depth", None)
        report(PASS, "camera frames", f"{w}x{h} at {cam.fps:.0f} fps")
        report(PASS if depth is not None else WARN, "LiDAR depth",
               f"{depth.shape[1]}x{depth.shape[0]}" if depth is not None else "no depth: the pen end is picked from the hand pose only")
        os.makedirs("captures", exist_ok=True)
        cv2.imwrite("captures/preflight_camera.png", f.image)
        print("         saved captures/preflight_camera.png (the raw view)")
        print(f"\n  >>> Hands OUT of the view, whole sheet visible, keep still: looking for the page ({seconds:.0f} s)", flush=True)
        locked = wait_until(lambda: not tr.calibration_status()[0] and tr.calibration is not None, seconds)
        if not locked:
            report(FAIL, "page lock", f"page not found in {seconds:.0f} s: is the whole sheet in view, flat, well lit, "
                                      "with nothing on it? (the last view is captures/preflight_camera.png)")
            cv2.imwrite("captures/preflight_camera.png", cam.latest().image)
            return
        c = tr.calibration
        report(PASS, "page lock", "corners " + " ".join(f"({x:.0f},{y:.0f})" for x, y in c.corners_px))
        snap = tr.snapshot()
        if snap is not None:
            cv2.imwrite("captures/preflight_page.png", snap)
            print("         saved captures/preflight_page.png (the straightened page the answer check sees)")
        print("\n  >>> Now hold the pen over the page, tip on the paper (watching for 8 s)", flush=True)
        seen = []
        end = time.time() + 8.0
        while time.time() < end:
            r = tr.read()
            if r is not None and r.pen is not None:
                seen.append((r.pen.x, r.pen.y))
            time.sleep(0.05)
        st = tr.stats()
        if seen:
            x, y = seen[-1]
            from src.contracts import PAGE_H_CM, PAGE_W_CM
            report(PASS, "pen tip", f"seen in {len(seen)} of ~160 reads, last at {x * PAGE_W_CM:.1f}, {y * PAGE_H_CM:.1f} cm "
                                    f"from the top-left (tracker {st['tracker_fps']:.0f} fps, {st['proc_ms']:.0f} ms/frame)")
        else:
            report(WARN, "pen tip", "not seen (was a pen in view? python tools/setup_live.py shows the live detection)")
    finally:
        tr.stop()


def check_voice() -> None:
    step("Voice")
    from src.voice import VoiceEngine
    v = VoiceEngine()
    path = "ElevenLabs" if v.client else f"offline {v._offline_name()}"
    t0 = time.time()
    v.speak("Illumin voice check. If you can hear this, the voice works.")
    wait_until(lambda: not v.is_speaking(), 20.0)
    report(PASS, "voice", f"{path}, finished in {time.time() - t0:.1f} s (did you hear it?)")
    v.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=PARTS, action="append", help="run just these parts (repeatable)")
    ap.add_argument("--taps", action="store_true", help="also ask for a single and a double tap")
    ap.add_argument("--no-voice", action="store_true", help="skip the spoken test phrase")
    ap.add_argument("--page-seconds", type=float, default=20.0, help="how long to look for the page")
    a = ap.parse_args()
    parts = a.only or [p for p in PARTS if not (p == "voice" and a.no_voice)]
    for p in parts:
        try:
            {"env": check_env, "files": check_files, "keys": check_keys,
             "pen": lambda: check_pen(a.taps), "camera": lambda: check_camera(a.page_seconds),
             "voice": check_voice}[p]()
        except Exception as e:  # noqa: BLE001 -- one broken check must not hide the others
            report(FAIL, p, f"check crashed: {type(e).__name__}: {e}")

    print("\n== Summary")
    for st, what, detail in RESULTS:
        if st != PASS:
            print(f"  [{st}] {what}: {detail}")
    fails = sum(st == FAIL for st, _, _ in RESULTS)
    warns = sum(st == WARN for st, _, _ in RESULTS)
    print(f"  {len(RESULTS) - fails - warns} pass, {warns} warn, {fails} fail")
    if not fails:
        print("\n  Ready: python -m src.main --real all --debug-keys")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
