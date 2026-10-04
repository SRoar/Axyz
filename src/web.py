"""
Web bridge: streams the live System state to the browser UI in ui/.     Owner: Dev 4.

    python -m src.main --web                 # serve http://127.0.0.1:8765  (UI + live data)
    python -m src.main --web 9000 --real all --debug-keys
    python -m src.main --headless --web      # no cv2 window; keep serving until Ctrl-C

Zero new dependencies: stdlib http.server + the cv2/numpy the project already uses.
Read-only on the System, except POST /api/key which forwards to DebugImu (--debug-keys).

Endpoints
    GET  /                 ui/index.html            (any file under ui/ is served; no directory escapes)
    GET  /events           text/event-stream        one full JSON snapshot per message (~20 Hz)
    GET  /api/snapshot     application/json         the latest snapshot (polling / debugging / tests)
    GET  /video.mjpg       multipart MJPEG          the tracker frame (404 until a frame exists)
    GET  /api/health       {"ok": true}
    POST /api/key          {"key": "3"}             same keys as --debug-keys (409 if not enabled)

SNAPSHOT SCHEMA v1  (ui/js/sources.js builds the SAME shape from its fake world;
tests/test_web.py compares the two key-sets so they cannot drift)

    v            1
    source       "live"
    t            System clock seconds (float)
    phase        Phase value: IDLE READING NAVIGATING WRITING AUDITING COMPLETE
    motion       MotionState value: STILL MOVING WRITING LIFTED
    question     {id, text, index (0-based), total, box_id} | null
    progress     [{id, status}]   status: pending | active | answered | skipped | unchecked
                 Taken from `brain.results` ({question_id: status}) when the brain exposes it; otherwise
                 inferred from the audit log + the brain's spoken phrases ("Skipping question.", ...).
    boxes        [{id, label, question_id, xmin, ymin, xmax, ymax, active}]   page-normalized
    pen          {x, y, confidence} | null
    guidance     {cmd, speech, dx_cm, dy_cm, dist_cm, in_box, write_status, pen_visible}
    haptic       {cmd, seq, age_s, active}     cmd = last HapticCmd; seq++ per command;
                 active = buzzers sounding NOW (firmware rules: GUIDE_*/WARN die 1.5 s after the
                 last refresh, LOCK/COMPLETE are one-shots, OFF is silence)
    voice        {speaking, level, level_source, text}   level 0..1 or null; "real" | "synthetic"
    imu          {samples: [[ax, ay, az], ...]}   (accelerometer only)
    taps         [{age_s, n}]                  taps in the last 1.6 s
    tap_stats    {single, double, keyboard, last: {n, age_s, keyboard} | null}   every tap since start
                 (single/double = reported by the pen; keyboard = injected with --debug-keys)
    pen_link     {kind, connected, port, lines_per_s, error}   kind: "serial" (real pen) | "fake"
    audit        {question_id, ink_present, ink_outside, confidence, note, age_s} | null
    events       [{seq, t, kind, text}]        newest 40 log lines (PHASE SPEAK SILENCE HAPTIC
                 BOX SNAPSHOT AUDIT ERROR); the UI dedupes by seq
    health       {errors: {component: n}, last_error}
    debug        {enabled, override}
    camera       {has_frame, label, rectified, page_found, page_search}   page_found = page calibrated
                 (frame is the straightened page); page_search = 0..1 progress while looking for it
"""
from __future__ import annotations

import json
import mimetypes
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.contracts import HapticCmd, Phase, Tap
from src.system import System

SCHEMA_VERSION = 1
DEFAULT_PORT = 8765
UI_DIR = Path(__file__).resolve().parents[1] / "ui"

GUIDE_OR_WARN = {"GUIDE_LEFT", "GUIDE_RIGHT", "GUIDE_BOTH", "WARN"}
ONE_SHOT_SHOW_S = {"LOCK": 0.5, "COMPLETE": 0.5}      # how long a one-shot reads as "buzzing"
HAPTIC_DEADMAN_S = 1.5                                # firmware dead-man for GUIDE_*/WARN
TAP_SHOW_S = 1.6
EVENT_KEEP = 40
IMU_SAMPLES = 80


def _r(x: float, n: int = 3) -> float:
    return round(float(x), n)


class Snapshotter:
    """Turns System state into the schema above.  No I/O; call observe() once per tick, build() any time."""

    def __init__(self, system: System, debug: Any = None) -> None:
        self.s = system
        self.debug = debug
        self._log_i = 0
        self._seq = 0
        self._events: Deque[Dict[str, Any]] = deque(maxlen=EVENT_KEEP)
        self._taps: Deque[Tuple[float, int]] = deque(maxlen=8)
        self._tap_counts = {1: 0, 2: 0}
        self._key_taps = 0
        self._last_tap: Optional[Tuple[float, int, bool]] = None
        self._rate = (None, 0, 0.0)                     # (t, lines_seen, lines/s) for the pen link
        self._last_inputs_t: Optional[float] = None
        self._haptic: Tuple[float, str] = (-1e9, "OFF")
        self._haptic_seq = 0
        self._status: Dict[str, str] = {qid: "pending" for qid in system.question_order}
        self._prev_index = 0
        self._frame: Any = None
        self._frame_id = 0
        self._voice_text = ""

    # ------------------------------------------------------------------ per tick
    def observe(self) -> None:
        s = self.s
        now = s.clock.now()
        inp = s.inputs
        if inp is not None and inp.t != self._last_inputs_t:
            self._last_inputs_t = inp.t
            for e in inp.imu_events:
                if e.tap != Tap.NONE:
                    n, typed = int(e.tap), bool(getattr(e, "injected", False))
                    self._taps.append((now, n))
                    self._last_tap = (now, n, typed)
                    if typed:
                        self._key_taps += 1
                    else:
                        self._tap_counts[n] = self._tap_counts.get(n, 0) + 1
        seen = getattr(s.c.imu, "lines_seen", None)
        if seen is not None:
            t0, n0, rate = self._rate
            if t0 is None:
                self._rate = (now, seen, 0.0)
            elif now - t0 >= 1.0:
                self._rate = (now, seen, (seen - n0) / (now - t0))
        rd = s.reading
        if rd is not None and rd.frame is not None:
            self._frame, self._frame_id = rd.frame, self._frame_id + 1

        index = s.c.brain.index
        while self._log_i < len(s.log):
            t, kind, text = s.log[self._log_i]
            self._log_i += 1
            self._seq += 1
            self._events.append({"seq": self._seq, "t": _r(t), "kind": kind, "text": text})
            if kind == "HAPTIC":
                self._haptic, self._haptic_seq = (t, text), self._haptic_seq + 1
            elif kind == "SPEAK":
                self._voice_text = text
                prev = self._question_id_at(self._prev_index)
                if text.startswith("Skipping question") and prev:
                    self._status[prev] = "skipped"
                elif text.startswith("I could not check") and prev:
                    self._status[prev] = "unchecked"
            elif kind == "AUDIT":
                qid, _, rest = text.partition(":")
                if qid in self._status and "ink=True" in rest:
                    self._status[qid] = "answered"
        self._prev_index = index

    def _question_id_at(self, i: int) -> Optional[str]:
        order = self.s.question_order
        return order[i] if 0 <= i < len(order) else None

    @property
    def frame(self) -> Tuple[Any, int]:
        return self._frame, self._frame_id

    # ------------------------------------------------------------------ build
    def build(self) -> Dict[str, Any]:
        s = self.s
        c = s.c
        now = s.clock.now()
        inp = s.inputs
        phase = c.brain.phase
        q = c.brain.question
        order = s.question_order
        cur_id = q.id if q else None
        progress = []
        results = getattr(c.brain, "results", None)         # optional Dev 3 hook: {question_id: "answered"|"skipped"|"unchecked"}
        if isinstance(results, dict):
            self._status.update({k: v for k, v in results.items() if k in self._status})
        for qid in order:
            st = self._status.get(qid, "pending")
            if st == "pending" and qid == cur_id and phase != Phase.COMPLETE:
                st = "active"
            progress.append({"id": qid, "status": st})

        box_q = {qq.box_id: (i, qq.id) for i, qq in enumerate(s.questions.values())}
        boxes = []
        for b in s.layout.values():
            i, qid = box_q.get(b.id, (len(boxes), ""))
            boxes.append({"id": b.id, "label": f"Answer {i + 1}", "question_id": qid,
                          "xmin": _r(b.xmin, 4), "ymin": _r(b.ymin, 4), "xmax": _r(b.xmax, 4),
                          "ymax": _r(b.ymax, 4),
                          "active": bool(s.current_box is not None and s.current_box.id == b.id)})

        pen = inp.pen if inp else None
        gd = inp.guidance if inp else None
        guidance = {
            "cmd": gd.cmd.value if gd and gd.cmd else None,
            "speech": gd.speech if gd else None,
            "dx_cm": _r(gd.dx_cm, 2) if gd else 0.0, "dy_cm": _r(gd.dy_cm, 2) if gd else 0.0,
            "dist_cm": _r(gd.dist_cm, 2) if gd else 0.0,
            "in_box": bool(gd.in_box) if gd else False,
            "write_status": gd.write_status.value if gd else "UNKNOWN",
            "pen_visible": bool(gd.pen_visible) if gd else False,
        }

        h_t, h_cmd = self._haptic
        age = now - h_t
        if h_cmd in ONE_SHOT_SHOW_S:
            active = age < ONE_SHOT_SHOW_S[h_cmd]
        elif h_cmd in GUIDE_OR_WARN:
            active = age < HAPTIC_DEADMAN_S
        else:
            active = False

        level, level_source = self._voice_level()
        samples = self._samples()
        la = s.last_audit
        audit = None
        if la is not None:
            t_a, r = la
            audit = {"question_id": r.question_id, "ink_present": bool(r.ink_present),
                     "ink_outside": bool(r.ink_outside), "confidence": _r(r.confidence, 2),
                     "note": r.note, "age_s": _r(now - t_a, 2)}

        return {
            "v": SCHEMA_VERSION, "source": "live", "t": _r(now),
            "phase": phase.value,
            "motion": (inp.motion.value if inp else "STILL"),
            "question": ({"id": q.id, "text": q.text, "index": c.brain.index, "total": len(order),
                          "box_id": q.box_id} if q else None),
            "progress": progress,
            "boxes": boxes,
            "pen": ({"x": _r(pen.x, 4), "y": _r(pen.y, 4), "confidence": _r(pen.confidence, 2)}
                    if pen else None),
            "guidance": guidance,
            "haptic": {"cmd": h_cmd, "seq": self._haptic_seq, "age_s": _r(min(age, 999.0), 2),
                       "active": bool(active)},
            "voice": {"speaking": bool(inp.speaking) if inp else False, "level": level,
                      "level_source": level_source, "text": self._voice_text},
            "imu": {"samples": samples},
            "taps": [{"age_s": _r(now - t), "n": n} for t, n in self._taps if now - t < TAP_SHOW_S],
            "tap_stats": {"single": self._tap_counts.get(1, 0), "double": self._tap_counts.get(2, 0),
                          "keyboard": self._key_taps,
                          "last": ({"n": self._last_tap[1], "age_s": _r(now - self._last_tap[0], 1),
                                    "keyboard": self._last_tap[2]} if self._last_tap else None)},
            "pen_link": self._pen_link(),
            "audit": audit,
            "events": list(self._events),
            "health": {"errors": dict(s.errors), "last_error": s.last_error},
            "debug": {"enabled": self.debug is not None,
                      "override": (getattr(getattr(self.debug, "override", None), "value", None)
                                   if self.debug is not None else None)},
            "camera": self._camera(),
        }

    # ------------------------------------------------------------------ helpers
    def _pen_link(self) -> Dict[str, Any]:
        imu = self.s.c.imu
        connected = getattr(imu, "connected", None)
        if connected is None:
            return {"kind": "fake", "connected": True, "port": "", "lines_per_s": 0.0, "error": ""}
        return {"kind": "serial", "connected": bool(connected), "port": str(getattr(imu, "port_name", "") or ""),
                "lines_per_s": _r(self._rate[2], 1), "error": str(getattr(imu, "last_error", "") or "")}

    def _camera(self) -> Dict[str, Any]:
        tr = self.s.c.tracker
        found, progress = True, 1.0
        fn = getattr(tr, "calibration_status", None)
        if callable(fn):                                   # real tracker: is the page calibrated yet?
            try:
                searching, prog, _ = fn()
                found = getattr(tr, "calibration", None) is not None
                progress = 1.0 if found and not searching else float(prog)
            except Exception:  # noqa: BLE001
                pass
        return {"has_frame": self._frame is not None,
                "label": str(getattr(tr, "source_name", "Overhead camera")),
                "rectified": found, "page_found": found, "page_search": _r(progress, 2)}

    def _voice_level(self) -> Tuple[Optional[float], str]:
        """Optional Voice.level() (0..1, RMS of the audio playing now).  Dev 3 may add it; the UI
        synthesises a speech-like envelope from `speaking` until then."""
        v = self.s.c.voice
        fn = getattr(v, "level", None)
        if fn is None:
            return None, "synthetic"
        try:
            val = fn() if callable(fn) else fn
            return (max(0.0, min(1.0, float(val))), "real") if val is not None else (None, "synthetic")
        except Exception:  # noqa: BLE001 -- a broken meter must not break the snapshot
            return None, "synthetic"

    def _samples(self) -> List[List[float]]:
        try:
            return [[_r(x.ax), _r(x.ay), _r(x.az)] for x in self.s.c.imu.recent_samples(IMU_SAMPLES)]
        except Exception:  # noqa: BLE001
            return []


def encode_frame(frame: Any, max_w: int = 960, quality: int = 80) -> Optional[bytes]:
    """BGR numpy frame -> JPEG bytes (downscaled so a laptop can serve several viewers)."""
    import cv2
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


# =========================================================================== HTTP
class WebBridge:
    def __init__(self, system: System, debug: Any = None, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT, ui_dir: Optional[Path] = None, hz: float = 20.0) -> None:
        self.snap = Snapshotter(system, debug)
        self.debug = debug
        self.host, self._port, self.hz = host, port, hz
        self.ui_dir = Path(ui_dir) if ui_dir else UI_DIR
        self._cond = threading.Condition()
        self._json = "{}"
        self._version = 0
        self._last_build = 0.0
        self._stopping = False
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ lifecycle
    @property
    def port(self) -> int:
        return self._server.server_address[1] if self._server else self._port

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("127.0.0.1", "0.0.0.0") else self.host
        return f"http://{host}:{self.port}/"

    def start(self) -> None:
        bridge = self

        class Handler(_Handler):
            pass
        Handler.bridge = bridge
        self._server = ThreadingHTTPServer((self.host, self._port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="web-bridge")
        self._thread.start()
        self.publish()                      # something to serve before the first tick

    def stop(self) -> None:
        self._stopping = True
        with self._cond:
            self._cond.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # ------------------------------------------------------------------ per tick (main thread)
    def observe(self) -> None:
        self.snap.observe()
        now = time.monotonic()
        if now - self._last_build >= 1.0 / self.hz:
            self.publish()

    def publish(self) -> None:
        self._last_build = time.monotonic()
        data = json.dumps(self.snap.build(), separators=(",", ":"), allow_nan=False)
        with self._cond:
            self._json = data
            self._version += 1
            self._cond.notify_all()

    # ------------------------------------------------------------------ used by handlers
    def latest(self) -> str:
        with self._cond:
            return self._json

    def wait_new(self, seen: int, timeout: float = 1.0) -> Tuple[int, str]:
        with self._cond:
            if self._version == seen and not self._stopping:
                self._cond.wait(timeout)
            return self._version, self._json

    def press_key(self, key: str) -> bool:
        if self.debug is None or not key:
            return False
        return bool(self.debug.handle_key(ord(key[0])))


class _Handler(BaseHTTPRequestHandler):
    bridge: WebBridge
    server_version = "IlluminBridge/1"

    def log_message(self, *a: Any) -> None:   # quiet: the 50 Hz loop owns the console
        pass

    # ---- helpers
    def _send(self, code: int, body: bytes, ctype: str, extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json_reply(self, code: int, obj: Any) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    # ---- routes
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/events":
            return self._sse()
        if path == "/api/snapshot":
            return self._send(200, self.bridge.latest().encode(), "application/json")
        if path == "/api/health":
            return self._json_reply(200, {"ok": True, "schema": SCHEMA_VERSION})
        if path == "/video.mjpg":
            return self._mjpeg()
        return self._static(path)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/key":
            return self._json_reply(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            key = str(json.loads(self.rfile.read(n) or b"{}").get("key", ""))
        except (ValueError, TypeError):
            return self._json_reply(400, {"error": "expected JSON {\"key\": \"3\"}"})
        if self.bridge.debug is None:
            return self._json_reply(409, {"error": "start with --debug-keys to enable key injection"})
        return self._json_reply(200, {"ok": self.bridge.press_key(key)})

    def _static(self, path: str) -> None:
        root = self.bridge.ui_dir.resolve()
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (root / rel).resolve()
        if root != target and root not in target.parents:       # no ../ escapes
            return self._json_reply(404, {"error": "not found"})
        if not target.is_file():
            return self._json_reply(404, {"error": "not found"})
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix == ".js":
            ctype = "text/javascript"
        if ctype.startswith("text/") or ctype in ("application/json",):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        seen = -1
        try:
            while not self.bridge._stopping:
                seen, data = self.bridge.wait_new(seen, timeout=1.0)
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _mjpeg(self) -> None:
        frame, _ = self.bridge.snap.frame
        if frame is None:
            return self._json_reply(404, {"error": "no camera frame yet"})
        boundary = "illumin"
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True
        last_id = -1
        try:
            while not self.bridge._stopping:
                frame, fid = self.bridge.snap.frame
                if frame is not None and fid != last_id:
                    last_id = fid
                    jpg = encode_frame(frame)
                    if jpg:
                        self.wfile.write(f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                                         f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                time.sleep(1.0 / 20.0)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
