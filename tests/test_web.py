"""Web bridge tests (Dev 4): snapshot schema + semantics, HTTP endpoints, SSE, MJPEG, key injection, safety."""
import http.client
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from src.contracts import Phase, SimClock, sample_layout, sample_questions
from src.factory import build
from src.main import DebugImu, make_system, parse_args
from src.system import System
from src.web import Snapshotter, WebBridge, encode_frame

Q, L = sample_questions(), sample_layout()
TOP_KEYS = {"v", "source", "t", "phase", "motion", "question", "progress", "boxes", "pen", "guidance",
            "haptic", "voice", "imu", "taps", "audit", "events", "health", "debug", "camera"}


def make(debug=False, render=True):
    clock = SimClock()
    comps = build([], clock, Q, L, render=render)
    dbg = None
    if debug:
        dbg = DebugImu(comps.imu, clock)
        comps.imu = dbg
    s = System(comps, clock, Q, L)
    s.start()
    return s, clock, dbg


def run_until(s, clock, snap, stop_t=None, until=None, limit=120.0):
    while clock.now() < (stop_t if stop_t is not None else limit):
        s.step()
        snap.observe()
        clock.advance(0.02)
        if until and until(s):
            break


# ------------------------------------------------------------------ snapshot builder
def test_snapshot_has_the_schema_and_is_strict_json():
    s, clock, _ = make()
    snap = Snapshotter(s)
    run_until(s, clock, snap, stop_t=9.5)
    d = snap.build()
    assert set(d) == TOP_KEYS and d["v"] == 1 and d["source"] == "live"
    json.dumps(d, allow_nan=False)
    assert d["phase"] in {p.value for p in Phase} and d["motion"] in {"STILL", "MOVING", "WRITING", "LIFTED"}
    assert [b["label"] for b in d["boxes"]] == ["Answer 1", "Answer 2", "Answer 3"]
    assert d["question"]["total"] == 3 and d["question"]["id"] == "q1"
    assert {"x", "y", "confidence"} == set(d["pen"]) and 0 <= d["pen"]["x"] <= 1
    assert set(d["haptic"]) == {"cmd", "seq", "age_s", "active"}
    assert d["camera"]["has_frame"] is True and d["debug"] == {"enabled": False, "override": None}
    assert len(d["imu"]["samples"]) > 10 and len(d["imu"]["samples"][0]) == 3


def test_active_box_and_events_are_dedupable():
    s, clock, _ = make()
    snap = Snapshotter(s)
    run_until(s, clock, snap, stop_t=2.0)
    d = snap.build()
    assert [b["id"] for b in d["boxes"] if b["active"]] == ["box1"]
    seqs = [e["seq"] for e in d["events"]]
    assert seqs == sorted(set(seqs)) and any(e["kind"] == "SPEAK" for e in d["events"])


def test_haptic_active_follows_firmware_rules():
    s, clock, _ = make()
    snap = Snapshotter(s)
    run_until(s, clock, snap, until=lambda sys_: snap._haptic[1].startswith("GUIDE"))
    assert snap.build()["haptic"]["active"] is True            # just sent a GUIDE_*
    for _ in range(int(1.6 / 0.02)):                           # no refresh for 1.6 s (brain stopped)
        clock.advance(0.02)
    d = snap.build()
    assert d["haptic"]["active"] is False and d["haptic"]["age_s"] >= 1.6


def test_one_shot_lock_reads_active_only_briefly():
    s, clock, _ = make()
    snap = Snapshotter(s)
    run_until(s, clock, snap, until=lambda sys_: snap._haptic[1] == "LOCK")
    assert snap.build()["haptic"]["active"] is True
    seq = snap.build()["haptic"]["seq"]
    for _ in range(40):
        clock.advance(0.02)
    d = snap.build()
    assert d["haptic"]["active"] is False and d["haptic"]["seq"] == seq


def test_progress_tracks_answered_and_active_then_complete():
    s, clock, _ = make()
    snap = Snapshotter(s)
    run_until(s, clock, snap, until=lambda sys_: sys_.c.brain.phase == Phase.COMPLETE)
    d = snap.build()
    assert d["phase"] == "COMPLETE"
    assert [p["status"] for p in d["progress"]] == ["answered"] * 3
    assert d["audit"] is not None and d["audit"]["ink_present"] is True


def test_skip_is_recorded_as_skipped_for_the_right_question():
    s, clock, _ = make()
    snap = Snapshotter(s)
    run_until(s, clock, snap, until=lambda sys_: sys_.c.brain.phase == Phase.READING)
    s.c.imu._taps_done = 0                                      # no scripted tap interference
    from src.contracts import ImuEvent, Tap
    orig_poll = s.c.imu.poll
    s.c.imu.poll = lambda: orig_poll() + [ImuEvent(clock.now(), tap=Tap.DOUBLE)]
    s.step(); snap.observe()
    s.c.imu.poll = orig_poll
    assert snap.build()["progress"][0]["status"] == "skipped"


def test_taps_are_reported_once_and_expire():
    s, clock, _ = make()
    snap = Snapshotter(s)
    run_until(s, clock, snap, stop_t=36.6)                      # scripted tap 1 at 36.3 s
    assert [t["n"] for t in snap.build()["taps"]] == [1]
    run_until(s, clock, snap, stop_t=39.0)
    assert snap.build()["taps"] == []


def test_voice_level_real_vs_synthetic():
    s, clock, _ = make()
    snap = Snapshotter(s)
    assert snap.build()["voice"]["level_source"] == "synthetic" and snap.build()["voice"]["level"] is None
    s.c.voice.level = lambda: 0.42
    v = snap.build()["voice"]
    assert v["level"] == 0.42 and v["level_source"] == "real"
    s.c.voice.level = lambda: (_ for _ in ()).throw(RuntimeError("meter died"))
    assert snap.build()["voice"]["level_source"] == "synthetic"   # never breaks the snapshot


def test_component_errors_surface_in_health():
    s, clock, _ = make()
    snap = Snapshotter(s)
    s.c.tracker.read = lambda: (_ for _ in ()).throw(RuntimeError("camera unplugged"))
    for _ in range(3):
        s.step(); snap.observe(); clock.advance(0.02)
    d = snap.build()
    assert d["health"]["errors"] == {"tracker": 3} and "camera unplugged" in d["health"]["last_error"]
    assert any(e["kind"] == "ERROR" for e in d["events"])


def test_encode_frame_returns_a_jpeg():
    s, clock, _ = make()
    s.step()
    jpg = encode_frame(s.reading.frame)
    assert jpg[:2] == b"\xff\xd8" and jpg[-2:] == b"\xff\xd9"


# ------------------------------------------------------------------ HTTP
@pytest.fixture
def bridge():
    s, clock, dbg = make(debug=True)
    b = WebBridge(s, dbg, port=0, ui_dir=Path("ui"))
    b.start()
    for _ in range(100):
        s.step(); b.observe(); clock.advance(0.02)
    b.publish()
    yield b
    b.stop()


def get(b, path, **kw):
    return urllib.request.urlopen(f"http://127.0.0.1:{b.port}{path}", timeout=5, **kw)


def test_serves_the_ui_index_and_assets(bridge):
    r = get(bridge, "/")
    body = r.read().decode()
    assert r.status == 200 and "text/html" in r.headers["Content-Type"] and "<html" in body.lower()
    js = get(bridge, "/js/app.js")
    assert "javascript" in js.headers["Content-Type"]
    assert get(bridge, "/fonts/instrument-sans-latin-wght-normal.woff2").read(4) == b"wOF2"


def test_static_server_blocks_path_traversal(bridge):
    for p in ("/../src/web.py", "/..%2fsrc/web.py", "/%2e%2e/src/web.py", "/js/../../README.md"):
        conn = http.client.HTTPConnection("127.0.0.1", bridge.port, timeout=5)
        conn.request("GET", p)
        r = conn.getresponse()
        assert r.status == 404, p
        conn.close()


def test_health_and_snapshot_endpoints(bridge):
    assert json.load(get(bridge, "/api/health"))["ok"] is True
    d = json.load(get(bridge, "/api/snapshot"))
    assert set(d) == TOP_KEYS and d["debug"]["enabled"] is True


def test_sse_streams_snapshots(bridge):
    conn = http.client.HTTPConnection("127.0.0.1", bridge.port, timeout=5)
    conn.request("GET", "/events")
    r = conn.getresponse()
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    line = r.fp.readline().decode()
    assert line.startswith("data: ") and set(json.loads(line[6:])) == TOP_KEYS
    conn.close()


def test_mjpeg_stream_sends_jpeg_parts(bridge):
    conn = http.client.HTTPConnection("127.0.0.1", bridge.port, timeout=5)
    conn.request("GET", "/video.mjpg")
    r = conn.getresponse()
    assert r.status == 200 and "multipart/x-mixed-replace" in r.getheader("Content-Type")
    head = r.fp.readline() + r.fp.readline() + r.fp.readline() + r.fp.readline()
    assert b"--illumin" in head and b"image/jpeg" in head
    assert r.fp.read(2) == b"\xff\xd8"
    conn.close()


def test_mjpeg_is_404_until_a_frame_exists():
    s, clock, dbg = make(render=False)                         # headless fakes: tracker gives no frame
    b = WebBridge(s, None, port=0, ui_dir=Path("ui"))
    b.start()
    try:
        s.step(); b.observe()
        with pytest.raises(urllib.error.HTTPError) as e:
            get(b, "/video.mjpg")
        assert e.value.code == 404
    finally:
        b.stop()


def post(b, path, obj):
    req = urllib.request.Request(f"http://127.0.0.1:{b.port}{path}", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=5)


def test_key_endpoint_drives_debug_imu(bridge):
    assert json.load(post(bridge, "/api/key", {"key": "3"})) == {"ok": True}
    assert bridge.debug.override.value == "WRITING"
    assert json.load(post(bridge, "/api/key", {"key": "t"})) == {"ok": True}
    assert json.load(post(bridge, "/api/key", {"key": "x"})) == {"ok": False}
    assert json.load(post(bridge, "/api/key", {"key": "0"})) == {"ok": True} and bridge.debug.override is None


def test_key_endpoint_is_409_without_debug_keys():
    s, clock, _ = make()
    b = WebBridge(s, None, port=0, ui_dir=Path("ui"))
    b.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            post(b, "/api/key", {"key": "3"})
        assert e.value.code == 409
    finally:
        b.stop()


def test_bad_post_bodies_do_not_crash_the_server(bridge):
    req = urllib.request.Request(f"http://127.0.0.1:{bridge.port}/api/key", data=b"not json", method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 400
    assert json.load(get(bridge, "/api/health"))["ok"] is True


def test_stop_releases_the_port_and_sse_clients():
    s, clock, _ = make()
    b = WebBridge(s, None, port=0, ui_dir=Path("ui"))
    b.start()
    port = b.port
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/events")
    conn.getresponse()
    b.stop()
    with pytest.raises(OSError):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1)


# ------------------------------------------------------------------ CLI
def test_cli_web_flag_parsing():
    assert parse_args([]).web is None
    assert parse_args(["--web"]).web == 8765
    assert parse_args(["--web", "9100"]).web == 9100


def test_web_and_fast_are_incompatible():
    with pytest.raises(SystemExit) as e:
        make_system(parse_args(["--web", "--fast", "--questions", "x", "--layout", "y"]))
    assert "--fast" in str(e.value)


def test_main_serves_then_stops_cleanly(tmp_path):
    """python -m src.main --headless --web: serves while running, exits on --max-seconds."""
    import threading
    from src.main import main
    results = {}

    def probe():
        import time
        for _ in range(50):
            time.sleep(0.1)
            try:
                results["d"] = json.load(urllib.request.urlopen("http://127.0.0.1:18765/api/snapshot", timeout=1))
                return
            except Exception:  # noqa: BLE001
                continue
    t = threading.Thread(target=probe); t.start()
    rc = main(["--headless", "--web", "18765", "--max-seconds", "2.5", "--questions", "x", "--layout", "y"])
    t.join()
    assert rc == 0 and set(results["d"]) == TOP_KEYS


# ------------------------------------------------------------------ the JS demo world must match the bridge
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed (UI parity checks need it)")


def _node(*args):
    out = subprocess.run([NODE, "ui/tests/fake_snapshot.js", *args], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(out.stdout)


def _shape(a, b, path="", errs=None):
    """Compare key sets of every dict present in both; compare the first element of lists."""
    errs = [] if errs is None else errs
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            errs.append(f"{path or '<root>'}: python-only={sorted(set(a) - set(b))} js-only={sorted(set(b) - set(a))}")
        for k in set(a) & set(b):
            _shape(a[k], b[k], f"{path}.{k}", errs)
    elif isinstance(a, list) and isinstance(b, list) and a and b:
        _shape(a[0], b[0], path + "[0]", errs)
    return errs


def _python_snapshot_after_full_flow():
    s, clock, _ = make()
    snap = Snapshotter(s)
    run_until(s, clock, snap, until=lambda sys_: sys_.c.brain.phase == Phase.COMPLETE)
    return snap.build()


@needs_node
def test_js_demo_snapshot_has_exactly_the_bridge_schema():
    py = _python_snapshot_after_full_flow()
    js = _node("snapshot", "20.5")                              # after Q1's audit, pen visible, phase IDLE
    assert set(py) == set(js) == TOP_KEYS
    errs = _shape(py, js)
    assert errs == [], errs


@needs_node
def test_js_demo_snapshot_during_writing_also_matches():
    s, clock, _ = make()
    snap = Snapshotter(s)
    run_until(s, clock, snap, stop_t=12.6)
    errs = _shape(snap.build(), _node("snapshot", "12.6"))
    assert errs == [], errs


@needs_node
def test_js_brain_port_follows_the_reference_brain():
    s, clock, _ = make()
    snap = Snapshotter(s)
    run_until(s, clock, snap, until=lambda sys_: sys_.c.brain.phase == Phase.COMPLETE)
    py_phases = [t for _, k, t in s.log if k == "PHASE"]
    js = _node("flow")
    js_phases = [e["text"] for e in js if e["kind"] == "PHASE"]
    # Both worlds answer q1 and q2 the same way; then Python answers q3 while the JS demo skips it
    # with a double tap. Compare everything up to the start of q3 (the third IDLE).
    def until_q3(phases):
        idles = [i for i, p in enumerate(phases) if p == "IDLE"]
        return phases[: idles[2] + 1]
    assert until_q3(js_phases) == until_q3(py_phases)
    assert js_phases[-1] == "COMPLETE"
    spoken = [e["text"] for e in js if e["kind"] == "SPEAK"]
    assert "Skipping question." in spoken and spoken[-1].startswith("That was the last question")
    assert [e["text"] for e in js if e["kind"] == "HAPTIC"].count("WARN") >= 1


def test_brain_results_hook_overrides_text_inference():
    s, clock, _ = make()
    snap = Snapshotter(s)
    s.c.brain.results = {"q2": "skipped", "bogus": "answered"}
    run_until(s, clock, snap, stop_t=1.0)
    st = {p["id"]: p["status"] for p in snap.build()["progress"]}
    assert st["q2"] == "skipped" and "bogus" not in st


def test_no_cv2_display_falls_back_to_headless_and_keeps_serving(monkeypatch, capsys):
    """WSL / opencv-python-headless: namedWindow raises cv2.error. main must not crash; --web keeps serving."""
    import threading
    import time
    import cv2
    from src.main import main

    def no_gui(*a, **k):
        raise cv2.error("The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support")
    monkeypatch.setattr(cv2, "namedWindow", no_gui)
    got = {}

    def probe():
        for _ in range(60):
            time.sleep(0.1)
            try:
                got["d"] = json.load(urllib.request.urlopen("http://127.0.0.1:18792/api/snapshot", timeout=1))
                return
            except Exception:  # noqa: BLE001
                continue
    th = threading.Thread(target=probe); th.start()
    rc = main(["--web", "18792", "--max-seconds", "2.5", "--questions", "x", "--layout", "y"])   # NOT --headless
    th.join()
    assert rc == 0 and set(got["d"]) == TOP_KEYS
    assert "no display for the cv2 HUD window" in capsys.readouterr().err
