"""
Fast threaded camera capture for the overhead camera (Logitech Brio 101 by default).  Owner: Dev 2.

Why a USB webcam delivers fewer frames than it claims, and what this module does about each:
  1. Pixel format: uncompressed YUY2 at 720p/1080p is capped at ~5-10 FPS by USB bandwidth.
     -> request MJPG (before AND after the resolution; some DirectShow drivers reset it).
  2. Backend: on Windows MSMF opens slowly and ignores most settings -> DirectShow.
  3. Auto-exposure in indoor light stretches the exposure past 33 ms and halves the frame rate.
     -> optional manual exposure, chosen by `tools/camera_check.py --bench` and saved to
        data/camera.json.
  4. Virtual cameras (NVIDIA Broadcast, OBS, ...) re-process every frame, add latency and often
     hold the physical camera exclusively -> select the physical camera by NAME.
  5. Reading in the processing loop drops frames whenever processing is slow and returns stale,
     buffered frames -> a dedicated grab thread that only ever keeps the NEWEST frame.

Every Dev 2 tool (camera_check, tune_marker, calibrate_page, prescan) and the PenTracker open the
camera through `make_camera(load_camera_config())`, so the settings tuned once are used everywhere.
"""
from __future__ import annotations

import json
import os
import platform
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")   # MSMF open hangs otherwise
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

CAMERA_CFG_PATH = "data/camera.json"

VIRTUAL_CAMERA_HINTS = ("nvidia broadcast", "obs", "virtual", "snap camera", "xsplit", "manycam")
BUSY_HINT = (
    "The camera is probably held by another program. Only one program can use a webcam at a time:\n"
    "  close NVIDIA Broadcast (or switch its camera source away from the Logitech), Teams, Zoom,\n"
    "  Discord, OBS and the Windows Camera app, then try again.\n"
    "  Run `python tools/camera_check.py --list` to see the camera names and indexes."
)

ExposureSetting = Union[None, str, float]   # None = don't touch, "auto", or a manual value


class CameraError(RuntimeError):
    pass


@dataclass
class CameraConfig:
    name: str = "Brio|Logitech"   # regex matched against device names (physical cameras only); "" = use index
    index: int = 1                # fallback when no name matches / names can't be listed
    backend: str = "auto"         # auto | dshow | msmf | avfoundation | v4l2 | any
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"          # "" = driver default
    buffer_size: int = 1
    exposure: ExposureSetting = None   # DirectShow: log2(seconds), -6 ~ 16 ms, -7 ~ 8 ms
    gain: Optional[float] = None       # brightens a short manual exposure (Logitech: 0-255)
    autofocus: Optional[bool] = None
    source: str = "webcam"        # webcam | record3d

    def replace(self, **kw: Any) -> "CameraConfig":
        d = asdict(self)
        d.update({k: v for k, v in kw.items() if v is not None})
        return CameraConfig(**d)


def load_camera_config(path: str = CAMERA_CFG_PATH, **overrides: Any) -> CameraConfig:
    cfg = CameraConfig()
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in fields(CameraConfig)}
        cfg = CameraConfig(**{k: v for k, v in data.items() if k in known})
    return cfg.replace(**overrides)


def save_camera_config(cfg: CameraConfig, path: str = CAMERA_CFG_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


# --------------------------------------------------------------------------- backends / devices
_BACKENDS = {
    "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "avfoundation": cv2.CAP_AVFOUNDATION,
    "v4l2": cv2.CAP_V4L2, "any": cv2.CAP_ANY,
}


def backend_id(name: str = "auto") -> int:
    if name != "auto":
        return _BACKENDS[name]
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def backend_name(api: int) -> str:
    return next((k for k, v in _BACKENDS.items() if v == api), str(api))


def list_devices(api: Optional[int] = None) -> List[Tuple[int, str]]:
    """(index, name) for every camera the backend can see. Empty if names can't be listed."""
    api = backend_id() if api is None else api
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError:
        return []
    try:
        out = []
        for c in enumerate_cameras(api):
            idx = c.index - api if c.index >= api > 0 else c.index
            out.append((idx, c.name or f"camera {idx}"))
        return out
    except Exception:
        return []


def is_virtual(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in VIRTUAL_CAMERA_HINTS)


def resolve_device(cfg: CameraConfig, api: int) -> Tuple[int, str]:
    devices = list_devices(api)
    if cfg.name:
        pat = re.compile(cfg.name, re.IGNORECASE)
        for idx, name in devices:
            if pat.search(name) and not is_virtual(name):
                return idx, name
    name = next((n for i, n in devices if i == cfg.index), f"index {cfg.index}")
    if cfg.name and devices:
        print(f"[webcam] no physical camera matches {cfg.name!r}; using {name} (index {cfg.index}). "
              f"Seen: {devices}")
    if is_virtual(name):
        print(f"[webcam] WARNING: {name} is a virtual camera: expect added latency and capped FPS.")
    return cfg.index, name


def fourcc_str(value: float) -> str:
    raw = (int(value) & 0xFFFFFFFF).to_bytes(4, "little")
    s = raw.decode("ascii", errors="ignore").strip("\x00 ")
    return s if len(s) == 4 and s.isprintable() else "?"


def set_exposure(cap: cv2.VideoCapture, api: int, exposure: ExposureSetting) -> None:
    """Exposure flags differ per backend: DirectShow uses 1/0 (auto/manual), V4L2 uses 3/1."""
    if exposure is None:
        return
    auto_on, auto_off = (1, 0) if api in (cv2.CAP_DSHOW, cv2.CAP_MSMF) else (3, 1)
    if exposure == "auto":
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_on)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_off)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))


def open_capture(cfg: CameraConfig, verbose: bool = True) -> Tuple[cv2.VideoCapture, Dict[str, Any]]:
    api = backend_id(cfg.backend)
    idx, name = resolve_device(cfg, api)
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(idx, api)
    if not cap.isOpened():
        cap.release()
        raise CameraError(f"could not open {name} (index {idx}, {backend_name(api)}).\n{BUSY_HINT}")
    four = cv2.VideoWriter_fourcc(*cfg.fourcc) if len(cfg.fourcc) == 4 else None
    if four is not None:
        cap.set(cv2.CAP_PROP_FOURCC, four)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    if four is not None and fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)) != cfg.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, four)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size)
    set_exposure(cap, api, cfg.exposure)
    if cfg.gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, float(cfg.gain))
    if cfg.autofocus is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if cfg.autofocus else 0)
    info = {
        "device": name, "index": idx, "backend": backend_name(api),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fourcc": fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)),
        "fps_claim": cap.get(cv2.CAP_PROP_FPS),
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "gain": cap.get(cv2.CAP_PROP_GAIN),
        "open_s": round(time.perf_counter() - t0, 2),
    }
    if verbose:
        print(f"[webcam] {info['device']} idx={idx} {info['backend']} {info['width']}x{info['height']} "
              f"{info['fourcc']} claim={info['fps_claim']:.0f}fps exposure={info['exposure']} "
              f"(opened in {info['open_s']}s)")
        if cfg.fourcc and info["fourcc"] not in (cfg.fourcc, "?"):
            print(f"[webcam] WARNING: asked for {cfg.fourcc}, driver gave {info['fourcc']}: FPS may be capped.")
    return cap, info


# --------------------------------------------------------------------------- frames
@dataclass
class Frame:
    id: int
    t: float              # capture time from the camera's time_fn (the system Clock)
    image: np.ndarray     # BGR


class _FrameSource:
    """Shared newest-frame / FPS bookkeeping for every camera source."""

    def __init__(self, time_fn: Callable[[], float]) -> None:
        self.time_fn = time_fn
        self.info: Dict[str, Any] = {}
        self._cond = threading.Condition()
        self._frame: Optional[Frame] = None
        self._stamps: Deque[float] = deque(maxlen=120)
        self._running = False

    def _publish(self, image: np.ndarray) -> None:
        t = self.time_fn()
        with self._cond:
            fid = self._frame.id + 1 if self._frame else 1
            self._frame = Frame(fid, t, image)
            self._stamps.append(time.perf_counter())
            self._cond.notify_all()

    def latest(self) -> Optional[Frame]:
        with self._cond:
            return self._frame

    def wait(self, after_id: int = 0, timeout: float = 1.0) -> Optional[Frame]:
        """Block until a frame newer than `after_id` exists (or timeout). Never returns a repeat."""
        end = time.perf_counter() + timeout
        with self._cond:
            while self._running and (self._frame is None or self._frame.id <= after_id):
                left = end - time.perf_counter()
                if left <= 0:
                    return None
                self._cond.wait(left)
            f = self._frame
            return f if f is not None and f.id > after_id else None

    @property
    def fps(self) -> float:
        """Frames actually DELIVERED per second over the last ~2 s (not the driver's claim)."""
        with self._cond:
            s = list(self._stamps)
        now = time.perf_counter()
        s = [x for x in s if now - x <= 2.0]
        if len(s) < 2 or now - s[-1] > 0.5:
            return 0.0
        return (len(s) - 1) / (s[-1] - s[0])

    def _wait_first(self, timeout: float, what: str) -> None:
        if self.wait(0, timeout) is None:
            self.stop()
            raise CameraError(f"{what} opened but delivered no frames in {timeout:.0f}s.\n{BUSY_HINT}")

    def stop(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()


class Webcam(_FrameSource):
    """USB webcam on its own grab thread. `latest()` / `wait()` never return a stale or repeated frame."""

    def __init__(self, cfg: Optional[CameraConfig] = None,
                 time_fn: Callable[[], float] = time.monotonic, verbose: bool = True) -> None:
        super().__init__(time_fn)
        self.cfg = cfg or load_camera_config()
        self.verbose = verbose
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._pending: Deque[Tuple[int, float]] = deque()
        self._exposure_pending: Deque[ExposureSetting] = deque()

    def start(self, first_frame_timeout: float = 6.0) -> "Webcam":
        self._cap, self.info = open_capture(self.cfg, self.verbose)
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, name="webcam-grab", daemon=True)
        self._thread.start()
        self._wait_first(first_frame_timeout, self.info.get("device", "camera"))
        return self

    def stop(self) -> None:
        super().stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def set_property(self, prop: int, value: float) -> None:
        """Thread-safe cap.set(): applied by the grab thread between two reads."""
        self._pending.append((prop, value))

    def set_exposure(self, exposure: ExposureSetting) -> None:
        self._exposure_pending.append(exposure)

    def get_property(self, prop: int) -> float:
        return self._cap.get(prop) if self._cap is not None else -1.0

    def open_driver_settings(self) -> None:
        """DirectShow only: the driver's own settings dialog (exposure, gain, RightLight, ...)."""
        self.set_property(cv2.CAP_PROP_SETTINGS, 1)

    def _grab_loop(self) -> None:
        fails, last_ok = 0, time.perf_counter()
        while self._running:
            cap = self._cap
            if cap is None:
                break
            ok, img = cap.read()
            if ok and img is not None:
                fails, last_ok = 0, time.perf_counter()
                self._publish(img)
            else:
                fails += 1
                if fails > 30 or time.perf_counter() - last_ok > 2.0:
                    self._reopen()
                    fails, last_ok = 0, time.perf_counter()
                else:
                    time.sleep(0.005)
            while self._pending:
                prop, value = self._pending.popleft()
                cap.set(prop, value)
            while self._exposure_pending:
                set_exposure(cap, backend_id(self.cfg.backend), self._exposure_pending.popleft())

    def _reopen(self) -> None:
        print("[webcam] camera stopped delivering frames; reopening...")
        if self._cap is not None:
            self._cap.release()
        while self._running:
            try:
                self._cap, self.info = open_capture(self.cfg, self.verbose)
                return
            except CameraError as e:
                print(f"[webcam] reopen failed: {str(e).splitlines()[0]}")
                time.sleep(1.0)


class Record3DCamera(_FrameSource):
    """iPhone/iPad over USB via the Record3D app (RGB only is used here)."""

    def __init__(self, cfg: Optional[CameraConfig] = None,
                 time_fn: Callable[[], float] = time.monotonic, verbose: bool = True) -> None:
        super().__init__(time_fn)
        self.cfg = cfg or load_camera_config()
        self._stream = None

    def start(self, first_frame_timeout: float = 6.0) -> "Record3DCamera":
        try:
            from record3d import Record3DStream
        except ImportError as e:
            raise CameraError("record3d is not installed (pip install record3d)") from e
        devs = Record3DStream.get_connected_devices()
        if not devs:
            raise CameraError("no Record3D device connected (open the Record3D app, USB streaming mode)")
        self._stream = Record3DStream()
        self._stream.on_new_frame = lambda: self._publish(
            cv2.cvtColor(np.asarray(self._stream.get_rgb_frame()), cv2.COLOR_RGB2BGR))
        self._stream.on_stream_stopped = lambda: print("[webcam] Record3D stream stopped")
        self._running = True
        self._stream.connect(devs[0])
        self.info = {"device": "Record3D", "index": 0, "backend": "record3d"}
        self._wait_first(first_frame_timeout, "Record3D")
        return self


def make_camera(cfg: Optional[CameraConfig] = None, time_fn: Callable[[], float] = time.monotonic,
                verbose: bool = True) -> _FrameSource:
    cfg = cfg or load_camera_config()
    if cfg.source == "record3d":
        return Record3DCamera(cfg, time_fn, verbose)
    return Webcam(cfg, time_fn, verbose)


def add_camera_args(ap) -> None:
    """Standard camera CLI flags shared by every Dev 2 tool (all optional; default = data/camera.json)."""
    ap.add_argument("--index", type=int, default=None, help="camera index (disables name matching)")
    ap.add_argument("--name", default=None, help="regex for the camera name, e.g. 'Brio'")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--backend", default=None, choices=sorted(_BACKENDS) + ["auto"])
    ap.add_argument("--source", default=None, choices=["webcam", "record3d"])
    ap.add_argument("--camera-config", default=CAMERA_CFG_PATH)


def config_from_args(a) -> CameraConfig:
    cfg = load_camera_config(a.camera_config, width=a.width, height=a.height, fps=a.fps,
                             backend=a.backend, source=a.source)
    if a.index is not None:
        cfg = cfg.replace(index=a.index, name="" if a.name is None else a.name)
    elif a.name is not None:
        cfg = cfg.replace(name=a.name)
    return cfg
