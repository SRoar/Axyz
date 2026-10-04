"""
PenTracker: overhead camera -> pen-tip PenState in PAGE-NORMALIZED coordinates.  Owner: Dev 2.
Contract: src.tracker.PenTracker(clock) with start / stop / read / snapshot (see contracts.Tracker).

    python -m src.tracker            # ONE command: finds the page, then tracks the pen tip / fingertip

What happens on start (no keys, no clicks):
  1. CALIBRATE: the tracker looks for the white page; once it has held still for 1.5 s the
     4 corners lock (saved to data/page_calibration.json) and that view of the empty scene
     becomes the background for tip detection. If the page can't be found within
     STARTUP_CALIB_S, the saved calibration is used.
  2. TRACK: src.hand_tip.TipDetector separates the pen from the hand (skin colour vs our pen's
     colours from data/pen_profile.json, made once with tools/capture_pen.py), picks the writing
     end (LiDAR depth when streaming from Record3D, else how the hand holds the pen) and follows
     the line to the nib. No pen in hand -> the fingertip. If a pen marker colour was tuned
     (tools/tune_marker.py -> data/marker_hsv.json) and the marker is visible, it is used instead.
  3. OFF THE PAGE: positions outside 0..1 are reported as-is (left of the page: x < 0, ...);
     guidance.page_side() names the side and the spoken cue says so.

Optional tools: tools/camera_check.py (camera / FPS), tools/calibrate_page.py (calibration window),
tools/tune_marker.py (marker colour), python -m src.prescan (answer boxes -> data/layout.json).

Threads: the camera grab thread keeps only the newest frame; the tracker thread processes every
new frame exactly once (detect -> homography -> One-Euro smoothing). read() never blocks: it
returns the latest measurement, extrapolated for up to HOLD_S with decaying confidence while the
pen is hidden, then pen=None.
"""
from __future__ import annotations

import argparse
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from src.contracts import PAGE_H_CM, PAGE_W_CM, Clock, PenState, RealClock, TrackerReading
from src.hand_tip import TipDetector
from src.marker import MARKER_PATH, MarkerDetector, load_marker_params
from src.page_calibration import CALIB_PATH, PageCalibration, StablePageDetector
from src.pen_profile import PROFILE_PATH, PenProfile
from src.webcam import CameraConfig, Frame, load_camera_config, make_camera

FRESH_S = 0.08          # a measurement younger than this is reported as-is
HOLD_S = 0.30           # occlusion: hold/extrapolate this long, then pen=None
EXTRAP_MAX_S = 0.15     # never extrapolate motion further than this
ROI_MARGIN = 0.30       # marker search: this far (page units) outside the calibrated page
MAX_OFF_PAGE = 1.0      # positions further off the page than this are nonsense (bad homography)
STARTUP_CALIB_S = 15.0  # look for the page this long on start before falling back to the saved file
SEARCH_DEBUG_PATH = "data/page_search_failed.png"
SEARCH_EVERY_S = 0.05   # page-finding rate while searching
WATCH_EVERY_S = 0.15    # ... and afterwards, while watching for a moved page
RELOCK_PX = 8.0         # a page seen this far from the calibrated corners is re-locked
CAMERA_RETRY_S = 3.0    # camera failed to start (phone locked, app not streaming): retry this often


@dataclass
class TipSample:
    """One raw (unsmoothed) tip detection: every processed frame with a tip on/near the page."""
    t: float
    x: float            # page-normalized (camera-normalized while uncalibrated)
    y: float
    source: str         # "marker" | "pen" | "finger"
    px: float           # full-frame pixels
    py: float


@dataclass
class TrackPoint:
    x: float                                   # full-frame pixels
    y: float
    area: float
    source: str                                # "marker" | "pen" | "finger"
    entry: Optional[Tuple[float, float]] = None   # tip mode: where the arm enters the view


class OneEuro:
    """One-Euro filter (Casiez 2012): light smoothing when slow, low lag when fast."""

    def __init__(self, min_cutoff: float = 2.0, beta: float = 1.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: Optional[float] = None
        self.dx = 0.0
        self.t: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.x, self.dx, self.t = None, 0.0, None

    def __call__(self, x: float, t: float) -> float:
        if self.x is None or self.t is None or t <= self.t:
            self.x, self.t = x, t
            return x
        dt = t - self.t
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x, self.t = a * x + (1 - a) * self.x, t
        return self.x


@dataclass
class Measurement:
    t: float            # capture time of the frame (system clock)
    x: float            # smoothed page-normalized position
    y: float
    vx: float           # page units / s
    vy: float
    px: float           # raw pixel position (for display / prediction)
    py: float
    area: float
    n_blobs: int


def hold_or_extrapolate(m: Optional[Measurement], now: float) -> Optional[PenState]:
    """Occlusion policy (pure, unit-tested): fresh -> as-is; <= HOLD_S -> extrapolate with
    decaying confidence; older -> None."""
    if m is None:
        return None
    age = max(0.0, now - m.t)
    if age <= FRESH_S:
        return PenState(m.t, m.x, m.y, 0.0, 1.0)
    if age > HOLD_S:
        return None
    k = min(age, EXTRAP_MAX_S)
    conf = max(0.0, 1.0 - (age - FRESH_S) / (HOLD_S - FRESH_S))
    return PenState(now, m.x + m.vx * k, m.y + m.vy * k, 0.0, conf)


class PenTracker:
    HUD_PX_PER_CM = 20.0        # TrackerReading.frame: page-rectified, 432 x 559
    SNAPSHOT_PX_PER_CM = 40.0   # snapshot(): page-rectified, 864 x 1118

    def __init__(self, clock: Clock, camera_cfg: Optional[CameraConfig] = None,
                 calib_path: str = CALIB_PATH, marker_path: str = MARKER_PATH,
                 hud_frame: bool = True, camera: Any = None, auto_calibrate: bool = True,
                 startup_calibrate: bool = True, detect_mode: str = "auto",
                 profile_path: str = PROFILE_PATH, right_handed: bool = True,
                 record_path: Optional[str] = None, freeze_after_lock: bool = False) -> None:
        """detect_mode: "auto" (marker if tuned + visible, else tip) | "marker" | "tip".
        profile_path: our pen's colours (tools/capture_pen.py); without it a generic pen model is used.
        record_path: write every raw tip detection to this CSV (t, x_cm, y_cm, source, px, py).
        freeze_after_lock: for a camera that's physically fixed (taped down, top-down): once the
        initial scan locks, stop re-evaluating the page corners every frame. Corner-detector wobble
        (a few px, autofocus/lighting) can otherwise exceed RELOCK_PX on its own and trigger a
        spurious re-lock - which looks like the box positions / page angle drifting even though
        nothing moved. Call recalibrate() to deliberately re-scan (e.g. the rig got bumped)."""
        self.clock = clock
        self.auto_calibrate = auto_calibrate
        self.detect_mode = detect_mode
        self.freeze_after_lock = freeze_after_lock
        self._page_auto = StablePageDetector(hold_s=1.5)
        self.camera_cfg = camera_cfg
        self.calib_path, self.marker_path = calib_path, marker_path
        self.hud_frame = hud_frame
        self.cam = camera
        self.detector = MarkerDetector(load_marker_params(marker_path))
        self._marker_tuned = os.path.exists(marker_path)
        self.pen_profile = PenProfile.load(profile_path)
        self.tip = TipDetector(right_handed=right_handed, profile=self.pen_profile)
        self._calib_file: Optional[PageCalibration] = PageCalibration.load(calib_path)
        self._calib: Optional[PageCalibration] = None
        self._calib_shape: Optional[Tuple[int, int]] = None
        self._searching = auto_calibrate and (self._calib_file is None or startup_calibrate)
        self._search_t0: Optional[float] = None
        self.page_locked_t: Optional[float] = None      # frame time of the last fresh page lock
        self._next_search_t = -1e9
        self._fx, self._fy = OneEuro(), OneEuro()
        self._lock = threading.Lock()
        self._meas: Optional[Measurement] = None
        self._last_det: Optional[TrackPoint] = None
        self._frame: Optional[Frame] = None
        self._hud: Optional[np.ndarray] = None
        self._recent: Deque[Frame] = deque(maxlen=8)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._proc_ms = 0.0
        self._detect_rate = 0.0
        self._proc_stamps: Deque[float] = deque(maxlen=90)
        self._samples: Deque[TipSample] = deque(maxlen=600)
        self.record_path = record_path
        self._rec: Any = None
        self._rec_flush_t = 0.0
        self.camera_error = ""                 # why the camera isn't streaming yet ("" = it is)

    # ------------------------------------------------------------------ contract
    def start(self) -> None:
        if self._running:
            return
        if self.cam is None:
            self.cam = make_camera(self.camera_cfg or load_camera_config(), time_fn=self.clock.now)
        try:
            self.cam.start()
            self.camera_error = ""
        except Exception as e:
            # e.g. the iPhone is locked or Record3D isn't streaming yet: keep retrying instead of
            # running the whole demo without a camera (the HUD shows camera_error meanwhile)
            self.camera_error = f"{type(e).__name__}: {e}"
            print(f"[tracker] camera not ready ({self.camera_error}); retrying every {CAMERA_RETRY_S:.0f} s")
            threading.Thread(target=self._retry_camera, name="camera-retry", daemon=True).start()
        if self._searching:
            print("[tracker] looking for the page: whole sheet in view, hands out, hold still...")
        elif self._calib_file is None:
            print(f"[tracker] no {self.calib_path}: reporting camera-normalized coords.")
        if self.record_path:
            os.makedirs(os.path.dirname(self.record_path) or ".", exist_ok=True)
            self._rec = open(self.record_path, "w", encoding="utf-8", newline="")
            self._rec.write("t,x_cm,y_cm,source,px,py\n")
            print(f"[tracker] recording every tip point to {self.record_path}")
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="pen-tracker", daemon=True)
        self._thread.start()

    def _retry_camera(self) -> None:
        while True:
            time.sleep(CAMERA_RETRY_S)              # sleep first: start() sets _running after spawning this
            if not self._running or not self.camera_error:
                return
            try:
                self.cam.start()
                self.camera_error = ""
                print("[tracker] camera connected")
            except Exception as e:  # noqa: BLE001
                self.camera_error = f"{type(e).__name__}: {e}"

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.cam is not None:
            self.cam.stop()
        if self._rec is not None:
            self._rec.close()
            self._rec = None

    def read(self) -> Optional[TrackerReading]:
        with self._lock:
            frame, meas, hud = self._frame, self._meas, self._hud
        if frame is None:
            return None
        now = self.clock.now()
        return TrackerReading(now, hold_or_extrapolate(meas, now), hud)

    def snapshot(self) -> Any:
        """Sharpest of the last few frames, page-rectified at SNAPSHOT_PX_PER_CM (raw if uncalibrated)."""
        with self._lock:
            frames = list(self._recent)
        if not frames:
            return None
        best = max(frames, key=lambda f: _sharpness(f.image))
        calib = self._calibration_for(best.image.shape)
        if calib is None:
            print("[tracker] snapshot() without page calibration: returning the raw camera frame")
            return best.image.copy()
        return calib.rectify(best.image, self.SNAPSHOT_PX_PER_CM, cv2.INTER_CUBIC)

    # ------------------------------------------------------------------ extras (tools / HUD)
    @property
    def calibration(self) -> Optional[PageCalibration]:
        return self._calib

    def reload(self) -> None:
        """Re-read data/page_calibration.json and data/marker_hsv.json (after re-tuning)."""
        self._calib_file = PageCalibration.load(self.calib_path)
        self._calib, self._calib_shape = None, None
        self._page_auto.reset()
        self.detector = MarkerDetector(load_marker_params(self.marker_path))
        self._marker_tuned = os.path.exists(self.marker_path)
        self._fx.reset()
        self._fy.reset()

    def recalibrate(self) -> None:
        """Hands-free re-calibration: look for the page again (hands out of view, hold still)."""
        self._page_auto.reset()
        self._search_t0 = None
        self._next_search_t = -1e9
        self._searching = True

    def reset_background(self) -> None:
        """Re-learn the empty scene for tip detection from the next frame (hands out of view)."""
        self.tip.reset()

    def calibration_status(self) -> Tuple[bool, float, Optional[np.ndarray]]:
        """(searching, progress 0..1, current page-corner guess in pixels) for the live view / HUD."""
        return self._searching, self._page_auto.progress, self._page_auto.corners

    def raw_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.image

    def last_detection(self) -> Optional[TrackPoint]:
        with self._lock:
            return self._last_det

    def drain_samples(self) -> list:
        """Every raw tip detection since the last call (oldest first), for sequence-level decisions."""
        with self._lock:
            out = list(self._samples)
            self._samples.clear()
        return out

    def stats(self) -> Dict[str, float]:
        with self._lock:
            s = list(self._proc_stamps)
        now = time.perf_counter()
        s = [x for x in s if now - x <= 2.0]
        stale = not s or now - s[-1] > 0.5
        proc_fps = 0.0 if stale or len(s) < 2 or s[-1] <= s[0] else (len(s) - 1) / (s[-1] - s[0])
        return {"camera_fps": self.cam.fps if self.cam is not None else 0.0, "tracker_fps": proc_fps,
                "proc_ms": self._proc_ms, "detect_rate": self._detect_rate}

    # ------------------------------------------------------------------ internals
    def _calibration_for(self, shape: Tuple[int, ...]) -> Optional[PageCalibration]:
        if self._calib_file is None:
            return None
        h, w = shape[:2]
        if self._calib_shape != (w, h):
            self._calib = self._calib_file.scaled_to(w, h)
            self._calib_shape = (w, h)
            self.detector.set_roi(self._calib.page_polygon_px(ROI_MARGIN))
        return self._calib

    def _search_page(self, img: np.ndarray, f: Frame) -> None:
        """Startup: look for the page until it locks. Afterwards keep watching at a low rate and
        re-lock if the page is seen, uncovered and still, somewhere else (page or camera moved).
        With freeze_after_lock, skip all of that once locked - see recalibrate() to undo."""
        if self.freeze_after_lock and not self._searching and self._calib_file is not None:
            return
        if self._search_t0 is None:
            self._search_t0 = f.t
        if f.t < self._next_search_t:
            return
        self._next_search_t = f.t + (SEARCH_EVERY_S if self._searching else WATCH_EVERY_S)
        h, w = img.shape[:2]
        if self._page_auto.update(img, f.t) >= 1.0:
            corners = self._page_auto.corners
            old = self._calib_file
            moved = old is None or old.image_size != (w, h) or np.abs(corners - old.corners_px).max() > RELOCK_PX
            if self._searching or moved:
                found = PageCalibration(corners, (w, h))
                found.save(self.calib_path)
                self._calib_file, self._calib, self._calib_shape = found, None, None
                self.page_locked_t = f.t
                if self._last_det is None or self._last_det.source == "marker":
                    self.tip.set_background(img)   # whole page visible and no hand: the scene is empty
                elif moved:
                    self.tip.reset()               # re-learn once the hand is out and the view is still
                print(f"[tracker] page {'locked' if self._searching else 'moved: re-locked'} "
                      f"automatically -> {self.calib_path}; tracking the pen")
            self._searching = False
        elif self._searching and self._calib_file is not None and f.t - self._search_t0 > STARTUP_CALIB_S:
            self._searching = False
            try:
                cv2.imwrite(SEARCH_DEBUG_PATH, img)
            except cv2.error:
                pass
            print(f"[tracker] page not found in {STARTUP_CALIB_S:.0f}s: using the saved calibration "
                  f"(view saved to {SEARCH_DEBUG_PATH}); it locks by itself whenever the page is "
                  "fully visible with hands out")

    def _detect(self, img: np.ndarray, prev: Optional[Measurement], t: float, w: int,
                depth: Optional[np.ndarray] = None, confidence: Optional[np.ndarray] = None) -> Optional[TrackPoint]:
        if self.detect_mode == "marker" or (self.detect_mode == "auto" and self._marker_tuned):
            predict = (prev.px, prev.py) if prev is not None and t - prev.t <= HOLD_S else None
            m = self.detector.detect(img, predict, gate_px=0.15 * w)
            if m is not None:
                return TrackPoint(m.x, m.y, m.area, "marker")
        if self.detect_mode in ("auto", "tip"):
            d = self.tip.detect(img, t, depth, confidence)
            if self.tip.scene_changed:
                self.tip.scene_changed = False
                if self.auto_calibrate and not self._searching and not self.freeze_after_lock:
                    print("[tracker] camera moved: finding the page again (keep the camera fixed)")
                    self.recalibrate()
                elif self.freeze_after_lock:
                    # page corners stay locked, but the hand/tip background reference is still
                    # safe (and useful) to refresh - it doesn't touch page_calibration.json
                    self.tip.reset()
            if d is not None:
                return TrackPoint(d.x, d.y, d.area, d.source, d.entry)
        return None

    def _loop(self) -> None:
        last_id = 0
        while self._running:
            f = self.cam.wait(last_id, timeout=0.5)
            if f is None:
                continue
            last_id = f.id
            try:
                self._process(f)
            except Exception as e:          # never let one bad frame kill tracking
                print(f"[tracker] frame {f.id} failed: {e!r}")

    def _process(self, f: Frame) -> None:
        t0 = time.perf_counter()
        img = f.image
        h, w = img.shape[:2]
        if self.auto_calibrate:
            self._search_page(img, f)
        calib = self._calibration_for(img.shape)

        prev = self._meas
        det = self._detect(img, prev, f.t, w, getattr(f, "depth", None), getattr(f, "confidence", None))

        meas = prev
        sample = None
        if det is not None:
            if calib is not None:
                u, v = calib.to_page(det.x, det.y)
            else:
                u, v = det.x / w, det.y / h
            if -MAX_OFF_PAGE <= u <= 1 + MAX_OFF_PAGE and -MAX_OFF_PAGE <= v <= 1 + MAX_OFF_PAGE:
                sample = TipSample(f.t, float(u), float(v), det.source, float(det.x), float(det.y))
                if prev is None or f.t - prev.t > HOLD_S:
                    self._fx.reset()
                    self._fy.reset()
                xs = self._fx(u * PAGE_W_CM, f.t) / PAGE_W_CM
                ys = self._fy(v * PAGE_H_CM, f.t) / PAGE_H_CM
                meas = Measurement(f.t, xs, ys, self._fx.dx / PAGE_W_CM, self._fy.dx / PAGE_H_CM,
                                   det.x, det.y, det.area, 1)
            else:
                det = None

        hud = None
        if self.hud_frame:
            hud = calib.rectify(img, self.HUD_PX_PER_CM) if calib is not None else img

        with self._lock:
            self._frame = f
            self._recent.append(f)
            self._meas = meas
            self._last_det = det
            self._hud = hud
            self._proc_stamps.append(time.perf_counter())
            if sample is not None:
                self._samples.append(sample)
        if sample is not None and self._rec is not None:
            self._rec.write(f"{sample.t:.4f},{sample.x * PAGE_W_CM:.2f},{sample.y * PAGE_H_CM:.2f},"
                            f"{sample.source},{sample.px:.1f},{sample.py:.1f}\n")
            if f.t - self._rec_flush_t >= 1.0:
                self._rec.flush()
                self._rec_flush_t = f.t
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._proc_ms = 0.9 * self._proc_ms + 0.1 * dt_ms if self._proc_ms else dt_ms
        self._detect_rate = 0.95 * self._detect_rate + 0.05 * (1.0 if det is not None else 0.0)


def _sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


# ---------------------------------------------------------------------- live view
def main() -> int:
    from src.contracts import load_layout, sample_layout
    from src.guidance import RecentTipJudge, box_offset_cm, box_status, page_side
    from src.page_calibration import click_corners
    from src.webcam import add_camera_args, config_from_args

    ap = argparse.ArgumentParser(description="Live pen tracker: finds the page, then tracks the pen tip / fingertip")
    add_camera_args(ap)
    ap.add_argument("--layout", default="data/layout.json")
    ap.add_argument("--mode", default="auto", choices=["auto", "tip", "marker"],
                    help="auto = marker if tuned and visible, else pen tip / fingertip (no tuning)")
    ap.add_argument("--use-saved-calibration", action="store_true",
                    help="skip the startup page search and use data/page_calibration.json")
    ap.add_argument("--left-handed", action="store_true", help="the writer holds the pen in the left hand")
    ap.add_argument("--no-record", action="store_true",
                    help="don't write the tip points to data/sessions/tip_<time>.csv")
    a = ap.parse_args()

    try:
        layout = load_layout(a.layout)
    except FileNotFoundError:
        layout = sample_layout()
    box_ids = list(layout)
    clock = RealClock()
    record = None if a.no_record else time.strftime("data/sessions/tip_%Y%m%d_%H%M%S.csv")
    tracker = PenTracker(clock, config_from_args(a), detect_mode=a.mode,
                         startup_calibrate=not a.use_saved_calibration, right_handed=not a.left_handed,
                         record_path=record)
    print(f"[tracker] pen model: {'our pen (' + PROFILE_PATH + ')' if tracker.pen_profile else 'generic'}"
          " - capture yours with: python tools/capture_pen.py")
    tracker.start()
    judge = RecentTipJudge()
    box = layout[box_ids[0]] if box_ids else None
    trail: Deque[Tuple[float, float, float]] = deque()
    last_print = 0.0
    show_mask = False
    print("keys: c = find the page again   b = re-learn empty background   m = click corners manually\n"
          "      1-9 = guidance to box N   0 = no box   f = show foreground mask   s = snapshot   ESC = quit")
    try:
        while True:
            reading = tracker.read()
            raw = tracker.raw_frame()
            if reading is None or raw is None:
                if cv2.waitKey(10) & 0xFF == 27:
                    break
                continue
            now = clock.now()
            pen = reading.pen
            st = tracker.stats()
            for s in tracker.drain_samples():
                if s.source != "finger":            # only real pen-tip detections vote
                    judge.add(s.t, s.x, s.y)
            verdict = judge.judge(now, layout.values(), box)

            side = page_side(pen.x, pen.y) if pen is not None else None
            cam_view = raw.copy()
            ch, cw = cam_view.shape[:2]
            fs = max(0.45, cw / 1600.0)
            calib = tracker.calibration
            searching, progress, guess = tracker.calibration_status()
            if searching:
                if guess is not None:
                    cv2.polylines(cam_view, [np.int32(guess)], True, (0, 165, 255), 3)
                cv2.rectangle(cam_view, (10, ch - 30), (10 + int(progress * (cw - 20)), ch - 18), (0, 255, 0), -1)
                cv2.putText(cam_view, "FINDING THE PAGE - whole sheet in view, hands out, hold still",
                            (10, ch - 40), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 165, 255), 2)
            elif calib is not None:
                cv2.polylines(cam_view, [np.int32(calib.page_polygon_px())], True, (0, 255, 0), 2)
            det = tracker.last_detection()
            if det is not None:
                if det.entry is not None:
                    cv2.line(cam_view, (int(det.entry[0]), int(det.entry[1])), (int(det.x), int(det.y)), (255, 120, 0), 1)
                    cv2.circle(cam_view, (int(det.entry[0]), int(det.entry[1])), 8, (255, 120, 0), 2)
                cv2.circle(cam_view, (int(det.x), int(det.y)), 10, (0, 0, 255), 2)
                label = det.source
                if pen is not None and calib is not None:
                    label += f" ({pen.x * PAGE_W_CM:.1f}, {pen.y * PAGE_H_CM:.1f}) cm"
                cv2.putText(cam_view, label, (int(det.x) + 12, int(det.y) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), 2)
            fps_col = (0, 255, 0) if st["camera_fps"] >= 25 else (0, 0, 255)
            cv2.putText(cam_view, f"camera {st['camera_fps']:.0f} fps  tracker {st['tracker_fps']:.0f} fps  "
                        f"{st['proc_ms']:.1f} ms  detect {st['detect_rate'] * 100:.0f}%",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, fs, fps_col, 2)
            if side:
                cv2.putText(cam_view, f"OFF PAGE: {side.upper()}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 1.6, (0, 0, 255), 3)
            cv2.imshow("tracker: camera", cam_view)
            if show_mask and tracker.tip.mask is not None:
                hand_m, pen_m = tracker.tip.mask, tracker.tip.pen_mask
                view = cv2.merge([hand_m // 2, hand_m // 2 if pen_m is None else cv2.max(hand_m // 2, pen_m), hand_m // 2])
                cv2.imshow("tracker: hand (grey) / pen (green)", view)

            page = reading.frame.copy() if reading.frame is not None else np.full((559, 432, 3), 255, np.uint8)
            ph, pw = page.shape[:2]
            for b in layout.values():
                col = (255, 120, 0) if box is not None and b.id == box.id else (160, 160, 160)
                cv2.rectangle(page, (int(b.xmin * pw), int(b.ymin * ph)), (int(b.xmax * pw), int(b.ymax * ph)), col, 2)
                cv2.putText(page, b.id, (int(b.xmin * pw) + 4, int(b.ymin * ph) + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            if pen is not None:
                trail.append((now, pen.x, pen.y))
            while trail and now - trail[0][0] > 2.0:
                trail.popleft()
            for (_, x0, y0), (_, x1, y1) in zip(trail, list(trail)[1:]):
                cv2.line(page, (int(x0 * pw), int(y0 * ph)), (int(x1 * pw), int(y1 * ph)), (0, 200, 255), 2)
            # the decision: recent tip points (dots), their median (cross) and the arrow from it to the box
            recent = judge.recent(now)
            for _, xc, yc in recent:
                cv2.circle(page, (int(xc / PAGE_W_CM * pw), int(yc / PAGE_H_CM * ph)), 2, (200, 0, 200), -1)
            if verdict is not None:
                mx, my = verdict.mx_cm / PAGE_W_CM, verdict.my_cm / PAGE_H_CM
                mpx, mpy = int(mx * pw), int(my * ph)
                vcol = (0, 160, 0) if verdict.inside else (255, 120, 0)
                cv2.drawMarker(page, (mpx, mpy), vcol, cv2.MARKER_CROSS, 18, 2)
                if not verdict.inside and verdict.dist_cm > 0:
                    tx = int(round((mx + verdict.dx_cm / PAGE_W_CM) * pw))
                    ty = int(round((my + verdict.dy_cm / PAGE_H_CM) * ph))
                    cv2.arrowedLine(page, (mpx, mpy), (tx, ty), vcol, 2, tipLength=0.12)
                    cv2.putText(page, f"{verdict.dist_cm:.1f} cm", ((mpx + tx) // 2 + 6, (mpy + ty) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, vcol, 2)
            # this frame's pen tip -> every answer box: offset to the nearest point of the box, in cm
            offsets = {b.id: box_offset_cm(pen.x, pen.y, b) for b in layout.values()} if pen is not None else {}
            if pen is not None:
                col = (0, 0, 255) if pen.confidence >= 0.99 else (0, 165, 255)
                px = int(min(pw - 6, max(5, pen.x * pw)))
                py = int(min(ph - 6, max(5, pen.y * ph)))
                if side:      # off the page: pin a ring to the nearest edge
                    cv2.circle(page, (px, py), 9, (0, 0, 255), 2)
                    cv2.putText(page, f"OFF PAGE: {side}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.circle(page, (px, py), 6, col, -1)

            if pen is not None:
                tip_line = f"x {pen.x * PAGE_W_CM:5.1f} cm  y {pen.y * PAGE_H_CM:5.1f} cm"
                tip_sub = f"({pen.x:.3f}, {pen.y:.3f}) of the page" + (f"  OFF PAGE: {side}" if side else "")
            else:
                tip_line, tip_sub = "not seen", ""
            inside = next((bid for bid, o in offsets.items() if o[2] == 0), None)
            status = verdict.text() if verdict is not None else "not enough recent pen points"
            panel = np.full((ph, 360, 3), 255, np.uint8)
            rows = [("PEN TIP on the paper (from top-left)", 0.5, (0, 0, 0), 1),
                    (tip_line, 0.65, (0, 0, 255), 2), (tip_sub, 0.42, (80, 80, 80), 1), ("", 0.4, 0, 1),
                    (f"DECISION (last {judge.window_s:.0f} s, {len(recent)} points):", 0.45, (0, 0, 0), 1),
                    (status, 0.5, (0, 160, 0) if verdict is not None and verdict.inside else (255, 120, 0), 2),
                    ("this frame: " + box_status(inside, box, offsets), 0.4, (120, 120, 120), 1),
                    ("", 0.4, 0, 1), ("all boxes (this frame):", 0.45, (0, 0, 0), 1)]
            for b in layout.values():
                active = box is not None and b.id == box.id
                if b.id not in offsets:
                    text = f"{b.id}:  -"
                else:
                    dx, dy, dist = offsets[b.id]
                    text = f"{b.id}:  IN THE BOX" if dist == 0 else f"{b.id}: {dist:4.1f} cm away"
                rows.append((("> " if active else "  ") + text, 0.45,
                             (0, 160, 0) if b.id == inside else (255, 120, 0) if active else (60, 60, 60),
                             2 if active or b.id == inside else 1))
            rows += [("", 0.4, 0, 1), ("1-9 = target box N   0 = none", 0.42, (120, 120, 120), 1)]
            y = 24
            for text, scale, colr, thick in rows:
                cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colr, thick)
                y += int(34 * scale) + 6
            cv2.imshow("tracker: page", np.hstack([page, panel]))

            if now - last_print >= 1.0:
                last_print = now
                warn = "  <-- below 25 FPS: run tools/camera_check.py --bench" if 0 < st["camera_fps"] < 25 else ""
                where = "SEARCHING FOR PAGE | " if searching else ""
                src = f" [{det.source}]" if det is not None else ""
                off = f" | OFF PAGE: {side}" if side else ""
                print(f"{where}trk {st['tracker_fps']:4.0f} fps | tip {tip_line}{src}{off} | {status}{warn}")

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("c"):
                tracker.recalibrate()
                print("looking for the page again: hands out of view, hold still")
            elif key == ord("b"):
                tracker.reset_background()
                print("re-learning the empty background from the next frame")
            elif key == ord("f"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("tracker: foreground")
            elif key == ord("m"):
                corners = click_corners(lambda: tracker.raw_frame(), auto=False)
                if corners is not None:
                    h, w = raw.shape[:2]
                    PageCalibration(corners, (w, h)).save(tracker.calib_path)
                    tracker.reload()
                    tracker.reset_background()
                    print(f"saved {tracker.calib_path}")
            elif key == ord("r"):
                tracker.reload()
                print("reloaded calibration + marker params")
            elif key == ord("s"):
                snap = tracker.snapshot()
                if snap is not None:
                    cv2.imwrite("data/snapshot.png", snap)
                    print(f"saved data/snapshot.png {snap.shape[1]}x{snap.shape[0]}")
            elif ord("1") <= key <= ord("9") and key - ord("1") < len(box_ids):
                box = layout[box_ids[key - ord("1")]]
            elif key == ord("0"):
                box = None
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
