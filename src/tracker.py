"""
PenTracker: overhead camera -> pen-tip PenState in PAGE-NORMALIZED coordinates.  Owner: Dev 2.
Contract: src.tracker.PenTracker(clock) with start / stop / read / snapshot (see contracts.Tracker).

    python -m src.tracker            # ONE command: finds the page, then tracks the pen tip / fingertip

What happens on start (no keys, no clicks):
  1. CALIBRATE: the tracker looks for the white page; once it has held still for 1.5 s the
     4 corners lock (saved to data/page_calibration.json) and that view of the empty scene
     becomes the background for tip detection. If the page can't be found within
     STARTUP_CALIB_S, the saved calibration is used.
  2. TRACK: the pen tip (or the fingertip, when pointing) is found without any colour tuning:
     src.hand_tip.TipDetector = the point of the hand/pen silhouette farthest from where the
     arm enters the view. If a pen marker colour was tuned (tools/tune_marker.py ->
     data/marker_hsv.json) and the marker is visible, it is used instead (more precise).
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
from src.webcam import CameraConfig, Frame, load_camera_config, make_camera

FRESH_S = 0.08          # a measurement younger than this is reported as-is
HOLD_S = 0.30           # occlusion: hold/extrapolate this long, then pen=None
EXTRAP_MAX_S = 0.15     # never extrapolate motion further than this
ROI_MARGIN = 0.30       # marker search: this far (page units) outside the calibrated page
MAX_OFF_PAGE = 1.0      # positions further off the page than this are nonsense (bad homography)
STARTUP_CALIB_S = 15.0  # look for the page this long on start before falling back to the saved file
SEARCH_DEBUG_PATH = "data/page_search_failed.png"


@dataclass
class TrackPoint:
    x: float                                   # full-frame pixels
    y: float
    area: float
    source: str                                # "marker" | "tip"
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
                 startup_calibrate: bool = True, detect_mode: str = "auto") -> None:
        """detect_mode: "auto" (marker if tuned + visible, else tip) | "marker" | "tip"."""
        self.clock = clock
        self.auto_calibrate = auto_calibrate
        self.detect_mode = detect_mode
        self._page_auto = StablePageDetector(hold_s=1.5)
        self.camera_cfg = camera_cfg
        self.calib_path, self.marker_path = calib_path, marker_path
        self.hud_frame = hud_frame
        self.cam = camera
        self.detector = MarkerDetector(load_marker_params(marker_path))
        self._marker_tuned = os.path.exists(marker_path)
        self.tip = TipDetector()
        self._calib_file: Optional[PageCalibration] = PageCalibration.load(calib_path)
        self._calib: Optional[PageCalibration] = None
        self._calib_shape: Optional[Tuple[int, int]] = None
        self._searching = auto_calibrate and (self._calib_file is None or startup_calibrate)
        self._search_t0: Optional[float] = None
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

    # ------------------------------------------------------------------ contract
    def start(self) -> None:
        if self._running:
            return
        if self.cam is None:
            self.cam = make_camera(self.camera_cfg or load_camera_config(), time_fn=self.clock.now)
        self.cam.start()
        if self._searching:
            print("[tracker] looking for the page: whole sheet in view, hands out, hold still...")
        elif self._calib_file is None:
            print(f"[tracker] no {self.calib_path}: reporting camera-normalized coords.")
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="pen-tracker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.cam is not None:
            self.cam.stop()

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
        if self._search_t0 is None:
            self._search_t0 = f.t
        if f.id % 3 == 0 and self._page_auto.update(img, f.t) >= 1.0:
            h, w = img.shape[:2]
            found = PageCalibration(self._page_auto.corners, (w, h))
            found.save(self.calib_path)
            self._calib_file, self._calib, self._calib_shape = found, None, None
            if self._last_det is None or self._last_det.source != "tip":
                self.tip.set_background(img)   # whole page visible and no hand: the scene is empty
            self._searching = False
            print(f"[tracker] page locked automatically -> {self.calib_path}; tracking the pen")
        elif self._calib_file is not None and f.t - self._search_t0 > STARTUP_CALIB_S:
            self._searching = False
            try:
                cv2.imwrite(SEARCH_DEBUG_PATH, img)
            except cv2.error:
                pass
            print(f"[tracker] page not found in {STARTUP_CALIB_S:.0f}s: using the saved calibration "
                  f"(view saved to {SEARCH_DEBUG_PATH}; press c to look again)")

    def _detect(self, img: np.ndarray, prev: Optional[Measurement], t: float, w: int) -> Optional[TrackPoint]:
        if self.detect_mode == "marker" or (self.detect_mode == "auto" and self._marker_tuned):
            predict = (prev.px, prev.py) if prev is not None and t - prev.t <= HOLD_S else None
            m = self.detector.detect(img, predict, gate_px=0.15 * w)
            if m is not None:
                return TrackPoint(m.x, m.y, m.area, "marker")
        if self.detect_mode in ("auto", "tip"):
            d = self.tip.detect(img, t)
            if d is not None:
                return TrackPoint(d.x, d.y, d.area, "tip", d.entry)
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
        if self._searching:
            self._search_page(img, f)
        calib = self._calibration_for(img.shape)

        prev = self._meas
        det = self._detect(img, prev, f.t, w)

        meas = prev
        if det is not None:
            if calib is not None:
                u, v = calib.to_page(det.x, det.y)
            else:
                u, v = det.x / w, det.y / h
            if -MAX_OFF_PAGE <= u <= 1 + MAX_OFF_PAGE and -MAX_OFF_PAGE <= v <= 1 + MAX_OFF_PAGE:
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
    from src.guidance import GuidanceEngine, page_side
    from src.page_calibration import click_corners
    from src.webcam import add_camera_args, config_from_args

    ap = argparse.ArgumentParser(description="Live pen tracker: finds the page, then tracks the pen tip / fingertip")
    add_camera_args(ap)
    ap.add_argument("--layout", default="data/layout.json")
    ap.add_argument("--mode", default="auto", choices=["auto", "tip", "marker"],
                    help="auto = marker if tuned and visible, else pen tip / fingertip (no tuning)")
    ap.add_argument("--use-saved-calibration", action="store_true",
                    help="skip the startup page search and use data/page_calibration.json")
    a = ap.parse_args()

    try:
        layout = load_layout(a.layout)
    except FileNotFoundError:
        layout = sample_layout()
    box_ids = list(layout)
    clock = RealClock()
    tracker = PenTracker(clock, config_from_args(a), detect_mode=a.mode,
                         startup_calibrate=not a.use_saved_calibration)
    tracker.start()
    guide = GuidanceEngine()
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
            g = guide.compute(pen, box)

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
                cv2.putText(cam_view, det.source, (int(det.x) + 12, int(det.y) - 12),
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
                cv2.imshow("tracker: foreground", tracker.tip.mask)

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
            if pen is not None:
                col = (0, 0, 255) if pen.confidence >= 0.99 else (0, 165, 255)
                px = int(min(pw - 6, max(5, pen.x * pw)))
                py = int(min(ph - 6, max(5, pen.y * ph)))
                if side:      # off the page: pin a ring to the nearest edge
                    cv2.circle(page, (px, py), 9, (0, 0, 255), 2)
                    cv2.putText(page, f"OFF PAGE: {side}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.circle(page, (px, py), 6, col, -1)
            lines = [
                f"pen ({pen.x:.3f}, {pen.y:.3f}) = ({pen.x * PAGE_W_CM:.1f}, {pen.y * PAGE_H_CM:.1f}) cm  "
                f"conf {pen.confidence:.2f}" if pen else "pen: NOT SEEN",
                f"box {box.id if box else '-'}  in_box={g.in_box}  {g.write_status.value}",
                f"to box: right {g.dx_cm:+.1f} cm, down {g.dy_cm:+.1f} cm ({g.dist_cm:.1f} cm)",
                f"{g.cmd.value if g.cmd else '-'}  {g.speech or ''}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(page, line, (6, ph - 66 + 17 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.imshow("tracker: page", page)

            if now - last_print >= 1.0:
                last_print = now
                warn = "  <-- below 25 FPS: run tools/camera_check.py --bench" if 0 < st["camera_fps"] < 25 else ""
                where = "SEARCHING FOR PAGE | " if searching else ""
                src = f" [{det.source}]" if det is not None else ""
                off = f" | OFF PAGE: {side}" if side else ""
                print(f"{where}cam {st['camera_fps']:5.1f} fps | trk {st['tracker_fps']:5.1f} fps | "
                      f"{st['proc_ms']:4.1f} ms | {lines[0]}{src}{off} | {lines[2]} | {lines[3]}{warn}")

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
