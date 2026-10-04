import threading
import time

import cv2
import numpy as np
import pytest

from src.contracts import SimClock, Tracker
from src.marker import MarkerDetector, build_mask, sample_hsv_range
from src.page_calibration import PageCalibration, StablePageDetector, detect_page_corners, order_corners
from src.guidance import page_side
from src.hand_tip import TipDetector
from src.tracker import FRESH_S, HOLD_S, Measurement, OneEuro, PenTracker, hold_or_extrapolate
from src.webcam import Frame

W, H = 1280, 720
CORNERS = np.float32([[400, 80], [880, 90], [900, 660], [380, 650]])   # slightly keystoned page
GREEN = {"h_lo": 50, "h_hi": 70, "s_lo": 120, "s_hi": 255, "v_lo": 80, "v_hi": 255, "min_area": 40}


def page_image(marker_uv=None, extra_blob_px=None):
    img = np.full((H, W, 3), 40, np.uint8)                       # dark desk
    cv2.fillConvexPoly(img, CORNERS.astype(np.int32), (235, 235, 235))   # white page
    calib = PageCalibration(CORNERS, (W, H))
    if marker_uv is not None:
        x, y = calib.to_pixels(*marker_uv)
        cv2.circle(img, (int(round(x)), int(round(y))), 7, (0, 200, 0), -1)
    if extra_blob_px is not None:
        cv2.circle(img, extra_blob_px, 7, (0, 200, 0), -1)
    return img


SKIN = (110, 150, 205)


def hand_image(tip_px, entry_px=(640, H + 40), base=None):
    """An arm reaching in from the bottom edge, narrowing to a pointed finger / pen tip at tip_px."""
    img = page_image() if base is None else base.copy()
    tip = np.float32(tip_px)
    entry = np.float32(entry_px)
    d = (tip - entry) / np.linalg.norm(tip - entry)
    n = np.float32([-d[1], d[0]])
    knuckle = tip - 70 * d
    cv2.line(img, tuple(map(int, entry)), tuple(map(int, knuckle)), SKIN, 60)
    tri = np.int32([knuckle + 18 * n, knuckle - 18 * n, tip])
    cv2.fillConvexPoly(img, tri, SKIN)
    return img


# --------------------------------------------------------------------------- calibration
def test_calibration_maps_corners_to_unit_square_and_back():
    c = PageCalibration(CORNERS, (W, H))
    for (x, y), (u, v) in zip(CORNERS, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        assert c.to_page(x, y) == pytest.approx((u, v), abs=1e-5)
    x, y = c.to_pixels(0.3, 0.7)
    assert c.to_page(x, y) == pytest.approx((0.3, 0.7), abs=1e-6)


def test_calibration_scaling_and_persistence(tmp_path):
    c = PageCalibration(CORNERS, (W, H))
    half = c.scaled_to(W // 2, H // 2)
    assert half.to_page(*(CORNERS[2] / 2)) == pytest.approx((1, 1), abs=1e-5)
    p = tmp_path / "calib.json"
    c.save(str(p))
    c2 = PageCalibration.load(str(p))
    assert np.allclose(c2.corners_px, CORNERS, atol=0.01) and c2.image_size == (W, H)
    assert PageCalibration.load(str(tmp_path / "missing.json")) is None


def test_rectify_size_and_content():
    c = PageCalibration(CORNERS, (W, H))
    page = c.rectify(page_image(), 20.0)
    assert page.shape[:2] == (559, 432)
    assert page[10:-10, 10:-10].mean() > 200      # all page, no desk


def test_order_and_auto_detect_corners():
    shuffled = CORNERS[[2, 0, 3, 1]]
    assert np.allclose(order_corners(shuffled), CORNERS)
    found = detect_page_corners(page_image())
    assert found is not None and np.abs(found - CORNERS).max() < 4


def test_stable_page_detector_waits_for_a_still_page():
    det = StablePageDetector(hold_s=1.5)
    img = page_image()
    assert det.update(img, 0.0) == 0.0
    assert 0.0 < det.update(img, 1.0) < 1.0
    assert det.update(img, 1.6) == 1.0 and np.abs(det.corners - CORNERS).max() < 4
    covered = img.copy()
    covered[:, :] = 40                                   # page gone (e.g. lifted away)
    assert det.update(covered, 1.7) == 1.0               # a brief miss is ignored
    assert det.update(covered, 2.3) == 0.0 and det.corners is None   # a long gap resets
    assert det.update(img, 2.4) == 0.0                   # countdown restarts
    wobble = page_image()
    det2 = StablePageDetector(hold_s=1.0)
    det2.update(wobble, 0.0)
    shifted = np.roll(wobble, 3, axis=1)                 # 3 px wobble keeps counting
    assert det2.update(shifted, 1.1) == 1.0


# --------------------------------------------------------------------------- marker
def test_marker_detector_downscaled_with_roi_is_subpixel_accurate():
    c = PageCalibration(CORNERS, (W, H))
    det = MarkerDetector(GREEN, max_work_width=320)
    det.set_roi(c.page_polygon_px(0.05))
    img = page_image((0.37, 0.61))
    d = det.detect(img)
    ex, ey = c.to_pixels(0.37, 0.61)
    assert d is not None and abs(d.x - round(ex)) < 1.0 and abs(d.y - round(ey)) < 1.0


def test_marker_outside_roi_is_ignored():
    c = PageCalibration(CORNERS, (W, H))
    det = MarkerDetector(GREEN)
    det.set_roi(c.page_polygon_px(0.05))
    assert det.detect(page_image(None, extra_blob_px=(100, 100))) is None
    det.set_roi(None)
    assert det.detect(page_image(None, extra_blob_px=(100, 100))) is not None


def test_marker_prediction_picks_the_nearby_blob():
    c = PageCalibration(CORNERS, (W, H))
    det = MarkerDetector(GREEN)
    img = page_image((0.2, 0.2))
    x2, y2 = c.to_pixels(0.8, 0.8)
    cv2.circle(img, (int(x2), int(y2)), 7, (0, 200, 0), -1)
    d = det.detect(img, predict=(x2 + 5, y2 - 5), gate_px=60)
    assert abs(d.x - x2) < 2 and abs(d.y - y2) < 2 and d.n_blobs == 2


def test_click_sampling_handles_red_hue_wrap():
    img = np.zeros((20, 20, 3), np.uint8)
    img[:, :10] = (0, 0, 220)        # pure red, hue 0
    img[:, 10:] = (40, 0, 220)       # red-magenta, hue ~175
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = sample_hsv_range(hsv, 10, 10, radius=6)
    assert v["h_lo"] > v["h_hi"]                       # wraps around 0
    assert build_mask(hsv, v)[2:-2, 2:-2].all()


# --------------------------------------------------------------------------- marker-free tip
def test_tip_detector_finds_the_pointed_end():
    det = TipDetector()
    det.set_background(page_image())
    for tip in [(620, 300), (500, 200), (850, 450), (250, 350)]:      # last one is off the page
        d = det.detect(hand_image(tip))
        assert d is not None, tip
        assert abs(d.x - tip[0]) < 4 and abs(d.y - tip[1]) < 4, (tip, d)
        assert d.entry[1] > H - 10                                      # arm enters at the bottom


def test_tip_detector_learns_background_only_once_the_view_is_still():
    det = TipDetector()
    for i in range(10):                                   # hand waving around: never learned
        assert det.detect(hand_image((400 + 30 * i, 300)), i / 30) is None
    assert not det.has_background
    t = 10 / 30
    while not det.has_background:                         # empty and still for ~0.7 s
        det.detect(page_image(), t)
        t += 1 / 30
        assert t < 2.0
    assert det.detect(hand_image((640, 300)), t) is not None


def test_tip_detector_absorbs_a_blob_that_never_moves():
    det = TipDetector()
    det.set_background(hand_image((600, 250)))            # bad background: a hand was in it
    ghost = page_image()                                  # hand gone -> ghost where it was
    seen = [det.detect(ghost, i / 30) is not None for i in range(int(8 * 30))]
    assert seen[0] and not seen[-1]                       # stuck at first, absorbed within ~6 s
    assert det.detect(hand_image((500, 300)), 9.0) is not None   # a real hand still shows up


def test_tip_detector_ignores_shadows_exposure_and_loose_blobs():
    bg = page_image()
    det = TipDetector()
    det.set_background(bg)
    shadow = bg.copy()
    shadow[400:, 500:800] = (shadow[400:, 500:800] * 0.7).astype(np.uint8)   # hand shadow from the edge
    assert det.detect(shadow) is None
    assert det.detect((bg * 0.8).astype(np.uint8)) is None                   # auto-exposure dip
    ink = bg.copy()
    cv2.circle(ink, (640, 360), 15, (30, 30, 30), -1)                         # not touching the border
    assert det.detect(ink) is None


# --------------------------------------------------------------------------- occlusion
def test_hold_or_extrapolate():
    m = Measurement(10.0, 0.5, 0.5, 0.2, -0.1, 0, 0, 50, 1)
    p = hold_or_extrapolate(m, 10.0 + FRESH_S / 2)
    assert (p.x, p.y, p.confidence, p.t) == (0.5, 0.5, 1.0, 10.0)
    p1 = hold_or_extrapolate(m, 10.15)
    p2 = hold_or_extrapolate(m, 10.25)
    assert 0 < p2.confidence < p1.confidence < 1
    assert p1.x == pytest.approx(0.5 + 0.2 * 0.15) and p1.y == pytest.approx(0.5 - 0.1 * 0.15)
    assert p2.x == pytest.approx(p1.x)              # extrapolation is capped
    assert hold_or_extrapolate(m, 10.0 + HOLD_S + 0.01) is None
    assert hold_or_extrapolate(None, 10.0) is None


def test_one_euro_smooths_jitter_and_follows_motion():
    f = OneEuro()
    rng = np.random.default_rng(0)
    still = [f(5.0 + rng.normal(0, 0.05), i / 30) for i in range(60)]
    assert np.std(still[20:]) < 0.03
    for i in range(60, 120):
        out = f(5.0 + (i - 60) * 0.5, i / 30)       # 15 cm/s
    assert abs(out - (5.0 + 59 * 0.5)) < 1.0


# --------------------------------------------------------------------------- PenTracker end-to-end
class ScriptedCamera:
    """Stands in for Webcam: the test pushes frames with chosen timestamps."""

    def __init__(self):
        self.fps = 30.0
        self._cond = threading.Condition()
        self._frame = None
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()

    def push(self, image, t):
        with self._cond:
            fid = self._frame.id + 1 if self._frame else 1
            self._frame = Frame(fid, t, image)
            self._cond.notify_all()
        return fid

    def latest(self):
        return self._frame

    def wait(self, after_id=0, timeout=1.0):
        with self._cond:
            self._cond.wait_for(lambda: not self._running or (self._frame and self._frame.id > after_id), timeout)
            f = self._frame
            return f if f is not None and f.id > after_id else None


@pytest.fixture
def tracker(tmp_path):
    calib = tmp_path / "calib.json"
    PageCalibration(CORNERS, (W, H)).save(str(calib))
    marker = tmp_path / "marker.json"
    import json
    marker.write_text(json.dumps(GREEN))
    clock = SimClock(100.0)
    cam = ScriptedCamera()
    tr = PenTracker(clock, calib_path=str(calib), marker_path=str(marker), camera=cam,
                    startup_calibrate=False, detect_mode="marker")
    tr.start()
    yield tr, cam, clock
    tr.stop()


def _push_and_wait(tr, cam, img, t):
    cam.push(img, t)
    end = time.time() + 2.0
    while time.time() < end:
        f = tr._frame
        if f is not None and f.t == t:
            return
        time.sleep(0.002)
    raise AssertionError("tracker did not process the frame")


def test_pen_tracker_end_to_end(tracker):
    tr, cam, clock = tracker
    assert isinstance(tr, Tracker)
    assert tr.read() is None                         # no frame yet
    for i in range(10):                              # pen moving right at 0.3 page/s
        t = 100.0 + i / 30
        clock.t = t
        _push_and_wait(tr, cam, page_image((0.30 + 0.01 * i, 0.50)), t)
    r = tr.read()
    assert r.pen is not None and r.pen.confidence == 1.0
    # light smoothing: < 1.5 mm of lag at 6.5 cm/s
    assert r.pen.x == pytest.approx(0.39, abs=0.007) and r.pen.y == pytest.approx(0.50, abs=0.005)
    assert r.frame.shape[:2] == (559, 432)           # page-rectified HUD frame

    # hand covers the pen: frames keep coming but without the marker
    t_last = clock.t
    for i in range(1, 12):
        t = t_last + i / 30
        clock.t = t
        _push_and_wait(tr, cam, page_image(None), t)
        p = tr.read().pen
        if t - t_last <= FRESH_S:
            assert p is not None and p.confidence == 1.0
        elif t - t_last <= HOLD_S:
            assert p is not None and 0 < p.confidence < 1 and p.x > 0.39   # extrapolated forward
        else:
            assert p is None

    snap = tr.snapshot()
    assert snap.shape[:2] == (1118, 864)


def test_pen_tracker_calibrates_itself_when_no_file(tmp_path):
    import json
    calib = tmp_path / "calib.json"
    marker = tmp_path / "marker.json"
    marker.write_text(json.dumps(GREEN))
    clock = SimClock(0.0)
    cam = ScriptedCamera()
    tr = PenTracker(clock, calib_path=str(calib), marker_path=str(marker), camera=cam)
    tr.start()
    try:
        img = page_image((0.5, 0.5))
        for i in range(1, 75):                           # 2.5 s of a still page at 30 fps
            clock.t = i / 30
            _push_and_wait(tr, cam, img, clock.t)
        assert calib.exists() and tr.calibration is not None
        saved = PageCalibration.load(str(calib))
        assert np.abs(saved.corners_px - CORNERS).max() < 4
        clock.t += 1 / 30
        _push_and_wait(tr, cam, img, clock.t)
        p = tr.read().pen
        assert p is not None and abs(p.x - 0.5) < 0.01 and abs(p.y - 0.5) < 0.01
    finally:
        tr.stop()


def test_pen_tracker_tracks_fingertip_on_and_off_the_page(tmp_path):
    calib_path = tmp_path / "calib.json"
    calib = PageCalibration(CORNERS, (W, H))
    calib.save(str(calib_path))
    clock = SimClock(0.0)
    cam = ScriptedCamera()
    tr = PenTracker(clock, calib_path=str(calib_path), marker_path=str(tmp_path / "none.json"),
                    camera=cam, startup_calibrate=False)          # no marker tuned -> tip mode
    tr.start()
    try:
        for i in range(1, 31):                                    # 1 s of empty, still scene = background
            clock.t = i / 30
            _push_and_wait(tr, cam, page_image(), clock.t)
        assert tr.tip.has_background
        for target in [(0.40, 0.60), (-0.15, 0.50), (1.20, 0.40)]:
            tip = calib.to_pixels(*target)
            for _ in range(20):
                clock.t += 1 / 30
                _push_and_wait(tr, cam, hand_image(tip), clock.t)
            p = tr.read().pen
            assert p is not None and abs(p.x - target[0]) < 0.02 and abs(p.y - target[1]) < 0.02, (target, p)
            assert tr.last_detection().source == "tip"
            assert page_side(p.x, p.y) == {0.40: None, -0.15: "left", 1.20: "right"}[target[0]]
    finally:
        tr.stop()


def test_pen_tracker_startup_search_relocks_and_learns_background(tmp_path):
    calib_path = tmp_path / "calib.json"
    PageCalibration(CORNERS + 25, (W, H)).save(str(calib_path))   # stale: page has moved since
    clock = SimClock(0.0)
    cam = ScriptedCamera()
    tr = PenTracker(clock, calib_path=str(calib_path), marker_path=str(tmp_path / "none.json"), camera=cam)
    tr.start()
    try:
        assert tr.calibration_status()[0]                          # searching at startup
        for i in range(1, 75):
            clock.t = i / 30
            _push_and_wait(tr, cam, page_image(), clock.t)
        assert not tr.calibration_status()[0]
        assert np.abs(tr.calibration.corners_px - CORNERS).max() < 4
        assert np.abs(PageCalibration.load(str(calib_path)).corners_px - CORNERS).max() < 4
        assert tr.tip.has_background
    finally:
        tr.stop()
