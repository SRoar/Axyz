import cv2
import numpy as np

from src.depth_plane import TablePlane, fit_inverse_plane
from src.hand_tip import TipDetector
from src.pen_profile import PenProfile, build_profile, segment_pen
from tests.test_tracker import PEN, SKIN, WOOD, H, W, hand_image, hand_with_pen, page_image

TABLE_Z = 0.40
DS = 4              # depth frames are low resolution (Record3D: 192 x 256)


def tilted_table(h=H // DS, w=W // DS):
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return (1.0 / (2.5 + 0.0004 * u - 0.0006 * v)).astype(np.float32)


def depth_for(img, pen=None, hand_h=0.05):
    """LiDAR-like depth for a synthetic frame: table plane, skin raised by hand_h, and the pen
    rising linearly from height h0 at p0 to h1 at p1 (pen = (p0, h0, p1, h1), full-res pixels)."""
    table = tilted_table()
    depth = table.copy()
    small = cv2.resize(img, (W // DS, H // DS), interpolation=cv2.INTER_NEAREST)
    skin = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb), (25, 138, 97), (255, 185, 128)) > 0
    skin &= np.abs(small.astype(int) - np.int32(SKIN)).sum(axis=2) < 30
    depth[skin] -= hand_h
    if pen is not None:
        (x0, y0), h0, (x1, y1), h1 = pen
        for k in range(20):
            f0, f1 = k / 20, (k + 1) / 20
            seg = np.zeros(depth.shape, np.uint8)
            cv2.line(seg, (int((x0 + f0 * (x1 - x0)) / DS), int((y0 + f0 * (y1 - y0)) / DS)),
                     (int((x0 + f1 * (x1 - x0)) / DS), int((y0 + f1 * (y1 - y0)) / DS)), 255, 5)
            hk = h0 + (f0 + f1) / 2 * (h1 - h0)
            depth[seg > 0] = table[seg > 0] - hk
    return depth


def test_inverse_plane_fit_and_height_map():
    table = tilted_table()
    depth = table.copy()
    depth[10:40, 20:60] -= 0.05                       # a hand-sized box 5 cm up
    assert fit_inverse_plane(depth, np.ones(depth.shape, bool)) is not None
    tp = TablePlane()
    hgt = tp.height(depth)
    assert np.nanmax(np.abs(hgt[100:, 100:])) < 0.002
    assert abs(float(np.nanmedian(hgt[15:35, 25:55])) - 0.05) < 0.003
    conf = np.full(depth.shape, 2, np.uint8)
    conf[0:5, :] = 0
    assert np.isnan(tp.height(depth, conf)[0:5, :]).all()      # low-confidence depth is unknown


def test_depth_decides_the_writing_end_against_the_hand_pose():
    # same picture as a normal grip, but depth says the end near the palm touches the paper:
    # the lower end writes, whatever the hand pose suggests
    tip, back, palm = (760, 420), (520, 300), (680, 470)
    img = hand_with_pen(tip, back, palm, entry_px=(palm[0], H + 40), base=page_image(desk=WOOD))
    d = TipDetector().detect(img)
    assert d.end_cue == "hand" and np.hypot(d.x - back[0], d.y - back[1]) < 20, d    # pose alone: wrong end
    depth = depth_for(img, pen=(tip, 0.003, back, 0.06))
    d = TipDetector().detect(img, depth=depth)
    assert d.source == "pen" and d.end_cue == "depth", d
    assert np.hypot(d.x - tip[0], d.y - tip[1]) < 20, d
    assert d.height_m is not None and d.height_m < 0.012


def test_depth_rejects_a_flat_pen_shaped_shadow():
    img = hand_image((620, 300), base=page_image(desk=WOOD))
    cv2.line(img, (600, 330), (420, 400), PEN, 18)       # dark strip lying flat on the paper
    d = TipDetector().detect(img, depth=depth_for(img))
    assert d is not None and d.source == "finger", d
    assert np.hypot(d.x - 620, d.y - 300) < 14, d


def test_pen_profile_capture_and_probability(tmp_path):
    tip, back, palm = (520, 300), (760, 420), (680, 470)
    img = hand_with_pen(tip, back, palm, entry_px=(palm[0], H + 40), base=page_image(desk=WOOD))
    page = np.float32([[400, 80], [880, 90], [900, 660], [380, 650]])
    seg = segment_pen(img, page)
    assert seg is not None
    ends = [seg.tip, seg.back]
    assert min(np.hypot(*np.subtract(e, tip)) for e in ends) < 25       # one end at the nib
    if np.hypot(*np.subtract(seg.tip, tip)) > np.hypot(*np.subtract(seg.back, tip)):
        seg.tip, seg.back = seg.back, seg.tip
    prof = build_profile(img, seg)
    p = prof.probability(img)
    on_pen = p[cv2.line(np.zeros((H, W), np.uint8), tip, back, 255, 8) > 0]
    assert np.median(on_pen) > 200
    assert np.percentile(p[100:200, 450:600], 99) < 50                  # paper
    assert np.percentile(p[640:700, 640:720], 99) < 50                  # skin (arm)
    path = str(tmp_path / "pen.json")
    prof.save(path)
    again = PenProfile.load(path)
    assert again is not None and again.image_size == (W, H)
    assert np.abs(again.probability(img).astype(int) - p.astype(int)).max() <= 2
    d = TipDetector(profile=again).detect(img)
    assert d.source == "pen" and np.hypot(d.x - tip[0], d.y - tip[1]) < 12, d
