"""
Track a hand moving toward / away from the camera using depth, with a sheet of
paper as a reference: hand height above the paper and position over it.

Everything is kept in memory - nothing is written to disk.

Start with the hand OUT of view so the background (and paper) can be learned (~1 s).
The paper should be fully visible (all 4 corners) and the phone must stay still.
Controls (click the window first):
  b  - re-learn background and paper (keep hand out of view)
  p  - re-detect paper only
  c  - mark the paper manually: click its 4 corners in the Hand depth or Camera window
  q  - quit
"""
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

from live_depth import LiveDepth, DEVICE_TYPE__TRUEDEPTH
from paper import Paper, hand_relative_to_paper


def fit_slope(t, z, now, window):
    """Slope of z over the last `window` seconds (units/s), ignoring NaNs; NaN if too few points."""
    t, z = np.asarray(t), np.asarray(z)
    sel = (t >= now - window) & np.isfinite(z)
    if sel.sum() < 3 or np.ptp(t[sel]) == 0:
        return float("nan")
    return float(np.polyfit(t[sel], z[sel], 1)[0])


class HandTracker:
    """
    Keeps a rolling history of depth frames (as flattened vectors) and a time series
    of the hand's depth, so you can compute how depth changes as the hand moves.
    """

    def __init__(self, frame_history=120, series_history=600, bg_frames=30,
                 min_delta=0.03, min_area=150, velocity_window=0.25, still_speed=0.10):
        self.bg_frames = bg_frames            # frames used to learn the background
        self.min_delta = min_delta            # m closer than background to count as hand
        self.min_area = min_area              # min pixels for a blob to be a hand
        self.velocity_window = velocity_window  # s of history used for the speed fit
        self.still_speed = still_speed        # m/s below which the hand is "still"

        # Raw depth frames as flattened vectors (~2 s at 60 fps)
        self.frames = deque(maxlen=frame_history)
        self.frame_times = deque(maxlen=frame_history)
        # Per-frame hand measurements (~10 s at 60 fps); NaN when no hand
        self.times = deque(maxlen=series_history)
        self.hand_depth = deque(maxlen=series_history)
        self.hand_area = deque(maxlen=series_history)

        self.reset_background()

    def reset_background(self):
        self._bg_samples = []
        self.background = None
        self.present = False

    def depth_series(self):
        """(times, hand depth in m) as numpy vectors; depth is NaN where no hand."""
        return np.array(self.times), np.array(self.hand_depth)

    def frame_matrix(self):
        """Recent depth frames stacked as an (N, H*W) matrix, one row per frame."""
        return np.stack(self.frames) if self.frames else np.empty((0, 0), np.float32)

    def update(self, depth, t):
        d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.frames.append(d.ravel().copy())
        self.frame_times.append(t)

        if self.background is None:
            self._bg_samples.append(d)
            if len(self._bg_samples) >= self.bg_frames:
                # Farthest depth seen per pixel, not the median: a hand/object is always
                # closer than the table behind it, so this recovers the true table surface
                # even if something briefly passed through part of the frame during
                # calibration - the median would instead bake that object in as background.
                stack = np.stack(self._bg_samples)
                bg = np.where(stack > 0, stack, -np.inf).max(axis=0)
                bg[~np.isfinite(bg)] = np.inf  # no background return -> anything valid is in front
                self.background = bg
                self._bg_samples = []
            return {"state": "calibrating",
                    "progress": len(self._bg_samples) / self.bg_frames,
                    "mask": np.zeros(d.shape, bool)}

        mask = self._hand_mask(d)
        area = int(mask.sum())
        present = area >= self.min_area
        hd = float(np.median(d[mask])) if present else float("nan")

        self.times.append(t)
        self.hand_depth.append(hd)
        self.hand_area.append(area if present else 0)

        event = None
        if present and not self.present:
            event = "entered"
        elif not present and self.present:
            event = "left"
        self.present = present

        velocity = self._velocity(t)
        if not present:
            state = "no hand"
        elif np.isnan(velocity):
            state = "hand detected"
        elif velocity < -self.still_speed:
            state = "moving closer"
        elif velocity > self.still_speed:
            state = "moving away"
        else:
            state = "holding still"

        return {"state": state, "event": event, "present": present, "depth": hd,
                "velocity": velocity, "area": area, "mask": mask,
                "frame_change": self._frame_change()}

    def _hand_mask(self, d):
        closer = (d > 0) & ((self.background - d) > self.min_delta)
        m = cv2.morphologyEx(closer.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            return np.zeros(d.shape, bool)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest

    def _velocity(self, now):
        """Slope of hand depth over the last velocity_window seconds (m/s). Negative = closer."""
        return fit_slope(self.times, self.hand_depth, now, self.velocity_window)

    def _frame_change(self):
        """Mean absolute per-pixel depth change between the last two frames (m)."""
        if len(self.frames) < 2:
            return 0.0
        a, b = self.frames[-2], self.frames[-1]
        valid = (a > 0) & (b > 0)
        return float(np.abs(b[valid] - a[valid]).mean()) if valid.any() else 0.0


def draw_plot(times, depths, width, height=120, span=5.0, title="hand depth", up_is_larger=False):
    """A value over the last `span` seconds as a small line plot."""
    img = np.full((height, width, 3), 30, np.uint8)
    if len(times) == 0:
        return img
    t, z = np.asarray(times), np.asarray(depths)
    sel = (t >= t[-1] - span) & np.isfinite(z)
    cv2.putText(img, f"{title}, last {span:.0f} s", (5, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (180, 180, 180), 1, cv2.LINE_AA)
    if sel.sum() < 2:
        return img
    lo, hi = float(z[sel].min()) - 0.02, float(z[sel].max()) + 0.02
    xs = ((t[sel] - (t[-1] - span)) / span * (width - 1)).astype(np.int32)
    frac = (z[sel] - lo) / (hi - lo)
    if up_is_larger:
        frac = 1 - frac
    ys = (20 + frac * (height - 30)).astype(np.int32)
    cv2.polylines(img, [np.stack([xs, ys], 1)], False, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"{lo:.2f}-{hi:.2f} m", (width - 95, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (180, 180, 180), 1, cv2.LINE_AA)
    return img


def detect_paper(rgb, tracker, K):
    """Find the paper using the learned (hand-free, denoised) background depth."""
    bg = tracker.background.copy()
    bg[~np.isfinite(bg)] = 0
    debug = {}
    paper = Paper.from_frame(rgb, bg, K, debug=debug)
    if paper is None:
        print(f"Paper not found ({debug.get('candidates', 0)} candidate shapes; "
              f"rejected: {debug.get('rejected') or 'none'})")
        print("  Make sure all 4 corners are in view, or press c and click the 4 corners.")
    else:
        report_paper(paper)
    return paper


def report_paper(paper):
    tilt = np.degrees(np.arccos(abs(paper.normal[2])))
    note = " (page runs off-frame - size/position are approximate; height above it is still accurate)" \
        if paper.approximate else ""
    print(f"Paper found: {paper.width * 100:.1f} x {paper.height * 100:.1f} cm, "
          f"{np.linalg.norm(paper.center):.3f} m away, tilted {tilt:.0f} deg from camera{note}")


CAMERA_VIEW_SCALE = 0.5  # Camera window is shown at half the RGB resolution


def main():
    stream = LiveDepth()
    stream.connect()
    tracker = HandTracker()
    paper = None
    find_paper = True  # detect paper once background is learned
    # Hand height above paper over time (~10 s), NaN when no hand
    paper_times, paper_height = deque(maxlen=600), deque(maxlen=600)
    scale = 2  # enlarge the small depth image for viewing
    clicks = None  # depth-pixel corner clicks while in manual paper mode, else None
    shapes = {"depth": None, "rgb": None}  # current frame shapes, for click coordinate conversion

    def on_depth_click(event, x, y, flags, param):
        # (x, y) are pixels in the enlarged depth image (top of the "Hand depth" window)
        if event == cv2.EVENT_LBUTTONDOWN and clicks is not None and len(clicks) < 4:
            clicks.append((x / scale, y / scale))

    def on_camera_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or clicks is None or len(clicks) >= 4:
            return
        if shapes["depth"] is None:
            return
        rgb_x, rgb_y = x / CAMERA_VIEW_SCALE, y / CAMERA_VIEW_SCALE
        dh, dw = shapes["depth"][:2]
        rh, rw = shapes["rgb"][:2]
        clicks.append((rgb_x * dw / rw, rgb_y * dh / rh))

    cv2.namedWindow("Hand depth")
    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Hand depth", on_depth_click)
    cv2.setMouseCallback("Camera", on_camera_click)
    print("Keep your hand out of view while the background is learned...")

    try:
        while True:
            if not stream.event.wait(5):
                print("No frames for 5 s - is Record3D still streaming?")
                continue
            depth = stream.session.get_depth_frame()
            rgb = stream.session.get_rgb_frame()
            stream.event.clear()
            t = time.monotonic()
            K = stream.depth_intrinsics(depth.shape, rgb.shape)
            if stream.session.get_device_type() == DEVICE_TYPE__TRUEDEPTH:
                depth, rgb = cv2.flip(depth, 1), cv2.flip(rgb, 1)
                K = (K[0], K[1], depth.shape[1] - 1 - K[2], K[3])
            shapes["depth"], shapes["rgb"] = depth.shape, rgb.shape

            r = tracker.update(depth, t)
            if r.get("event"):
                print(f"[{t:.2f}] hand {r['event']}" +
                      (f" at {r['depth']:.3f} m" if r["event"] == "entered" else ""))

            if find_paper and tracker.background is not None:
                paper = detect_paper(rgb, tracker, K)
                find_paper = False

            if clicks is not None and len(clicks) == 4 and tracker.background is not None:
                bg = tracker.background.copy()
                bg[~np.isfinite(bg)] = 0
                # clicks are already in depth-pixel space; rgb_shape=depth.shape makes
                # from_corners' internal RGB->depth scaling a no-op
                paper = Paper.from_corners(np.array(clicks), depth.shape, bg, K)
                if paper is None:
                    print("Not enough depth inside those corners - press c to try again")
                else:
                    report_paper(paper)
                clicks = None
                paper_times.clear()
                paper_height.clear()

            rel = None
            if paper is not None and r.get("present"):
                rel = hand_relative_to_paper(depth, r["mask"], paper, K)
            if paper is not None and r["state"] != "calibrating":
                paper_times.append(t)
                paper_height.append(rel["height"] if rel else float("nan"))

            # Display
            norm = np.clip(np.nan_to_num(depth) / 3.0, 0, 1)
            vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            vis[r["mask"]] = (0.5 * vis[r["mask"]] + [0, 127, 0]).astype(np.uint8)
            vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            if paper is not None:
                cv2.polylines(vis, [np.round(paper.corners_px * scale).astype(np.int32)],
                              True, (255, 255, 255), 2)
            if clicks is not None:
                for cx, cy in clicks:
                    cv2.circle(vis, (int(cx * scale), int(cy * scale)), 6, (0, 0, 255), -1)
                cv2.putText(vis, f"click corner {len(clicks) + 1} of 4 (c to cancel)",
                            (6, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2, cv2.LINE_AA)

            if r["state"] == "calibrating":
                lines = [f"learning background {r['progress']:.0%} - keep hand out"]
            else:
                lines = [r["state"]]
                if r["present"]:
                    v = r["velocity"]
                    lines.append(f"depth {r['depth']:.3f} m")
                    lines.append("speed --" if np.isnan(v) else f"speed {v:+.2f} m/s")
                if paper is None:
                    lines.append("no paper (p retry, c click corners)")
                elif rel:
                    pv = fit_slope(paper_times, paper_height, t, tracker.velocity_window)
                    lines.append(f"above paper {rel['height'] * 100:.1f} cm"
                                 f" (closest {rel['closest'] * 100:.1f})")
                    if not np.isnan(pv):
                        lines.append(f"  {pv * 100:+.0f} cm/s vs paper")
                    x, y = rel["xy"] * 100
                    approx = " ~" if paper.approximate else ""
                    lines.append(f"over paper{approx} x {x:+.1f} y {y:+.1f} cm" if rel["over_paper"]
                                 else f"off paper{approx} (x {x:+.1f} y {y:+.1f} cm)")
            for i, s in enumerate(lines):
                cv2.putText(vis, s, (6, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2, cv2.LINE_AA)

            if paper is not None:
                plot = draw_plot(paper_times, paper_height, vis.shape[1],
                                 title="height above paper", up_is_larger=True)
            else:
                plot = draw_plot(tracker.times, tracker.hand_depth, vis.shape[1])
            cv2.imshow("Hand depth", np.vstack([vis, plot]))

            # Camera view with the detected paper outlined, to check detection
            cam = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if paper is not None:
                to_rgb = np.array([rgb.shape[1] / depth.shape[1], rgb.shape[0] / depth.shape[0]])
                cv2.polylines(cam, [np.round(paper.corners_px * to_rgb).astype(np.int32)],
                              True, (0, 255, 0), 4)
            cam = cv2.resize(cam, None, fx=CAMERA_VIEW_SCALE, fy=CAMERA_VIEW_SCALE)
            if clicks is not None:
                to_rgb = np.array([rgb.shape[1] / depth.shape[1], rgb.shape[0] / depth.shape[0]])
                for cx, cy in clicks:
                    rx, ry = np.array([cx, cy]) * to_rgb * CAMERA_VIEW_SCALE
                    cv2.circle(cam, (int(rx), int(ry)), 5, (0, 0, 255), -1)
                cv2.putText(cam, f"click paper corner {len(clicks) + 1} of 4 (c to cancel)",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Camera", cam)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("b"):
                tracker.reset_background()
                paper, find_paper = None, True
                paper_times.clear()
                paper_height.clear()
                print("Re-learning background and paper - keep hand out of view")
            elif key == ord("p") and tracker.background is not None:
                paper = detect_paper(rgb, tracker, K)
                paper_times.clear()
                paper_height.clear()
            elif key == ord("c"):
                if clicks is None and tracker.background is not None:
                    clicks = []
                    print("Click the paper's 4 corners in the Hand depth or Camera window")
                else:
                    clicks = None
            elif key == ord("q"):
                break
    finally:
        stream.disconnect()
        cv2.destroyAllWindows()
        # record3d's native teardown can corrupt the heap on Windows; skip it
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
