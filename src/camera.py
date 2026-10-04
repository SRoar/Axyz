"""RGBD capture engine supporting Record3D USB streaming and standard camera fallback."""

import threading
from typing import Optional, Tuple
import cv2
import numpy as np
from src.config import config

# Dynamic import: fallback to webcam if Record3D is not installed
try:
    from record3d import Record3DStream
    RECORD3D_AVAILABLE = True
except ImportError:
    RECORD3D_AVAILABLE = False


class VisionTracker:
    def __init__(self, camera_index: int = config.CAMERA_INDEX):
        self.lock = threading.Lock()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.is_running = False
        self.desk_depth_m: Optional[float] = None
        self.use_record3d = False

        if RECORD3D_AVAILABLE:
            try:
                self.stream = Record3DStream()
                self.stream.on_new_frame = self._on_new_record3d_frame
                self.stream.on_stream_stopped = self._on_record3d_stopped
                self.stream.connect()
                self.is_running = True
                self.use_record3d = True
                print("[VisionTracker] Successfully connected to Record3D USB stream.")
            except Exception as e:
                print(f"[VisionTracker] Could not connect to Record3D device ({e}). Falling back to OpenCV.")
                self.use_record3d = False

        if not self.use_record3d:
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            self.is_running = self.cap.isOpened()
            print(f"[VisionTracker] Initialized standard OpenCV camera on index {camera_index}.")

    def _on_new_record3d_frame(self):
        with self.lock:
            # Depth frame in meters (float32)
            self.latest_depth = np.copy(self.stream.get_depth_frame())
            # RGB frame (H, W, 3) in uint8 -> convert to BGR for OpenCV
            rgb = np.copy(self.stream.get_rgb_frame())
            self.latest_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _on_record3d_stopped(self):
        print("[VisionTracker] Record3D stream stopped.")
        self.is_running = False

    def is_opened(self) -> bool:
        if self.use_record3d:
            return self.is_running
        return self.cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns:
            (success, bgr_image, depth_map_meters)
        """
        if self.use_record3d:
            with self.lock:
                if self.latest_rgb is None:
                    return False, None, None
                depth = self.latest_depth.copy() if self.latest_depth is not None else None
                return True, self.latest_rgb.copy(), depth
        else:
            ret, frame = self.cap.read()
            return ret, frame, None

    def calibrate_desk_plane(self, num_samples: int = 10):
        """Samples the resting desk depth in meters to baseline hand elevation."""
        if not self.use_record3d:
            self.desk_depth_m = 0.50
            return

        samples = []
        for _ in range(num_samples):
            ret, _, depth = self.read_frame()
            if ret and depth is not None:
                h, w = depth.shape
                # Sample the center 20% of the desk field
                center_crop = depth[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)]
                valid = center_crop[~np.isnan(center_crop) & (center_crop > 0.05)]
                if len(valid) > 0:
                    samples.append(np.median(valid))

        if samples:
            self.desk_depth_m = float(np.median(samples))
            print(f"[VisionTracker] Desk baseline locked at {self.desk_depth_m * 100:.1f} cm.")
        else:
            self.desk_depth_m = 0.50

    def estimate_hand_pose(
        self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Tuple[float, float]], float, Optional[Tuple[int, int, int, int]]]:
        """
        Segments the hand using HSV and estimates metric Z-lift.
        Returns:
            ((norm_x, norm_y), elevation_cm, (x, y, w, h))
        """
        h_frame, w_frame = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, 20, 60], dtype=np.uint8),
            np.array([25, 255, 255], dtype=np.uint8),
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0, None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 1200:
            return None, 0.0, None

        x, y, w, h = cv2.boundingRect(largest)
        # Use top-most contour point as pointer tip instead of palm center
        topmost = tuple(largest[largest[:, :, 1].argmin()][0])
        norm_x = topmost[0] / float(w_frame)
        norm_y = topmost[1] / float(h_frame)

        # Compute metric Z-lift (distance above desk)
        elevation_cm = 0.0
        if depth_map is not None:
            dh, dw = depth_map.shape[:2]
            dcx = int(norm_x * dw)
            dcy = int(norm_y * dh)

            y1, y2 = max(0, dcy - 3), min(dh, dcy + 4)
            x1, x2 = max(0, dcx - 3), min(dw, dcx + 4)
            patch = depth_map[y1:y2, x1:x2]

            valid = patch[~np.isnan(patch) & (patch > 0.05)]
            if len(valid) > 0 and self.desk_depth_m is not None:
                hand_depth_m = float(np.median(valid))
                # Hand is closer to the camera than desk, so desk_depth > hand_depth
                elevation_cm = max(0.0, (self.desk_depth_m - hand_depth_m) * 100.0)
        else:
            # Fallback optical perspective calculation if no depth map
            scale_ratio = float(w) / config.HAND_FLAT_WIDTH_PX
            elevation_cm = max(0.0, (scale_ratio - 1.0) * config.Z_SCALE_FACTOR)

        return (norm_x, norm_y), elevation_cm, (x, y, w, h)

    def release(self) -> None:
        self.is_running = False
        if not self.use_record3d and hasattr(self, "cap"):
            self.cap.release()