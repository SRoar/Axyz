"""High-resolution camera capture and optical depth estimation."""

import cv2
import numpy as np
from typing import Optional, Tuple
from src.config import config


class VisionTracker:
    def __init__(self, camera_index: int = config.CAMERA_INDEX):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        # Baseline width of hand at zero elevation
        self.baseline_px = config.HAND_FLAT_WIDTH_PX

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a single frame from the camera stream."""
        ret, frame = self.cap.read()
        return ret, frame

    def estimate_hand_pose(
        self, frame: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float]], float, Optional[Tuple[int, int, int, int]]]:
        """
        Extracts normalized hand coordinates (x_norm, y_norm) and estimated Z-lift in cm.
        Returns:
            ((norm_x, norm_y), elevation_cm, bounding_rect_px)
        """
        h_frame, w_frame = frame.shape[:2]

        # Convert to HSV to segment skin/hand module
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Standard skin/glove range (adjustable if using colored glove/marker)
        mask = cv2.inRange(
            hsv, 
            np.array([0, 20, 60], dtype=np.uint8), 
            np.array([25, 255, 255], dtype=np.uint8)
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0, None

        # Take the most prominent foreground contour
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 1000:
            return None, 0.0, None

        x, y, w, h = cv2.boundingRect(largest)

        # Normalized coordinates (0.0 to 1.0)
        norm_x = (x + w / 2.0) / w_frame
        norm_y = (y + h / 2.0) / h_frame

        # Depth ratio: when hand moves higher, it appears larger in pixels
        scale_ratio = float(w) / self.baseline_px
        elevation_cm = max(0.0, (scale_ratio - 1.0) * config.Z_SCALE_FACTOR)

        return (norm_x, norm_y), elevation_cm, (x, y, w, h)

    def release(self) -> None:
        self.cap.release()