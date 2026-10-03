"""Simulated Arduino bridge for testing haptic guidance without hardware."""

from dataclasses import dataclass
from src.config import config


@dataclass
class Telemetry:
    left_hz: int
    right_hz: int
    dx_cm: float
    dy_cm: float
    dist_cm: float
    elevation_cm: float
    is_locked: bool
    mock_occlusion_ratio: float


class MockHapticBridge:
    def __init__(self):
        self.locked = False
        self.left_hz = 0
        self.right_hz = 0

    def compute_guidance(
        self,
        hand_norm: tuple[float, float],
        target_norm: tuple[float, float],
        elevation_cm: float,
    ) -> Telemetry:
        # Physical metric offset in centimeters across the document plane
        dx_cm = (target_norm[0] - hand_norm[0]) * config.DOCUMENT_WIDTH_CM
        dy_cm = (target_norm[1] - hand_norm[1]) * config.DOCUMENT_HEIGHT_CM
        dist_cm = (dx_cm**2 + dy_cm**2) ** 0.5

        # Lock check: within target radius and resting flat on paper (< 2.0 cm lift)
        if dist_cm <= config.TARGET_LOCK_DIST_CM and elevation_cm <= config.TARGET_LOCK_MAX_ELEVATION_CM:
            self.locked = True
            self.left_hz = 1000
            self.right_hz = 1000
            occlusion_ratio = 0.15  # Palm shadow detected on document
            return Telemetry(
                self.left_hz, self.right_hz, dx_cm, dy_cm, dist_cm, elevation_cm, True, occlusion_ratio
            )

        self.locked = False
        occlusion_ratio = 0.95  # Ambient room light unblocked

        # Pulse frequency modulation: closer -> higher frequency (2 Hz to 45 Hz)
        base_freq = int(max(2, min(45, 60.0 / (dist_cm + 0.5))))

        DEADZONE_CM = 1.2
        if dx_cm < -DEADZONE_CM:  # Target is to the left of the hand
            self.left_hz = base_freq
            self.right_hz = 0
        elif dx_cm > DEADZONE_CM:  # Target is to the right of the hand
            self.left_hz = 0
            self.right_hz = base_freq
        else:  # Centered on X axis
            self.left_hz = base_freq
            self.right_hz = base_freq

        return Telemetry(
            self.left_hz, self.right_hz, dx_cm, dy_cm, dist_cm, elevation_cm, False, occlusion_ratio
        )