"""Configuration constants for TactileReader."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

    # Hardware Serial
    ARDUINO_PORT: str = os.getenv("ARDUINO_PORT", "/dev/cu.usbmodem1101")
    BAUD_RATE: int = 115200

    # Vision & Camera
    CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "1"))
    FRAME_WIDTH: int = 1920
    FRAME_HEIGHT: int = 1080

    # Physical Surface Dimensions (cm) - Standard Letter/A4
    DOCUMENT_WIDTH_CM: float = 21.5
    DOCUMENT_HEIGHT_CM: float = 28.0

    # Optical Depth Calibration
    # Hand contour pixel width when resting flat on the document
    HAND_FLAT_WIDTH_PX: float = 160.0
    # Elevation scalar to convert scale ratio to cm
    Z_SCALE_FACTOR: float = 10.0

    # Tolerances
    TARGET_LOCK_DIST_CM: float = 1.5
    TARGET_LOCK_MAX_ELEVATION_CM: float = 2.0


config = AppConfig()