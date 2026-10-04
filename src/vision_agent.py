"""Target localization using Gemini 2.5 Flash and Structured Outputs."""

import cv2
import numpy as np
from typing import Optional, Tuple
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from src.config import config


class TargetBoundingBox(BaseModel):
    label: str = Field(description="Name or brief description of the detected element")
    ymin: int = Field(description="Top edge coordinate normalized to 0-1000")
    xmin: int = Field(description="Left edge coordinate normalized to 0-1000")
    ymax: int = Field(description="Bottom edge coordinate normalized to 0-1000")
    xmax: int = Field(description="Right edge coordinate normalized to 0-1000")


class VisionAgent:
    def __init__(self, api_key: str = config.GEMINI_API_KEY):
        self.client = genai.Client(api_key=api_key)

    def locate_target(
        self, frame: np.ndarray, query: str
    ) -> Tuple[Optional[Tuple[float, float]], Optional[TargetBoundingBox]]:
        """
        Locates a requested visual target on the document.
        Returns:
            ((norm_center_x, norm_center_y), bounding_box) where coords are [0.0, 1.0].
        """
        success, encoded_img = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return None, None

        prompt = (
            f"You are a spatial reader for assistive navigation. Detect the exact physical location of: '{query}'. "
            "Return the tightest bounding box enclosing that region or input line, normalized from 0 to 1000."
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=encoded_img.tobytes(), mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TargetBoundingBox,
                    temperature=0.1,
                ),
            )
            box = TargetBoundingBox.model_validate_json(response.text)

            center_x = ((box.xmin + box.xmax) / 2.0) / 1000.0
            center_y = ((box.ymin + box.ymax) / 2.0) / 1000.0

            return (center_x, center_y), box

        except Exception as e:
            print(f"[VisionAgent] Target localization error: {e}")
            return None, None
