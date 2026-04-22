from __future__ import annotations

from typing import TypedDict

import cv2
import numpy as np

from pipeline.quality_screening._shared import (
    to_grayscale_uint8,
    validate_non_negative_number,
)

class SharpnessResult(TypedDict):
    passed: bool
    reason: str
    sharpness: float

def check_sharpness(
    image: np.ndarray,
    min_sharpness: float = 100.0,
) -> SharpnessResult:
    """Check whether an image meets the minimum sharpness threshold."""
    validate_non_negative_number("min_sharpness", min_sharpness)

    gray = to_grayscale_uint8(image)
    if gray.size == 0:
        return {
            "passed": False,
            "reason": "invalid_image",
            "sharpness": 0.0,
        }
    
    gray = to_grayscale_uint8(image)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    passed = sharpness >= min_sharpness
    return {
        "passed": passed,
        "reason": "" if passed else "low_sharpness",
        "sharpness": sharpness,
    }