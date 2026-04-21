from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from pipeline.quality_screening._shared import (
    to_grayscale_uint8,
    validate_non_negative_number,
)


def check_sharpness(
    image: np.ndarray,
    min_sharpness: float = 100.0,
) -> Dict[str, Any]:
    """Check whether an image meets the minimum sharpness threshold."""
    validate_non_negative_number("min_sharpness", min_sharpness)

    gray = to_grayscale_uint8(image)
    sharpness = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))

    passed = sharpness >= min_sharpness
    reason = "" if passed else "low_sharpness"

    return {
        "passed": passed,
        "reason": reason,
        "sharpness": sharpness,
    }