from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pipeline.quality_screening._shared import (
    validate_image_array,
    validate_non_negative_int,
)


def check_resolution(
    image: np.ndarray,
    min_width: int = 640,
    min_height: int = 480,
) -> Dict[str, Any]:
    """Check whether an image meets the minimum resolution threshold."""
    validate_image_array(image)
    validate_non_negative_int("min_width", min_width)
    validate_non_negative_int("min_height", min_height)

    height, width = image.shape[:2]

    passed = width >= min_width and height >= min_height
    reason = "" if passed else "low_resolution"

    return {
        "passed": passed,
        "reason": reason,
        "resolution": [width, height],
    }