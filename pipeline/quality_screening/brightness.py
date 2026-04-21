from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pipeline.quality_screening._shared import to_grayscale_uint8, validate_min_max


def check_brightness(
    image: np.ndarray,
    min_brightness: float = 40.0,
    max_brightness: float = 220.0,
) -> Dict[str, Any]:
    """Check whether an image falls within an acceptable brightness range."""
    validate_min_max(
        "min_brightness",
        min_brightness,
        "max_brightness",
        max_brightness,
    )

    gray = to_grayscale_uint8(image)
    brightness = float(np.mean(gray))
    passed = min_brightness <= brightness <= max_brightness

    if passed:
        reason = ""
    elif brightness < min_brightness:
        reason = "underexposed"
    else:
        reason = "overexposed"

    return {
        "passed": passed,
        "reason": reason,
        "brightness": brightness,
    }