from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pipeline.quality_screening._shared import (
    fraction_at_or_above_threshold,
    to_grayscale_uint8,
    validate_fraction,
    validate_min_max,
    validate_uint8_threshold,
)


def check_brightness(
    image: np.ndarray,
    min_brightness: float = 40.0,
    max_brightness: float = 220.0,
    bright_pixel_threshold: float = 150.0,
    min_bright_pixel_fraction: float = 0.01,
) -> Dict[str, Any]:
    """Check whether an image has acceptable global or localized brightness."""
    validate_min_max(
        "min_brightness",
        min_brightness,
        "max_brightness",
        max_brightness,
    )
    validate_uint8_threshold("bright_pixel_threshold", bright_pixel_threshold)
    validate_fraction("min_bright_pixel_fraction", min_bright_pixel_fraction)

    gray = to_grayscale_uint8(image)
    brightness = float(np.mean(gray))
    bright_pixel_fraction = fraction_at_or_above_threshold(
        gray,
        bright_pixel_threshold,
    )

    average_brightness_passed = min_brightness <= brightness <= max_brightness
    bright_region_passed = (
        brightness < min_brightness
        and bright_pixel_fraction >= min_bright_pixel_fraction
    )
    passed = average_brightness_passed or bright_region_passed

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
        "bright_pixel_fraction": bright_pixel_fraction,
        "bright_pixel_threshold": bright_pixel_threshold,
        "min_bright_pixel_fraction": min_bright_pixel_fraction,
    }
