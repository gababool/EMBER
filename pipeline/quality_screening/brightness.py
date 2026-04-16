from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def check_brightness(
    image: np.ndarray,
    min_brightness: float = 40.0,
    max_brightness: float = 220.0,
) -> Dict[str, Any]:
    """Check whether an image falls within an acceptable brightness range.

    Args:
        image: Input image as a numpy array (grayscale or BGR).
        min_brightness: Minimum acceptable average brightness (0-255).
        max_brightness: Maximum acceptable average brightness (0-255).

    Returns:
        A dict with:
            passed (bool): True if the image brightness is within range.
            reason (str): Empty string on pass; "underexposed" or "overexposed" on failure.
            brightness (float): Measured average brightness of the image.
    """
    if image is None:
        raise ValueError("image must not be None.")
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy ndarray.")
    if image.size == 0:
        raise ValueError("image must not be empty.")

    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("image must be a grayscale or BGR image.")

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
