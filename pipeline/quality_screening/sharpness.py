# Run tests with: uv run pytest tests/ -v

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def check_sharpness(
    image: np.ndarray,
    min_sharpness: float = 100.0,
) -> Dict[str, Any]:
    """Check whether an image meets the minimum sharpness threshold.

    Sharpness is measured as the variance of the Laplacian: a high value
    indicates a sharp image with clear edges while a low value indicates blur.

    Args:
        image: Input image as a numpy array (grayscale or BGR).
        min_sharpness: Minimum acceptable Laplacian variance.

    Returns:
        A dict with:
            passed (bool): True if the image meets the minimum sharpness.
            reason (str): Empty string on pass, "low_sharpness" on failure.
            sharpness (float): Measured Laplacian variance of the image.
    """
    if image is None:
        raise ValueError("image must not be None.")
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy ndarray.")
    if image.size == 0:
        raise ValueError("image must not be empty.")
    if min_sharpness < 0:
        raise ValueError("min_sharpness must not be negative.")

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))

    passed = sharpness >= min_sharpness
    reason = "" if passed else "low_sharpness"

    return {
        "passed": passed,
        "reason": reason,
        "sharpness": sharpness,
    }
