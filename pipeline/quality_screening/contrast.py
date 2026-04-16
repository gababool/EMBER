# Run tests with: uv run pytest tests/ -v

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def check_contrast(
    image: np.ndarray,
    min_contrast: float = 30.0,
    max_contrast: float = 120.0,
) -> Dict[str, Any]:
    """Check whether an image falls within an acceptable contrast range.

    Contrast is measured as the standard deviation of pixel values in the
    grayscale image: a low value indicates a washed-out image,
    while a very high value indicates an overexposed or extreme image.

    Args:
        image: Input image as a numpy array (grayscale or BGR).
        min_contrast: Minimum acceptable contrast (std dev of pixel values).
        max_contrast: Maximum acceptable contrast (std dev of pixel values).

    Returns:
        A dict with:
            passed (bool): True if the image contrast is within the accepted range.
            reason (str): Empty string on pass, "low_contrast" or "high_contrast" on failure.
            contrast (float): Measured standard deviation of pixel values.
    """
    if image is None:
        raise ValueError("image must not be None.")
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy ndarray.")
    if image.size == 0:
        raise ValueError("image must not be empty.")
    if min_contrast < 0 or max_contrast < 0:
        raise ValueError("min_contrast and max_contrast must not be negative.")
    if min_contrast > max_contrast:
        raise ValueError("min_contrast must not be greater than max_contrast.")

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray.astype(np.float64)))

    if contrast < min_contrast:
        return {"passed": False, "reason": "low_contrast", "contrast": contrast}
    if contrast > max_contrast:
        return {"passed": False, "reason": "high_contrast", "contrast": contrast}

    return {"passed": True, "reason": "", "contrast": contrast}
