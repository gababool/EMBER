# Run tests with: uv run pytest tests/ -v

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def check_resolution(
    image: np.ndarray,
    min_width: int = 640,
    min_height: int = 480,
) -> Dict[str, Any]:
    """Check whether an image meets the minimum resolution threshold.

    Args:
        image: Input image as a numpy array (grayscale or BGR).
        min_width: Minimum acceptable width in pixels.
        min_height: Minimum acceptable height in pixels.

    Returns:
        A dict with:
            passed (bool): True if the image meets the minimum resolution.
            reason (str): Empty string on pass; "low_resolution" on failure.
            resolution (list[int, int]): Actual [width, height] of the image.
    """
    if image is None:
        raise ValueError("image must not be None.")
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy ndarray.")
    if image.size == 0:
        raise ValueError("image must not be empty.")
    if min_width < 0 or min_height < 0:
        raise ValueError("min_width and min_height must not be negative.")

    height, width = image.shape[:2]

    passed = width >= min_width and height >= min_height
    reason = "" if passed else "low_resolution"

    return {
        "passed": passed,
        "reason": reason,
        "resolution": [width, height],
    }
