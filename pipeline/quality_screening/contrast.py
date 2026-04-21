from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pipeline.quality_screening._shared import to_grayscale_uint8, validate_min_max


def check_contrast(
    image: np.ndarray,
    min_contrast: float = 30.0,
    max_contrast: float = 120.0,
) -> Dict[str, Any]:
    """Check whether an image falls within an acceptable contrast range."""
    validate_min_max(
        "min_contrast",
        min_contrast,
        "max_contrast",
        max_contrast,
    )

    gray = to_grayscale_uint8(image)
    contrast = float(np.std(gray.astype(np.float64)))

    if contrast < min_contrast:
        return {"passed": False, "reason": "low_contrast", "contrast": contrast}
    if contrast > max_contrast:
        return {"passed": False, "reason": "high_contrast", "contrast": contrast}

    return {"passed": True, "reason": "", "contrast": contrast}