import numpy as np

# Run tests with: uv run pytest tests/ -v
from quality_screening.contrast import check_contrast


def test_normal_image_passes():
    # Half dark-gray (80), half light-gray (180) gives standard deviance around 50, within [30, 120]
    image = np.full((480, 640, 3), 80, dtype=np.uint8)
    image[240:, :] = 180
    result = check_contrast(image)
    assert result["passed"] is True


def test_low_contrast_image_fails():
    # Nearly uniform image (all pixels = 128) gives standard deviance of 0, below min_contrast of 30
    image = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = check_contrast(image)
    assert result["passed"] is False
    assert result["reason"] != ""
