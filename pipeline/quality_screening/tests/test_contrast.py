import numpy as np
import pytest

# Run tests with: uv run pytest tests/ -v
from pipeline.quality_screening.contrast import check_contrast


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


def test_result_contains_required_fields():
    image = np.full((480, 640, 3), 80, dtype=np.uint8)
    image[240:, :] = 180
    result = check_contrast(image)
    assert "passed" in result
    assert "reason" in result
    assert "contrast" in result
    assert isinstance(result["contrast"], float)


def test_high_contrast_image_fails():
    # Pure black/white checkerboard gives standard deviance of around 127, above max_contrast of 120
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[::2, ::2] = 255
    image[1::2, 1::2] = 255
    result = check_contrast(image)
    assert result["passed"] is False
    assert result["reason"] != ""


def test_custom_thresholds_can_be_overridden():
    # Uniform image fails the default min_contrast but passes a custom threshold of 0
    image = np.full((480, 640, 3), 128, dtype=np.uint8)
    assert check_contrast(image)["passed"] is False
    assert check_contrast(image, min_contrast=0.0)["passed"] is True


def test_negative_thresholds_are_not_allowed():
    image = np.full((480, 640, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError):
        check_contrast(image, min_contrast=-1.0, max_contrast=-1.0)


def test_none_image_is_not_allowed():
    with pytest.raises(ValueError):
        check_contrast(None)


def test_non_ndarray_image_is_not_allowed():
    with pytest.raises(TypeError):
        check_contrast([[1, 2], [3, 4]])


def test_empty_image_is_not_allowed():
    image = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        check_contrast(image)


def test_min_contrast_greater_than_max_contrast_is_not_allowed():
    image = np.full((480, 640, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError):
        check_contrast(image, min_contrast=100.0, max_contrast=50.0)
