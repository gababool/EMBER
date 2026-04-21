import numpy as np
import pytest

# Run tests with: uv run pytest tests/ -v
from pipeline.quality_screening.brightness import check_brightness


def test_brightness_check_passes_for_normal_lighting():
    image = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = check_brightness(image)
    assert result["passed"] is True
    assert result["reason"] == ""
    assert 0 <= result["brightness"] <= 255


def test_brightness_check_fails_for_too_dark_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = check_brightness(image)
    assert result["passed"] is False
    assert result["reason"] == "underexposed"
    assert result["brightness"] == 0.0


def test_brightness_check_fails_for_too_bright_image():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    result = check_brightness(image)
    assert result["passed"] is False
    assert result["reason"] == "overexposed"
    assert result["brightness"] == 255.0


def test_result_contains_required_fields():
    image = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = check_brightness(image)
    assert "passed" in result
    assert "reason" in result
    assert "brightness" in result
    assert isinstance(result["brightness"], float)


def test_custom_thresholds_can_be_overridden():
    # Too dark image fails default but passes custom low min
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert check_brightness(image)["passed"] is False
    assert check_brightness(image, min_brightness=0.0)["passed"] is True

    # Too bright image fails default but passes custom high max
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    assert check_brightness(image)["passed"] is False
    assert check_brightness(image, max_brightness=255.0)["passed"] is True

def test_empty_image_is_not_allowed():
    image = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        check_brightness(image)

def test_none_image_is_not_allowed():
    with pytest.raises(ValueError):
        check_brightness(None)

def test_non_ndarray_image_is_not_allowed():
    with pytest.raises(TypeError):
        check_brightness([[1, 2], [3, 4]])

def test_negative_threshold_is_not_allowed():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        check_brightness(image, min_brightness=-1.0)
        
def test_brightness_accepts_bgra_images():
    image = np.full((480, 640, 4), 128, dtype=np.uint8)
    result = check_brightness(image)
    assert result["passed"] is True

def test_brightness_rejects_nan_values():
    image = np.full((480, 640, 3), 0.5, dtype=np.float32)
    image[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        check_brightness(image)

def test_brightness_accepts_float_images_in_zero_to_one_range():
    image = np.full((480, 640, 3), 0.5, dtype=np.float32)
    result = check_brightness(image)
    assert result["passed"] is True
    assert 120.0 <= result["brightness"] <= 135.0
