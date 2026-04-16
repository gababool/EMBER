import numpy as np
import pytest

# Run tests with: uv run pytest tests/ -v
from pipeline.quality_screening.resolution import check_resolution


def test_large_image_passes():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = check_resolution(image)
    assert result["passed"] is True


def test_small_image_fails():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = check_resolution(image)
    assert result["passed"] is False
    assert result["reason"] != ""


def test_result_contains_required_fields():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = check_resolution(image)
    assert "passed" in result
    assert "reason" in result
    assert "resolution" in result
    assert result["resolution"] == [640, 480]


def test_custom_threshold_can_be_overridden():
    # 200x200 fails the default 640x480 but passes a custom 100x100 minimum
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    assert check_resolution(image)["passed"] is False
    assert check_resolution(image, min_width=100, min_height=100)["passed"] is True

def test_negative_thresholds_are_not_allowed():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises((ValueError, TypeError)):
        check_resolution(image, min_width=-1, min_height=-1)


def test_none_image_is_not_allowed():
    with pytest.raises(ValueError):
        check_resolution(None)


def test_non_ndarray_image_is_not_allowed():
    with pytest.raises(TypeError):
        check_resolution([[1, 2], [3, 4]])


def test_empty_image_is_not_allowed():
    image = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        check_resolution(image)