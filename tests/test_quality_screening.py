import numpy as np

from quality_screening.resolution import check_resolution
from quality_screening.brightness import check_brightness


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


def test_brightness_check_fails_for_overexposed_image():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    result = check_brightness(image)
    assert result["passed"] is False
    assert result["reason"] == "overexposed"
    assert result["brightness"] == 255.0


def test_brightness_thresholds_can_be_overridden():
    image = np.full((480, 640, 3), 60, dtype=np.uint8)
    assert check_brightness(image)["passed"] is True
    result = check_brightness(image, min_brightness=70.0)
    assert result["passed"] is False
    assert result["reason"] == "underexposed"
