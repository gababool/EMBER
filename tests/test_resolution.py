import numpy as np

# Run tests with: uv run pytest tests/ -v
from quality_screening.resolution import check_resolution


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
