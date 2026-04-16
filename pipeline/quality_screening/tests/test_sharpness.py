import cv2
import numpy as np
import pytest

# Run tests with: uv run pytest tests/ -v
from pipeline.quality_screening.sharpness import check_sharpness


def test_sharp_image_passes():
    # Image with strong edges produces high Laplacian variance
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (590, 430), (255, 255, 255), thickness=2)
    cv2.line(image, (0, 0), (640, 480), (255, 255, 255), thickness=2)
    result = check_sharpness(image)
    assert result["passed"] is True


def test_blurry_image_fails():
    # Heavy Gaussian blur destroys edges, producing almost zero Laplacian variance
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (51, 51), sigmaX=30)
    result = check_sharpness(image)
    assert result["passed"] is False
    assert result["reason"] != ""


def test_result_contains_required_fields():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (590, 430), (255, 255, 255), thickness=2)
    result = check_sharpness(image)
    assert "passed" in result
    assert "reason" in result
    assert "sharpness" in result
    assert isinstance(result["sharpness"], float)


def test_custom_threshold_can_be_overridden():
    # Blurry image fails the default threshold but passes a very low custom one
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (51, 51), sigmaX=30)
    assert check_sharpness(image)["passed"] is False
    assert check_sharpness(image, min_sharpness=0.1)["passed"] is True


def test_negative_threshold_is_not_allowed():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        check_sharpness(image, min_sharpness=-1.0)


def test_none_image_is_not_allowed():
    with pytest.raises(ValueError):
        check_sharpness(None)


def test_non_ndarray_image_is_not_allowed():
    with pytest.raises(TypeError):
        check_sharpness([[1, 2], [3, 4]])


def test_empty_image_is_not_allowed():
    image = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        check_sharpness(image)
