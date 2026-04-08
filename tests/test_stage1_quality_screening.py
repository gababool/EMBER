import numpy as np

from stage1_quality_screening import Stage1QualityConfig, run_quality_screening
from pipeline import process_image_pipeline


def test_passes_valid_image():
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.rectangle(image, (50, 50), (750, 550), (200, 100, 50), thickness=-1)
    cv2.line(image, (50, 50), (750, 550), (255, 255, 255), thickness=2)
    result = run_quality_screening(image)
    assert result["passed"] is True
    assert result["confidence"] > 0.7
    assert result["reasons"] == []


def test_fails_low_sharpness():
    image = np.random.randint(60, 200, (600, 800, 3), dtype=np.uint8)
    cv2 = __import__("cv2")
    image = cv2.GaussianBlur(image, (29, 29), sigmaX=15)
    result = run_quality_screening(image)
    assert result["passed"] is False
    assert "low_sharpness" in result["reasons"]


def test_fails_low_contrast():
    image = np.full((600, 800, 3), 120, dtype=np.uint8)
    result = run_quality_screening(image)
    assert result["passed"] is False
    assert "low_contrast" in result["reasons"]


def test_fails_low_resolution():
    image = np.full((100, 100, 3), 150, dtype=np.uint8)
    result = run_quality_screening(image)
    assert result["passed"] is False
    assert "low_resolution" in result["reasons"]


def test_fails_low_brightness():
    image = np.full((600, 800, 3), 10, dtype=np.uint8)
    result = run_quality_screening(image)
    assert result["passed"] is False
    assert "low_brightness" in result["reasons"]


def test_pipeline_skips_stage2_on_reject():
    image = np.full((100, 100, 3), 10, dtype=np.uint8)
    result = process_image_pipeline(image)
    assert result["stage1"]["passed"] is False
    assert result["stage2"] is None
