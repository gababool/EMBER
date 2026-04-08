import numpy as np
import pytest

from stage1_quality_screening.stage1_quality_screening import check_resolution


def test_large_image_passes():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = check_resolution(image)
    assert result["passed"] is True


def test_small_image_fails():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = check_resolution(image)
    assert result["passed"] is False
    assert result["reason"] != ""
