import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.quality_screening.config import QualityScreeningConfig
from pipeline.quality_screening.screening import run_quality_screening


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH


def _dark_image(base_value: int = 8) -> np.ndarray:
    return np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), base_value, dtype=np.uint8)


def _with_bright_fraction(
    bright_fraction: float,
    bright_value: int,
    base_value: int = 8,
) -> np.ndarray:
    image = _dark_image(base_value)
    bright_pixels = int(round(IMAGE_PIXELS * bright_fraction))

    if bright_pixels == 0:
        return image

    flat = image.reshape(-1, 3)
    flat[:bright_pixels] = bright_value
    return image


def _with_fire_patch(
    top: int,
    left: int,
    height: int,
    width: int,
    color: tuple[int, int, int],
    base_value: int = 8,
) -> np.ndarray:
    image = _dark_image(base_value)
    image[top : top + height, left : left + width] = color
    return image


@pytest.mark.parametrize(
    ("name", "image", "expected_brightness_passed"),
    [
        ("black_frame", _dark_image(0), False),
        ("very_dark_frame", _dark_image(8), False),
        ("dark_gray_frame", _dark_image(25), False),
        (
            "large_dim_glow_below_threshold",
            _with_bright_fraction(0.08, bright_value=149),
            False,
        ),
        (
            "tiny_bright_region_below_fraction",
            _with_bright_fraction(0.005, bright_value=220),
            False,
        ),
        (
            "exact_minimum_bright_fraction",
            _with_bright_fraction(0.01, bright_value=180),
            True,
        ),
        (
            "small_visible_fire_fraction",
            _with_bright_fraction(0.015, bright_value=210),
            True,
        ),
        (
            "larger_visible_fire_fraction",
            _with_bright_fraction(0.04, bright_value=230),
            True,
        ),
        (
            "warm_fire_patch",
            _with_fire_patch(200, 260, 64, 64, color=(30, 190, 255)),
            True,
        ),
        (
            "small_warm_fire_patch_below_fraction",
            _with_fire_patch(220, 300, 32, 32, color=(30, 190, 255)),
            False,
        ),
        (
            "white_hot_fire_patch",
            _with_fire_patch(160, 240, 80, 80, color=(240, 240, 240)),
            True,
        ),
        (
            "dim_smoke_only_patch",
            _with_fire_patch(160, 240, 120, 120, color=(90, 90, 90)),
            False,
        ),
    ],
)
def test_dark_images_are_screened_by_bright_pixel_fraction(
    name: str,
    image: np.ndarray,
    expected_brightness_passed: bool,
):
    config = QualityScreeningConfig(
        min_contrast=0.0,
        max_contrast=255.0,
        min_sharpness=0.0,
        min_texture=0.0,
        min_valid_tile_fraction=0.0,
        min_sharp_tile_fraction=0.0,
    )

    result = run_quality_screening(image, config=config)
    brightness_result = result["checks"]["brightness"]

    assert brightness_result["brightness"] < config.min_brightness, name
    assert brightness_result["passed"] is expected_brightness_passed, name
    if expected_brightness_passed:
        assert brightness_result["reason"] == "", name
    else:
        assert brightness_result["reason"] == "underexposed", name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
