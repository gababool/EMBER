import numpy as np
import pytest

# Run tests with: uv run pytest tests/ -v
from pipeline.quality_screening.brightness import check_brightness


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


@pytest.mark.parametrize(
    ("name", "image", "expected_passed"),
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
    expected_passed: bool,
):
    result = check_brightness(image)

    assert result["brightness"] < 40.0, name
    assert result["passed"] is expected_passed, name
    if expected_passed:
        assert result["reason"] == "", name
    else:
        assert result["reason"] == "underexposed", name
