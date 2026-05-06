import cv2
import numpy as np
import pytest

from pipeline.quality_screening.sharpness import check_sharpness


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TILE_SIZE = 128


def _blank_image(value: int = 32) -> np.ndarray:
    return np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), value, dtype=np.uint8)


def _draw_textured_tile(
    image: np.ndarray,
    row: int,
    col: int,
    *,
    blur: bool = False,
) -> None:
    y0 = row * TILE_SIZE
    x0 = col * TILE_SIZE
    tile = image[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE]

    tile[:] = 24
    cv2.rectangle(tile, (16, 16), (112, 112), (235, 235, 235), thickness=2)
    cv2.line(tile, (12, 116), (116, 12), (255, 255, 255), thickness=2)
    cv2.line(tile, (12, 64), (116, 64), (180, 180, 180), thickness=1)

    if blur:
        tile[:] = cv2.GaussianBlur(tile, (31, 31), sigmaX=10)


def _image_with_textured_tiles(
    sharp_tiles: int,
    blurred_tiles: int = 0,
) -> np.ndarray:
    image = _blank_image()
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for row, col in positions[:sharp_tiles]:
        _draw_textured_tile(image, row, col)

    start = sharp_tiles
    end = sharp_tiles + blurred_tiles
    for row, col in positions[start:end]:
        _draw_textured_tile(image, row, col, blur=True)

    return image


def test_sharp_textured_tiles_pass():
    image = _image_with_textured_tiles(sharp_tiles=4)

    result = check_sharpness(
        image,
        tile_size=TILE_SIZE,
        min_valid_tile_fraction=1.0,
        min_sharp_tile_fraction=1.0,
    )

    assert result["passed"] is True
    assert result["reason"] == ""
    assert result["total_tiles"] == 4
    assert result["valid_tiles"] == 4
    assert result["sharp_tiles"] == 4
    assert result["sharpness"] >= result["sharpness_threshold"]


def test_low_texture_image_fails_before_sharpness_scoring():
    image = _blank_image()

    result = check_sharpness(image, tile_size=TILE_SIZE)

    assert result["passed"] is False
    assert result["reason"] == "insufficient_textured_regions"
    assert result["total_tiles"] == 4
    assert result["valid_tiles"] == 0
    assert result["sharp_tiles"] == 0


def test_blurred_textured_tiles_fail_low_sharpness():
    image = _image_with_textured_tiles(sharp_tiles=0, blurred_tiles=4)

    result = check_sharpness(
        image,
        tile_size=TILE_SIZE,
        min_texture=1.0,
        min_valid_tile_fraction=1.0,
        min_sharp_tile_fraction=1.0,
    )

    assert result["passed"] is False
    assert result["reason"] == "low_sharpness"
    assert result["valid_tiles"] == 4
    assert result["sharp_tiles"] == 0
    assert result["sharpness"] < result["sharpness_threshold"]


def test_too_few_sharp_tiles_fail_even_when_percentile_is_high_enough():
    image = _image_with_textured_tiles(sharp_tiles=1, blurred_tiles=3)

    result = check_sharpness(
        image,
        min_sharpness=300.0,
        min_texture=1.0,
        tile_size=TILE_SIZE,
        min_valid_tile_fraction=1.0,
        min_sharp_tile_fraction=0.5,
        sharpness_percentile=100.0,
    )

    assert result["passed"] is False
    assert result["reason"] == "too_few_sharp_tiles"
    assert result["valid_tiles"] == 4
    assert result["sharp_tiles"] == 1
    assert result["sharpness"] >= result["sharpness_threshold"]


def test_result_contains_tile_based_fields():
    image = _image_with_textured_tiles(sharp_tiles=4)

    result = check_sharpness(image, tile_size=TILE_SIZE)

    assert "passed" in result
    assert "reason" in result
    assert "sharpness" in result
    assert "valid_tiles" in result
    assert "total_tiles" in result
    assert "sharp_tiles" in result
    assert "texture_threshold" in result
    assert "sharpness_threshold" in result
    assert "sharpness_percentile" in result
    assert isinstance(result["sharpness"], float)


def test_custom_tile_thresholds_can_be_overridden():
    image = _image_with_textured_tiles(sharp_tiles=1)

    assert check_sharpness(
        image,
        tile_size=TILE_SIZE,
        min_valid_tile_fraction=1.0,
    )["passed"] is False

    result = check_sharpness(
        image,
        tile_size=TILE_SIZE,
        min_valid_tile_fraction=0.25,
        min_sharp_tile_fraction=1.0,
    )

    assert result["passed"] is True
    assert result["valid_tiles"] == 1
    assert result["sharp_tiles"] == 1


def test_negative_threshold_is_not_allowed():
    image = _blank_image()
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


def test_zero_tile_size_is_not_allowed():
    image = _blank_image()
    with pytest.raises(ValueError):
        check_sharpness(image, tile_size=0)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("min_valid_tile_fraction", 1.1),
        ("min_sharp_tile_fraction", 1.1),
        ("sharpness_percentile", 101.0),
    ],
)
def test_fraction_and_percentile_arguments_are_bounded(argument: str, value: float):
    image = _blank_image()

    with pytest.raises(ValueError):
        check_sharpness(image, **{argument: value})
