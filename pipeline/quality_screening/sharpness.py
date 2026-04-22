from __future__ import annotations

from typing import TypedDict

import cv2
import numpy as np

from pipeline.quality_screening._shared import (
    to_grayscale_uint8,
    validate_non_negative_int,
    validate_non_negative_number,
)


class SharpnessResult(TypedDict):
    passed: bool
    reason: str
    sharpness: float
    valid_tiles: int
    total_tiles: int
    sharp_tiles: int
    texture_threshold: float
    sharpness_threshold: float
    sharpness_percentile: float


def _validate_fraction(name: str, value: float) -> None:
    validate_non_negative_number(name, value)
    if value > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")


def _validate_percentile(name: str, value: float) -> None:
    validate_non_negative_number(name, value)
    if value > 100.0:
        raise ValueError(f"{name} must be between 0.0 and 100.0.")


def _iter_tiles(
    image: np.ndarray,
    tile_size: int,
):
    height, width = image.shape[:2]

    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            yield image[y0:y1, x0:x1]


def _gradient_magnitude(tile: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(tile, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(tile, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _texture_score(tile: np.ndarray) -> float:
    """
    Texture gate: mean gradient magnitude.

    Low-texture tiles such as sky/smoke/haze may not contain enough structure
    for a meaningful blur judgment, so we exclude them from sharpness scoring.
    """
    grad_mag = _gradient_magnitude(tile)
    return float(np.mean(grad_mag))


def _tenengrad_score(tile: np.ndarray) -> float:
    """
    Tenengrad sharpness: mean squared gradient magnitude.
    """
    gx = cv2.Sobel(tile, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(tile, cv2.CV_64F, 0, 1, ksize=3)
    grad_sq = gx * gx + gy * gy
    return float(np.mean(grad_sq))


def check_sharpness(
    image: np.ndarray,
    min_sharpness: float = 300.0,
    min_texture: float = 8.0,
    tile_size: int = 128,
    min_valid_tile_fraction: float = 0.1,
    min_sharp_tile_fraction: float = 0.5,
    sharpness_percentile: float = 75.0,
) -> SharpnessResult:
    """
    Check image sharpness using tile-based Tenengrad with texture filtering.

    Strategy:
    - split image into non-overlapping tiles
    - ignore tiles with too little texture
    - compute Tenengrad sharpness on textured tiles only
    - require enough valid tiles
    - require enough sharp valid tiles
    - aggregate overall sharpness via percentile

    This is more robust than global Laplacian variance for smoke-heavy wildfire
    imagery, where large parts of the frame may be naturally soft.
    """
    validate_non_negative_number("min_sharpness", min_sharpness)
    validate_non_negative_number("min_texture", min_texture)
    validate_non_negative_int("tile_size", tile_size)
    _validate_fraction("min_valid_tile_fraction", min_valid_tile_fraction)
    _validate_fraction("min_sharp_tile_fraction", min_sharp_tile_fraction)
    _validate_percentile("sharpness_percentile", sharpness_percentile)

    if tile_size == 0:
        raise ValueError("tile_size must be greater than 0.")

    gray = to_grayscale_uint8(image)
    height, width = gray.shape[:2]

    total_tiles = 0
    valid_tile_scores: list[float] = []

    for tile in _iter_tiles(gray, tile_size):
        total_tiles += 1

        # Skip extremely small edge tiles that are too tiny to score reliably.
        if tile.shape[0] < 16 or tile.shape[1] < 16:
            continue

        texture = _texture_score(tile)
        if texture < min_texture:
            continue

        score = _tenengrad_score(tile)
        valid_tile_scores.append(score)

    valid_tiles = len(valid_tile_scores)

    if total_tiles == 0:
        return {
            "passed": False,
            "reason": "invalid_image",
            "sharpness": 0.0,
            "valid_tiles": 0,
            "total_tiles": 0,
            "sharp_tiles": 0,
            "texture_threshold": min_texture,
            "sharpness_threshold": min_sharpness,
            "sharpness_percentile": sharpness_percentile,
        }

    min_valid_tiles = max(1, int(np.ceil(total_tiles * min_valid_tile_fraction)))
    if valid_tiles < min_valid_tiles:
        return {
            "passed": False,
            "reason": "insufficient_textured_regions",
            "sharpness": 0.0,
            "valid_tiles": valid_tiles,
            "total_tiles": total_tiles,
            "sharp_tiles": 0,
            "texture_threshold": min_texture,
            "sharpness_threshold": min_sharpness,
            "sharpness_percentile": sharpness_percentile,
        }

    sharp_tiles = sum(score >= min_sharpness for score in valid_tile_scores)
    sharp_tile_fraction = sharp_tiles / valid_tiles

    aggregate_sharpness = float(
        np.percentile(valid_tile_scores, sharpness_percentile)
    )

    passed = (
        aggregate_sharpness >= min_sharpness
        and sharp_tile_fraction >= min_sharp_tile_fraction
    )

    if passed:
        reason = ""
    elif aggregate_sharpness < min_sharpness:
        reason = "low_sharpness"
    else:
        reason = "too_few_sharp_tiles"

    return {
        "passed": passed,
        "reason": reason,
        "sharpness": aggregate_sharpness,
        "valid_tiles": valid_tiles,
        "total_tiles": total_tiles,
        "sharp_tiles": sharp_tiles,
        "texture_threshold": min_texture,
        "sharpness_threshold": min_sharpness,
        "sharpness_percentile": sharpness_percentile,
    }