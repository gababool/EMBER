from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QualityScreeningConfig:
    min_brightness: float = 40.0
    max_brightness: float = 220.0
    min_contrast: float = 30.0
    max_contrast: float = 120.0
    min_width: int = 640
    min_height: int = 480

    # Tile-based sharpness settings
    min_sharpness: float = 100.0
    min_texture: float = 8.0
    tile_size: int = 128
    min_valid_tile_fraction: float = 0.1
    min_sharp_tile_fraction: float = 0.5
    sharpness_percentile: float = 30.0