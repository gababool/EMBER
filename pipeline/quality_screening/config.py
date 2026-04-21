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
    min_sharpness: float = 100.0