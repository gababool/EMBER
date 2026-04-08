from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Stage1QualityConfig:
    min_brightness: float = 40.0
    max_brightness: float = 220.0
    min_sharpness: float = 100.0
    min_width: int = 640
    min_height: int = 480
    min_contrast: float = 20.0

    @property
    def min_pixel_count(self) -> int:
        return self.min_width * self.min_height

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(source: Dict[str, Any]) -> "Stage1QualityConfig":
        return Stage1QualityConfig(
            min_brightness=float(source.get("min_brightness", 40.0)),
            max_brightness=float(source.get("max_brightness", 220.0)),
            min_sharpness=float(source.get("min_sharpness", 100.0)),
            min_width=int(source.get("min_width", 640)),
            min_height=int(source.get("min_height", 480)),
            min_contrast=float(source.get("min_contrast", 20.0)),
        )

    @staticmethod
    def from_yaml(path: Union[str, Path]) -> "Stage1QualityConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("Stage1QualityConfig YAML must contain a mapping at the top level.")
        return Stage1QualityConfig.from_dict(data)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _compute_metrics(image: np.ndarray) -> Dict[str, Any]:
    if image is None:
        raise ValueError("Image data must not be None.")
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy ndarray.")
    if image.size == 0:
        raise ValueError("Image data must not be empty.")

    gray = _to_gray(image)
    height, width = gray.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Image dimensions must be non-zero.")

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.var(laplacian))

    return {
        "brightness": brightness,
        "sharpness": sharpness,
        "resolution": [int(width), int(height)],
        "contrast": contrast,
    }


def _score_brightness(brightness: float, config: Stage1QualityConfig) -> float:
    if brightness < config.min_brightness:
        return _clamp(brightness / config.min_brightness, 0.0, 1.0)
    if brightness > config.max_brightness:
        return _clamp((config.max_brightness - (brightness - config.max_brightness)) / config.max_brightness, 0.0, 1.0)
    return 1.0


def _score_threshold(value: float, minimum: float) -> float:
    if minimum <= 0:
        return 1.0
    return _clamp(value / minimum, 0.0, 1.0)


def _score_resolution(width: int, height: int, config: Stage1QualityConfig) -> float:
    pixel_count = width * height
    return _score_threshold(pixel_count, config.min_pixel_count)


def _compute_confidence(metrics: Dict[str, Any], config: Stage1QualityConfig) -> float:
    brightness_score = _score_brightness(metrics["brightness"], config)
    sharpness_score = _score_threshold(metrics["sharpness"], config.min_sharpness)
    contrast_score = _score_threshold(metrics["contrast"], config.min_contrast)
    width, height = metrics["resolution"]
    resolution_score = _score_resolution(width, height, config)
    raw_confidence = (brightness_score + sharpness_score + contrast_score + resolution_score) / 4.0
    return float(round(raw_confidence, 3))


def run_quality_screening(
    image: np.ndarray,
    config: Optional[Union[Stage1QualityConfig, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if config is None:
        config = Stage1QualityConfig()
    elif isinstance(config, dict):
        config = Stage1QualityConfig.from_dict(config)
    elif not isinstance(config, Stage1QualityConfig):
        raise TypeError("config must be Stage1QualityConfig, dict, or None")

    metrics = _compute_metrics(image)
    reasons: List[str] = []

    if metrics["brightness"] < config.min_brightness:
        reasons.append("low_brightness")
    elif metrics["brightness"] > config.max_brightness:
        reasons.append("high_brightness")

    if metrics["sharpness"] < config.min_sharpness:
        reasons.append("low_sharpness")

    width, height = metrics["resolution"]
    if width * height < config.min_pixel_count:
        reasons.append("low_resolution")

    if metrics["contrast"] < config.min_contrast:
        reasons.append("low_contrast")

    passed = len(reasons) == 0
    confidence = _compute_confidence(metrics, config)

    return {
        "passed": passed,
        "confidence": confidence,
        "reasons": reasons,
        "metrics": metrics,
    }
