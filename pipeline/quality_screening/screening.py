from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from pipeline.quality_screening.brightness import check_brightness
from pipeline.quality_screening.config import QualityScreeningConfig
from pipeline.quality_screening.contrast import check_contrast
from pipeline.quality_screening.resolution import check_resolution
from pipeline.quality_screening.sharpness import check_sharpness


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk without altering its original channel layout.

    The image is loaded with cv2.IMREAD_UNCHANGED so that grayscale, BGR,
    and BGRA images can be handled downstream by the quality screening stage.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"image path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"image path is not a file: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"failed to load image from path: {path}")

    return image


def run_quality_screening(
    image: np.ndarray,
    config: QualityScreeningConfig | None = None,
    stop_on_first_failure: bool = False,
) -> Dict[str, Any]:
    """Run the full quality screening stage on an already loaded image.

    The checks are intentionally run in this order:
    1. resolution
    2. brightness
    3. contrast
    4. sharpness

    Resolution is checked first because it is cheap and can reject clearly
    unsuitable inputs early. The remaining checks evaluate image quality
    characteristics needed by later pipeline stages.
    """
    if config is None:
        config = QualityScreeningConfig()

    checks: Dict[str, Dict[str, Any]] = {}
    failed_checks: list[str] = []

    ordered_checks = [
        (
            "resolution",
            lambda img: check_resolution(
                img,
                min_width=config.min_width,
                min_height=config.min_height,
            ),
        ),
        (
            "brightness",
            lambda img: check_brightness(
                img,
                min_brightness=config.min_brightness,
                max_brightness=config.max_brightness,
                bright_pixel_threshold=config.bright_pixel_threshold,
                min_bright_pixel_fraction=config.min_bright_pixel_fraction,
            ),
        ),
        (
            "contrast",
            lambda img: check_contrast(
                img,
                min_contrast=config.min_contrast,
                max_contrast=config.max_contrast,
            ),
        ),
        (
            "sharpness",
            lambda img: check_sharpness(
                img,
                min_sharpness=config.min_sharpness,
                min_texture=config.min_texture,
                tile_size=config.tile_size,
                min_valid_tile_fraction=config.min_valid_tile_fraction,
                min_sharp_tile_fraction=config.min_sharp_tile_fraction,
                sharpness_percentile=config.sharpness_percentile,
            ),
        ),
    ]

    for name, check_fn in ordered_checks:
        result = check_fn(image)
        checks[name] = result

        if not result["passed"]:
            failed_checks.append(name)
            if stop_on_first_failure:
                break

    passed = len(failed_checks) == 0
    print("passed" if passed else "failed")
    return {
        "passed": passed,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def run_quality_screening_from_path(
    image_path: str | Path,
    config: QualityScreeningConfig | None = None,
    stop_on_first_failure: bool = False,
) -> Dict[str, Any]:
    """Load an image from disk and run the full quality screening stage.

    This is the main entrypoint other file-based pipeline stages should call.
    """
    image = load_image(image_path)
    result = run_quality_screening(
        image=image,
        config=config,
        stop_on_first_failure=stop_on_first_failure,
    )
    result["image_path"] = str(image_path)
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.quality_screening.screening",
        description="Run the quality screening pipeline on a single image.",
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to the image to screen (absolute or relative to the current working directory).",
    )
    parser.add_argument(
        "--stop-on-first-failure",
        action="store_true",
        help="Stop running checks as soon as one fails.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_quality_screening_from_path(
        image_path=args.image_path,
        stop_on_first_failure=args.stop_on_first_failure,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
