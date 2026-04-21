from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.quality_screening.screening import (
    run_quality_screening,
    run_quality_screening_from_path,
)


def test_quality_screening_runs_all_checks_in_sequence():
    image = np.full((480, 640, 3), 80, dtype=np.uint8)
    image[240:, :] = 180
    cv2.rectangle(image, (50, 50), (590, 430), (255, 255, 255), thickness=2)
    cv2.line(image, (0, 0), (639, 479), (255, 255, 255), thickness=2)

    result = run_quality_screening(image)

    assert result["passed"] is True
    assert list(result["checks"].keys()) == [
        "resolution",
        "brightness",
        "contrast",
        "sharpness",
    ]
    assert result["checks"]["resolution"]["passed"] is True
    assert result["checks"]["brightness"]["passed"] is True
    assert result["checks"]["contrast"]["passed"] is True
    assert result["checks"]["sharpness"]["passed"] is True
    assert result["failed_checks"] == []


def test_run_quality_screening_from_path_runs_all_checks(tmp_path: Path):
    image = np.full((480, 640, 3), 80, dtype=np.uint8)
    image[240:, :] = 180
    cv2.rectangle(image, (50, 50), (590, 430), (255, 255, 255), thickness=2)
    cv2.line(image, (0, 0), (639, 479), (255, 255, 255), thickness=2)

    image_path = tmp_path / "test_image.png"
    success = cv2.imwrite(str(image_path), image)
    assert success is True

    result = run_quality_screening_from_path(image_path)

    assert result["passed"] is True
    assert result["image_path"] == str(image_path)
    assert list(result["checks"].keys()) == [
        "resolution",
        "brightness",
        "contrast",
        "sharpness",
    ]
    assert result["failed_checks"] == []


def test_run_quality_screening_from_path_raises_for_missing_file(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.png"

    with pytest.raises(FileNotFoundError):
        run_quality_screening_from_path(missing_path)


def test_quality_screening_can_stop_on_first_failure():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = run_quality_screening(image, stop_on_first_failure=True)

    assert result["passed"] is False
    assert result["failed_checks"] == ["resolution"]
    assert list(result["checks"].keys()) == ["resolution"]


def test_dataset_image_runs_through_all_quality_checks_in_order():
    dataset_root = Path("wildfire-dataset")

    if not dataset_root.exists():
        pytest.skip("wildfire-dataset/ is not available locally.")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    dataset_images = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    ]

    if not dataset_images:
        pytest.skip("No image files found in wildfire-dataset/.")

    image_path = dataset_images[0]
    result = run_quality_screening_from_path(image_path)

    assert "checks" in result
    assert list(result["checks"].keys()) == [
        "resolution",
        "brightness",
        "contrast",
        "sharpness",
    ]
    assert "passed" in result
    assert "failed_checks" in result
    assert result["image_path"] == str(image_path)