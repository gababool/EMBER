import cv2
import numpy as np

from pipeline.quality_screening.screening import run_quality_screening


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