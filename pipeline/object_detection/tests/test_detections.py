import pytest

from pipeline.object_detection.detections import (
    format_detection_context,
    resolve_yolo_input_mode,
)


SAMPLE_DETECTION_RESULT = {
    "image_width": 1280,
    "image_height": 720,
    "detections": [
        {
            "index": 1,
            "label": "Fire",
            "class_id": 1,
            "confidence": 0.874,
            "bbox_xyxy": [420.0, 180.25, 690.5, 410.0],
        },
        {
            "index": 2,
            "label": "Smoke",
            "class_id": 0,
            "confidence": 0.743,
            "bbox_xyxy": [80.0, 100.0, 300.0, 220.0],
        },
    ],
}


def test_resolve_yolo_input_mode_keeps_legacy_annotation_flag():
    assert resolve_yolo_input_mode(use_annotation=True) == "annotated_image"
    assert resolve_yolo_input_mode(use_annotation=False) == "context_summary"


def test_resolve_yolo_input_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown YOLO input mode"):
        resolve_yolo_input_mode("boxes_in_the_moon")


def test_format_detection_context_summary_excludes_locations():
    formatted = format_detection_context(SAMPLE_DETECTION_RESULT, "context_summary")

    assert "YOLO DETECTION CONTEXT" in formatted
    assert "Detections: 2 boxes" in formatted
    assert "- 1: Fire, confidence 0.87" in formatted
    assert "- 2: Smoke, confidence 0.74" in formatted
    assert "bbox" not in formatted
    assert "Image size" not in formatted


def test_format_detection_context_locations_includes_coordinate_contract():
    formatted = format_detection_context(SAMPLE_DETECTION_RESULT, "context_locations")

    assert "Image size: 1280x720 px" in formatted
    assert "Bounding boxes: xyxy pixel coordinates, origin at top-left" in formatted
    assert "- 1: Fire, confidence 0.87, bbox xyxy=(420, 180.2, 690.5, 410)" in formatted
    assert "- 2: Smoke, confidence 0.74, bbox xyxy=(80, 100, 300, 220)" in formatted


def test_format_detection_context_handles_zero_detections():
    formatted = format_detection_context({"detections": []}, "context_locations")

    assert "Detections: 0 boxes" in formatted
    assert "bbox xyxy" not in formatted
