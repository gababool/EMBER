import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.yolo import run_yolo_for_llm, yolo_mode_tag


def test_yolo_mode_tag_distinguishes_benchmark_conditions():
    assert yolo_mode_tag(False) == "_noyolo"
    assert yolo_mode_tag(True, "annotated_image") == "_yolo_annotated"
    assert yolo_mode_tag(True, "context_summary") == "_yolo_summary"
    assert yolo_mode_tag(True, "context_locations") == "_yolo_boxes"


def test_context_modes_do_not_request_saved_annotation(monkeypatch, tmp_path):
    calls = []

    def fake_run_detection(image_path, model_name, show, save_annotation=True):
        calls.append(save_annotation)
        return {
            "passed": True,
            "image_path": image_path,
            "model_name": model_name,
            "image_width": 1280,
            "image_height": 720,
            "detections": [],
        }

    import pipeline.object_detection.run

    monkeypatch.setattr(pipeline.object_detection.run, "run", fake_run_detection)

    result = run_yolo_for_llm(
        tmp_path / "nofire.jpg",
        "best",
        "context_locations",
    )

    assert calls == [False]
    assert result["annotated_path"] is None
    assert result["llm_input_path"] == tmp_path / "nofire.jpg"
    assert "Detections: 0 boxes" in result["additional_context"]


def test_annotated_mode_requests_saved_annotation(monkeypatch, tmp_path):
    calls = []
    annotated_path = tmp_path / "result.png"

    def fake_run_detection(image_path, model_name, show, save_annotation=True):
        calls.append(save_annotation)
        return {
            "passed": True,
            "image_path": image_path,
            "model_name": model_name,
            "annotated_image_path": str(annotated_path),
            "image_width": 1280,
            "image_height": 720,
            "detections": [],
        }

    import pipeline.object_detection.run

    monkeypatch.setattr(pipeline.object_detection.run, "run", fake_run_detection)

    result = run_yolo_for_llm(
        tmp_path / "fire.jpg",
        "best",
        "annotated_image",
    )

    assert calls == [True]
    assert result["annotated_path"] == annotated_path
    assert result["llm_input_path"] == annotated_path
    assert result["additional_context"] is None
