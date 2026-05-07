"""Shared YOLO preprocessing helpers for benchmark entry points."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pipeline.object_detection.detections import (
    YoloInputMode,
    format_detection_context,
    resolve_yolo_input_mode,
)


def yolo_mode_tag(with_yolo: bool, yolo_input_mode: str = "annotated_image") -> str:
    if not with_yolo:
        return "_noyolo"

    mode = resolve_yolo_input_mode(yolo_input_mode)
    tags = {
        "annotated_image": "_yolo_annotated",
        "context_summary": "_yolo_summary",
        "context_locations": "_yolo_boxes",
    }
    return tags[mode]


def run_yolo_for_llm(
    image_path: str | Path,
    yolo_model_name: str,
    yolo_input_mode: str,
) -> dict[str, Any]:
    """Run YOLO once and prepare the corresponding LLM input payload."""
    from pipeline.object_detection.run import run as run_detection

    mode: YoloInputMode = resolve_yolo_input_mode(yolo_input_mode)
    image_path = Path(image_path)

    start = time.perf_counter()
    detection_result = run_detection(
        image_path=str(image_path),
        model_name=yolo_model_name,
        show=False,
        save_annotation=mode == "annotated_image",
    )
    duration_s = time.perf_counter() - start

    if detection_result is None:
        raise RuntimeError("YOLO preprocessing did not return a result.")
    if not detection_result.get("passed"):
        raise RuntimeError(detection_result.get("error", "YOLO preprocessing failed."))

    llm_input_path = (
        Path(detection_result["annotated_image_path"])
        if mode == "annotated_image"
        else image_path
    )
    detection_context = format_detection_context(detection_result, mode)

    return {
        "llm_input_path": llm_input_path,
        "additional_context": detection_context or None,
        "detection_result": detection_result,
        "detection_count": len(detection_result.get("detections", [])),
        "duration_s": round(duration_s, 3),
        "annotated_path": (
            Path(detection_result["annotated_image_path"])
            if detection_result.get("annotated_image_path")
            else None
        ),
        "yolo_input_mode": mode,
    }
