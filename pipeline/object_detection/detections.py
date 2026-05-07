from __future__ import annotations

from typing import Any, Literal


YoloInputMode = Literal["annotated_image", "context_summary", "context_locations"]
YOLO_INPUT_MODES: tuple[YoloInputMode, ...] = (
    "annotated_image",
    "context_summary",
    "context_locations",
)


def resolve_yolo_input_mode(
    mode: str | None = None,
    *,
    use_annotation: bool | None = None,
) -> YoloInputMode:
    if mode is None:
        return "annotated_image" if use_annotation is not False else "context_summary"
    if mode not in YOLO_INPUT_MODES:
        valid = ", ".join(YOLO_INPUT_MODES)
        raise ValueError(f"Unknown YOLO input mode '{mode}'. Valid options: {valid}")
    return mode  # type: ignore[return-value]


def format_detection_context(
    detection_result: dict[str, Any],
    mode: YoloInputMode,
) -> str:
    if mode == "annotated_image":
        return ""

    detections = detection_result.get("detections", [])
    lines = ["YOLO DETECTION CONTEXT"]
    lines.append(
        "Source: object-detection preprocessing; treat as a model-generated signal, not direct visual evidence."
    )

    image_width = detection_result.get("image_width")
    image_height = detection_result.get("image_height")
    if mode == "context_locations" and image_width and image_height:
        lines.append(f"Image size: {image_width}x{image_height} px")
        lines.append("Bounding boxes: xyxy pixel coordinates, origin at top-left")

    box_word = "box" if len(detections) == 1 else "boxes"
    lines.append(f"Detections: {len(detections)} {box_word}")

    if not detections:
        return "\n".join(lines)

    for detection in detections:
        index = detection.get("index", "?")
        label = detection.get("label", "Unknown")
        confidence = detection.get("confidence")
        confidence_text = (
            f"{confidence:.2f}" if isinstance(confidence, int | float) else "unknown"
        )
        line = f"- {index}: {label}, confidence {confidence_text}"

        if mode == "context_locations":
            bbox = detection.get("bbox_xyxy")
            if bbox is not None:
                line += f", bbox xyxy={_format_bbox(bbox)}"

        lines.append(line)

    return "\n".join(lines)


def _format_bbox(bbox: Any) -> str:
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return str(bbox)

    rounded = [round(value, 1) for value in values]
    return "(" + ", ".join(f"{value:g}" for value in rounded) + ")"
