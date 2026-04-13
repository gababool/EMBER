from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np


DEFAULT_MODEL_PATH = "models/yolo_mp.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25


def _default_model_loader(model_path: str) -> Any:
    """Load a YOLO-compatible model backend.

    The current implementation targets Ultralytics-style YOLO APIs. The
    dependency is optional so the Stage 2 interface can be developed and tested
    before the final YOLO-MP runtime is installed in the environment.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLO backend is not installed. Install the runtime dependency "
            "for YOLO-MP or pass a custom loader."
        ) from exc

    return YOLO(model_path)


def load_yolo_mp_model(
    model_path: str = DEFAULT_MODEL_PATH,
    loader: Callable[[str], Any] | None = None,
) -> Any:
    """Load the YOLO-MP model using the provided or default loader."""
    if not model_path:
        raise ValueError("model_path must not be empty.")

    model_loader = loader or _default_model_loader
    return model_loader(model_path)


def run_yolo_mp_detection(
    image: np.ndarray,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model: Any | None = None,
    loader: Callable[[str], Any] | None = None,
) -> Dict[str, Any]:
    """Run YOLO-MP inference and normalize detections for later pipeline stages.

    Args:
        image: Input image as a numpy array.
        model_path: Path to the YOLO-MP weights file.
        confidence_threshold: Minimum confidence threshold passed to the model.
        model: Optional preloaded model to reuse across calls.
        loader: Optional model loader override.

    Returns:
        A dict with:
            model_path (str): Resolved model path used for loading.
            confidence_threshold (float): Threshold used during inference.
            image_size (list[int, int]): Input [width, height].
            detections (list[dict]): Normalized detections.
            has_detections (bool): True when at least one detection is present.
            count (int): Number of detections.
    """
    if image is None:
        raise ValueError("image must not be None.")
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy ndarray.")
    if image.size == 0:
        raise ValueError("image must not be empty.")
    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0.")

    height, width = image.shape[:2]
    resolved_model_path = str(Path(model_path))
    loaded_model = model if model is not None else load_yolo_mp_model(resolved_model_path, loader=loader)
    raw_results = loaded_model.predict(image, conf=confidence_threshold, verbose=False)
    detections = _normalize_detections(raw_results)

    return {
        "model_path": resolved_model_path,
        "confidence_threshold": confidence_threshold,
        "image_size": [width, height],
        "detections": detections,
        "has_detections": bool(detections),
        "count": len(detections),
    }


def _normalize_detections(raw_results: Any) -> List[Dict[str, Any]]:
    if raw_results is None:
        return []

    results = list(raw_results)
    if not results:
        return []

    first_result = results[0]
    boxes = getattr(first_result, "boxes", None)
    if boxes is None:
        return []

    xyxy_values = _to_nested_list(getattr(boxes, "xyxy", []))
    confidence_values = _to_flat_list(getattr(boxes, "conf", []))
    class_values = _to_flat_list(getattr(boxes, "cls", []))
    names = getattr(first_result, "names", {})

    detections: List[Dict[str, Any]] = []
    for index, bbox in enumerate(xyxy_values):
        class_id = _safe_int(class_values[index]) if index < len(class_values) else None
        label = names.get(class_id, str(class_id)) if class_id is not None else ""
        confidence = _safe_float(confidence_values[index]) if index < len(confidence_values) else None
        detections.append(
            {
                "class_id": class_id,
                "label": label,
                "confidence": confidence,
                "bbox": [_safe_float(value) for value in bbox],
            }
        )

    return detections


def _to_nested_list(values: Any) -> List[List[float]]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        converted = values.tolist()
    else:
        converted = values
    return [list(item) for item in converted]


def _to_flat_list(values: Any) -> List[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        converted = values.tolist()
    else:
        converted = values
    return list(converted)


def _safe_float(value: Any) -> float:
    return float(value)


def _safe_int(value: Any) -> int:
    return int(value)
