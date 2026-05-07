import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.yolo import yolo_mode_tag


def test_yolo_mode_tag_distinguishes_benchmark_conditions():
    assert yolo_mode_tag(False) == "_noyolo"
    assert yolo_mode_tag(True, "annotated_image") == "_yolo_annotated"
    assert yolo_mode_tag(True, "context_summary") == "_yolo_summary"
    assert yolo_mode_tag(True, "context_locations") == "_yolo_boxes"
