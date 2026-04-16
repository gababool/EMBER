# my_module/config.py
from pathlib import Path
from ultralytics import YOLO


WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MODEL_NEXT_BEST = WEIGHTS_DIR / "next_best.pt"
MODEL_LAST = WEIGHTS_DIR / "last.pt"
MODEL_BEST = WEIGHTS_DIR / "best.pt"
MODEL_TEST = WEIGHTS_DIR / "test_1.pt"

model_best = YOLO(MODEL_BEST)
model_next_best = YOLO(MODEL_NEXT_BEST)
model_last = YOLO(MODEL_LAST)
model_test = YOLO(MODEL_TEST)