# my_module/config.py
from pathlib import Path
from ultralytics import YOLO


WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MODEL_BEST = WEIGHTS_DIR / "best.pt"
MODEL_LAST = WEIGHTS_DIR / "last.pt"

model_best = YOLO(MODEL_BEST)
model_last = YOLO(MODEL_LAST)