# my_module/config.py
from pathlib import Path
from ultralytics import YOLO


WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MODEL_3 = WEIGHTS_DIR / "model_3.pt"
MODEL_LAST = WEIGHTS_DIR / "last.pt"
MODEL_BEST = WEIGHTS_DIR / "best.pt"
MODEL_TEST = WEIGHTS_DIR / "test_1.pt"

model_best = YOLO(MODEL_BEST) #Newest Model, Currently the best one
model_3 = YOLO(MODEL_3)
model_last = YOLO(MODEL_LAST) 
model_test = YOLO(MODEL_TEST) # Very similar to the newest one in accuracy