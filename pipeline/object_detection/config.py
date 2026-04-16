from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

MODEL_PATHS = {
    "best": WEIGHTS_DIR / "best.pt",
    "last": WEIGHTS_DIR / "last.pt",
    "model_3": WEIGHTS_DIR / "model_3.pt",
    "test": WEIGHTS_DIR / "test_1.pt",
}