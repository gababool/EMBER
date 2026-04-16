from ultralytics import YOLO
from config import MODEL_PATHS


_models_cache = {}

class ModelLoadError(Exception):
    pass


def get_model(name: str) -> YOLO:

    if name in _models_cache:
        return _models_cache[name]
    
    if name not in MODEL_PATHS:
        raise ModelLoadError(f"Unknown model: '{name}'. Available: {list(MODEL_PATHS.keys())} ")
    
    path = MODEL_PATHS[name]

    try: 
        model = YOLO(path)
    except Exception as e:
        raise ModelLoadError(f"Failed to load model: '{name}' from {path}: {e}")
    
    _models_cache[name] = model
    return model