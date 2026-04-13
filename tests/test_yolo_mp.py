import numpy as np

from detection.yolo_mp import load_yolo_mp_model, run_yolo_mp_detection


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.predict_calls = []

    def predict(self, image, conf, verbose):
        self.predict_calls.append(
            {
                "shape": image.shape,
                "conf": conf,
                "verbose": verbose,
            }
        )
        return self.results


def test_model_loads_successfully_with_custom_loader():
    loaded_paths = []

    def fake_loader(model_path):
        loaded_paths.append(model_path)
        return {"model_path": model_path}

    result = load_yolo_mp_model("custom/yolo-mp.pt", loader=fake_loader)

    assert result == {"model_path": "custom/yolo-mp.pt"}
    assert loaded_paths == ["custom/yolo-mp.pt"]


def test_inference_runs_and_returns_structured_output():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    model = FakeModel(
        [
            FakeResult(
                FakeBoxes(
                    xyxy=[[10, 20, 110, 220]],
                    conf=[0.91],
                    cls=[2],
                ),
                names={2: "detected_object"},
            )
        ]
    )

    result = run_yolo_mp_detection(image, model=model, confidence_threshold=0.4)

    assert result["model_path"] == "models/yolo_mp.pt"
    assert result["confidence_threshold"] == 0.4
    assert result["image_size"] == [640, 480]
    assert result["has_detections"] is True
    assert result["count"] == 1
    assert result["detections"] == [
        {
            "class_id": 2,
            "label": "detected_object",
            "confidence": 0.91,
            "bbox": [10.0, 20.0, 110.0, 220.0],
        }
    ]
    assert model.predict_calls == [
        {
            "shape": (480, 640, 3),
            "conf": 0.4,
            "verbose": False,
        }
    ]


def test_empty_detection_results_are_handled_gracefully():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    model = FakeModel([])

    result = run_yolo_mp_detection(image, model=model)

    assert result["detections"] == []
    assert result["has_detections"] is False
    assert result["count"] == 0


def test_model_path_and_confidence_threshold_can_be_overridden():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    loaded_paths = []
    created_models = []

    def fake_loader(model_path):
        loaded_paths.append(model_path)
        model = FakeModel([])
        created_models.append(model)
        return model

    result = run_yolo_mp_detection(
        image,
        model_path="weights/override.pt",
        confidence_threshold=0.65,
        loader=fake_loader,
    )

    assert result["model_path"] == "weights/override.pt"
    assert result["confidence_threshold"] == 0.65
    assert loaded_paths == ["weights/override.pt"]
    assert created_models[0].predict_calls == [
        {
            "shape": (200, 300, 3),
            "conf": 0.65,
            "verbose": False,
        }
    ]
