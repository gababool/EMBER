from pathlib import Path
import random
import sys

from PySide6.QtCore import QObject, Signal, Slot


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.service import create_default_pipeline

DEMO_DATASET_DIR = (
    REPO_ROOT
    / "dataset"
    / "wildfire-dataset"
    / "the_wildfire_dataset_2n_version"
    / "test"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class PipelineWorker(QObject):
    image_selected = Signal(str)
    event_received = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        dataset_dir: Path = DEMO_DATASET_DIR,
        yolo_model: str = "best",
        llm_model: str = "ministral",
        parent=None,
    ):
        super().__init__(parent)
        self.dataset_dir = Path(dataset_dir)
        self.yolo_model = yolo_model
        self.llm_model = llm_model

    @Slot()
    def run(self):
        try:
            image_path = self._choose_random_image()
            self.image_selected.emit(str(image_path))

            runner = create_default_pipeline(
                yolo_model=self.yolo_model,
                llm_model=self.llm_model,
            )
            for event in runner.iter_events(image_path):
                self.event_received.emit(event)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    def _choose_random_image(self) -> Path:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Demo dataset not found: {self.dataset_dir}")

        image_paths = [
            path
            for path in self.dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not image_paths:
            raise FileNotFoundError(f"No demo images found in: {self.dataset_dir}")

        return random.choice(image_paths)
