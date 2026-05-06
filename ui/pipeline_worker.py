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
        prompt_file: str | None = None,
        context_file: str | None = None,
        skip_quality_screening: bool = False,
        use_annotation: bool = True,
        fixed_image_path: str | None = None,
        label_filter: str | None = None,
        max_quality_retries: int = 20,
        parent=None,
    ):
        super().__init__(parent)
        self.dataset_dir = Path(dataset_dir)
        self.yolo_model = yolo_model
        self.llm_model = llm_model
        self.prompt_file = prompt_file
        self.context_file = context_file
        self.skip_quality_screening = skip_quality_screening
        self.use_annotation = use_annotation
        self.fixed_image_path = Path(fixed_image_path) if fixed_image_path else None
        self.label_filter = label_filter
        self.max_quality_retries = max_quality_retries

    @Slot()
    def run(self):
        try:
            tried_paths = set()
            kwargs = dict(
                yolo_model=self.yolo_model,
                llm_model=self.llm_model,
                skip_quality_screening=self.skip_quality_screening,
            )
            if self.prompt_file:
                kwargs["prompt_file"] = self.prompt_file
            if self.context_file:
                kwargs["context_file"] = self.context_file
            kwargs["use_annotation"] = self.use_annotation
            runner = create_default_pipeline(**kwargs)

            if self.fixed_image_path is not None:
                self.image_selected.emit(str(self.fixed_image_path))
                for event in runner.iter_events(self.fixed_image_path):
                    self.event_received.emit(event)
            else:
                for _attempt in range(self.max_quality_retries):
                    image_path = self._choose_random_image(exclude=tried_paths, label_filter=self.label_filter)
                    tried_paths.add(image_path)
                    self.image_selected.emit(str(image_path))

                    retry_after_quality_failure = False
                    for event in runner.iter_events(image_path):
                        self.event_received.emit(event)
                        result = event.result
                        if (
                            result is not None
                            and result.stage_name == "quality_screening"
                            and not result.passed
                        ):
                            retry_after_quality_failure = True
                            break

                    if not retry_after_quality_failure:
                        break
                else:
                    self.failed.emit("No image passed quality screening after multiple attempts.")
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    def _choose_random_image(self, exclude: set[Path] | None = None, label_filter: str | None = None) -> Path:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Demo dataset not found: {self.dataset_dir}")

        exclude = exclude or set()
        image_paths = [
            path
            for path in self.dataset_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_SUFFIXES
                and path not in exclude
                and (label_filter is None or path.parent.name == label_filter)
            )
        ]
        if not image_paths:
            raise FileNotFoundError(f"No demo images found in: {self.dataset_dir}")

        return random.choice(image_paths)
