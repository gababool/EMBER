from pathlib import Path
import random
import sys
import threading

from PySide6.QtCore import QObject, Signal, Slot


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.context.schemas import CompassBearing, OperationalContext
from pipeline.context.wind import build_manual_wind, build_mock_wind
from pipeline.service import create_default_pipeline

DEMO_DATASET_DIR = (
    REPO_ROOT
    / "dataset"
    / "wildfire-dataset"
    / "the_wildfire_dataset_2n_version"
    / "test"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_CONTEXT_FILE = REPO_ROOT / "data" / "contexts" / "benchmark_scenario.json"
DEFAULT_PROMPT_FILE = REPO_ROOT / "pipeline" / "llm" / "prompts" / "c2v4prompt.txt"
WIND_MODE_NONE = "none"
WIND_MODE_MANUAL = "manual"
WIND_MODE_MOCKED = "mocked"


class PipelineWorker(QObject):
    image_selected = Signal(str)
    wind_updated = Signal(object)
    event_received = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        dataset_dir: Path = DEMO_DATASET_DIR,
        yolo_model: str = "best",
        llm_model: str = "ministral",
        wind_mode: str = WIND_MODE_NONE,
        wind_direction: CompassBearing = "SW",
        wind_speed_mps: float = 6.5,
        prompt_file: str | None = None,
        context_file: str | None = None,
        skip_quality_screening: bool = False,
        llm_enabled: bool = True,
        yolo_input_mode: str = "annotated_image",
        use_annotation: bool | None = None,
        fixed_image_path: str | None = None,
        label_filter: str | None = None,
        max_quality_retries: int = 20,
        parent=None,
    ):
        super().__init__(parent)
        self.dataset_dir = Path(dataset_dir)
        self.yolo_model = yolo_model
        self.llm_model = llm_model
        self.wind_mode = wind_mode
        self.wind_direction = wind_direction
        self.wind_speed_mps = wind_speed_mps
        self.prompt_file = prompt_file
        self.context_file = context_file
        self.skip_quality_screening = skip_quality_screening
        self.llm_enabled = llm_enabled
        if use_annotation is not None:
            yolo_input_mode = "annotated_image" if use_annotation else "context_summary"
        self.yolo_input_mode = yolo_input_mode
        self.fixed_image_path = Path(fixed_image_path) if fixed_image_path else None
        self.label_filter = label_filter
        self.max_quality_retries = max_quality_retries
        self._cancel_llm_event = threading.Event()

    @Slot()
    def run(self):
        try:
            tried_paths = set()
            if self.fixed_image_path is not None:
                self.image_selected.emit(str(self.fixed_image_path))
                runner = self._build_runner(self.fixed_image_path)
                for event in runner.iter_events(self.fixed_image_path):
                    self.event_received.emit(event)
            else:
                for _attempt in range(self.max_quality_retries):
                    image_path = self._choose_random_image(exclude=tried_paths, label_filter=self.label_filter)
                    tried_paths.add(image_path)
                    self.image_selected.emit(str(image_path))
                    runner = self._build_runner(image_path)

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

    @Slot()
    def cancel_llm_inference(self):
        self._cancel_llm_event.set()

    def _build_runner(self, image_path: Path):
        operational_context = self._build_operational_context(image_path)
        self.wind_updated.emit(self._wind_status_payload(operational_context))

        kwargs = dict(
            yolo_model=self.yolo_model,
            llm_model=self.llm_model,
            skip_quality_screening=self.skip_quality_screening,
            llm_enabled=self.llm_enabled,
            should_cancel_llm=self._cancel_llm_event.is_set,
            yolo_input_mode=self.yolo_input_mode,
            operational_context=operational_context,
        )
        if self.prompt_file:
            kwargs["prompt_file"] = self.prompt_file
        if self.context_file:
            kwargs["context_file"] = self.context_file
        return create_default_pipeline(**kwargs)

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

    def _build_operational_context(self, image_path: Path) -> OperationalContext | None:
        context_file = Path(self.context_file) if self.context_file else None
        if context_file is None and self.wind_mode != WIND_MODE_NONE:
            context_file = DEFAULT_CONTEXT_FILE

        if context_file is None:
            return None

        if not context_file.exists():
            if self.wind_mode != WIND_MODE_NONE:
                raise FileNotFoundError(
                    f"Operational context not found: {context_file}"
                )
            return None

        context = OperationalContext.model_validate_json(
            context_file.read_text(encoding="utf-8")
        )

        if self.wind_mode == WIND_MODE_NONE:
            wind = None
        elif self.wind_mode == WIND_MODE_MANUAL:
            wind = build_manual_wind(self.wind_direction, self.wind_speed_mps)
        elif self.wind_mode == WIND_MODE_MOCKED:
            wind = build_mock_wind(f"ui-demo:{image_path.stem}:{random.random()}")
        else:
            raise ValueError(f"Unknown wind mode: {self.wind_mode}")

        return context.model_copy(update={"wind": wind})

    @staticmethod
    def _wind_status_payload(context: OperationalContext | None) -> dict:
        if context is None or context.wind is None:
            return {"text": "NO WIND", "direction": None, "speed_mps": None}
        return {
            "text": f"{context.wind.speed_mps:.1f}m/s {context.wind.direction_compass}",
            "direction": context.wind.direction_compass,
            "speed_mps": context.wind.speed_mps,
        }
