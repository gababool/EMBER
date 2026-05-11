from pathlib import Path
import shutil
import tempfile
import uuid

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from ui.assets.design import (
        DEFAULT_THEME,
        FONT_SIZE_LG,
        FONT_SIZE_MD,
        FONT_SIZE_SM,
        FONT_SIZE_XL,
        FONT_SIZE_XS,
        Theme,
        app_font,
    )
    from ui.components.classification_badge import ClassificationBadge, CorrectnessBadge
    from ui.components.history_strip import HistoryStrip
    from ui.components.image_preview_dialog import ImagePreviewDialog
    from ui.components.model_combo_box import ModelComboBox
    from ui.components.result_panel import ResultPanel
    from pipeline.context.wind import build_mock_wind
    from pipeline.llm.inference import BENCHMARK_MODEL_TAGS, PROMPTS_DIR
    from pipeline.object_detection.detections import YOLO_INPUT_MODES
    from ui.pipeline_worker import DEMO_DATASET_DIR
except ImportError:
    from assets.design import (
        DEFAULT_THEME,
        FONT_SIZE_LG,
        FONT_SIZE_MD,
        FONT_SIZE_SM,
        FONT_SIZE_XL,
        FONT_SIZE_XS,
        Theme,
        app_font,
    )
    from components.classification_badge import ClassificationBadge, CorrectnessBadge
    from components.history_strip import HistoryStrip
    from components.image_preview_dialog import ImagePreviewDialog
    from components.model_combo_box import ModelComboBox
    from components.result_panel import ResultPanel
    from pipeline.context.wind import build_mock_wind
    from pipeline.llm.inference import BENCHMARK_MODEL_TAGS, PROMPTS_DIR
    from pipeline.object_detection.detections import YOLO_INPUT_MODES
    from pipeline_worker import DEMO_DATASET_DIR


HISTORY_IMAGE_DIR = Path(tempfile.gettempdir()) / "ember_ui_history"
HEADER_CONTROL_WIDTH = 150
COMPACT_HEADER_CONTROL_WIDTH = 118
HEADER_CONTROL_HEIGHT = 40
CONTEXTS_DIR = Path(__file__).resolve().parents[2] / "data" / "contexts"
_CONTEXT_MARKER = "structured operational context block"


def _prompt_requires_context(prompt_path: str) -> bool:
    try:
        return _CONTEXT_MARKER in Path(prompt_path).read_text(encoding="utf-8")
    except OSError:
        return False


class PipelineDashboard(QWidget):
    image_picked = Signal(str)
    cancel_llm_requested = Signal()
    llm_started = Signal()

    def __init__(self, theme: Theme = DEFAULT_THEME, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.expected_label = None
        self.current_run = None
        self.run_history = []
        self.compact_layout = False
        self.last_image_path = None
        self.current_preview_image_path = None
        self._current_preview_pixmap = QPixmap()
        self._generated_mock_wind = None
        self.setObjectName("PipelineDashboard")
        self.setAutoFillBackground(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        # Row 1: title + action buttons
        title_row = QWidget()
        title_row_layout = QHBoxLayout(title_row)
        title_row_layout.setContentsMargins(0, 0, 0, 0)
        title_row_layout.setSpacing(12)

        title_block = QWidget()
        title_block.setMinimumWidth(0)
        title_layout = QVBoxLayout(title_block)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)

        self.title_label = QLabel("Pipeline Demo")
        self.title_label.setFont(app_font(FONT_SIZE_XL, bold=True))
        self.title_label.setMinimumWidth(0)

        self.subtitle_label = QLabel("Run a wildfire image through screening, detection, and reasoning.")
        self.subtitle_label.setFont(app_font(FONT_SIZE_SM))
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setMinimumWidth(0)

        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.subtitle_label)

        self.pick_button = QPushButton("Pick Image")
        self.pick_button.setFont(app_font(FONT_SIZE_MD, bold=True))
        self.pick_button.setCursor(Qt.PointingHandCursor)
        self.pick_button.setFixedHeight(40)
        self.pick_button.clicked.connect(self._open_image_picker)

        self.rerun_button = QPushButton("Re-run")
        self.rerun_button.setFont(app_font(FONT_SIZE_MD, bold=True))
        self.rerun_button.setCursor(Qt.PointingHandCursor)
        self.rerun_button.setFixedHeight(40)
        self.rerun_button.setEnabled(False)

        self.start_button = QPushButton("Test Image")
        self.start_button.setFont(app_font(FONT_SIZE_MD, bold=True))
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.setFixedSize(HEADER_CONTROL_WIDTH, HEADER_CONTROL_HEIGHT)

        self.stop_llm_button = QPushButton("Stop LLM")
        self.stop_llm_button.setFont(app_font(FONT_SIZE_MD, bold=True))
        self.stop_llm_button.setCursor(Qt.PointingHandCursor)
        self.stop_llm_button.setFixedSize(HEADER_CONTROL_WIDTH, HEADER_CONTROL_HEIGHT)
        self.stop_llm_button.setEnabled(False)
        self.stop_llm_button.setVisible(False)
        self.stop_llm_button.clicked.connect(self._request_llm_cancel)

        title_row_layout.addWidget(title_block, 1)
        title_row_layout.addWidget(self.pick_button)
        title_row_layout.addWidget(self.rerun_button)
        title_row_layout.addWidget(self.start_button)
        title_row_layout.addWidget(self.stop_llm_button)

        # Row 2: settings controls
        controls_row = QWidget()
        controls_row_layout = QHBoxLayout(controls_row)
        controls_row_layout.setContentsMargins(0, 0, 0, 0)
        controls_row_layout.setSpacing(8)

        self.image_filter_select = ModelComboBox(theme)
        self.image_filter_select.setMinimumWidth(100)
        for label, data in [("Any", ""), ("Fire", "fire"), ("No Fire", "nofire")]:
            self.image_filter_select.addItem(label, data)

        self.yolo_mode_select = ModelComboBox(theme)
        self.yolo_mode_select.setMinimumWidth(170)
        yolo_mode_labels = {
            "annotated_image": "YOLO: Annotated Image",
            "disabled": "YOLO: Off",
            "context_summary": "YOLO: Box Summary",
            "context_locations": "YOLO: Box Location",
        }
        for mode in ("annotated_image", "disabled", "context_summary", "context_locations"):
            if mode not in YOLO_INPUT_MODES:
                continue
            self.yolo_mode_select.addItem(yolo_mode_labels[mode], mode)

        self.skip_quality_button = QPushButton("Quality: ON")
        self.skip_quality_button.setFont(app_font(FONT_SIZE_MD, bold=True))
        self.skip_quality_button.setCursor(Qt.PointingHandCursor)
        self.skip_quality_button.setFixedHeight(36)
        self.skip_quality_button.setCheckable(True)
        self.skip_quality_button.toggled.connect(self._on_skip_quality_toggled)

        self.llm_toggle_button = QPushButton("LLM: ON")
        self.llm_toggle_button.setFont(app_font(FONT_SIZE_MD, bold=True))
        self.llm_toggle_button.setCursor(Qt.PointingHandCursor)
        self.llm_toggle_button.setFixedHeight(36)
        self.llm_toggle_button.setCheckable(True)
        self.llm_toggle_button.setChecked(True)
        self.llm_toggle_button.toggled.connect(self._on_llm_enabled_toggled)

        self.context_select = ModelComboBox(theme)
        self.context_select.setMinimumWidth(150)
        for ctx_path in sorted(CONTEXTS_DIR.glob("*.json")):
            self.context_select.addItem(ctx_path.stem, str(ctx_path))
        self.context_select.setVisible(False)

        self.prompt_select = ModelComboBox(theme)
        self.prompt_select.setMinimumWidth(130)
        for prompt_path in sorted(PROMPTS_DIR.glob("*.txt")):
            self.prompt_select.addItem(prompt_path.stem, str(prompt_path))
        self.prompt_select.currentIndexChanged.connect(self._on_prompt_changed)

        self.model_select = ModelComboBox(theme)
        for model_tag in BENCHMARK_MODEL_TAGS:
            self.model_select.addItem(model_tag, model_tag)
        self.model_select.setFixedSize(HEADER_CONTROL_WIDTH, HEADER_CONTROL_HEIGHT)

        self.wind_mode_select = ModelComboBox(theme)
        self.wind_mode_select.addItem("No wind", "none")
        self.wind_mode_select.addItem("Manual", "manual")
        self.wind_mode_select.addItem("Mocked", "mocked")
        self.wind_mode_select.setFixedSize(HEADER_CONTROL_WIDTH, HEADER_CONTROL_HEIGHT)
        self.wind_mode_select.currentIndexChanged.connect(self._sync_wind_controls)

        self.wind_direction_select = ModelComboBox(theme)
        for direction in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
            self.wind_direction_select.addItem(direction, direction)
        self.wind_direction_select.setCurrentText("SW")
        self.wind_direction_select.setFixedSize(HEADER_CONTROL_WIDTH, HEADER_CONTROL_HEIGHT)

        self.wind_speed_input = WindSpeedControl(theme)
        self.wind_speed_input.setFixedSize(HEADER_CONTROL_WIDTH, HEADER_CONTROL_HEIGHT)

        controls_row_layout.addWidget(self.image_filter_select)
        controls_row_layout.addWidget(self.yolo_mode_select)
        controls_row_layout.addWidget(self.skip_quality_button)
        controls_row_layout.addWidget(self.llm_toggle_button)
        controls_row_layout.addStretch(1)
        controls_row_layout.addWidget(self.context_select)
        controls_row_layout.addWidget(self.prompt_select)
        controls_row_layout.addWidget(self.model_select)
        controls_row_layout.addWidget(self.wind_mode_select)
        controls_row_layout.addWidget(self.wind_direction_select)
        controls_row_layout.addWidget(self.wind_speed_input)

        header_layout.addWidget(title_row)
        header_layout.addWidget(controls_row)

        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(18)
        self.grid.setVerticalSpacing(18)

        self.quality_body = QWidget()
        quality_layout = QVBoxLayout(self.quality_body)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        quality_layout.setSpacing(10)
        self.quality_rows = []
        self.quality_row_map = {}
        for check_name in ("Resolution", "Brightness", "Contrast", "Sharpness"):
            row = self._make_check_row(check_name)
            self.quality_rows.append(row)
            self.quality_row_map[check_name.lower()] = row
            quality_layout.addWidget(row)
        quality_layout.addStretch()

        self.image_placeholder = ClickableImageLabel("Processed image will appear here")
        self.image_placeholder.setAlignment(Qt.AlignCenter)
        self.image_placeholder.setMinimumSize(360, 220)
        self.image_placeholder.clicked.connect(self._open_image_preview)

        self.open_preview_button = QPushButton("Open Preview")
        self.open_preview_button.setFont(app_font(FONT_SIZE_SM, bold=True))
        self.open_preview_button.setCursor(Qt.PointingHandCursor)
        self.open_preview_button.setFixedHeight(32)
        self.open_preview_button.setEnabled(False)
        self.open_preview_button.clicked.connect(self._open_image_preview)

        self.image_body = QWidget()
        image_layout = QVBoxLayout(self.image_body)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(10)
        image_layout.addWidget(self.image_placeholder, 1)
        image_layout.addWidget(self.open_preview_button, 0, Qt.AlignRight)

        self.reasoning_text = QTextEdit()
        self.reasoning_text.setReadOnly(True)
        self.reasoning_text.setPlainText("LLM reasoning will appear here after the demo runs.")
        self.reasoning_text.setFont(app_font(FONT_SIZE_SM))

        self.reasoning_body = QWidget()
        reasoning_layout = QVBoxLayout(self.reasoning_body)
        reasoning_layout.setContentsMargins(0, 0, 0, 0)
        reasoning_layout.setSpacing(12)

        self.classification_badge = ClassificationBadge(theme)
        self.correctness_badge = CorrectnessBadge(theme)
        reasoning_layout.addWidget(self.classification_badge)
        reasoning_layout.addWidget(self.correctness_badge)
        reasoning_layout.addWidget(self.reasoning_text, 1)

        self.quality_panel = ResultPanel(
            "Quality Screening",
            "Checklist for image suitability before detection.",
            self.quality_body,
            theme,
        )
        self.image_panel = ResultPanel(
            "Processed Image",
            "Detection output with bounding boxes.",
            self.image_body,
            theme,
        )
        self.reasoning_panel = ResultPanel(
            "LLM Reasoning",
            "Structured assessment and recommendation.",
            self.reasoning_body,
            theme,
        )

        self.grid.addWidget(self.quality_panel, 0, 0)
        self.grid.addWidget(self.image_panel, 0, 1)
        self.grid.addWidget(self.reasoning_panel, 0, 2)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 2)
        self.grid.setColumnStretch(2, 2)

        self.history_strip = HistoryStrip(theme)
        self.history_strip.run_selected.connect(self.load_history_run)

        layout.addWidget(header)
        layout.addLayout(self.grid, 1)
        layout.addWidget(self.history_strip)

        self.apply_theme(theme)
        self._on_prompt_changed(self.prompt_select.current_index)
        self._sync_wind_controls()

    def reset_demo(self):
        self.subtitle_label.setText("Selecting a random wildfire dataset image...")
        self.start_button.setEnabled(False)
        self.rerun_button.setEnabled(False)
        self.pick_button.setEnabled(False)
        self.model_select.setEnabled(False)
        self.wind_mode_select.setEnabled(False)
        self.wind_direction_select.setEnabled(False)
        self.wind_speed_input.setEnabled(False)
        self.prompt_select.setEnabled(False)
        self.yolo_mode_select.setEnabled(False)
        self.context_select.setEnabled(False)
        self.image_filter_select.setEnabled(False)
        self.skip_quality_button.setEnabled(False)
        self.llm_toggle_button.setEnabled(False)
        self.stop_llm_button.setEnabled(False)
        self._set_llm_cancel_visible(False)
        self.start_button.setText("Running")
        self.expected_label = None
        self._start_current_run_record()
        self._reset_stage_views()

    def _start_current_run_record(self):
        self.current_run = {
            "image_path": None,
            "expected_label": None,
            "quality_result": None,
            "detection_result": None,
            "reasoning_result": None,
            "error": None,
            "history_image_path": None,
            "llm_model": self.selected_llm_model(),
            "llm_enabled": self.llm_inference_enabled(),
            "yolo_input_mode": self.selected_yolo_input_mode(),
            "wind": self.selected_wind_config(),
        }

    def _reset_stage_views(self):
        skip_quality = self.skip_quality_button.isChecked()
        self.quality_panel.setVisible(not skip_quality)
        for row in self.quality_rows:
            row.set_state("pending")
        self._clear_processed_image("Processed image will appear here")
        self.classification_badge.set_classification(
            "pending" if self.llm_inference_enabled() else "disabled"
        )
        self.correctness_badge.reset()
        if self.llm_inference_enabled():
            self.reasoning_text.setPlainText("Waiting for LLM reasoning...")
        else:
            self.reasoning_text.setPlainText(
                "LLM inference disabled. Quality screening and YOLO will still run."
            )

    def set_demo_finished(self):
        self.start_button.setEnabled(True)
        self.rerun_button.setEnabled(self.last_image_path is not None)
        self.pick_button.setEnabled(True)
        self.wind_mode_select.setEnabled(True)
        self._sync_wind_controls()
        self.yolo_mode_select.setEnabled(True)
        self.image_filter_select.setEnabled(True)
        self.skip_quality_button.setEnabled(True)
        self.llm_toggle_button.setEnabled(True)
        self._sync_llm_controls()
        self.stop_llm_button.setEnabled(False)
        self._set_llm_cancel_visible(False)
        self.start_button.setText("Test Image")
        self.quality_panel.setVisible(True)
        self._save_current_run_to_history()

    def selected_llm_model(self) -> str:
        return self.model_select.currentData()

    def llm_inference_enabled(self) -> bool:
        return self.llm_toggle_button.isChecked()

    def selected_wind_config(self) -> dict:
        return {
            "mode": self.wind_mode_select.currentData(),
            "direction": self.wind_direction_select.currentData(),
            "speed_mps": self.wind_speed_input.value(),
        }

    def wind_status_text(self) -> str:
        if not self.llm_inference_enabled():
            self._generated_mock_wind = None
            return "NO WIND"

        config = self.selected_wind_config()
        if config["mode"] == "none":
            self._generated_mock_wind = None
            return "NO WIND"
        if config["mode"] == "mocked":
            if self._generated_mock_wind is None:
                preview_wind = build_mock_wind("ui-preview")
                self._generated_mock_wind = {
                    "direction": preview_wind.direction_compass,
                    "speed_mps": preview_wind.speed_mps,
                }
            return (
                f"{self._generated_mock_wind['speed_mps']:.1f}m/s "
                f"{self._generated_mock_wind['direction']}"
            )
        self._generated_mock_wind = None
        return f"{config['speed_mps']:.1f}m/s {config['direction']}"

    def set_generated_wind(self, direction: str | None, speed_mps: float | None):
        if self.wind_mode_select.currentData() != "mocked":
            return
        if direction and speed_mps is not None:
            self._generated_mock_wind = {
                "direction": direction,
                "speed_mps": speed_mps,
            }
        if direction:
            self.wind_direction_select.setCurrentText(direction)
        if speed_mps is not None:
            self.wind_speed_input.setValue(speed_mps)
        self._sync_wind_controls()

    def selected_prompt_file(self) -> str:
        return self.prompt_select.currentData()

    def selected_context_file(self) -> str | None:
        if not self.llm_inference_enabled() or not self.context_select.isVisible():
            return None
        return self.context_select.currentData()

    def selected_yolo_input_mode(self) -> str:
        return self.yolo_mode_select.currentData() or "annotated_image"

    def use_annotation(self) -> bool:
        return self.selected_yolo_input_mode() == "annotated_image"

    def _on_prompt_changed(self, _index: int):
        self._sync_llm_controls()

    def selected_image_filter(self) -> str | None:
        data = self.image_filter_select.currentData()
        return data or None

    def skip_quality_screening(self) -> bool:
        return self.skip_quality_button.isChecked()

    def selected_rerun_image(self) -> str | None:
        return self.last_image_path

    def _open_image_picker(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Pick Image",
            str(DEMO_DATASET_DIR),
            "Images (*.jpg *.jpeg *.png)",
        )
        if path:
            self.image_picked.emit(path)

    def _on_skip_quality_toggled(self, checked: bool):
        self.skip_quality_button.setText("Quality: OFF" if checked else "Quality: ON")
        self._style_skip_quality_button(self.theme)

    def _on_llm_enabled_toggled(self, checked: bool):
        self.llm_toggle_button.setText("LLM: ON" if checked else "LLM: OFF")
        self._style_llm_toggle_button(self.theme)
        self._sync_llm_controls()
        if not checked and self.current_run is None:
            self.classification_badge.set_classification("disabled")
            self.correctness_badge.reset()
            self.reasoning_text.setPlainText(
                "LLM inference disabled. Quality screening and YOLO will still run."
            )
        elif self.current_run is None:
            self.classification_badge.set_classification("pending")
            self.reasoning_text.setPlainText("LLM reasoning will appear here after the demo runs.")

    def _sync_llm_controls(self):
        visible = self.llm_inference_enabled()
        enabled = visible and self.llm_toggle_button.isEnabled()
        context_visible = visible and _prompt_requires_context(
            self.prompt_select.currentData() or ""
        )

        self.model_select.setVisible(visible)
        self.prompt_select.setVisible(visible)
        self.context_select.setVisible(context_visible)
        self.model_select.setEnabled(enabled)
        self.prompt_select.setEnabled(enabled)
        self.context_select.setEnabled(enabled and context_visible)
        self._sync_wind_controls()

    def _request_llm_cancel(self):
        self.stop_llm_button.setEnabled(False)
        self.stop_llm_button.setText("Cancelled")
        self.reasoning_text.setPlainText("LLM inference cancelled.")
        self.cancel_llm_requested.emit()

    def cancel_llm_immediately(self, status: str = "cancelled"):
        self.set_reasoning_result({
            "status": status,
            "parsed": {},
            "raw_response": "",
            "model_name": self.selected_llm_model(),
            "prompt_file": self.selected_prompt_file(),
        })
        self.set_demo_finished()

    def set_llm_background_stopping(self, stopping: bool):
        status = self.classification_badge.classification
        if status not in {"cancelled", "timed_out"}:
            return
        action_text = (
            "LLM inference timed out after 2 minutes."
            if status == "timed_out"
            else "LLM inference cancelled."
        )
        if stopping:
            self.reasoning_text.setPlainText(
                f"{action_text} Existing quality screening and YOLO results were preserved.\n\n"
                "Ollama is still stopping in the background."
            )
        else:
            self.reasoning_text.setPlainText(
                f"{action_text} Existing quality screening and YOLO results were preserved."
            )

    def _set_llm_cancel_visible(self, visible: bool):
        self.start_button.setVisible(not visible)
        self.stop_llm_button.setVisible(visible)

    def set_selected_image(self, image_path: str):
        if self.current_run is None:
            self._start_current_run_record()
            self._reset_stage_views()

        path = Path(image_path)
        self.last_image_path = str(path)
        label = path.parent.name
        self.expected_label = label
        if self.current_run is not None:
            self.current_run["image_path"] = str(path)
            self.current_run["expected_label"] = label
        self.correctness_badge.set_expected_label(label)
        self.subtitle_label.setText(f"Running demo image: {label}/{path.name}")

    def set_pipeline_error(self, message: str):
        if self.current_run is not None:
            self.current_run["error"] = message
        self.reasoning_text.setPlainText(f"Pipeline error:\n{message}")

    def handle_pipeline_event(self, event):
        if event.event_type == "stage_started":
            self._set_stage_running(event.stage_name)
            return

        if event.result is None:
            return

        result = event.result
        if result.stage_name == "quality_screening":
            self.set_quality_result(result.output)
            if not result.passed:
                if result.error:
                    self.set_pipeline_error(result.error)
                self._save_current_run_to_history(quality_failed=True)
                return
        elif result.stage_name == "object_detection":
            self.set_detection_result(result.output)
        elif result.stage_name == "llm_reasoning":
            self.set_reasoning_result(result.output)

        if not result.passed and result.error:
            self.set_pipeline_error(result.error)

    def set_quality_result(self, result: dict):
        if self.current_run is not None:
            self.current_run["quality_result"] = result
        checks = result.get("checks", {})
        for check_name, row in self.quality_row_map.items():
            check_result = checks.get(check_name)
            if check_result is None:
                row.set_state("pending")
                continue
            row.set_state("passed" if check_result.get("passed") else "failed")

    def set_detection_result(self, result: dict):
        if self.current_run is not None:
            self.current_run["detection_result"] = result
        if not result.get("passed"):
            self._clear_processed_image(result.get("error", "Object detection failed"))
            return

        annotated_path = result.get("annotated_image_path")
        if not annotated_path:
            self._clear_processed_image("No annotated image returned")
            return

        self._load_image_into_processed_box(annotated_path)

    def set_reasoning_result(self, result: dict):
        if self.current_run is not None:
            self.current_run["reasoning_result"] = result
        status = result.get("status")
        if status == "disabled":
            self.stop_llm_button.setEnabled(False)
            self._set_llm_cancel_visible(False)
            self.classification_badge.set_classification("disabled")
            self.correctness_badge.reset()
            self.reasoning_text.setPlainText(
                "LLM inference disabled. Quality screening and YOLO completed without model reasoning."
            )
            return
        if status == "cancelled":
            self.stop_llm_button.setEnabled(False)
            self.stop_llm_button.setText("Stop LLM")
            self._set_llm_cancel_visible(False)
            self.classification_badge.set_classification("cancelled")
            self.correctness_badge.reset()
            self.reasoning_text.setPlainText(
                "LLM inference cancelled. Existing quality screening and YOLO results were preserved."
            )
            return
        if status == "timed_out":
            self.stop_llm_button.setEnabled(False)
            self.stop_llm_button.setText("Stop LLM")
            self._set_llm_cancel_visible(False)
            self.classification_badge.set_classification("timed_out")
            self.correctness_badge.reset()
            self.reasoning_text.setPlainText(
                "LLM inference timed out after 2 minutes. Existing quality screening and YOLO results were preserved."
            )
            return

        self.stop_llm_button.setEnabled(False)
        self.stop_llm_button.setText("Stop LLM")
        self._set_llm_cancel_visible(False)
        parsed = result.get("parsed", {})
        if not parsed:
            self.reasoning_text.setPlainText("No reasoning returned.")
            return

        classification = parsed.get("classification", "unknown")
        reasoning = parsed.get("reasoning", "")
        recommendation = parsed.get("recommendation", "")
        situation_brief = parsed.get("situation_brief") or ""
        tactical_priority = parsed.get("tactical_priority") or ""
        self.classification_badge.set_classification(classification)
        self.correctness_badge.set_prediction(classification)

        parts = []
        if situation_brief:
            parts.append(f"Situation Brief:\n{situation_brief}")
        if reasoning:
            parts.append(f"Reasoning:\n{reasoning}")
        if tactical_priority:
            parts.append(f"Tactical Priority:\n{tactical_priority}")
        if recommendation:
            parts.append(f"Recommendation:\n{recommendation}")
        self.reasoning_text.setPlainText("\n\n".join(parts) if parts else "No reasoning returned.")

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.setStyleSheet(f"""
            QWidget#PipelineDashboard {{
                background-color: {theme.bg_main};
                color: {theme.text_primary};
            }}
        """)
        self.title_label.setStyleSheet(self._label_style(theme.text_primary))
        self.subtitle_label.setStyleSheet(self._label_style(theme.text_muted))
        self.start_button.setStyleSheet(f"""
            QPushButton {{
                color: #ffffff;
                background-color: {theme.accent_orange};
                border: 1px solid {theme.accent_orange};
                border-radius: 5px;
                padding: 6px 18px;
            }}
            QPushButton:hover {{
                background-color: {theme.accent_blue};
                border-color: {theme.accent_blue};
            }}
            QPushButton:pressed {{
                background-color: {theme.accent_cyan};
                border-color: {theme.accent_cyan};
            }}
            QPushButton:disabled {{
                color: {theme.text_primary};
                background-color: {theme.bg_panel_alt};
                border-color: {theme.border};
            }}
        """)
        self.pick_button.setStyleSheet(f"""
            QPushButton {{
                color: {theme.text_primary};
                background-color: {theme.bg_panel_alt};
                border: 1px solid {theme.border};
                border-radius: 5px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                border-color: {theme.accent_cyan};
            }}
            QPushButton:pressed {{
                background-color: {theme.bg_panel};
            }}
            QPushButton:disabled {{
                color: {theme.text_muted};
                border-color: {theme.border};
            }}
        """)
        self.rerun_button.setStyleSheet(f"""
            QPushButton {{
                color: {theme.text_primary};
                background-color: {theme.bg_panel_alt};
                border: 1px solid {theme.border};
                border-radius: 5px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                border-color: {theme.accent_orange};
            }}
            QPushButton:pressed {{
                background-color: {theme.bg_panel};
            }}
            QPushButton:disabled {{
                color: {theme.text_muted};
                border-color: {theme.border};
            }}
        """)
        self.open_preview_button.setStyleSheet(f"""
            QPushButton {{
                color: {theme.text_primary};
                background-color: {theme.bg_panel_alt};
                border: 1px solid {theme.border};
                border-radius: 5px;
                padding: 5px 12px;
            }}
            QPushButton:hover {{
                border-color: {theme.accent_cyan};
            }}
            QPushButton:pressed {{
                background-color: {theme.bg_panel};
            }}
            QPushButton:disabled {{
                color: {theme.text_muted};
                border-color: {theme.border};
            }}
        """)
        self.model_select.apply_theme(theme)
        self.wind_mode_select.apply_theme(theme)
        self.wind_direction_select.apply_theme(theme)
        self.wind_speed_input.apply_theme(theme)
        self.prompt_select.apply_theme(theme)
        self.context_select.apply_theme(theme)
        self.image_filter_select.apply_theme(theme)
        self.yolo_mode_select.apply_theme(theme)
        self._style_skip_quality_button(theme)
        self._style_llm_toggle_button(theme)
        self._style_stop_llm_button(theme)

        for panel in (self.quality_panel, self.image_panel, self.reasoning_panel):
            panel.apply_theme(theme)

        for row in self.quality_rows:
            row.apply_theme(theme)

        self.classification_badge.apply_theme(theme)
        self.correctness_badge.apply_theme(theme)
        self.history_strip.apply_theme(theme)
        self.image_placeholder.setStyleSheet(f"""
            QLabel {{
                color: {theme.text_muted};
                background-color: transparent;
                border: none;
            }}
        """)
        self.reasoning_text.setStyleSheet(f"""
            QTextEdit {{
                color: {theme.text_primary};
                background-color: {theme.bg_panel_alt};
                border: 1px solid {theme.border};
                border-radius: 4px;
                padding: 10px;
            }}
        """)

    def _make_check_row(self, label: str):
        row = CheckRow(label, self.theme)
        return row

    def _sync_wind_controls(self):
        llm_visible = self.llm_inference_enabled()
        wind_enabled = llm_visible and self.llm_toggle_button.isEnabled()
        manual = self.wind_mode_select.currentData() == "manual"
        controls_enabled = wind_enabled and manual
        self.wind_mode_select.setVisible(llm_visible)
        self.wind_direction_select.setVisible(llm_visible and manual)
        self.wind_speed_input.setVisible(llm_visible and manual)
        self.wind_mode_select.setEnabled(wind_enabled)
        self.wind_direction_select.setEnabled(controls_enabled)
        self.wind_speed_input.setEnabled(controls_enabled)

    def close_open_popups(self):
        self.model_select.close_popup()
        self.prompt_select.close_popup()
        self.context_select.close_popup()
        self.image_filter_select.close_popup()
        self.wind_mode_select.close_popup()
        self.wind_direction_select.close_popup()

    def resizeEvent(self, event):
        self.close_open_popups()
        super().resizeEvent(event)
        self._refresh_processed_pixmap()
        self._apply_responsive_layout()

    def _apply_responsive_layout(self):
        compact = self.width() < 1250
        if compact == self.compact_layout:
            return

        self.compact_layout = compact
        self._set_header_control_width(
            COMPACT_HEADER_CONTROL_WIDTH if compact else HEADER_CONTROL_WIDTH
        )
        for panel in (self.quality_panel, self.image_panel, self.reasoning_panel):
            self.grid.removeWidget(panel)

        if compact:
            self.grid.addWidget(self.quality_panel, 0, 0)
            self.grid.addWidget(self.image_panel, 1, 0)
            self.grid.addWidget(self.reasoning_panel, 2, 0)
            self.grid.setColumnStretch(0, 1)
            self.grid.setColumnStretch(1, 0)
            self.grid.setColumnStretch(2, 0)
        else:
            self.grid.addWidget(self.quality_panel, 0, 0)
            self.grid.addWidget(self.image_panel, 0, 1)
            self.grid.addWidget(self.reasoning_panel, 0, 2)
            self.grid.setColumnStretch(0, 1)
            self.grid.setColumnStretch(1, 2)
            self.grid.setColumnStretch(2, 2)

    def _set_header_control_width(self, width: int):
        for widget in (
            self.model_select,
            self.wind_mode_select,
            self.wind_direction_select,
            self.wind_speed_input,
            self.start_button,
            self.stop_llm_button,
        ):
            widget.setFixedSize(width, HEADER_CONTROL_HEIGHT)

    def _set_stage_running(self, stage_name: str):
        if stage_name == "quality_screening":
            for row in self.quality_rows:
                row.set_state("running")
        elif stage_name == "object_detection":
            self._clear_processed_image("Running object detection...")
        elif stage_name == "llm_reasoning":
            self.stop_llm_button.setText("Stop LLM")
            self.stop_llm_button.setEnabled(True)
            self._set_llm_cancel_visible(True)
            self.llm_started.emit()
            self.reasoning_text.setPlainText("Running LLM reasoning...")

    def _save_current_run_to_history(self, quality_failed: bool = False):
        if not self.current_run or not self.current_run.get("image_path"):
            return

        run_record = dict(self.current_run)
        run_record["quality_failed"] = quality_failed
        detection_result = run_record.get("detection_result") or {}
        annotated_path = detection_result.get("annotated_image_path")
        history_image_path = self._copy_history_image(annotated_path)
        if history_image_path is None:
            history_image_path = self._copy_history_image(run_record.get("image_path"))

        run_record["history_image_path"] = str(history_image_path) if history_image_path else None
        run_record["classification"] = self._classification_from_run(run_record)
        self.run_history.append(run_record)
        self.current_run = None
        self.history_strip.add_run(run_record)

    def load_history_run(self, index: int):
        if index < 0 or index >= len(self.run_history):
            return

        self.current_run = None
        run_record = self.run_history[index]
        image_path = Path(run_record.get("image_path", ""))
        self.last_image_path = str(image_path) if run_record.get("image_path") else self.last_image_path
        self.rerun_button.setEnabled(self.last_image_path is not None)
        expected_label = run_record.get("expected_label")
        self.expected_label = expected_label
        self.subtitle_label.setText(
            f"Loaded history image: {expected_label}/{image_path.name}"
        )

        self.reset_loaded_state(expected_label)

        quality_result = run_record.get("quality_result")
        if quality_result:
            self.set_quality_result(quality_result)

        detection_result = dict(run_record.get("detection_result") or {})
        if run_record.get("history_image_path"):
            detection_result["annotated_image_path"] = run_record["history_image_path"]
            detection_result["passed"] = detection_result.get("passed", True)
        if detection_result:
            self.set_detection_result(detection_result)
        elif run_record.get("image_path"):
            self._load_image_into_processed_box(run_record["image_path"])

        reasoning_result = run_record.get("reasoning_result")
        if reasoning_result:
            self.set_reasoning_result(reasoning_result)
        elif run_record.get("llm_enabled") is False:
            self.set_reasoning_result({"status": "disabled"})
        elif run_record.get("error"):
            self.reasoning_text.setPlainText(f"Pipeline error:\n{run_record['error']}")

        self.current_run = None

    def reset_loaded_state(self, expected_label: str | None):
        for row in self.quality_rows:
            row.set_state("pending")
        self._clear_processed_image("Processed image will appear here")
        self.classification_badge.set_classification("pending")
        self.correctness_badge.reset()
        if expected_label:
            self.correctness_badge.set_expected_label(expected_label)
        self.reasoning_text.setPlainText("No LLM reasoning stored for this run.")

    def _load_image_into_processed_box(self, image_path: str):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self._clear_processed_image(f"Could not load image:\n{image_path}")
            return
        self.current_preview_image_path = image_path
        self._current_preview_pixmap = pixmap
        self.image_placeholder.setText("")
        self.image_placeholder.setCursor(Qt.PointingHandCursor)
        self.open_preview_button.setEnabled(True)
        self._refresh_processed_pixmap()

    def _clear_processed_image(self, message: str):
        self.image_placeholder.setPixmap(QPixmap())
        self.image_placeholder.setText(message)
        self.image_placeholder.unsetCursor()
        self.current_preview_image_path = None
        self._current_preview_pixmap = QPixmap()
        self.open_preview_button.setEnabled(False)

    def _refresh_processed_pixmap(self):
        if self._current_preview_pixmap.isNull():
            return
        self.image_placeholder.setPixmap(
            self._current_preview_pixmap.scaled(
                self.image_placeholder.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _open_image_preview(self):
        if not self.current_preview_image_path:
            return
        dialog = ImagePreviewDialog(self.current_preview_image_path, self.theme, self)
        parent_window = self.window()
        parent_geometry = parent_window.geometry()
        dialog.resize(
            max(900, int(parent_geometry.width() * 0.9)),
            max(640, int(parent_geometry.height() * 0.86)),
        )
        dialog.move(parent_geometry.center() - dialog.rect().center())
        dialog.image_view.fit_image()
        dialog.exec()

    @staticmethod
    def _copy_history_image(image_path: str | None):
        if not image_path:
            return None

        source = Path(image_path)
        if not source.exists():
            return None

        HISTORY_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        target = HISTORY_IMAGE_DIR / f"{uuid.uuid4().hex}{source.suffix}"
        shutil.copy2(source, target)
        return target

    @staticmethod
    def _classification_from_run(run_record: dict) -> str:
        reasoning_result = run_record.get("reasoning_result") or {}
        status = reasoning_result.get("status")
        if status in {"disabled", "cancelled", "timed_out"}:
            return status
        if run_record.get("llm_enabled") is False:
            return "disabled"
        parsed = reasoning_result.get("parsed", {})
        return parsed.get("classification", "pending")

    def _style_skip_quality_button(self, theme: Theme):
        skipped = self.skip_quality_button.isChecked()
        border_color, text_color, background_color, hover_border, hover_text, hover_background = (
            self._toggle_style_values(theme, active=not skipped)
        )
        self.skip_quality_button.setStyleSheet(f"""
            QPushButton {{
                color: {text_color};
                background-color: {background_color};
                border: 1px solid {border_color};
                border-radius: 5px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                color: {hover_text};
                background-color: {hover_background};
                border-color: {hover_border};
            }}
            QPushButton:disabled {{
                color: {theme.text_muted};
                border-color: {theme.border};
            }}
        """)

    def _style_llm_toggle_button(self, theme: Theme):
        enabled = self.llm_toggle_button.isChecked()
        border_color, text_color, background_color, hover_border, hover_text, hover_background = (
            self._toggle_style_values(theme, active=enabled)
        )
        self.llm_toggle_button.setStyleSheet(f"""
            QPushButton {{
                color: {text_color};
                background-color: {background_color};
                border: 1px solid {border_color};
                border-radius: 5px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                color: {hover_text};
                background-color: {hover_background};
                border-color: {hover_border};
            }}
            QPushButton:disabled {{
                color: {theme.text_muted};
                border-color: {theme.border};
            }}
        """)

    @staticmethod
    def _toggle_style_values(theme: Theme, active: bool):
        if active:
            return (
                theme.border,
                theme.text_primary,
                theme.bg_panel_alt,
                theme.accent_orange,
                theme.text_primary,
                theme.bg_panel_alt,
            )

        if theme.name == "light":
            return (
                theme.danger,
                theme.danger,
                "#fff1f3",
                theme.danger,
                theme.danger,
                "#ffe3e7",
            )

        return (
            theme.danger,
            theme.danger,
            "rgba(255, 90, 102, 38)",
            theme.danger,
            theme.text_primary,
            "rgba(255, 90, 102, 72)",
        )

    def _style_stop_llm_button(self, theme: Theme):
        self.stop_llm_button.setStyleSheet(f"""
            QPushButton {{
                color: {theme.danger};
                background-color: {theme.bg_panel_alt};
                border: 1px solid {theme.danger};
                border-radius: 5px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                color: #ffffff;
                background-color: {theme.danger};
                border-color: {theme.danger};
            }}
            QPushButton:disabled {{
                color: {theme.text_muted};
                background-color: {theme.bg_panel_alt};
                border-color: {theme.border};
            }}
        """)

    @staticmethod
    def _label_style(color: str) -> str:
        return f"""
            QLabel {{
                color: {color};
                background: transparent;
                border: none;
            }}
        """


class ClickableImageLabel(QLabel):
    clicked = Signal()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap() and not self.pixmap().isNull():
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class CheckRow(QFrame):
    def __init__(self, text: str, theme: Theme = DEFAULT_THEME, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.state = "pending"
        self.setFixedHeight(42)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(10)

        self.icon_label = QLabel("○")
        self.icon_label.setFont(app_font(FONT_SIZE_LG, bold=True))
        self.icon_label.setFixedWidth(24)
        self.icon_label.setAlignment(Qt.AlignCenter)

        self.text_label = QLabel(text)
        self.text_label.setFont(app_font(FONT_SIZE_MD, bold=True))

        self.state_label = QLabel("Pending")
        self.state_label.setFont(app_font(FONT_SIZE_SM))
        self.state_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label, 1)
        layout.addWidget(self.state_label)

        self.apply_theme(theme)

    def set_state(self, state: str):
        state = state.lower()
        self.state = state
        if state == "passed":
            self.icon_label.setText("✓")
            self.state_label.setText("Passed")
            color = self.theme.accent_green
        elif state == "failed":
            self.icon_label.setText("✕")
            self.state_label.setText("Failed")
            color = self.theme.danger
        elif state == "running":
            self.icon_label.setText("…")
            self.state_label.setText("Running")
            color = self.theme.accent_blue
        else:
            self.icon_label.setText("○")
            self.state_label.setText("Pending")
            color = self.theme.text_muted

        self.icon_label.setStyleSheet(self._label_style(color))
        self.state_label.setStyleSheet(self._label_style(color))

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.setStyleSheet(f"""
            CheckRow {{
                background-color: {theme.bg_panel_alt};
                border: 1px solid {theme.border};
                border-radius: 4px;
            }}
        """)
        self.text_label.setStyleSheet(self._label_style(theme.text_primary))
        self.set_state(self.state)

    @staticmethod
    def _label_style(color: str) -> str:
        return f"""
            QLabel {{
                color: {color};
                background: transparent;
                border: none;
            }}
        """


class WindSpeedControl(QFrame):
    valueChanged = Signal(float)

    def __init__(self, theme: Theme = DEFAULT_THEME, parent=None):
        super().__init__(parent)
        self.theme = theme
        self._value = 6.5
        self._minimum = 0.0
        self._maximum = 40.0
        self._step = 0.5
        self.setObjectName("WindSpeedControl")
        self.setFixedHeight(40)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(0)

        self.value_label = QLabel()
        self.value_label.setFont(app_font(FONT_SIZE_SM, bold=True))
        self.value_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        button_column = QWidget()
        button_layout = QVBoxLayout(button_column)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)

        self.up_button = QPushButton("▲")
        self.down_button = QPushButton("▼")
        for button in (self.up_button, self.down_button):
            button.setCursor(Qt.PointingHandCursor)
            button.setFont(app_font(FONT_SIZE_XS, bold=True))
            button.setFixedSize(30, 20)

        self.up_button.clicked.connect(self.increase)
        self.down_button.clicked.connect(self.decrease)
        button_layout.addWidget(self.up_button)
        button_layout.addWidget(self.down_button)

        layout.addWidget(self.value_label, 1)
        layout.addWidget(button_column)

        self._refresh_label()
        self.apply_theme(theme)

    def value(self) -> float:
        return self._value

    def setValue(self, value: float, snap_to_step: bool = False):
        if snap_to_step:
            value = round(value / self._step) * self._step
        value = max(self._minimum, min(self._maximum, round(value, 1)))
        if value == self._value:
            return
        self._value = value
        self._refresh_label()
        self.valueChanged.emit(self._value)

    def increase(self):
        self.setValue(self._value + self._step, snap_to_step=True)

    def decrease(self):
        self.setValue(self._value - self._step, snap_to_step=True)

    def _refresh_label(self):
        self.value_label.setText(f"{self._value:.1f} m/s")

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.setStyleSheet(f"""
            QFrame#WindSpeedControl {{
                background-color: {theme.bg_panel_alt};
                border: 1px solid {theme.border};
                border-radius: 5px;
            }}
            QFrame#WindSpeedControl:hover {{
                border-color: {theme.accent_orange};
            }}
            QPushButton {{
                color: {theme.text_muted};
                background-color: {theme.bg_panel_alt};
                border: none;
                border-left: 1px solid {theme.border};
            }}
            QPushButton:hover {{
                color: {theme.text_primary};
                background-color: {theme.bg_panel};
            }}
            QPushButton:pressed {{
                color: {theme.text_primary};
                background-color: {theme.accent_orange};
            }}
        """)
        self.value_label.setStyleSheet(self._label_style(theme.text_primary))

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        color = self.theme.text_primary if enabled else self.theme.text_muted
        self.value_label.setStyleSheet(self._label_style(color))
        self.up_button.setEnabled(enabled)
        self.down_button.setEnabled(enabled)

    @staticmethod
    def _label_style(color: str) -> str:
        return f"""
            QLabel {{
                color: {color};
                background: transparent;
                border: none;
            }}
        """
