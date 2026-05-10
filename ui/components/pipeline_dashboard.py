from pathlib import Path
import shutil
import tempfile
import uuid

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
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
PREVIEW_ZOOM_STEP = 1.20
PREVIEW_WHEEL_DELTA_PER_STEP = 150


def _prompt_requires_context(prompt_path: str) -> bool:
    try:
        return _CONTEXT_MARKER in Path(prompt_path).read_text(encoding="utf-8")
    except OSError:
        return False


class PipelineDashboard(QWidget):
    image_picked = Signal(str)

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

        title_row_layout.addWidget(title_block, 1)
        title_row_layout.addWidget(self.pick_button)
        title_row_layout.addWidget(self.rerun_button)
        title_row_layout.addWidget(self.start_button)

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
            "yolo_input_mode": self.selected_yolo_input_mode(),
            "wind": self.selected_wind_config(),
        }

    def _reset_stage_views(self):
        skip_quality = self.skip_quality_button.isChecked()
        self.quality_panel.setVisible(not skip_quality)
        for row in self.quality_rows:
            row.set_state("pending")
        self._clear_processed_image("Processed image will appear here")
        self.classification_badge.set_classification("pending")
        self.correctness_badge.reset()
        self.reasoning_text.setPlainText("Waiting for LLM reasoning...")

    def set_demo_finished(self):
        self.start_button.setEnabled(True)
        self.rerun_button.setEnabled(self.last_image_path is not None)
        self.pick_button.setEnabled(True)
        self.model_select.setEnabled(True)
        self.wind_mode_select.setEnabled(True)
        self._sync_wind_controls()
        self.prompt_select.setEnabled(True)
        self.yolo_mode_select.setEnabled(True)
        self.context_select.setEnabled(True)
        self.image_filter_select.setEnabled(True)
        self.skip_quality_button.setEnabled(True)
        self.start_button.setText("Test Image")
        self.quality_panel.setVisible(True)
        self._save_current_run_to_history()

    def selected_llm_model(self) -> str:
        return self.model_select.currentData()

    def selected_wind_config(self) -> dict:
        return {
            "mode": self.wind_mode_select.currentData(),
            "direction": self.wind_direction_select.currentData(),
            "speed_mps": self.wind_speed_input.value(),
        }

    def wind_status_text(self) -> str:
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
        if not self.context_select.isVisible():
            return None
        return self.context_select.currentData()

    def selected_yolo_input_mode(self) -> str:
        return self.yolo_mode_select.currentData() or "annotated_image"

    def use_annotation(self) -> bool:
        return self.selected_yolo_input_mode() == "annotated_image"

    def _on_prompt_changed(self, _index: int):
        self.context_select.setVisible(
            _prompt_requires_context(self.prompt_select.currentData() or "")
        )

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
        manual = self.wind_mode_select.currentData() == "manual"
        controls_enabled = manual and self.wind_mode_select.isEnabled()
        self.wind_direction_select.setVisible(manual)
        self.wind_speed_input.setVisible(manual)
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
        ):
            widget.setFixedSize(width, HEADER_CONTROL_HEIGHT)

    def _set_stage_running(self, stage_name: str):
        if stage_name == "quality_screening":
            for row in self.quality_rows:
                row.set_state("running")
        elif stage_name == "object_detection":
            self._clear_processed_image("Running object detection...")
        elif stage_name == "llm_reasoning":
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
        parsed = reasoning_result.get("parsed", {})
        return parsed.get("classification", "pending")

    def _style_skip_quality_button(self, theme: Theme):
        checked = self.skip_quality_button.isChecked()
        border_color = theme.danger if checked else theme.border
        text_color = theme.danger if checked else theme.text_primary
        self.skip_quality_button.setStyleSheet(f"""
            QPushButton {{
                color: {text_color};
                background-color: {theme.bg_panel_alt};
                border: 1px solid {border_color};
                border-radius: 5px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                border-color: {theme.danger};
                color: {theme.danger};
            }}
            QPushButton:disabled {{
                color: {theme.text_muted};
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


class ZoomableImageView(QGraphicsView):
    zoom_changed = Signal(int)

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene = QGraphicsScene(self)
        self.scene.addItem(self.pixmap_item)
        self.setScene(self.scene)

        self.fit_to_view = True
        self.min_zoom = 0.10
        self.max_zoom = 12.0
        self.zoom_scale = 1.0

        self.setAlignment(Qt.AlignCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if not delta:
            event.ignore()
            return
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.zoom_by(PREVIEW_ZOOM_STEP ** (delta / PREVIEW_WHEEL_DELTA_PER_STEP))
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_to_view:
            self.fit_image()

    def zoom_by(self, factor: float):
        next_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom_scale * factor))
        if next_zoom == self.zoom_scale:
            return
        self.fit_to_view = False
        scale_factor = next_zoom / self.zoom_scale
        self.scale(scale_factor, scale_factor)
        self.zoom_scale = next_zoom
        self.zoom_changed.emit(int(self.zoom_scale * 100))

    def fit_image(self):
        if self.pixmap_item.pixmap().isNull():
            return
        self.fit_to_view = True
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.zoom_scale = self.transform().m11()
        self.zoom_changed.emit(int(self.zoom_scale * 100))


class ImagePreviewDialog(QDialog):
    def __init__(self, image_path: str, theme: Theme = DEFAULT_THEME, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.image_path = image_path
        self.source_pixmap = QPixmap(image_path)

        self.setWindowTitle("Image Preview")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)

        self.title_label = QLabel(Path(image_path).name)
        self.title_label.setFont(app_font(FONT_SIZE_MD, bold=True))

        self.zoom_out_button = QPushButton("-")
        self.fit_button = QPushButton("Fit")
        self.zoom_in_button = QPushButton("+")
        self.close_button = QPushButton("Close")
        for button in (
            self.zoom_out_button,
            self.fit_button,
            self.zoom_in_button,
            self.close_button,
        ):
            button.setCursor(Qt.PointingHandCursor)
            button.setFont(app_font(FONT_SIZE_SM, bold=True))
            button.setFixedHeight(34)

        self.zoom_out_button.setFixedWidth(42)
        self.zoom_in_button.setFixedWidth(42)
        self.fit_button.setFixedWidth(64)
        self.close_button.setFixedWidth(84)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setFont(app_font(FONT_SIZE_SM, bold=True))
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setFixedWidth(70)

        toolbar.addWidget(self.title_label, 1)
        toolbar.addWidget(self.zoom_out_button)
        toolbar.addWidget(self.zoom_label)
        toolbar.addWidget(self.zoom_in_button)
        toolbar.addWidget(self.fit_button)
        toolbar.addWidget(self.close_button)

        self.image_view = ZoomableImageView(self.source_pixmap)
        self.image_view.zoom_changed.connect(lambda value: self.zoom_label.setText(f"{value}%"))
        self.zoom_out_button.clicked.connect(lambda: self.image_view.zoom_by(1 / PREVIEW_ZOOM_STEP))
        self.zoom_in_button.clicked.connect(lambda: self.image_view.zoom_by(PREVIEW_ZOOM_STEP))
        self.fit_button.clicked.connect(self.image_view.fit_image)
        self.close_button.clicked.connect(self.accept)

        layout.addLayout(toolbar)
        layout.addWidget(self.image_view, 1)

        self.apply_theme(theme)
        self.image_view.fit_image()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.image_view.zoom_by(PREVIEW_ZOOM_STEP)
            return
        if event.key() == Qt.Key_Minus:
            self.image_view.zoom_by(1 / PREVIEW_ZOOM_STEP)
            return
        if event.key() == Qt.Key_0:
            self.image_view.fit_image()
            return
        super().keyPressEvent(event)

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {theme.bg_main};
                color: {theme.text_primary};
            }}
            QGraphicsView {{
                background-color: {theme.bg_panel};
                border: 1px solid {theme.border};
                border-radius: 6px;
            }}
            QScrollBar:vertical, QScrollBar:horizontal {{
                background-color: {theme.bg_panel_alt};
                border: none;
                width: 10px;
                height: 10px;
            }}
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
                background-color: {theme.border};
                border-radius: 5px;
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{
                width: 0px;
                height: 0px;
            }}
        """)
        self.title_label.setStyleSheet(self._label_style(theme.text_primary))
        self.zoom_label.setStyleSheet(self._label_style(theme.text_muted))
        for button in (
            self.zoom_out_button,
            self.fit_button,
            self.zoom_in_button,
            self.close_button,
        ):
            button.setStyleSheet(f"""
                QPushButton {{
                    color: {theme.text_primary};
                    background-color: {theme.bg_panel_alt};
                    border: 1px solid {theme.border};
                    border-radius: 5px;
                    padding: 5px 10px;
                }}
                QPushButton:hover {{
                    border-color: {theme.accent_cyan};
                }}
                QPushButton:pressed {{
                    background-color: {theme.bg_panel};
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
