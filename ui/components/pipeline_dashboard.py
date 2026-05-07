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
        Theme,
        app_font,
    )
    from ui.components.classification_badge import ClassificationBadge, CorrectnessBadge
    from ui.components.history_strip import HistoryStrip
    from ui.components.model_combo_box import ModelComboBox
    from ui.components.result_panel import ResultPanel
    from pipeline.llm.inference import BENCHMARK_MODEL_TAGS, PROMPTS_DIR
    from ui.pipeline_worker import DEMO_DATASET_DIR
except ImportError:
    from assets.design import (
        DEFAULT_THEME,
        FONT_SIZE_LG,
        FONT_SIZE_MD,
        FONT_SIZE_SM,
        FONT_SIZE_XL,
        Theme,
        app_font,
    )
    from components.classification_badge import ClassificationBadge, CorrectnessBadge
    from components.history_strip import HistoryStrip
    from components.model_combo_box import ModelComboBox
    from components.result_panel import ResultPanel
    from pipeline.llm.inference import BENCHMARK_MODEL_TAGS, PROMPTS_DIR
    from pipeline_worker import DEMO_DATASET_DIR


HISTORY_IMAGE_DIR = Path(tempfile.gettempdir()) / "ember_ui_history"
CONTEXTS_DIR = Path(__file__).resolve().parents[2] / "data" / "contexts"
CONTEXT_PROMPTS = {"c2v2prompt", "c2v3prompt", "c2v4prompt"}


class PipelineDashboard(QWidget):
    image_picked = Signal(str)

    def __init__(self, theme: Theme = DEFAULT_THEME, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.expected_label = None
        self.current_run = None
        self.run_history = []
        self.last_image_path = None
        self.setObjectName("PipelineDashboard")
        self.setAutoFillBackground(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(16)

        title_block = QWidget()
        title_layout = QVBoxLayout(title_block)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)

        self.title_label = QLabel("Pipeline Demo")
        self.title_label.setFont(app_font(FONT_SIZE_XL, bold=True))

        self.subtitle_label = QLabel("Run a wildfire image through screening, detection, and reasoning.")
        self.subtitle_label.setFont(app_font(FONT_SIZE_SM))

        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.subtitle_label)

        self.rerun_button = QPushButton("Re-run")
        self.rerun_button.setFont(app_font(FONT_SIZE_MD, bold=True))
        self.rerun_button.setCursor(Qt.PointingHandCursor)
        self.rerun_button.setFixedHeight(40)
        self.rerun_button.setEnabled(False)

        self.start_button = QPushButton("Test Image")
        self.start_button.setFont(app_font(FONT_SIZE_MD, bold=True))
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.setFixedHeight(40)

        self.model_select = ModelComboBox(theme)
        for model_tag in BENCHMARK_MODEL_TAGS:
            self.model_select.addItem(model_tag, model_tag)

        self.prompt_select = ModelComboBox(theme)
        self.prompt_select.setMinimumWidth(140)
        for prompt_path in sorted(PROMPTS_DIR.glob("*.txt")):
            self.prompt_select.addItem(prompt_path.stem, str(prompt_path))
        self.prompt_select.currentIndexChanged.connect(self._on_prompt_changed)

        self.context_select = ModelComboBox(theme)
        self.context_select.setMinimumWidth(160)
        for ctx_path in sorted(CONTEXTS_DIR.glob("*.json")):
            self.context_select.addItem(ctx_path.stem, str(ctx_path))
        self.context_select.setVisible(False)

        self.image_filter_select = ModelComboBox(theme)
        self.image_filter_select.setMinimumWidth(110)
        for label, data in [("Any", ""), ("Fire", "fire"), ("No Fire", "nofire")]:
            self.image_filter_select.addItem(label, data)

        self.pick_button = QPushButton("Pick Image")
        self.pick_button.setFont(app_font(FONT_SIZE_MD, bold=True))
        self.pick_button.setCursor(Qt.PointingHandCursor)
        self.pick_button.setFixedHeight(40)
        self.pick_button.clicked.connect(self._open_image_picker)

        self.annotation_button = QPushButton("Annotation: ON")
        self.annotation_button.setFont(app_font(FONT_SIZE_MD, bold=True))
        self.annotation_button.setCursor(Qt.PointingHandCursor)
        self.annotation_button.setFixedHeight(40)
        self.annotation_button.setCheckable(True)
        self.annotation_button.toggled.connect(self._on_annotation_toggled)

        self.skip_quality_button = QPushButton("Quality: ON")
        self.skip_quality_button.setFont(app_font(FONT_SIZE_MD, bold=True))
        self.skip_quality_button.setCursor(Qt.PointingHandCursor)
        self.skip_quality_button.setFixedHeight(40)
        self.skip_quality_button.setCheckable(True)
        self.skip_quality_button.toggled.connect(self._on_skip_quality_toggled)

        header_layout.addWidget(title_block, 1)
        header_layout.addWidget(self.image_filter_select)
        header_layout.addWidget(self.pick_button)
        header_layout.addWidget(self.skip_quality_button)
        header_layout.addWidget(self.annotation_button)
        header_layout.addWidget(self.context_select)
        header_layout.addWidget(self.prompt_select)
        header_layout.addWidget(self.model_select)
        header_layout.addWidget(self.rerun_button)
        header_layout.addWidget(self.start_button)

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

        self.image_placeholder = QLabel("Processed image will appear here")
        self.image_placeholder.setAlignment(Qt.AlignCenter)
        self.image_placeholder.setMinimumSize(360, 220)

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
            self.image_placeholder,
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

    def reset_demo(self):
        self.subtitle_label.setText("Selecting a random wildfire dataset image...")
        self.start_button.setEnabled(False)
        self.rerun_button.setEnabled(False)
        self.pick_button.setEnabled(False)
        self.model_select.setEnabled(False)
        self.prompt_select.setEnabled(False)
        self.annotation_button.setEnabled(False)
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
        }

    def _reset_stage_views(self):
        skip_quality = self.skip_quality_button.isChecked()
        self.quality_panel.setVisible(not skip_quality)
        for row in self.quality_rows:
            row.set_state("pending")
        self.image_placeholder.setPixmap(QPixmap())
        self.image_placeholder.setText("Processed image will appear here")
        self.classification_badge.set_classification("pending")
        self.correctness_badge.reset()
        self.reasoning_text.setPlainText("Waiting for LLM reasoning...")

    def set_demo_finished(self):
        self.start_button.setEnabled(True)
        self.rerun_button.setEnabled(self.last_image_path is not None)
        self.pick_button.setEnabled(True)
        self.model_select.setEnabled(True)
        self.prompt_select.setEnabled(True)
        self.annotation_button.setEnabled(True)
        self.context_select.setEnabled(True)
        self.image_filter_select.setEnabled(True)
        self.skip_quality_button.setEnabled(True)
        self.start_button.setText("Test Image")
        self.quality_panel.setVisible(True)
        self._save_current_run_to_history()

    def selected_llm_model(self) -> str:
        return self.model_select.currentData()

    def selected_prompt_file(self) -> str:
        return self.prompt_select.currentData()

    def selected_context_file(self) -> str | None:
        if not self.context_select.isVisible():
            return None
        return self.context_select.currentData()

    def use_annotation(self) -> bool:
        return not self.annotation_button.isChecked()

    def _on_annotation_toggled(self, checked: bool):
        self.annotation_button.setText("Annotation: OFF" if checked else "Annotation: ON")
        self._style_annotation_button(self.theme)

    def _on_prompt_changed(self, _index: int):
        stem = Path(self.prompt_select.currentData() or "").stem
        self.context_select.setVisible(stem in CONTEXT_PROMPTS)

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
            self.image_placeholder.setPixmap(QPixmap())
            self.image_placeholder.setText(result.get("error", "Object detection failed"))
            return

        annotated_path = result.get("annotated_image_path")
        if not annotated_path:
            self.image_placeholder.setText("No annotated image returned")
            return

        pixmap = QPixmap(annotated_path)
        if pixmap.isNull():
            self.image_placeholder.setText(f"Could not load image:\n{annotated_path}")
            return

        self.image_placeholder.setText("")
        self.image_placeholder.setPixmap(
            pixmap.scaled(
                self.image_placeholder.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

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
        self.model_select.apply_theme(theme)
        self.prompt_select.apply_theme(theme)
        self.context_select.apply_theme(theme)
        self.image_filter_select.apply_theme(theme)
        self._style_annotation_button(theme)
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
                background-color: {theme.bg_panel_alt};
                border: 1px dashed {theme.border};
                border-radius: 4px;
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

    def _set_stage_running(self, stage_name: str):
        if stage_name == "quality_screening":
            for row in self.quality_rows:
                row.set_state("running")
        elif stage_name == "object_detection":
            self.image_placeholder.setPixmap(QPixmap())
            self.image_placeholder.setText("Running object detection...")
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
        self.image_placeholder.setPixmap(QPixmap())
        self.image_placeholder.setText("Processed image will appear here")
        self.classification_badge.set_classification("pending")
        self.correctness_badge.reset()
        if expected_label:
            self.correctness_badge.set_expected_label(expected_label)
        self.reasoning_text.setPlainText("No LLM reasoning stored for this run.")

    def _load_image_into_processed_box(self, image_path: str):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_placeholder.setText(f"Could not load image:\n{image_path}")
            return
        self.image_placeholder.setText("")
        self.image_placeholder.setPixmap(
            pixmap.scaled(
                self.image_placeholder.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

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

    def _style_annotation_button(self, theme: Theme):
        checked = self.annotation_button.isChecked()
        border_color = theme.danger if checked else theme.border
        text_color = theme.danger if checked else theme.text_primary
        self.annotation_button.setStyleSheet(f"""
            QPushButton {{
                color: {text_color};
                background-color: {theme.bg_panel_alt};
                border: 1px solid {border_color};
                border-radius: 5px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{ border-color: {theme.danger}; color: {theme.danger}; }}
            QPushButton:disabled {{ color: {theme.text_muted}; border-color: {theme.border}; }}
        """)

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
