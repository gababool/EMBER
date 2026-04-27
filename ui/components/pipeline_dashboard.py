from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
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
    from ui.components.result_panel import ResultPanel
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
    from components.result_panel import ResultPanel


class PipelineDashboard(QWidget):
    def __init__(self, theme: Theme = DEFAULT_THEME, parent=None):
        super().__init__(parent)
        self.theme = theme
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

        self.start_button = QPushButton("Start Demo")
        self.start_button.setFont(app_font(FONT_SIZE_MD, bold=True))
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.setFixedHeight(40)

        header_layout.addWidget(title_block, 1)
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
            self.reasoning_text,
            theme,
        )

        self.grid.addWidget(self.quality_panel, 0, 0)
        self.grid.addWidget(self.image_panel, 0, 1)
        self.grid.addWidget(self.reasoning_panel, 0, 2)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 2)
        self.grid.setColumnStretch(2, 2)

        layout.addWidget(header)
        layout.addLayout(self.grid, 1)

        self.apply_theme(theme)

    def reset_demo(self):
        self.subtitle_label.setText("Selecting a random wildfire dataset image...")
        self.start_button.setEnabled(False)
        self.start_button.setText("Running")
        for row in self.quality_rows:
            row.set_state("pending")
        self.image_placeholder.setPixmap(QPixmap())
        self.image_placeholder.setText("Processed image will appear here")
        self.reasoning_text.setPlainText("Waiting for LLM reasoning...")

    def set_demo_finished(self):
        self.start_button.setEnabled(True)
        self.start_button.setText("Start Demo")

    def set_selected_image(self, image_path: str):
        path = Path(image_path)
        label = path.parent.name
        self.subtitle_label.setText(f"Running demo image: {label}/{path.name}")

    def set_pipeline_error(self, message: str):
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
        elif result.stage_name == "object_detection":
            self.set_detection_result(result.output)
        elif result.stage_name == "llm_reasoning":
            self.set_reasoning_result(result.output)

        if not result.passed and result.error:
            self.set_pipeline_error(result.error)

    def set_quality_result(self, result: dict):
        checks = result.get("checks", {})
        for check_name, row in self.quality_row_map.items():
            check_result = checks.get(check_name)
            if check_result is None:
                row.set_state("pending")
                continue
            row.set_state("passed" if check_result.get("passed") else "failed")

    def set_detection_result(self, result: dict):
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
        parsed = result.get("parsed", {})
        if not parsed:
            self.reasoning_text.setPlainText("No reasoning returned.")
            return

        classification = parsed.get("classification", "unknown")
        reasoning = parsed.get("reasoning", "")
        recommendation = parsed.get("recommendation", "")
        self.reasoning_text.setPlainText(
            f"Classification: {classification}\n\n"
            f"Reasoning:\n{reasoning}\n\n"
            f"Recommendation:\n{recommendation}"
        )

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

        for panel in (self.quality_panel, self.image_panel, self.reasoning_panel):
            panel.apply_theme(theme)

        for row in self.quality_rows:
            row.apply_theme(theme)

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
