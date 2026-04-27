from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel

try:
    from ui.assets.design import DEFAULT_THEME, FONT_SIZE_LG, FONT_SIZE_SM, Theme, app_font
except ImportError:
    from assets.design import DEFAULT_THEME, FONT_SIZE_LG, FONT_SIZE_SM, Theme, app_font


ICONS_DIR = Path(__file__).resolve().parents[1] / "assets" / "icons"


class ClassificationBadge(QFrame):
    def __init__(self, theme: Theme = DEFAULT_THEME, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.classification = "pending"
        self.setFixedHeight(58)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(12)

        self.icon_label = QLabel()
        self.icon_label.setFixedSize(28, 28)
        self.icon_label.setAlignment(Qt.AlignCenter)

        text_layout = QHBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(10)

        self.title_label = QLabel("Awaiting Result")
        self.title_label.setFont(app_font(FONT_SIZE_LG, bold=True))

        self.detail_label = QLabel("LLM classification")
        self.detail_label.setFont(app_font(FONT_SIZE_SM))

        text_layout.addWidget(self.title_label)
        text_layout.addStretch()
        text_layout.addWidget(self.detail_label)

        layout.addWidget(self.icon_label)
        layout.addLayout(text_layout, 1)

        self.apply_theme(theme)

    def set_classification(self, classification: str):
        self.classification = classification
        self._render()

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self._render()

    def _render(self):
        color, title, detail, icon_name = self._classification_style()
        self.setStyleSheet(f"""
            ClassificationBadge {{
                background-color: {self.theme.bg_panel_alt};
                border: 1px solid {color};
                border-radius: 5px;
            }}
        """)
        self.title_label.setText(title)
        self.title_label.setStyleSheet(self._label_style(color))
        self.detail_label.setText(detail)
        self.detail_label.setStyleSheet(self._label_style(self.theme.text_muted))

        if icon_name:
            self.icon_label.setPixmap(self._load_icon(icon_name, color, 28))
        else:
            self.icon_label.setText("○")
            self.icon_label.setStyleSheet(self._label_style(self.theme.text_muted))

    def _classification_style(self):
        if self.classification == "fire_detected":
            return (
                self.theme.accent_orange,
                "Fire",
                "fire_detected",
                "flame.svg",
            )
        if self.classification == "no_fire_detected":
            return (
                self.theme.accent_green,
                "No Fire",
                "no_fire_detected",
                "shield-check.svg",
            )
        if self.classification == "uncertain":
            return (
                self.theme.accent_blue,
                "Uncertain",
                "uncertain",
                "triangle-alert.svg",
            )
        return (
            self.theme.text_muted,
            "Awaiting Result",
            "LLM classification",
            None,
        )

    @staticmethod
    def _load_icon(icon_name: str, color: str, size: int) -> QPixmap:
        svg = (ICONS_DIR / icon_name).read_text(encoding="utf-8")
        svg = svg.replace("currentColor", color)
        renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        return pixmap

    @staticmethod
    def _label_style(color: str) -> str:
        return f"""
            QLabel {{
                color: {color};
                background: transparent;
                border: none;
            }}
        """
