from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

try:
    from ui.assets.design import DEFAULT_THEME, FONT_SIZE_SM, FONT_SIZE_XS, Theme, app_font
    from ui.components.icon_utils import load_svg_icon
except ImportError:
    from assets.design import DEFAULT_THEME, FONT_SIZE_SM, FONT_SIZE_XS, Theme, app_font
    from components.icon_utils import load_svg_icon


class StatMeter(QWidget):
    def __init__(
        self,
        label: str,
        icon_name: str,
        accent_color: str,
        theme: Theme = DEFAULT_THEME,
        parent=None,
    ):
        super().__init__(parent)
        self.label = label
        self.icon_name = icon_name
        self.accent_color = accent_color
        self.theme = theme
        self.value = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.icon_label = QLabel()
        self.icon_label.setFixedSize(18, 18)
        self.icon_label.setAlignment(Qt.AlignCenter)

        text_block = QWidget()
        text_layout = QVBoxLayout(text_block)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.name_label = QLabel(label)
        self.name_label.setFont(app_font(FONT_SIZE_XS, bold=True))

        self.value_label = QLabel("--%")
        self.value_label.setFont(app_font(FONT_SIZE_SM, bold=True))
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setMinimumWidth(38)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedSize(92, 6)

        row.addWidget(self.name_label)
        row.addWidget(self.value_label)

        text_layout.addLayout(row)
        text_layout.addWidget(self.bar)

        layout.addWidget(self.icon_label)
        layout.addWidget(text_block)

        self.apply_theme(theme)

    def set_value(self, value: float | int | None):
        self.value = value
        if value is None:
            self.value_label.setText("N/A")
            self.bar.setValue(0)
            return

        bounded = max(0, min(100, int(round(value))))
        self.value_label.setText(f"{bounded}%")
        self.bar.setValue(bounded)

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.icon_label.setPixmap(load_svg_icon(self.icon_name, self.accent_color, 18))
        self.name_label.setStyleSheet(self._label_style(theme.text_muted))
        self.value_label.setStyleSheet(self._label_style(theme.text_primary))
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {theme.bg_panel_alt};
                border: 1px solid {theme.border};
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {self.accent_color};
                border-radius: 2px;
            }}
        """)
        self.set_value(self.value)

    @staticmethod
    def _label_style(color: str) -> str:
        return f"""
            QLabel {{
                color: {color};
                background: transparent;
                border: none;
            }}
        """
