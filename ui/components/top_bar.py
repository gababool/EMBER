from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from ui.assets.design import (
        DEFAULT_THEME,
        FONT_SIZE_LG,
        FONT_SIZE_SM,
        FONT_SIZE_XL,
        FONT_SIZE_XS,
        Theme,
        app_font,
    )
    from ui.components.status_item import StatusItem
except ImportError:
    from assets.design import (
        DEFAULT_THEME,
        FONT_SIZE_LG,
        FONT_SIZE_SM,
        FONT_SIZE_XL,
        FONT_SIZE_XS,
        Theme,
        app_font,
    )
    from components.status_item import StatusItem


class TopBar(QWidget):
    def __init__(self, theme: Theme = DEFAULT_THEME, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setFixedHeight(72)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.main_row = QFrame()
        self.main_row.setFixedHeight(68)

        row_layout = QHBoxLayout(self.main_row)
        row_layout.setContentsMargins(20, 8, 20, 8)
        row_layout.setSpacing(16)

        logo_widget = QWidget()
        logo_layout = QVBoxLayout(logo_widget)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.setSpacing(0)

        self.logo_label = QLabel("EMBER")
        self.logo_label.setFont(app_font(FONT_SIZE_XL, bold=True))

        self.version_label = QLabel("V.0.3.1")
        self.version_label.setFont(app_font(FONT_SIZE_XS))

        logo_layout.addWidget(self.logo_label)
        logo_layout.addWidget(self.version_label)

        row_layout.addWidget(logo_widget)
        row_layout.addStretch()

        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(18)

        self.wind_item = StatusItem("⇄", "5m/s nw", icon_color=theme.accent_cyan)
        self.gpu_item = StatusItem(
            "▣",
            "GPU 43%",
            icon_color=theme.accent_purple,
            text_color=theme.accent_green,
        )

        self.mode_label = QLabel("OFFLINE MODE")
        self.mode_label.setFont(app_font(FONT_SIZE_LG, bold=True))

        self.theme_button = QPushButton()
        self.theme_button.setFont(app_font(FONT_SIZE_SM, bold=True))
        self.theme_button.setCursor(Qt.PointingHandCursor)
        self.theme_button.setFixedHeight(32)

        self.signal_label = QLabel("⌁")
        self.signal_label.setFont(app_font(FONT_SIZE_LG, bold=True))

        status_layout.addWidget(self.wind_item)
        status_layout.addWidget(self.gpu_item)
        status_layout.addWidget(self.mode_label)
        status_layout.addWidget(self.theme_button)
        status_layout.addWidget(self.signal_label)

        row_layout.addWidget(status_widget)

        self.accent_line = QFrame()
        self.accent_line.setFixedHeight(4)

        outer_layout.addWidget(self.main_row)
        outer_layout.addWidget(self.accent_line)

        self.apply_theme(theme)

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.main_row.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.bg_topbar};
                border: none;
            }}
        """)
        self.logo_label.setStyleSheet(self._label_style(theme.accent_orange))
        self.version_label.setStyleSheet(self._label_style(theme.text_muted))
        self.wind_item.setIconColor(theme.accent_cyan)
        self.wind_item.apply_theme(theme)
        self.gpu_item.setIconColor(theme.accent_purple)
        self.gpu_item.setTextColor(theme.accent_green)
        self.gpu_item.apply_theme(theme)
        self.mode_label.setStyleSheet(self._label_style(theme.accent_blue))
        self.signal_label.setStyleSheet(self._label_style(theme.accent_orange))
        self.theme_button.setText("LIGHT" if theme.name == "dark" else "DARK")
        self.theme_button.setStyleSheet(f"""
            QPushButton {{
                color: {theme.text_primary};
                background-color: {theme.bg_panel_alt};
                border: 1px solid {theme.border};
                border-radius: 4px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                border-color: {theme.accent_orange};
            }}
            QPushButton:pressed {{
                background-color: {theme.bg_panel};
            }}
        """)
        self.accent_line.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.accent_orange};
                border: none;
            }}
        """)

    def set_version(self, version: str):
        self.version_label.setText(version)

    def set_logo_text(self, text: str):
        self.logo_label.setText(text)

    def set_wind(self, text: str):
        self.wind_item.setText(text)

    def set_gpu(self, text: str):
        self.gpu_item.setText(text)

    def set_mode(self, text: str):
        self.mode_label.setText(text)

    @staticmethod
    def _label_style(color: str) -> str:
        return f"""
            QLabel {{
                color: {color};
                background: transparent;
                border: none;
            }}
        """
