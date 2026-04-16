from PySide6.QtWidgets import (
    QWidget,
    QFrame,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
)

from components.status_item import StatusItem
from assets.design import (
    BG_TOPBAR,
    TEXT_MUTED,
    ACCENT_ORANGE,
    ACCENT_BLUE,
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_PURPLE,
    FONT_SIZE_XS,
    FONT_SIZE_XL,
    FONT_SIZE_LG,
    app_font,
)


class TopBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(72)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        main_row = QFrame()
        main_row.setFixedHeight(68)
        main_row.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_TOPBAR};
                border: none;
            }}
        """)

        row_layout = QHBoxLayout(main_row)
        row_layout.setContentsMargins(20, 8, 20, 8)
        row_layout.setSpacing(16)

        # Left side: logo + version
        logo_widget = QWidget()
        logo_layout = QVBoxLayout(logo_widget)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.setSpacing(0)

        self.logo_label = QLabel("EMBER")
        self.logo_label.setFont(app_font(FONT_SIZE_XL, bold=True))
        self.logo_label.setStyleSheet(f"""
            QLabel {{
                color: {ACCENT_ORANGE};
                background: transparent;
                border: none;
            }}
        """)

        self.version_label = QLabel("V.0.3.1")
        self.version_label.setFont(app_font(FONT_SIZE_XS))
        self.version_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_MUTED};
                background: transparent;
                border: none;
            }}
        """)

        logo_layout.addWidget(self.logo_label)
        logo_layout.addWidget(self.version_label)

        row_layout.addWidget(logo_widget)
        row_layout.addStretch()

        # Right side: status area
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(18)

        self.wind_item = StatusItem("⇄", "5m/s nw", icon_color=ACCENT_CYAN)
        self.gpu_item = StatusItem("▣", "GPU 43%", icon_color=ACCENT_PURPLE, text_color=ACCENT_GREEN)

        self.mode_label = QLabel("OFFLINE MODE")
        self.mode_label.setFont(app_font(FONT_SIZE_LG, bold=True))
        self.mode_label.setStyleSheet(f"""
            QLabel {{
                color: {ACCENT_BLUE};
                background: transparent;
                border: none;
            }}
        """)

        self.signal_label = QLabel("⌁")
        self.signal_label.setFont(app_font(FONT_SIZE_LG, bold=True))
        self.signal_label.setStyleSheet(f"""
            QLabel {{
                color: {ACCENT_ORANGE};
                background: transparent;
                border: none;
            }}
        """)

        status_layout.addWidget(self.wind_item)
        status_layout.addWidget(self.gpu_item)
        status_layout.addWidget(self.mode_label)
        status_layout.addWidget(self.signal_label)

        row_layout.addWidget(status_widget)

        # Bottom accent line
        accent_line = QFrame()
        accent_line.setFixedHeight(4)
        accent_line.setStyleSheet(f"""
            QFrame {{
                background-color: {ACCENT_ORANGE};
                border: none;
            }}
        """)

        outer_layout.addWidget(main_row)
        outer_layout.addWidget(accent_line)

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