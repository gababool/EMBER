from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout
)

from components.status_item import StatusItem


class TopBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(72)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        main_row = QFrame()
        main_row.setStyleSheet("""
            QFrame {
                background-color: #1b1d21;
                border: none;
            }
        """)
        main_row.setFixedHeight(68)

        row_layout = QHBoxLayout(main_row)
        row_layout.setContentsMargins(20, 8, 20, 8)
        row_layout.setSpacing(16)

        # Left logo area
        logo_widget = QWidget()
        logo_layout = QVBoxLayout(logo_widget)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.setSpacing(0)

        self.logo_label = QLabel("EMBER")
        logo_font = QFont("Arial", 20)
        logo_font.setBold(True)
        self.logo_label.setFont(logo_font)
        self.logo_label.setStyleSheet("""
            QLabel {
                color: #ff8c2b;
                letter-spacing: 2px;
                background: transparent;
            }
        """)

        self.version_label = QLabel("V.0.3.1")
        self.version_label.setStyleSheet("""
            QLabel {
                color: #7a7f87;
                font-size: 10px;
                background: transparent;
            }
        """)

        logo_layout.addWidget(self.logo_label)
        logo_layout.addWidget(self.version_label)

        row_layout.addWidget(logo_widget)
        row_layout.addStretch()

        # Right status area
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(18)

        self.wind_item = StatusItem("⇄", "5m/s nw", icon_color="#00d7ff")
        self.gpu_item = StatusItem("▣", "GPU 43%", icon_color="#d34dff", text_color="#49d86f")

        self.offline_label = QLabel("OFFLINE MODE")
        self.offline_label.setStyleSheet("""
            QLabel {
                color: #00a6ff;
                font-size: 14px;
                font-weight: 800;
                background: transparent;
            }
        """)

        self.signal_label = QLabel("⌁")
        self.signal_label.setStyleSheet("""
            QLabel {
                color: #ff8c2b;
                font-size: 18px;
                font-weight: 700;
                background: transparent;
            }
        """)

        status_layout.addWidget(self.wind_item)
        status_layout.addWidget(self.gpu_item)
        status_layout.addWidget(self.offline_label)
        status_layout.addWidget(self.signal_label)

        row_layout.addWidget(status_widget)

        # Bottom accent line
        accent_line = QFrame()
        accent_line.setFixedHeight(4)
        accent_line.setStyleSheet("background-color: #ff8c2b; border: none;")

        outer_layout.addWidget(main_row)
        outer_layout.addWidget(accent_line)

    def set_version(self, version_text: str):
        self.version_label.setText(version_text)

    def set_wind(self, text: str):
        self.wind_item.setText(text)

    def set_gpu(self, text: str):
        self.gpu_item.setText(text)

    def set_mode(self, text: str):
        self.offline_label.setText(text)