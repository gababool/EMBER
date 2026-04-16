from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout

from assets.design import (
    TEXT_PRIMARY,
    FONT_SIZE_LG,
    app_font,
)


class StatusItem(QWidget):
    def __init__(
        self,
        icon_text: str,
        text: str,
        icon_color: str,
        text_color: str = TEXT_PRIMARY,
        parent=None
    ):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.icon_label = QLabel(icon_text)
        self.icon_label.setFont(app_font(FONT_SIZE_LG, bold=True))
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                color: {icon_color};
                background: transparent;
                border: none;
            }}
        """)

        self.text_label = QLabel(text)
        self.text_label.setFont(app_font(FONT_SIZE_LG, bold=True))
        self.text_label.setStyleSheet(f"""
            QLabel {{
                color: {text_color};
                background: transparent;
                border: none;
            }}
        """)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)

    def setText(self, text: str):
        self.text_label.setText(text)

    def setIconText(self, icon_text: str):
        self.icon_label.setText(icon_text)

    def setIconColor(self, color: str):
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                background: transparent;
                border: none;
            }}
        """)

    def setTextColor(self, color: str):
        self.text_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                background: transparent;
                border: none;
            }}
        """)