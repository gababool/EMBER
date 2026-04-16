from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout


class StatusItem(QWidget):
    def __init__(self, icon_text, text, icon_color="#00d7ff", text_color="#cfd3d8", parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.icon_label = QLabel(icon_text)
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                color: {icon_color};
                font-size: 15px;
                font-weight: 700;
                background: transparent;
            }}
        """)

        self.text_label = QLabel(text)
        self.text_label.setStyleSheet(f"""
            QLabel {{
                color: {text_color};
                font-size: 14px;
                font-weight: 600;
                background: transparent;
            }}
        """)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)

    def setText(self, text):
        self.text_label.setText(text)

    def setIconText(self, icon_text):
        self.icon_label.setText(icon_text)