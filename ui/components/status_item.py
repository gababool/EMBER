from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout

try:
    from ui.assets.design import DEFAULT_THEME, FONT_SIZE_LG, Theme, app_font
except ImportError:
    from assets.design import DEFAULT_THEME, FONT_SIZE_LG, Theme, app_font


class StatusItem(QWidget):
    def __init__(
        self,
        icon_text: str,
        text: str,
        icon_color: str,
        text_color: str | None = None,
        parent=None
    ):
        super().__init__(parent)
        self.icon_color = icon_color
        self.text_color = text_color
        self.theme = DEFAULT_THEME

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.icon_label = QLabel(icon_text)
        self.icon_label.setFont(app_font(FONT_SIZE_LG, bold=True))
        self.icon_label.setStyleSheet(self._label_style(self.icon_color))

        self.text_label = QLabel(text)
        self.text_label.setFont(app_font(FONT_SIZE_LG, bold=True))
        self.text_label.setStyleSheet(
            self._label_style(self.text_color or self.theme.text_primary)
        )

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)

    def setText(self, text: str):
        self.text_label.setText(text)

    def setIconText(self, icon_text: str):
        self.icon_label.setText(icon_text)

    def setIconColor(self, color: str):
        self.icon_color = color
        self.icon_label.setStyleSheet(self._label_style(color))

    def setTextColor(self, color: str):
        self.text_color = color
        self.text_label.setStyleSheet(self._label_style(color))

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.icon_label.setStyleSheet(self._label_style(self.icon_color))
        self.text_label.setStyleSheet(
            self._label_style(self.text_color or self.theme.text_primary)
        )

    @staticmethod
    def _label_style(color: str) -> str:
        return f"""
            QLabel {{
                color: {color};
                background: transparent;
                border: none;
            }}
        """
