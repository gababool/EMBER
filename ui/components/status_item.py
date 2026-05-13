from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout

try:
    from ui.assets.design import DEFAULT_THEME, FONT_SIZE_LG, Theme, app_font
    from ui.components.icon_utils import load_svg_icon
except ImportError:
    from assets.design import DEFAULT_THEME, FONT_SIZE_LG, Theme, app_font
    from components.icon_utils import load_svg_icon


class StatusItem(QWidget):
    def __init__(
        self,
        icon_text: str,
        text: str,
        icon_color: str,
        icon_name: str | None = None,
        text_color: str | None = None,
        parent=None
    ):
        super().__init__(parent)
        self.icon_color = icon_color
        self.icon_name = icon_name
        self.text_color = text_color
        self.theme = DEFAULT_THEME

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.icon_label = QLabel(icon_text)
        self.icon_label.setFixedSize(22, 22)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFont(app_font(FONT_SIZE_LG, bold=True))

        self.text_label = QLabel(text)
        self.text_label.setFont(app_font(FONT_SIZE_LG, bold=True))
        self.text_label.setStyleSheet(
            self._label_style(self.text_color or self.theme.text_primary)
        )

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)
        self._refresh_icon()

    def setText(self, text: str):
        self.text_label.setText(text)

    def setIconText(self, icon_text: str):
        self.icon_label.setText(icon_text)

    def setIconColor(self, color: str):
        self.icon_color = color
        self._refresh_icon()

    def setTextColor(self, color: str):
        self.text_color = color
        self.text_label.setStyleSheet(self._label_style(color))

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self._refresh_icon()
        self.text_label.setStyleSheet(
            self._label_style(self.text_color or self.theme.text_primary)
        )

    def _refresh_icon(self):
        if self.icon_name:
            self.icon_label.setPixmap(load_svg_icon(self.icon_name, self.icon_color, 20))
            self.icon_label.setText("")
            self.icon_label.setStyleSheet("background: transparent; border: none;")
            return
        self.icon_label.setPixmap(QPixmap())
        self.icon_label.setStyleSheet(self._label_style(self.icon_color))

    @staticmethod
    def _label_style(color: str) -> str:
        return f"""
            QLabel {{
                color: {color};
                background: transparent;
                border: none;
            }}
        """
