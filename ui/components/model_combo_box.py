from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

try:
    from ui.assets.design import DEFAULT_THEME, FONT_SIZE_SM, Theme, app_font
    from ui.components.icon_utils import load_svg_icon
except ImportError:
    from assets.design import DEFAULT_THEME, FONT_SIZE_SM, Theme, app_font
    from components.icon_utils import load_svg_icon


class ModelComboBox(QFrame):
    currentIndexChanged = Signal(int)

    def __init__(self, theme: Theme = DEFAULT_THEME, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.items = []
        self.current_index = -1
        self.popup = None
        self.setFixedHeight(40)
        self.setMinimumWidth(170)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        self.text_label = QLabel()
        self.text_label.setFont(app_font(FONT_SIZE_SM, bold=True))

        self.arrow_label = QLabel()
        self.arrow_label.setFixedSize(16, 16)
        self.arrow_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.text_label, 1)
        layout.addWidget(self.arrow_label)

        self.apply_theme(theme)

    def addItem(self, text: str, user_data=None):
        self.items.append((text, user_data if user_data is not None else text))
        if self.current_index == -1:
            self.setCurrentIndex(0)

    def currentData(self):
        if self.current_index < 0:
            return None
        return self.items[self.current_index][1]

    def currentText(self):
        if self.current_index < 0:
            return ""
        return self.items[self.current_index][0]

    def setCurrentIndex(self, index: int):
        if index < 0 or index >= len(self.items):
            return
        self.current_index = index
        self.text_label.setText(self.items[index][0])
        self.currentIndexChanged.emit(index)

    def setCurrentText(self, text: str):
        for index, (item_text, _data) in enumerate(self.items):
            if item_text == text:
                self.setCurrentIndex(index)
                return

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.isEnabled():
            self.toggle_popup()
        super().mousePressEvent(event)

    def toggle_popup(self):
        if self.popup and self.popup.isVisible():
            self.popup.close()
            return
        self.show_popup()

    def show_popup(self):
        self.close_popup()
        self.popup = QFrame(None, Qt.Popup)
        self.popup.setObjectName("ModelPopup")
        self.popup.setFixedWidth(self.width())
        self.popup.destroyed.connect(lambda: setattr(self, "popup", None))

        layout = QVBoxLayout(self.popup)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        for index, (text, _data) in enumerate(self.items):
            button = QPushButton(text)
            button.setFont(app_font(FONT_SIZE_SM, bold=True))
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedHeight(40)
            button.clicked.connect(lambda checked=False, i=index: self._select_from_popup(i))
            button.setProperty("selected", index == self.current_index)
            layout.addWidget(button)

        self._style_popup()
        self.popup.move(self.mapToGlobal(QPoint(0, self.height() + 2)))
        self.popup.show()

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.setStyleSheet(f"""
            ModelComboBox {{
                color: {theme.text_primary};
                background-color: {theme.bg_panel_alt};
                border: 1px solid {theme.border};
                border-radius: 5px;
            }}
            ModelComboBox:hover {{
                border-color: {theme.accent_orange};
            }}
            ModelComboBox:disabled {{
                color: {theme.text_muted};
                border-color: {theme.border};
            }}
        """)
        self.text_label.setStyleSheet(self._label_style(theme.text_primary))
        self.arrow_label.setPixmap(load_svg_icon("chevron-down.svg", theme.text_muted, 16))
        if self.popup:
            self._style_popup()

    def _select_from_popup(self, index: int):
        self.setCurrentIndex(index)
        self.close_popup()

    def close_popup(self):
        if self.popup:
            popup = self.popup
            self.popup = None
            popup.close()

    def setEnabled(self, enabled: bool):
        if not enabled:
            self.close_popup()
        super().setEnabled(enabled)

    def hideEvent(self, event):
        self.close_popup()
        super().hideEvent(event)

    def moveEvent(self, event):
        self.close_popup()
        super().moveEvent(event)

    def resizeEvent(self, event):
        self.close_popup()
        super().resizeEvent(event)

    def focusOutEvent(self, event):
        self.close_popup()
        super().focusOutEvent(event)

    def _style_popup(self):
        if not self.popup:
            return

        self.popup.setStyleSheet(f"""
            QFrame#ModelPopup {{
                background-color: {self.theme.bg_panel};
                border: 1px solid {self.theme.border};
                border-radius: 5px;
            }}
            QPushButton {{
                color: {self.theme.text_primary};
                background-color: {self.theme.bg_panel};
                border: none;
                padding: 6px 10px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {self.theme.bg_panel_alt};
            }}
            QPushButton[selected="true"] {{
                color: {self.theme.text_primary};
                background-color: {self.theme.bg_panel_alt};
            }}
        """)

    @staticmethod
    def _label_style(color: str) -> str:
        return f"""
            QLabel {{
                color: {color};
                background: transparent;
                border: none;
            }}
        """
