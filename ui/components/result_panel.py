from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

try:
    from ui.assets.design import (
        DEFAULT_THEME,
        FONT_SIZE_LG,
        FONT_SIZE_SM,
        Theme,
        app_font,
    )
except ImportError:
    from assets.design import (
        DEFAULT_THEME,
        FONT_SIZE_LG,
        FONT_SIZE_SM,
        Theme,
        app_font,
    )


class ResultPanel(QFrame):
    def __init__(
        self,
        title: str,
        subtitle: str,
        body: QWidget,
        theme: Theme = DEFAULT_THEME,
        parent=None,
    ):
        super().__init__(parent)
        self.theme = theme
        self.setMinimumHeight(260)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(18, 16, 18, 18)
        self.layout.setSpacing(12)

        self.title_label = QLabel(title)
        self.title_label.setFont(app_font(FONT_SIZE_LG, bold=True))

        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setFont(app_font(FONT_SIZE_SM))
        self.subtitle_label.setWordWrap(True)

        self.body = body

        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.subtitle_label)
        self.layout.addWidget(self.body, 1)

        self.apply_theme(theme)

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.setStyleSheet(f"""
            ResultPanel {{
                background-color: {theme.bg_panel};
                border: 1px solid {theme.border};
                border-radius: 6px;
            }}
        """)
        self.title_label.setStyleSheet(f"""
            QLabel {{
                color: {theme.text_primary};
                background: transparent;
                border: none;
            }}
        """)
        self.subtitle_label.setStyleSheet(f"""
            QLabel {{
                color: {theme.text_muted};
                background: transparent;
                border: none;
            }}
        """)
