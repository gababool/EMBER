from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

try:
    from ui.assets.design import DEFAULT_THEME, FONT_SIZE_SM, FONT_SIZE_XS, Theme, app_font
    from ui.components.icon_utils import load_svg_icon
except ImportError:
    from assets.design import DEFAULT_THEME, FONT_SIZE_SM, FONT_SIZE_XS, Theme, app_font
    from components.icon_utils import load_svg_icon


class HistoryStrip(QFrame):
    run_selected = Signal(int)

    def __init__(self, theme: Theme = DEFAULT_THEME, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.items = []
        self.active_index = None
        self.setFixedHeight(132)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        self.title_label = QLabel("Recent Processed Images")
        self.title_label.setFont(app_font(FONT_SIZE_SM, bold=True))

        self.count_label = QLabel("0")
        self.count_label.setFont(app_font(FONT_SIZE_XS, bold=True))
        self.count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(self.count_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        self.items_widget = QWidget()
        self.items_layout = QHBoxLayout(self.items_widget)
        self.items_layout.setContentsMargins(0, 0, 0, 0)
        self.items_layout.setSpacing(10)
        self.items_layout.addStretch()
        self.scroll_area.setWidget(self.items_widget)

        layout.addLayout(header)
        layout.addWidget(self.scroll_area, 1)

        self.apply_theme(theme)

    def add_run(self, run_record: dict):
        index = len(self.items)
        item = HistoryItem(index, run_record, self.theme)
        item.clicked.connect(lambda checked=False, i=index: self.select_run(i))
        self.items_layout.insertWidget(index, item)
        self.items.append(item)
        self.count_label.setText(str(len(self.items)))
        self.select_run(index)

    def select_run(self, index: int):
        if index < 0 or index >= len(self.items):
            return

        self.active_index = index
        for item_index, item in enumerate(self.items):
            item.set_selected(item_index == index)
        self.run_selected.emit(index)

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.setStyleSheet(f"""
            HistoryStrip {{
                background-color: {theme.bg_panel};
                border: 1px solid {theme.border};
                border-radius: 6px;
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:horizontal {{
                background: {theme.bg_panel_alt};
                height: 6px;
                border-radius: 3px;
            }}
            QScrollBar::handle:horizontal {{
                background: {theme.border};
                border-radius: 3px;
            }}
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {{
                width: 0;
            }}
        """)
        self.title_label.setStyleSheet(self._label_style(theme.text_primary))
        self.count_label.setStyleSheet(self._label_style(theme.text_muted))
        self.items_widget.setStyleSheet("background: transparent;")
        for item in self.items:
            item.apply_theme(theme)

    @staticmethod
    def _label_style(color: str) -> str:
        return f"""
            QLabel {{
                color: {color};
                background: transparent;
                border: none;
            }}
        """


class HistoryItem(QPushButton):
    def __init__(self, index: int, run_record: dict, theme: Theme = DEFAULT_THEME, parent=None):
        super().__init__(parent)
        self.index = index
        self.run_record = run_record
        self.theme = theme
        self.selected = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(142, 68)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        thumb_frame = QFrame()
        thumb_frame.setFixedSize(52, 52)
        thumb_layout = QVBoxLayout(thumb_frame)
        thumb_layout.setContentsMargins(0, 0, 0, 0)
        thumb_layout.setSpacing(0)

        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(52, 52)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)

        self.guess_icon_label = QLabel(self.thumbnail_label)
        self.guess_icon_label.setFixedSize(20, 20)
        self.guess_icon_label.move(4, 28)
        self.guess_icon_label.setAlignment(Qt.AlignCenter)

        self.correctness_icon_label = QLabel(self.thumbnail_label)
        self.correctness_icon_label.setFixedSize(20, 20)
        self.correctness_icon_label.move(28, 28)
        self.correctness_icon_label.setAlignment(Qt.AlignCenter)

        thumb_layout.addWidget(self.thumbnail_label)

        text_block = QWidget()
        text_layout = QVBoxLayout(text_block)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        expected = run_record.get("expected_label", "unknown")
        self.title_label = QLabel(f"{index + 1}. {expected}")
        self.title_label.setFont(app_font(FONT_SIZE_SM, bold=True))

        status = run_record.get("classification", "pending").replace("_", " ")
        self.subtitle_label = QLabel(status)
        self.subtitle_label.setFont(app_font(FONT_SIZE_XS))
        self.subtitle_label.setWordWrap(True)

        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.subtitle_label)
        text_layout.addStretch()

        layout.addWidget(thumb_frame)
        layout.addWidget(text_block, 1)

        self._set_thumbnail()
        self.apply_theme(theme)

    def set_selected(self, selected: bool):
        self.selected = selected
        self.apply_theme(self.theme)

    def apply_theme(self, theme: Theme):
        self.theme = theme
        border = theme.accent_orange if self.selected else theme.border
        self.setStyleSheet(f"""
            HistoryItem {{
                background-color: {theme.bg_panel_alt};
                border: 1px solid {border};
                border-radius: 5px;
                text-align: left;
            }}
            HistoryItem:hover {{
                border-color: {theme.accent_blue};
            }}
        """)
        self.title_label.setStyleSheet(self._label_style(theme.text_primary))
        self.subtitle_label.setStyleSheet(self._label_style(theme.text_muted))
        self.thumbnail_label.setStyleSheet(f"""
            QLabel {{
                background-color: {theme.bg_panel};
                border: 1px solid {theme.border};
                border-radius: 4px;
                color: {theme.text_muted};
            }}
        """)
        badge_style = f"""
            QLabel {{
                background-color: {theme.bg_panel};
                border: 1px solid {theme.border};
                border-radius: 10px;
            }}
        """
        self.guess_icon_label.setStyleSheet(badge_style)
        self.correctness_icon_label.setStyleSheet(badge_style)
        self._set_guess_icon()
        self._set_correctness_icon()

    def _set_thumbnail(self):
        image_path = self.run_record.get("history_image_path") or self.run_record.get("image_path")
        pixmap = QPixmap(str(image_path)) if image_path else QPixmap()
        if pixmap.isNull():
            self.thumbnail_label.setText("No img")
            return

        self.thumbnail_label.setText("")
        self.thumbnail_label.setPixmap(
            pixmap.scaled(
                self.thumbnail_label.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
        )
        self._set_guess_icon()
        self._set_correctness_icon()

    def _set_guess_icon(self):
        classification = self.run_record.get("classification")
        if classification == "fire_detected":
            self.guess_icon_label.setPixmap(load_svg_icon("flame.svg", self.theme.accent_orange, 14))
        elif classification == "no_fire_detected":
            self.guess_icon_label.setPixmap(load_svg_icon("shield-check.svg", self.theme.accent_green, 14))
        else:
            self.guess_icon_label.setPixmap(QPixmap())

    def _set_correctness_icon(self):
        correctness = self._correctness()
        if correctness == "correct":
            self.correctness_icon_label.setPixmap(
                load_svg_icon("circle-check.svg", self.theme.accent_green, 14)
            )
        elif correctness == "wrong":
            self.correctness_icon_label.setPixmap(
                load_svg_icon("circle-x.svg", self.theme.danger, 14)
            )
        else:
            self.correctness_icon_label.setPixmap(QPixmap())

    def _correctness(self):
        expected = self.run_record.get("expected_label")
        classification = self.run_record.get("classification")
        if classification not in {"fire_detected", "no_fire_detected"}:
            return "unknown"
        if (
            expected == "fire"
            and classification == "fire_detected"
        ) or (
            expected == "nofire"
            and classification == "no_fire_detected"
        ):
            return "correct"
        return "wrong"

    @staticmethod
    def _label_style(color: str) -> str:
        return f"""
            QLabel {{
                color: {color};
                background: transparent;
                border: none;
            }}
        """
