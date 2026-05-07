from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from ui.assets.design import DEFAULT_THEME, FONT_SIZE_SM, FONT_SIZE_XS, Theme, app_font
except ImportError:
    from assets.design import DEFAULT_THEME, FONT_SIZE_SM, FONT_SIZE_XS, Theme, app_font

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
THUMB_SIZE = 120
COLUMNS = 6
BATCH_SIZE = 30  # thumbnails loaded per timer tick


class LazyThumb(QPushButton):
    """Thumbnail button that defers pixmap loading until first paint."""

    def __init__(self, image_path: Path, theme: Theme, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self._loaded = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(THUMB_SIZE + 16, THUMB_SIZE + 28)
        self.setToolTip(image_path.name)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.img_label = QLabel()
        self.img_label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.img_label.setText("…")

        stem = image_path.stem
        name_label = QLabel(stem[:16] + ("…" if len(stem) > 16 else ""))
        name_label.setFont(app_font(FONT_SIZE_XS))
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        name_label.setStyleSheet(f"color: {theme.text_muted}; background: transparent; border: none;")

        layout.addWidget(self.img_label)
        layout.addWidget(name_label)

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme.bg_panel_alt};
                border: 1px solid {theme.border};
                border-radius: 5px;
            }}
            QPushButton:hover {{ border-color: {theme.accent_orange}; }}
            QPushButton:pressed {{ background-color: {theme.bg_panel}; }}
        """)
        self.img_label.setStyleSheet(f"color: {theme.text_muted}; background: transparent; border: none;")

    def load_pixmap(self):
        if self._loaded:
            return
        self._loaded = True
        pixmap = QPixmap(str(self.image_path))
        if not pixmap.isNull():
            self.img_label.setText("")
            self.img_label.setPixmap(
                pixmap.scaled(THUMB_SIZE, THUMB_SIZE, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            )


class ImagePickerDialog(QDialog):
    image_picked = Signal(str)

    def __init__(self, dataset_dir: Path, theme: Theme = DEFAULT_THEME, parent=None):
        super().__init__(parent)
        self.dataset_dir = dataset_dir
        self.theme = theme
        self.setWindowTitle("Pick Image")
        self.resize(900, 580)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.setFont(app_font(FONT_SIZE_SM, bold=True))

        self._pending: list[list[LazyThumb]] = []  # one list per tab

        for label, folder in [("Fire", "fire"), ("No Fire", "nofire")]:
            tab, pending = self._make_tab(dataset_dir / folder)
            self.tabs.addTab(tab, label)
            self._pending.append(pending)

        layout.addWidget(self.tabs)
        self._apply_theme(theme)

        # Load first batch immediately, rest in background
        QTimer.singleShot(0, self._load_next_batch)

    def _make_tab(self, folder_path: Path) -> tuple[QWidget, list[LazyThumb]]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)

        images = sorted(
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ) if folder_path.exists() else []

        pending = []
        for i, image_path in enumerate(images):
            thumb = LazyThumb(image_path, self.theme)
            thumb.clicked.connect(lambda checked=False, p=str(image_path): self._pick(p))
            grid.addWidget(thumb, i // COLUMNS, i % COLUMNS)
            pending.append(thumb)

        if not images:
            empty = QLabel("No images found")
            empty.setAlignment(Qt.AlignCenter)
            empty.setFont(app_font(FONT_SIZE_SM))
            grid.addWidget(empty, 0, 0)

        scroll.setWidget(container)
        return scroll, pending

    def _load_next_batch(self):
        loaded = 0
        for pending in self._pending:
            while pending and loaded < BATCH_SIZE:
                pending.pop(0).load_pixmap()
                loaded += 1

        if any(self._pending):
            QTimer.singleShot(0, self._load_next_batch)

    def _pick(self, image_path: str):
        self.image_picked.emit(image_path)
        self.accept()

    def _apply_theme(self, theme: Theme):
        self.setStyleSheet(f"""
            QDialog {{ background-color: {theme.bg_main}; }}
            QTabWidget::pane {{
                background-color: {theme.bg_panel};
                border: 1px solid {theme.border};
                border-radius: 5px;
            }}
            QTabBar::tab {{
                background-color: {theme.bg_panel_alt};
                color: {theme.text_muted};
                border: 1px solid {theme.border};
                padding: 6px 20px;
                font-family: "JetBrains Mono";
            }}
            QTabBar::tab:selected {{
                background-color: {theme.bg_panel};
                color: {theme.text_primary};
                border-bottom: 2px solid {theme.accent_orange};
            }}
            QScrollArea {{ background: transparent; border: none; }}
            QWidget {{ background: transparent; }}
        """)
