from pathlib import Path

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QCursor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

try:
    from ui.assets.design import (
        DEFAULT_THEME,
        FONT_SIZE_MD,
        FONT_SIZE_SM,
        Theme,
        app_font,
    )
except ImportError:
    from assets.design import (
        DEFAULT_THEME,
        FONT_SIZE_MD,
        FONT_SIZE_SM,
        Theme,
        app_font,
    )


PREVIEW_ZOOM_STEP = 1.20
PREVIEW_PINCH_SENSITIVITY = 0.75


class ZoomableImageView(QGraphicsView):
    zoom_changed = Signal(int)

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene = QGraphicsScene(self)
        self.scene.addItem(self.pixmap_item)
        self.setScene(self.scene)

        self.fit_to_view = True
        self.min_zoom = 0.20
        self.max_zoom = 12.0
        self.zoom_scale = 1.0

        self.setAlignment(Qt.AlignCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)

    def wheelEvent(self, event):
        super().wheelEvent(event)

    def viewportEvent(self, event):
        if self._handle_native_zoom_event(event):
            return True
        return super().viewportEvent(event)

    def event(self, event):
        if self._handle_native_zoom_event(event):
            return True
        return super().event(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_to_view:
            self.fit_image()

    def zoom_by(self, factor: float, anchor_pos=None):
        next_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom_scale * factor))
        if next_zoom == self.zoom_scale:
            return

        self.fit_to_view = False
        scale_factor = next_zoom / self.zoom_scale

        if anchor_pos is None:
            self.scale(scale_factor, scale_factor)
        else:
            scene_pos = self.mapToScene(anchor_pos)
            self.scale(scale_factor, scale_factor)
            new_viewport_pos = self.mapFromScene(scene_pos)
            delta = new_viewport_pos - anchor_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())

        self.zoom_scale = next_zoom
        self.zoom_changed.emit(int(self.zoom_scale * 100))

    def _handle_native_zoom_event(self, event):
        if event.type() != QEvent.NativeGesture:
            return False
        if event.gestureType() != Qt.ZoomNativeGesture:
            return False

        value = event.value()
        if value == 0:
            return True

        anchor_pos = self._event_viewport_pos(event)
        factor = max(0.8, 1.0 + value * PREVIEW_PINCH_SENSITIVITY)
        self.zoom_by(factor, anchor_pos=anchor_pos)
        return True

    def _event_viewport_pos(self, event):
        if hasattr(event, "globalPosition"):
            pos = self.viewport().mapFromGlobal(event.globalPosition().toPoint())
            if self.viewport().rect().contains(pos):
                return pos
        if hasattr(event, "globalPos"):
            pos = self.viewport().mapFromGlobal(event.globalPos())
            if self.viewport().rect().contains(pos):
                return pos
        if not hasattr(event, "position"):
            return self.viewport().mapFromGlobal(QCursor.pos())

        pos = event.position().toPoint()
        if self.viewport().rect().contains(pos):
            return pos
        if self.rect().contains(pos):
            return self.viewport().mapFrom(self, pos)
        cursor_pos = self.viewport().mapFromGlobal(QCursor.pos())
        if self.viewport().rect().contains(cursor_pos):
            return cursor_pos
        return self.viewport().rect().center()

    def fit_image(self):
        if self.pixmap_item.pixmap().isNull():
            return
        self.fit_to_view = True
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.zoom_scale = self.transform().m11()
        self.zoom_changed.emit(int(self.zoom_scale * 100))


class ImagePreviewDialog(QDialog):
    def __init__(self, image_path: str, theme: Theme = DEFAULT_THEME, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.image_path = image_path
        self.source_pixmap = QPixmap(image_path)

        self.setWindowTitle("Image Preview")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)

        self.title_label = QLabel(Path(image_path).name)
        self.title_label.setFont(app_font(FONT_SIZE_MD, bold=True))

        self.zoom_out_button = QPushButton("-")
        self.fit_button = QPushButton("Fit")
        self.zoom_in_button = QPushButton("+")
        self.close_button = QPushButton("Close")
        for button in (
            self.zoom_out_button,
            self.fit_button,
            self.zoom_in_button,
            self.close_button,
        ):
            button.setCursor(Qt.PointingHandCursor)
            button.setFont(app_font(FONT_SIZE_SM, bold=True))
            button.setFixedHeight(34)

        self.zoom_out_button.setFixedWidth(42)
        self.zoom_in_button.setFixedWidth(42)
        self.fit_button.setFixedWidth(64)
        self.close_button.setFixedWidth(84)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setFont(app_font(FONT_SIZE_SM, bold=True))
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setFixedWidth(70)

        toolbar.addWidget(self.title_label, 1)
        toolbar.addWidget(self.zoom_out_button)
        toolbar.addWidget(self.zoom_label)
        toolbar.addWidget(self.zoom_in_button)
        toolbar.addWidget(self.fit_button)
        toolbar.addWidget(self.close_button)

        self.image_view = ZoomableImageView(self.source_pixmap)
        self.image_view.zoom_changed.connect(lambda value: self.zoom_label.setText(f"{value}%"))
        self.zoom_out_button.clicked.connect(lambda: self.image_view.zoom_by(1 / PREVIEW_ZOOM_STEP))
        self.zoom_in_button.clicked.connect(lambda: self.image_view.zoom_by(PREVIEW_ZOOM_STEP))
        self.fit_button.clicked.connect(self.image_view.fit_image)
        self.close_button.clicked.connect(self.accept)

        layout.addLayout(toolbar)
        layout.addWidget(self.image_view, 1)

        self.apply_theme(theme)
        self.image_view.fit_image()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.image_view.zoom_by(PREVIEW_ZOOM_STEP)
            return
        if event.key() == Qt.Key_Minus:
            self.image_view.zoom_by(1 / PREVIEW_ZOOM_STEP)
            return
        if event.key() == Qt.Key_0:
            self.image_view.fit_image()
            return
        super().keyPressEvent(event)

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {theme.bg_main};
                color: {theme.text_primary};
            }}
            QGraphicsView {{
                background-color: {theme.bg_panel};
                border: 1px solid {theme.border};
                border-radius: 6px;
            }}
            QScrollBar:vertical, QScrollBar:horizontal {{
                background-color: {theme.bg_panel_alt};
                border: none;
                width: 10px;
                height: 10px;
            }}
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
                background-color: {theme.border};
                border-radius: 5px;
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{
                width: 0px;
                height: 0px;
            }}
        """)
        self.title_label.setStyleSheet(self._label_style(theme.text_primary))
        self.zoom_label.setStyleSheet(self._label_style(theme.text_muted))
        for button in (
            self.zoom_out_button,
            self.fit_button,
            self.zoom_in_button,
            self.close_button,
        ):
            button.setStyleSheet(f"""
                QPushButton {{
                    color: {theme.text_primary};
                    background-color: {theme.bg_panel_alt};
                    border: 1px solid {theme.border};
                    border-radius: 5px;
                    padding: 5px 10px;
                }}
                QPushButton:hover {{
                    border-color: {theme.accent_cyan};
                }}
                QPushButton:pressed {{
                    background-color: {theme.bg_panel};
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
