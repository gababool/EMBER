from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


ICONS_DIR = Path(__file__).resolve().parents[1] / "assets" / "icons"


def load_svg_icon(icon_name: str, color: str, size: int) -> QPixmap:
    svg = (ICONS_DIR / icon_name).read_text(encoding="utf-8")
    svg = svg.replace("currentColor", color)
    return render_svg(svg, size)


def load_svg_asset(icon_name: str, size: int) -> QPixmap:
    svg = (ICONS_DIR / icon_name).read_text(encoding="utf-8")
    return render_svg(svg, size)


def render_svg(svg: str, size: int) -> QPixmap:
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return pixmap
