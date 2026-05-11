from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

try:
    from ui.assets.design import DEFAULT_THEME, FONT_SIZE_SM, FONT_SIZE_XS, Theme, app_font
except ImportError:
    from assets.design import DEFAULT_THEME, FONT_SIZE_SM, FONT_SIZE_XS, Theme, app_font


CONTEXTS_DIR = Path(__file__).resolve().parents[2] / "data" / "contexts"
SWEDEN_MAP_PATH = Path(__file__).resolve().parents[1] / "assets" / "Sweden_Map.png"

SCENARIO_DESCRIPTIONS = {
    "scenario_01": "Wetland forest, lakes, limited access",
    "scenario_02": "Forest edge near settlement and power lines",
    "scenario_03": "High Coast forest with steep terrain",
    "scenario_04": "Rural forest with dense nearby assets",
    "scenario_05": "Protected mire, isolated wet terrain",
    "scenario_06": "Remote northern forest, sparse infrastructure",
}


# Approximate geographic bounds of ui/assets/Sweden_Map.png. The asset is a
# Web-Mercator style basemap screenshot, so marker placement uses Mercator Y.
MAP_MIN_LON = -10.0
MAP_MAX_LON = 41.3
MAP_MIN_LAT = 52.0
MAP_MAX_LAT = 72.0



@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    description: str
    latitude: float
    longitude: float
    region: str
    context_path: Path

    @property
    def marker_label(self) -> str:
        return str(int(self.scenario_id.replace("scenario_", "")))


def load_scenarios(contexts_dir: Path = CONTEXTS_DIR) -> list[Scenario]:
    scenarios = []
    for context_path in sorted(contexts_dir.glob("scenario_*/scenario_*.json")):
        try:
            data = json.loads(context_path.read_text(encoding="utf-8"))
            coordinates = data["coordinates"]
            region_data = data.get("region", {})
            scenario_id = context_path.stem
            title = scenario_id.replace("_", " ").title()
            region = region_data.get("name") or "Unknown region"
            description = SCENARIO_DESCRIPTIONS.get(
                scenario_id,
                f"{data.get('land_cover', 'unknown').replace('_', ' ')} near {region}",
            )
            scenarios.append(
                Scenario(
                    scenario_id=scenario_id,
                    title=title,
                    description=description,
                    latitude=float(coordinates["latitude"]),
                    longitude=float(coordinates["longitude"]),
                    region=region,
                    context_path=context_path,
                )
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return scenarios


class ScenarioMap(QWidget):
    scenario_selected = Signal(object)

    def __init__(
        self,
        scenarios: list[Scenario] | None = None,
        theme: Theme = DEFAULT_THEME,
        parent=None,
    ):
        super().__init__(parent)
        self.theme = theme
        self.scenarios = scenarios or load_scenarios()
        self.selected_scenario = self.scenarios[0] if self.scenarios else None
        self._marker_points: dict[str, QPointF] = {}
        self._map_pixmap = QPixmap(str(SWEDEN_MAP_PATH))
        self.setMinimumSize(420, 360)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def selected_context_file(self) -> str | None:
        if self.selected_scenario is None:
            return None
        return str(self.selected_scenario.context_path)

    def set_selected_context_file(self, context_file: str | None):
        if not context_file:
            return
        target = Path(context_file).resolve()
        for scenario in self.scenarios:
            if scenario.context_path.resolve() == target:
                self.selected_scenario = scenario
                self.update()
                return

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(self.theme.bg_panel_alt))

        content = QRectF(self.rect()).adjusted(18, 18, -18, -18)
        map_rect = content.adjusted(10, 28, -10, -104)
        image_rect = self._image_rect(map_rect)

        self._draw_overview_background(painter, content)
        self._draw_sweden_map(painter, image_rect)
        self._draw_markers(painter, image_rect)
        self._draw_selected_card(painter, content)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        click_pos = QPointF(event.position())
        nearest = None
        nearest_distance = 18.0
        for scenario in self.scenarios:
            point = self._marker_points.get(scenario.scenario_id)
            if point is None:
                continue
            distance = ((point.x() - click_pos.x()) ** 2 + (point.y() - click_pos.y()) ** 2) ** 0.5
            if distance <= nearest_distance:
                nearest = scenario
                nearest_distance = distance

        if nearest is not None:
            self.selected_scenario = nearest
            self.scenario_selected.emit(nearest)
            self.update()

    def _draw_overview_background(self, painter: QPainter, content: QRectF):
        painter.setPen(QPen(QColor(self.theme.border), 1))
        painter.setBrush(QColor(self.theme.bg_panel))
        painter.drawRoundedRect(content, 5, 5)

        painter.setFont(app_font(FONT_SIZE_XS, bold=True))
        painter.setPen(QColor(self.theme.text_muted))
        painter.drawText(content.adjusted(12, 10, -12, -10), Qt.AlignTop | Qt.AlignLeft, "OFFLINE SCENARIO OVERVIEW")

    def _draw_sweden_map(self, painter: QPainter, image_rect: QRectF):
        if self._map_pixmap.isNull():
            painter.setPen(QPen(QColor(self.theme.border), 1))
            painter.setBrush(QColor(self.theme.bg_main))
            painter.drawRoundedRect(image_rect, 5, 5)
            painter.setPen(QColor(self.theme.text_muted))
            painter.drawText(image_rect, Qt.AlignCenter, "Sweden map asset not found")
            return

        painter.setPen(QPen(QColor(self.theme.border), 1))
        painter.setBrush(QColor(self.theme.bg_main))
        painter.drawRoundedRect(image_rect, 5, 5)
        painter.drawPixmap(image_rect.toRect(), self._map_pixmap)

    def _draw_markers(self, painter: QPainter, map_rect: QRectF):
        self._marker_points = {}
        painter.setFont(app_font(FONT_SIZE_SM, bold=True))
        metrics = QFontMetrics(painter.font())

        for scenario in self.scenarios:
            point = self._project(scenario.longitude, scenario.latitude, map_rect)
            self._marker_points[scenario.scenario_id] = point
            selected = scenario == self.selected_scenario
            fill_color = QColor(self.theme.accent_orange if selected else self.theme.accent_cyan)
            text_color = QColor("#ffffff" if selected else "#061018")
            radius = 16 if selected else 13

            painter.setPen(QPen(QColor(self.theme.bg_panel), 4))
            painter.setBrush(fill_color)
            painter.drawEllipse(point, radius, radius)

            label = scenario.marker_label
            label_rect = QRectF(
                point.x() - metrics.horizontalAdvance(label) / 2 - 4,
                point.y() - 10,
                metrics.horizontalAdvance(label) + 8,
                20,
            )
            painter.setPen(text_color)
            painter.drawText(label_rect, Qt.AlignCenter, label)

            if selected:
                painter.setPen(QPen(fill_color, 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(point, radius + 7, radius + 7)

    def _draw_selected_card(self, painter: QPainter, content: QRectF):
        if self.selected_scenario is None:
            return

        scenario = self.selected_scenario
        card = QRectF(content.left() + 12, content.bottom() - 86, content.width() - 24, 70)
        painter.setPen(QPen(QColor(self.theme.accent_orange), 1))
        painter.setBrush(QColor(self.theme.bg_panel_alt))
        painter.drawRoundedRect(card, 5, 5)

        painter.setFont(app_font(FONT_SIZE_SM, bold=True))
        painter.setPen(QColor(self.theme.text_primary))
        painter.drawText(card.adjusted(12, 8, -12, -38), Qt.AlignLeft | Qt.AlignVCenter, scenario.title)

        painter.setFont(app_font(FONT_SIZE_XS))
        painter.setPen(QColor(self.theme.text_muted))
        details = f"{scenario.region} | {scenario.description}"
        painter.drawText(card.adjusted(12, 32, -12, -8), Qt.AlignLeft | Qt.TextWordWrap, details)

    @staticmethod
    def _project(lon: float, lat: float, rect: QRectF) -> QPointF:
        x = rect.left() + ((lon - MAP_MIN_LON) / (MAP_MAX_LON - MAP_MIN_LON)) * rect.width()
        lat = max(min(lat, MAP_MAX_LAT), MAP_MIN_LAT)
        min_y = ScenarioMap._mercator_y(MAP_MIN_LAT)
        max_y = ScenarioMap._mercator_y(MAP_MAX_LAT)
        y_fraction = (max_y - ScenarioMap._mercator_y(lat)) / (max_y - min_y)
        y = rect.top() + y_fraction * rect.height()
        return QPointF(x, y)

    def _image_rect(self, target: QRectF) -> QRectF:
        if self._map_pixmap.isNull():
            return target

        image_ratio = self._map_pixmap.width() / self._map_pixmap.height()
        target_ratio = target.width() / target.height()
        if target_ratio > image_ratio:
            height = target.height()
            width = height * image_ratio
        else:
            width = target.width()
            height = width / image_ratio
        return QRectF(
            target.center().x() - width / 2,
            target.center().y() - height / 2,
            width,
            height,
        )

    @staticmethod
    def _mercator_y(lat: float) -> float:
        import math

        radians = math.radians(lat)
        return math.log(math.tan(math.pi / 4 + radians / 2))
