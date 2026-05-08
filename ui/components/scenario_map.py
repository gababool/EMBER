from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

try:
    from ui.assets.design import DEFAULT_THEME, FONT_SIZE_SM, FONT_SIZE_XS, Theme, app_font
except ImportError:
    from assets.design import DEFAULT_THEME, FONT_SIZE_SM, FONT_SIZE_XS, Theme, app_font


CONTEXTS_DIR = Path(__file__).resolve().parents[2] / "data" / "contexts"

SCENARIO_DESCRIPTIONS = {
    "scenario_01": "Wetland forest, lakes, limited access",
    "scenario_02": "Forest edge near settlement and power lines",
    "scenario_03": "High Coast forest with steep terrain",
    "scenario_04": "Rural forest with dense nearby assets",
    "scenario_05": "Protected mire, isolated wet terrain",
    "scenario_06": "Remote northern forest, sparse infrastructure",
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    description: str
    latitude: float
    longitude: float
    region: str
    context_path: Path


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

        content = self.rect().adjusted(18, 18, -18, -18)
        map_rect = QRectF(content)
        painter.setPen(QPen(QColor(self.theme.border), 1))
        painter.setBrush(QColor(self.theme.bg_panel))
        painter.drawRoundedRect(map_rect, 5, 5)

        self._draw_sweden_hint(painter, map_rect)
        self._draw_markers(painter, map_rect)

        if self.selected_scenario:
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

    def _draw_sweden_hint(self, painter: QPainter, map_rect: QRectF):
        painter.save()
        painter.setPen(QPen(QColor(self.theme.border), 1))
        painter.setBrush(QColor(self.theme.bg_main))

        outline = [
            (11.0, 55.3),
            (13.0, 55.1),
            (16.2, 56.2),
            (18.7, 57.8),
            (20.5, 60.4),
            (22.8, 64.0),
            (24.1, 66.8),
            (22.2, 68.8),
            (19.2, 68.3),
            (16.9, 66.0),
            (15.1, 63.2),
            (13.9, 60.6),
            (12.0, 58.1),
        ]
        points = [self._project(lon, lat, map_rect) for lon, lat in outline]
        painter.drawPolygon(QPolygonF(points))

        painter.setFont(app_font(FONT_SIZE_XS, bold=True))
        painter.setPen(QColor(self.theme.text_muted))
        painter.drawText(map_rect.adjusted(12, 10, -12, -10), Qt.AlignTop | Qt.AlignLeft, "SWEDEN")
        painter.restore()

    def _draw_markers(self, painter: QPainter, map_rect: QRectF):
        self._marker_points = {}
        painter.setFont(app_font(FONT_SIZE_XS, bold=True))
        metrics = QFontMetrics(painter.font())

        for scenario in self.scenarios:
            point = self._project(scenario.longitude, scenario.latitude, map_rect)
            self._marker_points[scenario.scenario_id] = point
            selected = scenario == self.selected_scenario
            color = QColor(self.theme.accent_orange if selected else self.theme.accent_cyan)
            radius = 8 if selected else 6

            painter.setPen(QPen(QColor(self.theme.bg_main), 3))
            painter.setBrush(color)
            painter.drawEllipse(point, radius, radius)

            label = scenario.scenario_id.replace("scenario_", "S")
            label_width = metrics.horizontalAdvance(label) + 10
            label_rect = QRectF(point.x() + 10, point.y() - 11, label_width, 22)
            painter.setPen(QPen(QColor(self.theme.border), 1))
            painter.setBrush(QColor(self.theme.bg_panel))
            painter.drawRoundedRect(label_rect, 4, 4)
            painter.setPen(color)
            painter.drawText(label_rect, Qt.AlignCenter, label)

    def _draw_selected_card(self, painter: QPainter, content):
        scenario = self.selected_scenario
        card = QRectF(content.left() + 14, content.bottom() - 94, content.width() - 28, 76)
        painter.setPen(QPen(QColor(self.theme.accent_orange), 1))
        painter.setBrush(QColor(self.theme.bg_panel))
        painter.drawRoundedRect(card, 5, 5)

        painter.setFont(app_font(FONT_SIZE_SM, bold=True))
        painter.setPen(QColor(self.theme.text_primary))
        painter.drawText(card.adjusted(12, 8, -12, -38), Qt.AlignLeft | Qt.AlignVCenter, scenario.title)

        painter.setFont(app_font(FONT_SIZE_XS))
        painter.setPen(QColor(self.theme.text_muted))
        details = f"{scenario.region} | {scenario.description}"
        painter.drawText(card.adjusted(12, 34, -12, -8), Qt.AlignLeft | Qt.TextWordWrap, details)

    @staticmethod
    def _project(lon: float, lat: float, rect: QRectF) -> QPointF:
        min_lon, max_lon = 10.5, 24.8
        min_lat, max_lat = 55.0, 69.2
        x = rect.left() + ((lon - min_lon) / (max_lon - min_lon)) * rect.width()
        y = rect.bottom() - ((lat - min_lat) / (max_lat - min_lat)) * rect.height()
        return QPointF(x, y)
