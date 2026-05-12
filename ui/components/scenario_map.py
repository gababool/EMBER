from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

try:
    from ui.assets.design import DEFAULT_THEME, FONT_SIZE_SM, FONT_SIZE_XS, Theme, app_font
except ImportError:
    from assets.design import DEFAULT_THEME, FONT_SIZE_SM, FONT_SIZE_XS, Theme, app_font


CONTEXTS_DIR = Path(__file__).resolve().parents[2] / "data" / "contexts"
OSM_DIR = Path(__file__).resolve().parents[2] / "data" / "gis" / "osm"
VECTOR_CACHE_DIR = Path(tempfile.gettempdir()) / "ember_osm_vector_cache"
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
VECTOR_CACHE_VERSION = 4



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

    @property
    def osm_pbf_path(self) -> Path:
        return OSM_DIR / f"{self.scenario_id}-50km.osm.pbf"


@dataclass(frozen=True)
class VectorFeature:
    layer: str
    geometry: list[tuple[float, float]]
    name: str | None = None
    closed: bool = False
    bounds: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ScenarioVectorData:
    features: list[VectorFeature]
    bounds: tuple[float, float, float, float] | None
    source_path: Path
    loaded_from_cache: bool = False


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


def load_scenario_vector_data(scenario: Scenario) -> ScenarioVectorData:
    source_path = scenario.osm_pbf_path
    if not source_path.exists():
        return ScenarioVectorData([], None, source_path)

    cached = _read_vector_cache(source_path)
    if cached is not None:
        return cached

    data = _extract_osm_vectors(source_path)
    _write_vector_cache(data)
    return data


def _read_vector_cache(source_path: Path) -> ScenarioVectorData | None:
    cache_path = _vector_cache_path(source_path)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("cache_version") != VECTOR_CACHE_VERSION:
            return None
        if payload.get("source_mtime") != source_path.stat().st_mtime:
            return None
        features = [
            VectorFeature(
                layer=item["layer"],
                geometry=[tuple(point) for point in item["geometry"]],
                name=item.get("name"),
                closed=bool(item.get("closed")),
                bounds=tuple(item["bounds"]) if item.get("bounds") else None,
            )
            for item in payload.get("features", [])
        ]
        bounds = payload.get("bounds")
        return ScenarioVectorData(
            features=features,
            bounds=tuple(bounds) if bounds else None,
            source_path=source_path,
            loaded_from_cache=True,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_vector_cache(data: ScenarioVectorData):
    try:
        VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_version": VECTOR_CACHE_VERSION,
            "source": str(data.source_path),
            "source_mtime": data.source_path.stat().st_mtime,
            "bounds": data.bounds,
            "features": [
                {
                    "layer": feature.layer,
                    "geometry": feature.geometry,
                    "name": feature.name,
                    "closed": feature.closed,
                    "bounds": feature.bounds,
                }
                for feature in data.features
            ],
        }
        _vector_cache_path(data.source_path).write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        return


def _vector_cache_path(source_path: Path) -> Path:
    return VECTOR_CACHE_DIR / f"{source_path.stem}.vector.json"


def _extract_osm_vectors(source_path: Path) -> ScenarioVectorData:
    import osmium

    class OSMVectorHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.features: list[VectorFeature] = []
            self.bounds = None
            self._wkb = osmium.geom.WKBFactory()

        def node(self, node):
            tags = dict(node.tags)
            place = tags.get("place")
            if place not in {"city", "town", "village", "hamlet", "isolated_dwelling"}:
                return
            name = tags.get("name")
            if not name:
                return
            self._add_feature(
                VectorFeature(
                    layer="settlement",
                    geometry=[(float(node.location.lon), float(node.location.lat))],
                    name=name,
                )
            )

        def way(self, way):
            tags = dict(way.tags)
            layer = self._layer_for_way(tags)
            if layer is None:
                return

            geometry = []
            try:
                for node in way.nodes:
                    geometry.append((float(node.lon), float(node.lat)))
            except Exception:
                return

            if len(geometry) < 2:
                return

            closed = geometry[0] == geometry[-1]
            thinned_geometry = _thin_geometry(geometry, layer)
            self._add_feature(
                VectorFeature(
                    layer=layer,
                    geometry=thinned_geometry,
                    name=tags.get("name"),
                    closed=closed,
                    bounds=_geometry_bounds(thinned_geometry),
                )
            )

        def area(self, area):
            tags = dict(area.tags)
            layer = self._layer_for_area(tags)
            if layer is None:
                return

            try:
                geometry_parts = _multipolygon_geometry_parts(self._wkb.create_multipolygon(area))
            except Exception:
                return

            for geometry in geometry_parts:
                if len(geometry) < 3:
                    continue
                thinned_geometry = _thin_geometry(geometry, layer)
                self._add_feature(
                    VectorFeature(
                        layer=layer,
                        geometry=thinned_geometry,
                        name=tags.get("name"),
                        closed=True,
                        bounds=_geometry_bounds(thinned_geometry),
                    )
                )

        def _add_feature(self, feature: VectorFeature):
            self.features.append(feature)
            for lon, lat in feature.geometry:
                if self.bounds is None:
                    self.bounds = (lon, lat, lon, lat)
                else:
                    min_lon, min_lat, max_lon, max_lat = self.bounds
                    self.bounds = (
                        min(min_lon, lon),
                        min(min_lat, lat),
                        max(max_lon, lon),
                        max(max_lat, lat),
                    )

        @staticmethod
        def _layer_for_way(tags: dict) -> str | None:
            if "waterway" in tags:
                return "waterway"
            if tags.get("power") in {"line", "minor_line"}:
                return "power"

            highway = tags.get("highway")
            if highway in {"motorway", "trunk", "primary", "secondary"}:
                return "road_major"
            if highway in {
                "motorway_link",
                "trunk_link",
                "primary_link",
                "secondary_link",
                "tertiary",
                "tertiary_link",
                "unclassified",
                "residential",
                "living_street",
                "service",
                "pedestrian",
                "road",
            }:
                return "road_minor"
            if highway in {"track", "path", "footway", "cycleway", "bridleway"}:
                return "track"
            return None

        @staticmethod
        def _layer_for_area(tags: dict) -> str | None:
            if (
                tags.get("natural") == "water"
                or tags.get("landuse") in {"reservoir", "basin"}
                or tags.get("waterway") == "riverbank"
            ):
                return "water"
            if tags.get("boundary") == "protected_area" or tags.get("leisure") == "nature_reserve":
                return "protected"
            return None

    handler = OSMVectorHandler()
    handler.apply_file(str(source_path), locations=True)
    return ScenarioVectorData(handler.features, handler.bounds, source_path)


def _geometry_bounds(geometry: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    lons = [point[0] for point in geometry]
    lats = [point[1] for point in geometry]
    return min(lons), min(lats), max(lons), max(lats)


def _geometry_centroid(geometry: list[tuple[float, float]]) -> tuple[float, float]:
    lon_sum = sum(point[0] for point in geometry)
    lat_sum = sum(point[1] for point in geometry)
    count = max(len(geometry), 1)
    return lon_sum / count, lat_sum / count


def _multipolygon_geometry_parts(wkb_hex: str) -> list[list[tuple[float, float]]]:
    from shapely import wkb as shapely_wkb

    geometry = shapely_wkb.loads(wkb_hex, hex=True)
    if geometry.geom_type == "Polygon":
        return [_ring_to_points(geometry.exterior.coords)]
    if geometry.geom_type == "MultiPolygon":
        return [_ring_to_points(polygon.exterior.coords) for polygon in geometry.geoms]
    return []


def _ring_to_points(coords) -> list[tuple[float, float]]:
    return [(float(lon), float(lat)) for lon, lat, *_ in coords]


def _thin_geometry(geometry: list[tuple[float, float]], layer: str) -> list[tuple[float, float]]:
    if layer in {"road_major", "road_minor"} or len(geometry) <= 120:
        return geometry
    target_points = 260 if layer in {"water", "protected"} else 120
    step = max(1, len(geometry) // target_points)
    thinned = geometry[::step]
    if thinned[-1] != geometry[-1]:
        thinned.append(geometry[-1])
    return thinned


class ScenarioMap(QWidget):
    scenario_selected = Signal(object)
    scenario_loaded = Signal(object)
    scenario_unloaded = Signal()

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
        self.loaded_scenario = None
        self._loaded_context = None
        self._vector_data = None
        self._features_by_layer: dict[str, list[VectorFeature]] = {}
        self._detail_center_merc = None
        self._detail_scale = None
        self._drag_last_pos = None
        self._context_highlights_visible = False
        self._marker_points: dict[str, QPointF] = {}
        self._map_pixmap = QPixmap(str(SWEDEN_MAP_PATH))

        self.load_scenario_button = QPushButton("Load Scenario", self)
        self.load_scenario_button.setCursor(Qt.PointingHandCursor)
        self.load_scenario_button.setFont(app_font(FONT_SIZE_XS, bold=True))
        self.load_scenario_button.clicked.connect(self._load_selected_scenario)

        self.zoom_in_button = QPushButton("+", self)
        self.zoom_out_button = QPushButton("-", self)
        for button in (self.zoom_in_button, self.zoom_out_button):
            button.setCursor(Qt.PointingHandCursor)
            button.setFont(app_font(FONT_SIZE_SM, bold=True))
            button.setFixedSize(34, 30)
            button.setVisible(False)
        self.zoom_in_button.clicked.connect(lambda: self._zoom_detail(1.25))
        self.zoom_out_button.clicked.connect(lambda: self._zoom_detail(1 / 1.25))

        self.exit_detail_button = QPushButton("Back", self)
        self.exit_detail_button.setCursor(Qt.PointingHandCursor)
        self.exit_detail_button.setFont(app_font(FONT_SIZE_XS, bold=True))
        self.exit_detail_button.setFixedSize(70, 30)
        self.exit_detail_button.setVisible(False)
        self.exit_detail_button.clicked.connect(self._exit_detail_view)

        self.setMinimumSize(420, 360)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.apply_theme(theme)

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
        self.load_scenario_button.setStyleSheet(f"""
            QPushButton {{
                color: #ffffff;
                background-color: {theme.accent_orange};
                border: 1px solid {theme.accent_orange};
                border-radius: 4px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: {theme.accent_cyan};
                border-color: {theme.accent_cyan};
            }}
            QPushButton:pressed {{
                background-color: {theme.accent_blue};
                border-color: {theme.accent_blue};
            }}
        """)
        zoom_style = f"""
            QPushButton {{
                color: {theme.text_primary};
                background-color: {theme.bg_panel_alt};
                border: 1px solid {theme.border};
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border-color: {theme.accent_orange};
            }}
            QPushButton:pressed {{
                background-color: {theme.bg_panel};
            }}
        """
        self.zoom_in_button.setStyleSheet(zoom_style)
        self.zoom_out_button.setStyleSheet(zoom_style)
        self.exit_detail_button.setStyleSheet(zoom_style)
        self.update()

    def set_context_highlights_visible(self, visible: bool):
        self._context_highlights_visible = visible
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(self.theme.bg_panel_alt))

        content = QRectF(self.rect()).adjusted(18, 18, -18, -18)
        if self.loaded_scenario is None:
            map_rect = content.adjusted(10, 28, -10, -104)
            image_rect = self._image_rect(map_rect)
            self._draw_overview_background(painter, content)
            self._draw_sweden_map(painter, image_rect)
            self._draw_markers(painter, image_rect)
            self._draw_selected_card(painter, content)
            self.load_scenario_button.setVisible(self.selected_scenario is not None)
            self.zoom_in_button.setVisible(False)
            self.zoom_out_button.setVisible(False)
            self.exit_detail_button.setVisible(False)
        else:
            self.load_scenario_button.setVisible(False)
            self._position_zoom_buttons(content)
            self.zoom_in_button.setVisible(True)
            self.zoom_out_button.setVisible(True)
            self.exit_detail_button.setVisible(True)
            self._draw_detail_view(painter, content)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        if self.loaded_scenario is not None:
            self._drag_last_pos = QPointF(event.position())
            self.setCursor(Qt.ClosedHandCursor)
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

    def mouseMoveEvent(self, event):
        if self.loaded_scenario is None or self._drag_last_pos is None:
            return
        if self._detail_scale is None or self._detail_center_merc is None:
            return
        pos = QPointF(event.position())
        delta = pos - self._drag_last_pos
        self._drag_last_pos = pos
        self._detail_center_merc = QPointF(
            self._detail_center_merc.x() - delta.x() / self._detail_scale,
            self._detail_center_merc.y() + delta.y() / self._detail_scale,
        )
        self._clamp_detail_center(self._detail_map_rect(QRectF(self.rect()).adjusted(18, 18, -18, -18)))
        self.update()

    def mouseReleaseEvent(self, event):
        if self.loaded_scenario is not None and event.button() == Qt.LeftButton:
            self._drag_last_pos = None
            self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, event):
        if self.loaded_scenario is None:
            super().wheelEvent(event)
            return

        content = QRectF(self.rect()).adjusted(18, 18, -18, -18)
        detail_rect = self._detail_map_rect(content)
        self._initialize_detail_view(detail_rect)
        pos = QPointF(event.position())
        if not detail_rect.contains(pos):
            return

        factor = 1.22 if event.angleDelta().y() > 0 else 1 / 1.22
        self._zoom_detail(factor, anchor_pos=pos)
        event.accept()

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
            text_color = QColor(self.theme.bg_panel if selected else self.theme.bg_main)
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
        self._position_load_button(card)
        painter.setPen(QPen(QColor(self.theme.accent_orange), 1))
        painter.setBrush(QColor(self.theme.bg_panel_alt))
        painter.drawRoundedRect(card, 5, 5)

        painter.setFont(app_font(FONT_SIZE_SM, bold=True))
        painter.setPen(QColor(self.theme.text_primary))
        painter.drawText(card.adjusted(12, 8, -12, -38), Qt.AlignLeft | Qt.AlignVCenter, scenario.title)

        painter.setFont(app_font(FONT_SIZE_XS))
        painter.setPen(QColor(self.theme.text_muted))
        details = f"{scenario.region} | {scenario.description}"
        painter.drawText(card.adjusted(12, 32, -132, -8), Qt.AlignLeft | Qt.TextWordWrap, details)

    def _position_load_button(self, card: QRectF):
        self.load_scenario_button.setGeometry(
            int(card.right() - 124),
            int(card.top() + 18),
            108,
            34,
        )

    def _position_zoom_buttons(self, content: QRectF):
        left = int(content.left() + 22)
        top = int(content.top() + 52)
        self.zoom_in_button.move(left, top)
        self.zoom_out_button.move(left, top + 34)
        self.exit_detail_button.move(int(content.right() - 92), top)

    def _exit_detail_view(self):
        self.loaded_scenario = None
        self._loaded_context = None
        self._vector_data = None
        self._features_by_layer = {}
        self._detail_center_merc = None
        self._detail_scale = None
        self._drag_last_pos = None
        self._context_highlights_visible = False
        self.setCursor(Qt.PointingHandCursor)
        self.scenario_unloaded.emit()
        self.update()

    def _zoom_detail(self, factor: float, anchor_pos: QPointF | None = None):
        if self.loaded_scenario is None:
            return

        content = QRectF(self.rect()).adjusted(18, 18, -18, -18)
        detail_rect = self._detail_map_rect(content)
        self._initialize_detail_view(detail_rect)
        if self._detail_scale is None or self._detail_center_merc is None:
            return

        anchor_pos = anchor_pos or detail_rect.center()
        anchor = self._screen_to_merc(anchor_pos, detail_rect)
        min_scale = self._minimum_detail_scale(detail_rect)
        next_scale = max(min_scale, min(7_000_000, self._detail_scale * factor))
        self._detail_center_merc = QPointF(
            anchor.x() - (anchor_pos.x() - detail_rect.center().x()) / next_scale,
            anchor.y() + (anchor_pos.y() - detail_rect.center().y()) / next_scale,
        )
        self._detail_scale = next_scale
        self._clamp_detail_center(detail_rect)
        self.update()

    def _load_selected_scenario(self):
        if self.selected_scenario is None:
            return
        self.load_scenario_button.setEnabled(False)
        self.load_scenario_button.setText("Loading...")
        QApplication.processEvents()
        self.loaded_scenario = self.selected_scenario
        self._loaded_context = self._read_context(self.loaded_scenario)
        try:
            self._vector_data = load_scenario_vector_data(self.loaded_scenario)
            self._features_by_layer = self._group_features_by_layer(self._vector_data.features)
            self._detail_center_merc = None
            self._detail_scale = None
            self._context_highlights_visible = False
            self.setCursor(Qt.OpenHandCursor)
            self.scenario_loaded.emit(self.loaded_scenario)
        finally:
            self.load_scenario_button.setText("Load Scenario")
            self.load_scenario_button.setEnabled(True)
            self.update()

    def _draw_detail_view(self, painter: QPainter, content: QRectF):
        scenario = self.loaded_scenario
        context = self._loaded_context or {}
        detail_rect = self._detail_map_rect(content)
        self._initialize_detail_view(detail_rect)
        self._enforce_detail_bounds(detail_rect)

        painter.setPen(QPen(QColor(self.theme.border), 1))
        painter.setBrush(QColor(self.theme.bg_panel))
        painter.drawRoundedRect(content, 5, 5)

        painter.setFont(app_font(FONT_SIZE_XS, bold=True))
        painter.setPen(QColor(self.theme.text_muted))
        painter.drawText(content.adjusted(12, 10, -12, -10), Qt.AlignTop | Qt.AlignLeft, "LOADED SCENARIO DETAIL")

        self._draw_vector_map(painter, detail_rect)
        self._draw_context_overlays(painter, detail_rect, context)

        card = QRectF(content.left() + 12, content.bottom() - 78, content.width() - 24, 62)
        painter.setPen(QPen(QColor(self.theme.accent_orange), 1))
        painter.setBrush(QColor(self.theme.bg_panel_alt))
        painter.drawRoundedRect(card, 5, 5)
        painter.setFont(app_font(FONT_SIZE_SM, bold=True))
        painter.setPen(QColor(self.theme.text_primary))
        painter.drawText(card.adjusted(12, 6, -12, -34), Qt.AlignLeft | Qt.AlignVCenter, scenario.title)
        painter.setFont(app_font(FONT_SIZE_XS))
        painter.setPen(QColor(self.theme.text_muted))
        painter.drawText(
            card.adjusted(12, 28, -12, -6),
            Qt.AlignLeft | Qt.TextWordWrap,
            self._detail_summary(context),
        )

    def _draw_vector_map(self, painter: QPainter, rect: QRectF):
        painter.save()
        painter.setClipRect(rect)
        painter.setPen(QPen(QColor(self.theme.border), 1))
        painter.setBrush(QColor(self.theme.map_bg))
        painter.drawRoundedRect(rect, 5, 5)

        if self._vector_data is None or not self._vector_data.features:
            painter.setPen(QColor(self.theme.text_muted))
            painter.drawText(rect, Qt.AlignCenter, f"No local OSM vector data found:\n{self.loaded_scenario.osm_pbf_path.name}")
            painter.restore()
            return

        layer_order = self._visible_layers_for_scale()
        viewport = self._visible_geo_bounds(rect)
        for layer in layer_order:
            for feature in self._features_by_layer.get(layer, []):
                if self._feature_in_view(feature, viewport):
                    self._draw_vector_feature(painter, rect, feature)
        painter.restore()

    def _draw_vector_feature(self, painter: QPainter, rect: QRectF, feature: VectorFeature):
        if feature.layer == "settlement":
            point = self._geo_to_screen(*feature.geometry[0], rect)
            if not rect.adjusted(-12, -12, 12, 12).contains(point):
                return
            painter.setPen(QPen(QColor(self.theme.map_bg), 2))
            painter.setBrush(QColor(self.theme.map_settlement))
            painter.drawEllipse(point, 4, 4)
            if feature.name and self._detail_scale and self._detail_scale > 220_000:
                painter.setFont(app_font(FONT_SIZE_XS, bold=True))
                painter.setPen(QColor(self.theme.text_primary))
                painter.drawText(QRectF(point.x() + 7, point.y() - 10, 130, 20), Qt.AlignLeft | Qt.AlignVCenter, feature.name[:22])
            return

        path = QPainterPath()
        geometry = self._geometry_for_scale(feature)
        first = self._geo_to_screen(*geometry[0], rect)
        path.moveTo(first)
        for lon, lat in geometry[1:]:
            path.lineTo(self._geo_to_screen(lon, lat, rect))

        if feature.layer == "water":
            painter.setPen(QPen(QColor(self.theme.map_water_stroke), 1))
            painter.setBrush(QColor(self.theme.map_water_fill))
            painter.drawPath(path)
        elif feature.layer == "protected":
            painter.setPen(QPen(QColor(self.theme.map_protected_stroke), 1, Qt.DashLine))
            painter.setBrush(self._theme_color(self.theme.map_protected_fill, 52))
            painter.drawPath(path)
        elif feature.layer == "building":
            if self._detail_scale and self._detail_scale < 250_000:
                return
            painter.setPen(QPen(QColor(self.theme.map_building), 1))
            painter.setBrush(QColor(self.theme.map_building))
            painter.drawPath(path)
        else:
            pen = self._pen_for_layer(feature.layer)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

    def _pen_for_layer(self, layer: str) -> QPen:
        if layer == "road_major":
            return QPen(QColor(self.theme.map_road_major), 3.0)
        if layer == "road_minor":
            return QPen(QColor(self.theme.map_road_minor), 1.8)
        if layer == "track":
            return QPen(QColor(self.theme.map_track), 1.2, Qt.DashLine)
        if layer == "power":
            return QPen(QColor(self.theme.map_power), 1.1, Qt.DashLine)
        if layer == "waterway":
            return QPen(QColor(self.theme.map_waterway), 1.4)
        return QPen(QColor(self.theme.border), 1)

    @staticmethod
    def _theme_color(value: str, alpha: int | None = None) -> QColor:
        color = QColor(value)
        if alpha is not None:
            color.setAlpha(alpha)
        return color

    def _visible_layers_for_scale(self) -> tuple[str, ...]:
        scale = self._detail_scale or 0
        return tuple(layer for layer in self._layer_draw_order() if scale >= self._minimum_scale_for_layer(layer))

    @staticmethod
    def _layer_draw_order() -> tuple[str, ...]:
        return ("protected", "water", "waterway", "track", "road_minor", "road_major", "power", "settlement")

    @staticmethod
    def _minimum_scale_for_layer(layer: str) -> int:
        thresholds = {
            "protected": 120_000,
            "water": 0,
            "waterway": 120_000,
            "track": 360_000,
            "road_minor": 120_000,
            "road_major": 0,
            "power": 120_000,
            "settlement": 360_000,
        }
        return thresholds.get(layer, 0)

    def _geometry_for_scale(self, feature: VectorFeature) -> list[tuple[float, float]]:
        scale = self._detail_scale or 0
        if scale >= 360_000 or len(feature.geometry) <= 16:
            return feature.geometry
        step = 4 if scale < 120_000 else 2
        geometry = feature.geometry[::step]
        if geometry[-1] != feature.geometry[-1]:
            geometry.append(feature.geometry[-1])
        return geometry

    @staticmethod
    def _group_features_by_layer(features: list[VectorFeature]) -> dict[str, list[VectorFeature]]:
        grouped: dict[str, list[VectorFeature]] = {}
        for feature in features:
            grouped.setdefault(feature.layer, []).append(feature)
        return grouped

    def _visible_geo_bounds(self, rect: QRectF) -> tuple[float, float, float, float]:
        top_left = self._screen_to_lon_lat(rect.topLeft(), rect)
        bottom_right = self._screen_to_lon_lat(rect.bottomRight(), rect)
        min_lon = min(top_left[0], bottom_right[0])
        max_lon = max(top_left[0], bottom_right[0])
        min_lat = min(top_left[1], bottom_right[1])
        max_lat = max(top_left[1], bottom_right[1])
        lon_pad = (max_lon - min_lon) * 0.08
        lat_pad = (max_lat - min_lat) * 0.08
        return min_lon - lon_pad, min_lat - lat_pad, max_lon + lon_pad, max_lat + lat_pad

    @staticmethod
    def _feature_in_view(feature: VectorFeature, bounds: tuple[float, float, float, float]) -> bool:
        min_lon, min_lat, max_lon, max_lat = bounds
        feature_bounds = feature.bounds or _geometry_bounds(feature.geometry)
        feature_min_lon, feature_min_lat, feature_max_lon, feature_max_lat = feature_bounds
        return not (
            feature_max_lon < min_lon
            or feature_min_lon > max_lon
            or feature_max_lat < min_lat
            or feature_min_lat > max_lat
        )

    def _draw_context_overlays(self, painter: QPainter, rect: QRectF, context: dict):
        scenario_point = self._geo_to_screen(self.loaded_scenario.longitude, self.loaded_scenario.latitude, rect)
        painter.setPen(QPen(QColor(self.theme.bg_panel), 4))
        painter.setBrush(QColor(self.theme.accent_orange))
        painter.drawEllipse(scenario_point, 8, 8)
        painter.setFont(app_font(FONT_SIZE_XS, bold=True))
        painter.setPen(QColor(self.theme.text_primary))
        painter.drawText(QRectF(scenario_point.x() + 12, scenario_point.y() - 12, 100, 24), Qt.AlignLeft | Qt.AlignVCenter, "Scenario")

        if not self._context_highlights_visible:
            return

        self._draw_context_road_highlights(painter, rect, context)

        for water in context.get("water_sources", []):
            label = water.get("name") or water.get("source_type", "water")
            coordinates = self._context_water_coordinates(water)
            self._draw_context_point(
                painter,
                rect,
                coordinates[0],
                coordinates[1],
                QColor(self.theme.accent_cyan),
                label,
            )

    def _draw_context_road_highlights(self, painter: QPainter, rect: QRectF, context: dict):
        primary = (context.get("roads") or {}).get("primary_access")
        if not primary:
            return

        label = primary.get("name") or "primary road"
        features = self._named_features(("road_major", "road_minor", "track"), primary.get("name"))
        if not features:
            self._draw_context_point(
                painter,
                rect,
                *self._context_offset_coordinates(primary),
                QColor(self.theme.accent_orange),
                label,
                square=True,
            )
            return

        for feature in features:
            self._draw_highlighted_line_feature(painter, rect, feature, QColor(self.theme.accent_orange))

        label_feature = min(features, key=self._feature_distance_to_scenario)
        label_lon, label_lat = self._feature_label_coordinates(label_feature)
        self._draw_context_label(
            painter,
            rect,
            label_lon,
            label_lat,
            QColor(self.theme.accent_orange),
            label,
            square=True,
        )

    def _draw_highlighted_line_feature(self, painter: QPainter, rect: QRectF, feature: VectorFeature, color: QColor):
        if len(feature.geometry) < 2:
            return

        path = QPainterPath()
        first = self._geo_to_screen(*feature.geometry[0], rect)
        path.moveTo(first)
        for lon, lat in feature.geometry[1:]:
            path.lineTo(self._geo_to_screen(lon, lat, rect))

        painter.setPen(QPen(QColor(self.theme.bg_panel), 6.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        painter.setPen(QPen(color, 3.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPath(path)

    def _draw_context_point(
        self,
        painter: QPainter,
        rect: QRectF,
        lon: float,
        lat: float,
        color: QColor,
        label: str,
        square: bool = False,
    ):
        point = self._geo_to_screen(lon, lat, rect)
        painter.setPen(QPen(QColor(self.theme.bg_panel), 2))
        painter.setBrush(color)
        if square:
            painter.drawRoundedRect(QRectF(point.x() - 5, point.y() - 5, 10, 10), 2, 2)
        else:
            painter.drawEllipse(point, 5, 5)
        painter.setFont(app_font(FONT_SIZE_XS))
        painter.setPen(QColor(self.theme.text_primary))
        painter.drawText(QRectF(point.x() + 8, point.y() - 10, 150, 20), Qt.AlignLeft | Qt.AlignVCenter, str(label)[:24])

    def _draw_context_label(
        self,
        painter: QPainter,
        rect: QRectF,
        lon: float,
        lat: float,
        color: QColor,
        label: str,
        square: bool = False,
    ):
        point = self._geo_to_screen(lon, lat, rect)
        painter.setPen(QPen(QColor(self.theme.bg_panel), 2))
        painter.setBrush(color)
        if square:
            painter.drawRoundedRect(QRectF(point.x() - 5, point.y() - 5, 10, 10), 2, 2)
        else:
            painter.drawEllipse(point, 5, 5)
        painter.setFont(app_font(FONT_SIZE_XS, bold=True))
        painter.setPen(QColor(self.theme.text_primary))
        painter.drawText(QRectF(point.x() + 9, point.y() - 11, 170, 22), Qt.AlignLeft | Qt.AlignVCenter, str(label)[:26])

    def _context_water_coordinates(self, water: dict) -> tuple[float, float]:
        name = water.get("name")
        if name:
            feature = self._named_feature("water", name)
            if feature is not None:
                return self._feature_label_coordinates(feature)
        return self._context_offset_coordinates(water)

    def _context_offset_coordinates(self, item: dict) -> tuple[float, float]:
        return self._offset_lon_lat(
            self.loaded_scenario.longitude,
            self.loaded_scenario.latitude,
            item.get("distance_m") or 0,
            item.get("bearing"),
        )

    def _named_feature(self, layer: str, name: str) -> VectorFeature | None:
        features = self._named_features((layer,), name)
        return features[0] if features else None

    def _named_features(self, layers: tuple[str, ...], name: str | None) -> list[VectorFeature]:
        if not name:
            return []
        target = self._normalize_feature_name(name)
        matches = []
        for layer in layers:
            for feature in self._features_by_layer.get(layer, []):
                if feature.name and self._normalize_feature_name(feature.name) == target:
                    matches.append(feature)
        return matches

    @staticmethod
    def _normalize_feature_name(name: str) -> str:
        return " ".join(name.casefold().split())

    @staticmethod
    def _feature_label_coordinates(feature: VectorFeature) -> tuple[float, float]:
        if feature.bounds is not None:
            min_lon, min_lat, max_lon, max_lat = feature.bounds
            return (min_lon + max_lon) / 2, (min_lat + max_lat) / 2
        return _geometry_centroid(feature.geometry)

    def _feature_distance_to_scenario(self, feature: VectorFeature) -> float:
        lon, lat = self._feature_label_coordinates(feature)
        return (lon - self.loaded_scenario.longitude) ** 2 + (lat - self.loaded_scenario.latitude) ** 2

    @staticmethod
    def _detail_map_rect(content: QRectF) -> QRectF:
        return content.adjusted(12, 40, -12, -92)

    def _initialize_detail_view(self, rect: QRectF):
        if self._detail_center_merc is not None and self._detail_scale is not None:
            return

        min_lon, min_lat, max_lon, max_lat = self._initial_detail_bounds()

        min_merc = self._mercator_project(min_lon, min_lat)
        max_merc = self._mercator_project(max_lon, max_lat)
        width = max(abs(max_merc.x() - min_merc.x()), 0.0001)
        height = max(abs(max_merc.y() - min_merc.y()), 0.0001)
        self._detail_center_merc = QPointF(
            (min_merc.x() + max_merc.x()) / 2,
            (min_merc.y() + max_merc.y()) / 2,
        )
        self._detail_scale = min(rect.width() / width, rect.height() / height) * 0.92
        self._enforce_detail_bounds(rect)

    def _enforce_detail_bounds(self, rect: QRectF):
        if self._detail_scale is None:
            return
        self._detail_scale = max(self._detail_scale, self._minimum_detail_scale(rect))
        self._clamp_detail_center(rect)

    def _minimum_detail_scale(self, rect: QRectF) -> float:
        bounds = self._scenario_bounds()
        if bounds is None:
            return 3_000

        min_lon, min_lat, max_lon, max_lat = bounds
        min_merc = self._mercator_project(min_lon, min_lat)
        max_merc = self._mercator_project(max_lon, max_lat)
        width = max(abs(max_merc.x() - min_merc.x()), 0.0001)
        height = max(abs(max_merc.y() - min_merc.y()), 0.0001)
        return max(rect.width() / width, rect.height() / height)

    def _initial_detail_bounds(self) -> tuple[float, float, float, float]:
        context = self._loaded_context or {}
        points = [(self.loaded_scenario.longitude, self.loaded_scenario.latitude)]
        for water in context.get("water_sources", []):
            distance_m = water.get("distance_m")
            bearing = water.get("bearing")
            if not isinstance(distance_m, (int, float)):
                continue
            points.append(
                self._offset_lon_lat(
                    self.loaded_scenario.longitude,
                    self.loaded_scenario.latitude,
                    float(distance_m),
                    bearing,
                )
            )

        if len(points) == 1:
            fallback_radius_m = 4_000
            lat_padding = fallback_radius_m / 111_320
            lon_padding = fallback_radius_m / (
                111_320 * max(math.cos(math.radians(self.loaded_scenario.latitude)), 0.2)
            )
            return (
                self.loaded_scenario.longitude - lon_padding,
                self.loaded_scenario.latitude - lat_padding,
                self.loaded_scenario.longitude + lon_padding,
                self.loaded_scenario.latitude + lat_padding,
            )

        min_lon = min(point[0] for point in points)
        max_lon = max(point[0] for point in points)
        min_lat = min(point[1] for point in points)
        max_lat = max(point[1] for point in points)
        padding_m = 900
        lat_padding = padding_m / 111_320
        lon_padding = padding_m / (
            111_320 * max(math.cos(math.radians(self.loaded_scenario.latitude)), 0.2)
        )
        return (
            min_lon - lon_padding,
            min_lat - lat_padding,
            max_lon + lon_padding,
            max_lat + lat_padding,
        )

    def _clamp_detail_center(self, rect: QRectF):
        if self._detail_center_merc is None or self._detail_scale is None:
            return

        bounds = self._scenario_bounds()
        if bounds is None:
            return

        min_lon, min_lat, max_lon, max_lat = bounds
        min_merc = self._mercator_project(min_lon, min_lat)
        max_merc = self._mercator_project(max_lon, max_lat)
        min_x = min(min_merc.x(), max_merc.x())
        max_x = max(min_merc.x(), max_merc.x())
        min_y = min(min_merc.y(), max_merc.y())
        max_y = max(min_merc.y(), max_merc.y())

        half_width = rect.width() / (2 * self._detail_scale)
        half_height = rect.height() / (2 * self._detail_scale)
        center_x = self._clamp_axis(self._detail_center_merc.x(), min_x + half_width, max_x - half_width, min_x, max_x)
        center_y = self._clamp_axis(self._detail_center_merc.y(), min_y + half_height, max_y - half_height, min_y, max_y)
        self._detail_center_merc = QPointF(center_x, center_y)

    def _scenario_bounds(self) -> tuple[float, float, float, float] | None:
        if self.loaded_scenario is None:
            return None
        return self._bounds_around_scenario(50_000)

    @staticmethod
    def _clamp_axis(value: float, lower: float, upper: float, min_value: float, max_value: float) -> float:
        if lower > upper:
            return (min_value + max_value) / 2
        return max(lower, min(upper, value))

    def _bounds_around_scenario(self, size_m: float) -> tuple[float, float, float, float]:
        half_size_m = size_m / 2
        lat_padding = half_size_m / 111_320
        lon_padding = half_size_m / (
            111_320 * max(math.cos(math.radians(self.loaded_scenario.latitude)), 0.2)
        )
        return (
            self.loaded_scenario.longitude - lon_padding,
            self.loaded_scenario.latitude - lat_padding,
            self.loaded_scenario.longitude + lon_padding,
            self.loaded_scenario.latitude + lat_padding,
        )

    def _geo_to_screen(self, lon: float, lat: float, rect: QRectF) -> QPointF:
        merc = self._mercator_project(lon, lat)
        return QPointF(
            rect.center().x() + (merc.x() - self._detail_center_merc.x()) * self._detail_scale,
            rect.center().y() - (merc.y() - self._detail_center_merc.y()) * self._detail_scale,
        )

    def _screen_to_merc(self, point: QPointF, rect: QRectF) -> QPointF:
        return QPointF(
            self._detail_center_merc.x() + (point.x() - rect.center().x()) / self._detail_scale,
            self._detail_center_merc.y() - (point.y() - rect.center().y()) / self._detail_scale,
        )

    def _screen_to_lon_lat(self, point: QPointF, rect: QRectF) -> tuple[float, float]:
        merc = self._screen_to_merc(point, rect)
        return math.degrees(merc.x()), self._inverse_mercator_y(merc.y())

    @staticmethod
    def _mercator_project(lon: float, lat: float) -> QPointF:
        lat = max(min(lat, 85.0), -85.0)
        return QPointF(math.radians(lon), ScenarioMap._mercator_y(lat))

    @staticmethod
    def _offset_lon_lat(lon: float, lat: float, distance_m: float, bearing: str | None) -> tuple[float, float]:
        bearing_degrees = {
            "N": 0,
            "NE": 45,
            "E": 90,
            "SE": 135,
            "S": 180,
            "SW": 225,
            "W": 270,
            "NW": 315,
        }.get(bearing or "N", 0)
        angle = math.radians(bearing_degrees)
        north_m = math.cos(angle) * distance_m
        east_m = math.sin(angle) * distance_m
        lat_offset = north_m / 111_320
        lon_offset = east_m / (111_320 * max(math.cos(math.radians(lat)), 0.2))
        return lon + lon_offset, lat + lat_offset

    @staticmethod
    def _read_context(scenario: Scenario) -> dict:
        try:
            return json.loads(scenario.context_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _detail_summary(context: dict) -> str:
        assets = context.get("assets_at_risk") or {}
        roads = context.get("roads") or {}
        water_count = len(context.get("water_sources", []))
        building_count = assets.get("buildings_within_radius", 0)
        primary = roads.get("primary_access") or {}
        primary_name = primary.get("name") or "primary access route"
        return f"{water_count} water sources | {building_count} buildings in radius | {primary_name}"

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
        radians = math.radians(lat)
        return math.log(math.tan(math.pi / 4 + radians / 2))

    @staticmethod
    def _inverse_mercator_y(y: float) -> float:
        return math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)
