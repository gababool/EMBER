from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget

try:
    from ui.assets.design import DEFAULT_THEME, Theme
except ImportError:
    from assets.design import DEFAULT_THEME, Theme


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


class ScenarioMapBridge(QObject):
    scenarioClicked = Signal(str)

    @Slot(str)
    def selectScenario(self, scenario_id: str):
        self.scenarioClicked.emit(scenario_id)


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
        self._map_loaded = False

        self.web_view = QWebEngineView(self)
        self.web_view.setContextMenuPolicy(Qt.NoContextMenu)
        self.web_view.loadFinished.connect(self._on_load_finished)

        self.bridge = ScenarioMapBridge(self)
        self.bridge.scenarioClicked.connect(self._select_scenario_by_id)
        self.channel = QWebChannel(self.web_view.page())
        self.channel.registerObject("scenarioBridge", self.bridge)
        self.web_view.page().setWebChannel(self.channel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.web_view)

        self.setMinimumSize(420, 360)
        self._render_map()

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
                self._sync_selected_marker()
                return

    def apply_theme(self, theme: Theme):
        self.theme = theme
        self._render_map()

    def _on_load_finished(self, ok: bool):
        self._map_loaded = ok
        if ok:
            self._sync_selected_marker()

    def _select_scenario_by_id(self, scenario_id: str):
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                self.selected_scenario = scenario
                self._sync_selected_marker()
                self.scenario_selected.emit(scenario)
                return

    def _sync_selected_marker(self):
        if not self._map_loaded or self.selected_scenario is None:
            return
        scenario_id = json.dumps(self.selected_scenario.scenario_id)
        self.web_view.page().runJavaScript(f"selectScenario({scenario_id});")

    def _render_map(self):
        self._map_loaded = False
        self.web_view.setHtml(self._html(), QUrl("https://www.openstreetmap.org/"))

    def _html(self) -> str:
        scenarios = [
            {
                "id": scenario.scenario_id,
                "label": scenario.marker_label,
                "title": scenario.title,
                "description": scenario.description,
                "region": scenario.region,
                "lat": scenario.latitude,
                "lon": scenario.longitude,
            }
            for scenario in self.scenarios
        ]
        selected_id = self.selected_scenario.scenario_id if self.selected_scenario else ""
        payload = json.dumps(scenarios)
        selected = json.dumps(selected_id)
        theme = {
            "bg_main": self.theme.bg_main,
            "bg_panel": self.theme.bg_panel,
            "bg_panel_alt": self.theme.bg_panel_alt,
            "border": self.theme.border,
            "text_primary": self.theme.text_primary,
            "text_muted": self.theme.text_muted,
            "accent_orange": self.theme.accent_orange,
            "accent_cyan": self.theme.accent_cyan,
            "shadow": self.theme.shadow,
        }
        theme_payload = json.dumps(theme)
        return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIINfQPDND6yLkPZIFwHGVxfiT4hE7hL6K8="
    crossorigin=""
  />
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin="">
  </script>
  <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
  <style>
    .leaflet-pane,
    .leaflet-tile,
    .leaflet-marker-icon,
    .leaflet-marker-shadow,
    .leaflet-tile-container,
    .leaflet-pane > svg,
    .leaflet-pane > canvas,
    .leaflet-zoom-box,
    .leaflet-image-layer,
    .leaflet-layer {{
      position: absolute;
      left: 0;
      top: 0;
    }}
    .leaflet-container {{
      overflow: hidden;
      -webkit-tap-highlight-color: transparent;
    }}
    .leaflet-tile,
    .leaflet-marker-icon,
    .leaflet-marker-shadow {{
      user-select: none;
      -webkit-user-drag: none;
    }}
    .leaflet-tile {{
      filter: inherit;
      visibility: hidden;
    }}
    .leaflet-tile-loaded {{
      visibility: inherit;
    }}
    .leaflet-zoom-box {{
      width: 0;
      height: 0;
      box-sizing: border-box;
      z-index: 800;
    }}
    .leaflet-overlay-pane svg {{
      user-select: none;
    }}
    .leaflet-control {{
      position: relative;
      z-index: 800;
      pointer-events: auto;
    }}
    .leaflet-top,
    .leaflet-bottom {{
      position: absolute;
      z-index: 1000;
      pointer-events: none;
    }}
    .leaflet-top {{
      top: 0;
    }}
    .leaflet-right {{
      right: 0;
    }}
    .leaflet-bottom {{
      bottom: 0;
    }}
    .leaflet-left {{
      left: 0;
    }}
    .leaflet-control {{
      float: left;
      clear: both;
    }}
    .leaflet-right .leaflet-control {{
      float: right;
    }}
    .leaflet-top .leaflet-control {{
      margin-top: 10px;
    }}
    .leaflet-bottom .leaflet-control {{
      margin-bottom: 10px;
    }}
    .leaflet-left .leaflet-control {{
      margin-left: 10px;
    }}
    .leaflet-right .leaflet-control {{
      margin-right: 10px;
    }}
    .leaflet-control-zoom {{
      border: 1px solid {self.theme.border};
      border-radius: 5px;
      overflow: hidden;
    }}
    .leaflet-control-zoom a {{
      display: block;
      width: 28px;
      height: 28px;
      line-height: 28px;
      text-align: center;
      text-decoration: none;
      font-weight: 700;
    }}
    .leaflet-control-attribution {{
      padding: 3px 6px;
      margin: 0;
    }}
    .leaflet-tooltip {{
      position: absolute;
      padding: 6px 8px;
      border-radius: 4px;
      border: 1px solid {self.theme.border};
      background: {self.theme.bg_panel};
      color: {self.theme.text_primary};
      font-size: 11px;
      white-space: nowrap;
      pointer-events: none;
      box-shadow: 0 3px 14px rgba(0, 0, 0, 0.25);
    }}
    html, body, #map {{
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: {self.theme.bg_panel_alt};
      font-family: "JetBrains Mono", monospace;
    }}
    .leaflet-container {{
      background: {self.theme.bg_panel_alt};
      color: {self.theme.text_primary};
    }}
    .leaflet-control-zoom a {{
      background: {self.theme.bg_panel};
      color: {self.theme.text_primary};
      border-color: {self.theme.border};
    }}
    .leaflet-control-attribution {{
      background: rgba(255, 255, 255, 0.78);
      font-size: 10px;
    }}
    .scenario-marker {{
      width: 28px;
      height: 28px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #061018;
      background: {self.theme.accent_cyan};
      border: 3px solid {self.theme.bg_panel};
      box-shadow: 0 2px 12px rgba(0, 0, 0, 0.38);
      font: 700 13px "JetBrains Mono", monospace;
      cursor: pointer;
    }}
    .scenario-marker.selected {{
      background: {self.theme.accent_orange};
      color: #ffffff;
      transform: scale(1.16);
      box-shadow: 0 0 0 4px rgba(255, 140, 43, 0.25), 0 3px 16px rgba(0, 0, 0, 0.45);
    }}
    .scenario-card {{
      position: absolute;
      left: 14px;
      right: 14px;
      bottom: 14px;
      z-index: 500;
      padding: 12px 14px;
      border-radius: 5px;
      border: 1px solid {self.theme.accent_orange};
      background: {self.theme.bg_panel};
      color: {self.theme.text_primary};
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.32);
      pointer-events: none;
    }}
    .scenario-card-title {{
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 5px;
    }}
    .scenario-card-detail {{
      color: {self.theme.text_muted};
      font-size: 11px;
      line-height: 1.35;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="scenario-card" id="scenario-card"></div>
  <script>
    const scenarios = {payload};
    const selectedInitial = {selected};
    const theme = {theme_payload};
    let scenarioBridge = null;
    const markers = new Map();

    new QWebChannel(qt.webChannelTransport, function(channel) {{
      scenarioBridge = channel.objects.scenarioBridge;
    }});

    const map = L.map("map", {{
      zoomControl: true,
      preferCanvas: false
    }});

    L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 18,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }}).addTo(map);

    const bounds = [];

    function markerHtml(scenario, selected) {{
      const cls = selected ? "scenario-marker selected" : "scenario-marker";
      return `<div class="${{cls}}">${{scenario.label}}</div>`;
    }}

    function setCard(scenario) {{
      const card = document.getElementById("scenario-card");
      if (!scenario) {{
        card.style.display = "none";
        return;
      }}
      card.style.display = "block";
      card.innerHTML = `
        <div class="scenario-card-title">${{scenario.title}}</div>
        <div class="scenario-card-detail">${{scenario.region}} | ${{scenario.description}}</div>
      `;
    }}

    function selectScenario(id) {{
      let selectedScenario = null;
      for (const scenario of scenarios) {{
        const marker = markers.get(scenario.id);
        const selected = scenario.id === id;
        marker.setIcon(L.divIcon({{
          html: markerHtml(scenario, selected),
          className: "",
          iconSize: [34, 34],
          iconAnchor: [17, 17]
        }}));
        if (selected) selectedScenario = scenario;
      }}
      setCard(selectedScenario);
    }}

    for (const scenario of scenarios) {{
      const marker = L.marker([scenario.lat, scenario.lon], {{
        icon: L.divIcon({{
          html: markerHtml(scenario, scenario.id === selectedInitial),
          className: "",
          iconSize: [34, 34],
          iconAnchor: [17, 17]
        }})
      }}).addTo(map);
      marker.bindTooltip(`${{scenario.title}}<br>${{scenario.region}}`, {{
        direction: "top",
        opacity: 0.95
      }});
      marker.on("click", function() {{
        selectScenario(scenario.id);
        if (scenarioBridge) scenarioBridge.selectScenario(scenario.id);
      }});
      markers.set(scenario.id, marker);
      bounds.push([scenario.lat, scenario.lon]);
    }}

    if (bounds.length > 0) {{
      map.fitBounds(bounds, {{ padding: [42, 42], maxZoom: 6 }});
      selectScenario(selectedInitial || scenarios[0].id);
    }} else {{
      map.setView([62.0, 15.0], 5);
      setCard(null);
    }}

    window.selectScenario = selectScenario;
  </script>
</body>
</html>
"""
