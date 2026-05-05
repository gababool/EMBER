"""GIS extraction for structured operational wildfire context.

Reads a local OpenStreetMap .pbf file and extracts geographic features
for a given coordinate pair, producing a fully populated OperationalContext.

All spatial operations are performed offline against the pre-loaded .pbf.
No network access is required at runtime.

Usage (CLI):
    uv run python -m pipeline.context.extract \\
        --lat 59.8 --lon 16.1 \\
        --output data/contexts/scenario_01.json

    uv run python -m pipeline.context.extract \\
        --lat 59.8 --lon 16.1 \\
        --pbf data/gis/osm/vastmanland-50km.osm.pbf \\
        --output data/contexts/scenario_01.json

Coordinate projection:
    OSM data is stored in WGS84 (lat/lon degrees). All distance and area
    calculations are performed in SWEREF99 TM (EPSG:3006), the Swedish
    national projected coordinate system where 1 unit = 1 metre.

Region lookup:
    Swedish municipality and county boundaries are stored as multipolygon
    relations in OSM. Geofabrik extracts clip member ways at the extract
    boundary, preventing osmium from assembling complete area geometries.
    Region lookup instead uses a two-pass approach: collect relation names
    and member way IDs in pass 1, collect way node coordinates in pass 2,
    then build convex hull polygons per relation for point-in-polygon lookup.
    Convex hulls are a deliberate approximation sufficient for municipality-
    level containment checks.

Pre-computation approach:
    The .pbf is loaded once per process into in-memory GeoDataFrames.
    Cached GeoDataFrames are reused for subsequent calls. At inference time,
    scenario JSONs are loaded directly without touching the .pbf.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmium
from shapely.geometry import MultiPoint, Point

from pipeline.context.schemas import (
    AssetsAtRisk,
    CompassBearing,
    Coordinates,
    ExtractionMetadata,
    LandCoverType,
    NamedFeature,
    NamedFeatureType,
    OperationalContext,
    Region,
    Road,
    RoadClass,
    Roads,
    Settlement,
    SettlementType,
    WaterSource,
    WaterSourceType,
    WaterSupplyCategory,
)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_PBF = Path("data/gis/osm/vastmanland-50km.osm.pbf")

SWEREF99_TM = "EPSG:3006"
WGS84 = "EPSG:4326"

# Search radii (metres).
SETTLEMENT_RADIUS_M = 20_000
WATER_RADIUS_M = 10_000
ROAD_RADIUS_M = 5_000
NAMED_FEATURE_RADIUS_M = 10_000
ASSETS_RADIUS_M = 2_000

# Maximum results per category.
MAX_SETTLEMENTS = 5
MAX_WATER_SOURCES = 5
MAX_TRACKS = 3
MAX_NAMED_FEATURES = 5

# Water supply classification threshold.
# Lakes and reservoirs above this area are classified as heavy supply.
HEAVY_SUPPLY_AREA_M2 = 10_000

PAVED_ROAD_CLASSES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "service",
}

SALTWATER_WATER_TAGS = {"sea", "ocean", "bay"}

WATER_TYPE_MAP: dict[str, WaterSourceType] = {
    "lake": "lake",
    "reservoir": "reservoir",
    "pond": "pond",
    "basin": "reservoir",
    "wetland": "wetland",
    "river": "river",
    "stream": "stream",
    "canal": "river",
    "ditch": "stream",
}

SETTLEMENT_TYPE_MAP: dict[str, SettlementType] = {
    "city": "city",
    "town": "town",
    "village": "village",
    "hamlet": "hamlet",
    "suburb": "suburb",
    "isolated_dwelling": "isolated_dwelling",
    "farm": "farm",
}

NAMED_FEATURE_TYPE_MAP: dict[str, NamedFeatureType] = {
    "peak": "peak",
    "ridge": "ridge",
    "valley": "valley",
    "wood": "forest",
    "island": "island",
}

PERMANENT_BUILDING_TAGS = {
    "house", "residential", "apartments", "detached", "terrace",
    "semidetached_house", "bungalow",
}

# ── Module-level cache ────────────────────────────────────────────────────────

_cache: dict[str, Any] = {}
_pbf_path: Path | None = None


# ── OSM handlers ──────────────────────────────────────────────────────────────

class _WaterHandler(osmium.SimpleHandler):
    """Collect freshwater polygon and line features."""

    def __init__(self):
        super().__init__()
        self.features: list[dict] = []
        self._wkb = osmium.geom.WKBFactory()

    def area(self, a):
        tags = dict(a.tags)
        water_tag = tags.get("water", "")
        natural_tag = tags.get("natural", "")
        waterway_tag = tags.get("waterway", "")

        is_water = natural_tag == "water" or waterway_tag in ("riverbank",)
        if not is_water:
            return
        if water_tag in SALTWATER_WATER_TAGS:
            return
        if tags.get("place") == "sea":
            return

        try:
            wkb = self._wkb.create_multipolygon(a)
            geom = _load_wkb(wkb)
        except Exception:
            return

        source_type = WATER_TYPE_MAP.get(water_tag, "lake")
        self.features.append({
            "name": tags.get("name"),
            "source_type": source_type,
            "geometry": geom,
            "is_polygon": True,
        })

    def way(self, w):
        tags = dict(w.tags)
        waterway_tag = tags.get("waterway", "")
        if waterway_tag not in ("river", "stream", "canal", "ditch"):
            return
        try:
            wkb = self._wkb.create_linestring(w)
            geom = _load_wkb(wkb)
        except Exception:
            return

        source_type = WATER_TYPE_MAP.get(waterway_tag, "stream")
        self.features.append({
            "name": tags.get("name"),
            "source_type": source_type,
            "geometry": geom,
            "is_polygon": False,
        })


class _RoadHandler(osmium.SimpleHandler):
    """Collect road and track features."""

    def __init__(self):
        super().__init__()
        self.features: list[dict] = []
        self._wkb = osmium.geom.WKBFactory()

    def way(self, w):
        tags = dict(w.tags)
        highway = tags.get("highway", "")
        if not highway:
            return
        if highway in ("footway", "cycleway", "steps", "pedestrian",
                       "bridleway", "corridor", "elevator", "construction"):
            return
        try:
            wkb = self._wkb.create_linestring(w)
            geom = _load_wkb(wkb)
        except Exception:
            return

        surface = tags.get("surface")
        tracktype = tags.get("tracktype")
        road_class = _map_road_class(highway)
        accessible = _is_vehicle_accessible(road_class, surface, tracktype)

        # Parse maxweight tag — may be "7.5", "30", or absent.
        max_weight: float | None = None
        raw_weight = tags.get("maxweight")
        if raw_weight:
            try:
                max_weight = float(raw_weight)
            except ValueError:
                pass

        has_bridge = tags.get("bridge", "") == "yes"

        self.features.append({
            "name": tags.get("name") or tags.get("ref"),
            "road_class": road_class,
            "surface": surface,
            "tracktype": tracktype,
            "vehicle_accessible": accessible,
            "max_weight_tonnes": max_weight,
            "has_bridge": has_bridge,
            "geometry": geom,
        })


class _SettlementHandler(osmium.SimpleHandler):
    """Collect populated place nodes."""

    def __init__(self):
        super().__init__()
        self.features: list[dict] = []
        self._wkb = osmium.geom.WKBFactory()

    def node(self, n):
        tags = dict(n.tags)
        place = tags.get("place", "")
        if place not in SETTLEMENT_TYPE_MAP:
            return
        try:
            wkb = self._wkb.create_point(n)
            geom = _load_wkb(wkb)
        except Exception:
            return

        self.features.append({
            "name": tags.get("name"),
            "settlement_type": SETTLEMENT_TYPE_MAP[place],
            "geometry": geom,
        })


class _LandCoverHandler(osmium.SimpleHandler):
    """Collect landuse and natural area polygons for land-cover lookup."""

    def __init__(self):
        super().__init__()
        self.features: list[dict] = []
        self._wkb = osmium.geom.WKBFactory()

    def area(self, a):
        tags = dict(a.tags)
        landuse = tags.get("landuse", "")
        natural = tags.get("natural", "")

        cover = _classify_land_cover(landuse, natural, tags)
        if cover is None:
            return

        try:
            wkb = self._wkb.create_multipolygon(a)
            geom = _load_wkb(wkb)
        except Exception:
            return

        self.features.append({
            "land_cover": cover,
            "geometry": geom,
            "area_m2": None,  # computed after projection
        })


class _NamedFeatureHandler(osmium.SimpleHandler):
    """Collect named natural features (peaks, ridges, valleys, etc.)."""

    def __init__(self):
        super().__init__()
        self.features: list[dict] = []
        self._wkb = osmium.geom.WKBFactory()

    def node(self, n):
        tags = dict(n.tags)
        natural = tags.get("natural", "")
        name = tags.get("name")
        if not name or natural not in NAMED_FEATURE_TYPE_MAP:
            return
        try:
            wkb = self._wkb.create_point(n)
            geom = _load_wkb(wkb)
        except Exception:
            return

        self.features.append({
            "name": name,
            "feature_type": NAMED_FEATURE_TYPE_MAP[natural],
            "geometry": geom,
        })

    def area(self, a):
        tags = dict(a.tags)
        natural = tags.get("natural", "")
        name = tags.get("name")
        if not name or natural not in ("wood",):
            return
        try:
            wkb = self._wkb.create_multipolygon(a)
            geom = _load_wkb(wkb)
        except Exception:
            return

        self.features.append({
            "name": name,
            "feature_type": "forest",
            "geometry": geom,
        })


class _AssetsHandler(osmium.SimpleHandler):
    """Collect buildings, power lines, and protected areas."""

    def __init__(self):
        super().__init__()
        self.buildings: list[dict] = []
        self.power_lines: list[dict] = []
        self.protected_areas: list[dict] = []
        self._wkb = osmium.geom.WKBFactory()

    def node(self, n):
        tags = dict(n.tags)
        if tags.get("power") == "tower":
            try:
                wkb = self._wkb.create_point(n)
                self.power_lines.append({"geometry": _load_wkb(wkb)})
            except Exception:
                pass

    def way(self, w):
        tags = dict(w.tags)
        if tags.get("power") == "line":
            try:
                wkb = self._wkb.create_linestring(w)
                self.power_lines.append({"geometry": _load_wkb(wkb)})
            except Exception:
                pass

    def area(self, a):
        tags = dict(a.tags)

        # Buildings.
        building = tags.get("building", "")
        if building and building != "no":
            try:
                wkb = self._wkb.create_multipolygon(a)
                self.buildings.append({
                    "building": building,
                    "geometry": _load_wkb(wkb),
                })
            except Exception:
                pass
            return

        # Protected areas.
        is_reserve = (
            tags.get("leisure") == "nature_reserve"
            or tags.get("boundary") == "protected_area"
        )
        if is_reserve:
            name = tags.get("name")
            try:
                wkb = self._wkb.create_multipolygon(a)
                self.protected_areas.append({
                    "name": name,
                    "geometry": _load_wkb(wkb),
                })
            except Exception:
                pass


# ── Region handlers (two-pass, no area assembly) ───────────────────────────────

class _RelationCollector(osmium.SimpleHandler):
    """Pass 1: collect admin relation metadata and member way IDs."""

    def __init__(self):
        super().__init__()
        self.relations: dict[int, dict] = {}

    def relation(self, r):
        tags = dict(r.tags)
        if tags.get("boundary") != "administrative":
            return
        level = tags.get("admin_level", "")
        if level not in ("4", "7"):
            return
        name = tags.get("name")
        if not name:
            return
        way_ids = {m.ref for m in r.members if m.type == "w"}
        self.relations[r.id] = {
            "name": name,
            "admin_level": level,
            "way_ids": way_ids,
        }


class _BoundaryWayCollector(osmium.SimpleHandler):
    """Pass 2: collect node coordinates for boundary member ways."""

    def __init__(self, target_way_ids: set[int]):
        super().__init__()
        self.target_way_ids = target_way_ids
        self.way_nodes: dict[int, list[tuple[float, float]]] = {}

    def way(self, w):
        if w.id not in self.target_way_ids:
            return
        try:
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
        except Exception:
            return
        if len(coords) >= 2:
            self.way_nodes[w.id] = coords


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _load_wkb(wkb_hex: str):
    from shapely import wkb as shapely_wkb
    return shapely_wkb.loads(wkb_hex, hex=True)


def _make_point_gdf(lat: float, lon: float) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame({"geometry": [Point(lon, lat)]}, crs=WGS84)
    return gdf.to_crs(SWEREF99_TM)


def _to_gdf(features: list[dict], crs: str = WGS84) -> gpd.GeoDataFrame:
    if not features:
        return gpd.GeoDataFrame(columns=["geometry"], crs=SWEREF99_TM)
    gdf = gpd.GeoDataFrame(features, crs=crs)
    return gdf.to_crs(SWEREF99_TM)


def _build_region_polygons(
    relations: dict,
    way_nodes: dict,
) -> list[dict]:
    """Build convex hull polygons for admin boundaries from way node coordinates.

    Geofabrik extracts clip boundary ways at the extract edge, so complete
    closed rings cannot always be formed. Convex hulls from all member way
    nodes are used instead — a deliberate approximation sufficient for
    point-in-polygon checks at municipality or county scale.
    """
    features = []
    for rel_id, rel in relations.items():
        all_coords: list[tuple[float, float]] = []
        for way_id in rel["way_ids"]:
            all_coords.extend(way_nodes.get(way_id, []))

        if len(all_coords) < 3:
            continue

        try:
            hull = MultiPoint(all_coords).convex_hull
            if hull.is_empty or hull.geom_type not in ("Polygon", "MultiPolygon"):
                continue
        except Exception:
            continue

        features.append({
            "name": rel["name"],
            "admin_level": rel["admin_level"],
            "geometry": hull,
        })

    return features


# ── Utility functions ─────────────────────────────────────────────────────────

def _degrees_to_compass(degrees: float) -> CompassBearing:
    directions: list[CompassBearing] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return directions[round(degrees / 45) % 8]


def _compute_bearing(from_geom, to_geom) -> CompassBearing:
    """Compute compass bearing from one projected geometry to another.

    Geometries must be in SWEREF99 TM so coordinate differences correspond
    to north/east offsets in metres.
    """
    from_pt = from_geom.centroid if from_geom.geom_type != "Point" else from_geom
    to_pt = to_geom.centroid if to_geom.geom_type != "Point" else to_geom
    dx = to_pt.x - from_pt.x
    dy = to_pt.y - from_pt.y
    return _degrees_to_compass(math.degrees(math.atan2(dx, dy)) % 360)


def _map_road_class(highway: str) -> RoadClass:
    known: set[RoadClass] = {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "service", "track", "path",
    }
    return highway if highway in known else "unknown"  # type: ignore[return-value]


def _is_vehicle_accessible(
    road_class: RoadClass,
    surface: str | None,
    tracktype: str | None,
) -> bool:
    """Derive vehicle accessibility for a fire truck from OSM tags.

    Paved road classes are always considered accessible. Tracks are assessed
    via tracktype grade or surface material. When neither is available the
    conservative assumption is that the track is not accessible, since we
    cannot confirm passability without evidence.
    """
    if road_class in PAVED_ROAD_CLASSES:
        return True
    if road_class == "track":
        if tracktype in ("grade1", "grade2"):
            return True
        if tracktype in ("grade3", "grade4", "grade5"):
            return False
        solid_surfaces = {
            "asphalt", "concrete", "paved", "compacted", "gravel",
            "fine_gravel", "paving_stones",
        }
        if surface and surface in solid_surfaces:
            return True
        return False
    return False


def _classify_land_cover(
    landuse: str,
    natural: str,
    tags: dict,
) -> LandCoverType | None:
    if natural == "water":
        return "water"
    if natural == "wetland" or landuse == "wetland":
        return "wetland"
    if natural in ("wood",) or landuse == "forest":
        leaf_type = tags.get("leaf_type", "")
        if leaf_type == "needleleaved":
            return "coniferous_forest"
        if leaf_type == "broadleaved":
            return "broadleaf_forest"
        if leaf_type == "mixed":
            return "mixed_forest"
        return "forest"
    if natural == "scrub":
        return "shrubland"
    if natural in ("grassland", "heath"):
        return "grassland"
    if natural == "bare_rock" or landuse == "quarry":
        return "bare_ground"
    if natural in ("glacier", "snow"):
        return "snow_or_ice"
    if landuse in ("farmland", "meadow", "orchard", "vineyard", "allotments"):
        return "agricultural"
    if landuse in ("residential", "commercial", "industrial", "retail",
                   "construction", "military"):
        return "urban"
    if landuse == "grass" or natural == "grassland":
        return "grassland"
    return None


def _classify_water_supply(
    source_type: WaterSourceType,
    area_m2: float | None,
) -> WaterSupplyCategory:
    """Classify a water source as heavy or light supply.

    Heavy supply (large lakes, reservoirs, rivers) is suitable for tanker
    filling and sustained pumping. Light supply (small ponds, streams, ditches,
    wetlands) is suitable for portable pumps only. The 10 000 m² threshold for
    polygon bodies was chosen to distinguish operationally meaningful lake-sized
    sources from small ponds and drainage features.
    """
    if source_type in ("river", "reservoir"):
        return "heavy"
    if source_type in ("stream", "wetland"):
        return "light"
    if source_type in ("lake", "pond"):
        if area_m2 is not None:
            return "heavy" if area_m2 >= HEAVY_SUPPLY_AREA_M2 else "light"
        return "unknown"
    return "unknown"


def _nan_to_none(val):
    """Convert a pandas NaN to None for optional Pydantic string fields."""
    if val is None:
        return None
    try:
        if math.isnan(float(val)):
            return None
    except (TypeError, ValueError):
        pass
    return val


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_pbf(pbf_path: Path) -> None:
    """Load all feature categories from the .pbf into module-level cache."""
    global _cache, _pbf_path

    pbf_str = str(pbf_path)
    print(f"Loading OSM data from {pbf_path} (this may take a moment)...")

    water_handler = _WaterHandler()
    water_handler.apply_file(pbf_str, locations=True, idx="flex_mem")

    road_handler = _RoadHandler()
    road_handler.apply_file(pbf_str, locations=True, idx="flex_mem")

    settlement_handler = _SettlementHandler()
    settlement_handler.apply_file(pbf_str, locations=True, idx="flex_mem")

    land_cover_handler = _LandCoverHandler()
    land_cover_handler.apply_file(pbf_str, locations=True, idx="flex_mem")

    named_feature_handler = _NamedFeatureHandler()
    named_feature_handler.apply_file(pbf_str, locations=True, idx="flex_mem")

    assets_handler = _AssetsHandler()
    assets_handler.apply_file(pbf_str, locations=True, idx="flex_mem")

    # Region loading: two-pass without area assembly.
    relation_collector = _RelationCollector()
    relation_collector.apply_file(pbf_str)

    all_way_ids: set[int] = set()
    for rel in relation_collector.relations.values():
        all_way_ids.update(rel["way_ids"])

    way_collector = _BoundaryWayCollector(all_way_ids)
    way_collector.apply_file(pbf_str, locations=True, idx="flex_mem")

    region_features = _build_region_polygons(
        relation_collector.relations,
        way_collector.way_nodes,
    )

    # Compute land cover polygon areas after projection (needed for wetland
    # area threshold used in land cover priority logic).
    land_cover_gdf = _to_gdf(land_cover_handler.features)
    if not land_cover_gdf.empty and "area_m2" in land_cover_gdf.columns:
        land_cover_gdf["area_m2"] = land_cover_gdf.geometry.area

    _cache = {
        "water": _to_gdf(water_handler.features),
        "roads": _to_gdf(road_handler.features),
        "settlements": _to_gdf(settlement_handler.features),
        "land_cover": land_cover_gdf,
        "regions": _to_gdf(region_features),
        "named_features": _to_gdf(named_feature_handler.features),
        "buildings": _to_gdf(assets_handler.buildings),
        "power_lines": _to_gdf(assets_handler.power_lines),
        "protected_areas": _to_gdf(assets_handler.protected_areas),
    }
    _pbf_path = pbf_path
    print("OSM data loaded.")


def _ensure_loaded(pbf_path: Path) -> None:
    if not _cache or _pbf_path != pbf_path:
        _load_pbf(pbf_path)


# ── Query functions ───────────────────────────────────────────────────────────

def _query_region(point_gdf: gpd.GeoDataFrame) -> Region:
    """Look up municipality and county containing the observation point.

    Uses convex hull approximations of admin boundaries. Returns both
    municipality name and county (admin_area) when both are found.
    """
    regions_gdf = _cache["regions"]
    if regions_gdf.empty:
        return Region(name="Unknown region")

    point = point_gdf.geometry.iloc[0]
    containing = regions_gdf[regions_gdf.geometry.contains(point)]

    municipality = None
    county = None
    for _, row in containing.iterrows():
        if row["admin_level"] == "7" and not municipality:
            municipality = row["name"]
        if row["admin_level"] == "4" and not county:
            county = row["name"]

    # Always populate admin_area with county regardless of whether
    # municipality was found. Previously admin_area was set to None
    # when no municipality was found, losing the county name.
    name = municipality or county or "Unknown region"
    return Region(name=name, admin_area=county)


def _query_land_cover(point_gdf: gpd.GeoDataFrame) -> LandCoverType:
    """Determine the dominant land-cover class at the observation point.

    When multiple land-cover polygons overlap the point, the highest-priority
    category wins. Within the same priority tier, the largest polygon wins.
    This prevents small wetland polygons from overriding a large forest polygon
    at the same location — a common artefact in Swedish OSM data where drainage
    wetlands are mapped inside larger forest areas.
    """
    land_cover_gdf = _cache["land_cover"]
    if land_cover_gdf.empty:
        return "unknown"

    point = point_gdf.geometry.iloc[0]
    containing = land_cover_gdf[land_cover_gdf.geometry.contains(point)].copy()

    if containing.empty:
        return "unknown"

    priority: dict[LandCoverType, int] = {
        "water": 0,
        "urban": 1,
        "snow_or_ice": 2,
        "coniferous_forest": 3,
        "broadleaf_forest": 3,
        "mixed_forest": 3,
        "forest": 4,
        "wetland": 5,
        "shrubland": 6,
        "agricultural": 7,
        "grassland": 8,
        "bare_ground": 9,
        "unknown": 10,
    }

    containing["_priority"] = containing["land_cover"].map(
        lambda c: priority.get(c, 99)
    )

    # Compute area if not already available (fallback for missing column).
    if "area_m2" not in containing.columns or containing["area_m2"].isna().all():
        containing["area_m2"] = containing.geometry.area

    # Sort by priority first, then by area descending within the same tier.
    containing = containing.sort_values(
        ["_priority", "area_m2"], ascending=[True, False]
    )

    return containing.iloc[0]["land_cover"]


def _query_water_sources(
    point_gdf: gpd.GeoDataFrame,
    radius_m: float,
) -> list[WaterSource]:
    """Find nearby freshwater sources within radius_m.

    Sorted by a combined score that weights distance and road accessibility.
    A water source that is slightly further away but has better road access
    ranks higher than a closer but isolated source. Score = distance_m +
    0.5 * nearest_road_distance_m, where road distance defaults to the
    search radius when no road is found (effectively penalising inaccessibility).
    """
    water_gdf = _cache["water"]
    roads_gdf = _cache["roads"]
    if water_gdf.empty:
        return []

    point = point_gdf.geometry.iloc[0]

    water_gdf = water_gdf.copy()
    water_gdf["distance_m"] = water_gdf.geometry.distance(point)
    nearby = water_gdf[water_gdf["distance_m"] <= radius_m].copy()

    if nearby.empty:
        return []

    accessible_roads = (
        roads_gdf[roads_gdf["vehicle_accessible"] == True].copy()  # noqa: E712
        if not roads_gdf.empty else gpd.GeoDataFrame()
    )

    results_raw = []
    for _, row in nearby.iterrows():
        geom = row["geometry"]
        distance_m = float(row["distance_m"])

        area_m2: float | None = None
        if row.get("is_polygon", False):
            area_m2 = float(geom.area)

        nearest_road_dist: float | None = None
        if not accessible_roads.empty:
            dists = accessible_roads.geometry.distance(geom.centroid)
            if not dists.empty:
                nearest_road_dist = float(dists.min())

        supply_category = _classify_water_supply(row["source_type"], area_m2)

        # Operational relevance score: lower is better.
        road_penalty = nearest_road_dist if nearest_road_dist is not None else radius_m
        score = distance_m + 0.5 * road_penalty

        results_raw.append((score, WaterSource(
            name=_nan_to_none(row.get("name")),
            source_type=row["source_type"],
            supply_category=supply_category,
            distance_m=distance_m,
            bearing=_compute_bearing(point, geom.centroid),
            area_m2=area_m2,
            nearest_road_distance_m=nearest_road_dist,
        )))

    results_raw.sort(key=lambda x: x[0])
    return [ws for _, ws in results_raw[:MAX_WATER_SOURCES]]


def _query_roads(
    point_gdf: gpd.GeoDataFrame,
    radius_m: float,
) -> Roads:
    """Find roads within radius_m of the observation point."""
    roads_gdf = _cache["roads"]
    if roads_gdf.empty:
        return Roads()

    point = point_gdf.geometry.iloc[0]

    roads_gdf = roads_gdf.copy()
    roads_gdf["distance_m"] = roads_gdf.geometry.distance(point)
    nearby = roads_gdf[roads_gdf["distance_m"] <= radius_m].copy()

    if nearby.empty:
        return Roads()

    paved = nearby[
        (nearby["vehicle_accessible"] == True) &  # noqa: E712
        (nearby["road_class"] != "track")
    ].sort_values("distance_m")

    primary_access: Road | None = None
    if not paved.empty:
        row = paved.iloc[0]
        primary_access = Road(
            name=_nan_to_none(row.get("name")),
            road_class=row["road_class"],
            surface=_nan_to_none(row.get("surface")),
            tracktype=_nan_to_none(row.get("tracktype")),
            vehicle_accessible=bool(row["vehicle_accessible"]),
            max_weight_tonnes=_nan_to_none(row.get("max_weight_tonnes")),
            has_bridge=bool(row.get("has_bridge", False)),
            distance_m=float(row["distance_m"]),
            bearing=_compute_bearing(point, row["geometry"].centroid),
        )

    tracks = nearby[nearby["road_class"] == "track"].sort_values("distance_m").head(MAX_TRACKS)
    nearby_tracks: list[Road] = []
    for _, row in tracks.iterrows():
        nearby_tracks.append(Road(
            name=_nan_to_none(row.get("name")),
            road_class="track",
            surface=_nan_to_none(row.get("surface")),
            tracktype=_nan_to_none(row.get("tracktype")),
            vehicle_accessible=bool(row["vehicle_accessible"]),
            max_weight_tonnes=_nan_to_none(row.get("max_weight_tonnes")),
            has_bridge=bool(row.get("has_bridge", False)),
            distance_m=float(row["distance_m"]),
            bearing=_compute_bearing(point, row["geometry"].centroid),
        ))

    return Roads(primary_access=primary_access, nearby_tracks=nearby_tracks)


def _query_settlements(
    point_gdf: gpd.GeoDataFrame,
    radius_m: float,
) -> list[Settlement]:
    """Find nearby settlements within radius_m, sorted by distance."""
    settlements_gdf = _cache["settlements"]
    if settlements_gdf.empty:
        return []

    point = point_gdf.geometry.iloc[0]

    settlements_gdf = settlements_gdf.copy()
    settlements_gdf["distance_m"] = settlements_gdf.geometry.distance(point)
    nearby = settlements_gdf[
        settlements_gdf["distance_m"] <= radius_m
    ].sort_values("distance_m").head(MAX_SETTLEMENTS)

    results: list[Settlement] = []
    for _, row in nearby.iterrows():
        results.append(Settlement(
            name=_nan_to_none(row.get("name")),
            settlement_type=row["settlement_type"],
            distance_m=float(row["distance_m"]),
            bearing=_compute_bearing(point, row["geometry"]),
        ))
    return results


def _query_named_features(
    point_gdf: gpd.GeoDataFrame,
    radius_m: float,
) -> list[NamedFeature]:
    """Find nearby named natural features within radius_m, sorted by distance."""
    features_gdf = _cache["named_features"]
    if features_gdf.empty:
        return []

    point = point_gdf.geometry.iloc[0]

    features_gdf = features_gdf.copy()
    features_gdf["distance_m"] = features_gdf.geometry.distance(point)
    nearby = features_gdf[
        features_gdf["distance_m"] <= radius_m
    ].sort_values("distance_m").head(MAX_NAMED_FEATURES)

    results: list[NamedFeature] = []
    for _, row in nearby.iterrows():
        name = row.get("name")
        if not name:
            continue
        geom = row["geometry"]
        centroid = geom.centroid if geom.geom_type != "Point" else geom
        results.append(NamedFeature(
            name=name,
            feature_type=row["feature_type"],
            distance_m=float(row["distance_m"]),
            bearing=_compute_bearing(point, centroid),
        ))
    return results


def _query_assets_at_risk(
    point_gdf: gpd.GeoDataFrame,
    radius_m: float,
) -> AssetsAtRisk:
    """Summarise structures and infrastructure at risk within radius_m.

    Building counts and type classification reflect what is mapped in OSM,
    which is reliable in populated areas but sparse in remote rural zones.
    Counts should be treated as lower bounds on actual structure density.
    """
    buildings_gdf = _cache["buildings"]
    power_gdf = _cache["power_lines"]
    protected_gdf = _cache["protected_areas"]

    point = point_gdf.geometry.iloc[0]

    # Buildings within radius.
    building_count = 0
    has_permanent = False
    if not buildings_gdf.empty:
        buildings_gdf = buildings_gdf.copy()
        buildings_gdf["distance_m"] = buildings_gdf.geometry.distance(point)
        nearby_buildings = buildings_gdf[buildings_gdf["distance_m"] <= radius_m]
        building_count = len(nearby_buildings)
        if not nearby_buildings.empty and "building" in nearby_buildings.columns:
            has_permanent = nearby_buildings["building"].isin(
                PERMANENT_BUILDING_TAGS
            ).any()

    # Power lines within radius.
    power_present = False
    if not power_gdf.empty:
        power_gdf = power_gdf.copy()
        power_gdf["distance_m"] = power_gdf.geometry.distance(point)
        power_present = (power_gdf["distance_m"] <= radius_m).any()

    # Protected area containment.
    protected_name: str | None = None
    if not protected_gdf.empty:
        containing = protected_gdf[protected_gdf.geometry.contains(point)]
        if not containing.empty:
            raw_name = containing.iloc[0].get("name")
            protected_name = _nan_to_none(raw_name)

    return AssetsAtRisk(
        buildings_within_radius=int(building_count),
        has_permanent_structures=bool(has_permanent),
        power_lines_present=bool(power_present),
        protected_area=protected_name,
        assets_radius_m=float(radius_m),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def extract_context(
    lat: float,
    lon: float,
    pbf_path: Path = DEFAULT_PBF,
) -> OperationalContext:
    """Extract a fully populated OperationalContext for the given coordinates.

    Loads the .pbf on first call (slow), then reuses cached GeoDataFrames for
    subsequent calls (fast). Terrain is not populated in the initial
    implementation; the field is set to None.

    Args:
        lat: Latitude in WGS84 decimal degrees.
        lon: Longitude in WGS84 decimal degrees.
        pbf_path: Path to the local .osm.pbf file.

    Returns:
        A fully validated OperationalContext.
    """
    _ensure_loaded(pbf_path)

    point_gdf = _make_point_gdf(lat, lon)

    region = _query_region(point_gdf)
    land_cover = _query_land_cover(point_gdf)
    water_sources = _query_water_sources(point_gdf, WATER_RADIUS_M)
    roads = _query_roads(point_gdf, ROAD_RADIUS_M)
    settlements = _query_settlements(point_gdf, SETTLEMENT_RADIUS_M)
    named_features = _query_named_features(point_gdf, NAMED_FEATURE_RADIUS_M)
    assets_at_risk = _query_assets_at_risk(point_gdf, ASSETS_RADIUS_M)

    metadata = ExtractionMetadata(
        extracted_at=datetime.now(timezone.utc),
        osm_dataset=pbf_path.name,
        settlement_radius_m=float(SETTLEMENT_RADIUS_M),
        water_source_radius_m=float(WATER_RADIUS_M),
        track_radius_m=float(ROAD_RADIUS_M),
        named_feature_radius_m=float(NAMED_FEATURE_RADIUS_M),
        assets_radius_m=float(ASSETS_RADIUS_M),
    )

    return OperationalContext(
        coordinates=Coordinates(latitude=lat, longitude=lon),
        region=region,
        land_cover=land_cover,
        terrain=None,
        water_sources=water_sources,
        roads=roads,
        settlements=settlements,
        assets_at_risk=assets_at_risk,
        named_features=named_features,
        wind=None,
        extraction_metadata=metadata,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract operational GIS context for a coordinate pair from a "
            "local OSM .pbf file and write the result as JSON."
        )
    )
    parser.add_argument("--lat", type=float, required=True, help="Latitude (WGS84)")
    parser.add_argument("--lon", type=float, required=True, help="Longitude (WGS84)")
    parser.add_argument(
        "--pbf", type=Path, default=DEFAULT_PBF,
        help=f"Path to .osm.pbf file (default: {DEFAULT_PBF})",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output path for the JSON file (e.g. data/contexts/scenario_01.json)",
    )
    args = parser.parse_args()

    if not args.pbf.exists():
        print(f"Error: .pbf file not found: {args.pbf}", file=sys.stderr)
        sys.exit(1)

    context = extract_context(lat=args.lat, lon=args.lon, pbf_path=args.pbf)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(context.model_dump_json(indent=2))

    print(f"Context written to {args.output}")
    region_str = context.region.name
    if context.region.admin_area:
        region_str += f", {context.region.admin_area}"
    print(f"  Region:          {region_str}")
    print(f"  Land cover:      {context.land_cover}")
    print(f"  Settlements:     {len(context.settlements)}")
    print(f"  Water sources:   {len(context.water_sources)}")
    print(f"  Named features:  {len(context.named_features)}")
    a = context.assets_at_risk
    print(f"  Buildings:       {a.buildings_within_radius} "
          f"({'permanent' if a.has_permanent_structures else 'no permanent'})")
    print(f"  Power lines:     {'yes' if a.power_lines_present else 'no'}")
    print(f"  Protected area:  {a.protected_area or 'none'}")
    primary = context.roads.primary_access
    if primary:
        weight = f", max {primary.max_weight_tonnes}t" if primary.max_weight_tonnes else ""
        bridge = ", has bridge" if primary.has_bridge else ""
        print(f"  Primary road:    {primary.road_class} "
              f"({primary.distance_m:.0f}m {primary.bearing}{weight}{bridge})")
    else:
        print("  Primary road:    none within search radius")


if __name__ == "__main__":
    main()