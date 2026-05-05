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
    national projected coordinate system where 1 unit = 1 metre. Results
    are converted back to WGS84 bearings and metre distances for the schema.

Pre-computation approach:
    The .pbf is loaded once per process into in-memory GeoDataFrames.
    For the expected usage pattern (5-10 scenario extractions per run),
    this is faster than re-reading the file per query. The loaded data is
    cached at module level after the first call to extract_context().
    At inference time, scenario JSONs are loaded directly without touching
    the .pbf at all.

This script was developed with Claude
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmium
import osmium.filter
from shapely.geometry import Point, MultiPolygon
from shapely.ops import unary_union

from pipeline.context.schemas import (
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
)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_PBF = Path("data/gis/osm/vastmanland-50km.osm.pbf")

# SWEREF99 TM — Swedish national projected CRS. All distance/area calculations
# use this CRS so that 1 unit == 1 metre.
SWEREF99_TM = "EPSG:3006"
WGS84 = "EPSG:4326"

# Search radii (metres).
SETTLEMENT_RADIUS_M = 20_000
WATER_RADIUS_M = 10_000
ROAD_RADIUS_M = 5_000
NAMED_FEATURE_RADIUS_M = 10_000

# Maximum results per category.
MAX_SETTLEMENTS = 5
MAX_WATER_SOURCES = 5
MAX_TRACKS = 3
MAX_NAMED_FEATURES = 5

# OSM highway values that count as vehicle-accessible roads (not tracks).
PAVED_ROAD_CLASSES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "service",
}

# Saltwater tags to exclude from water sources.
# OSM marks seas and oceans with natural=water + water=sea/ocean,
# or place=sea, or natural=coastline.
SALTWATER_WATER_TAGS = {"sea", "ocean", "bay"}

# OSM water tags mapped to WaterSourceType.
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

# OSM place tags mapped to SettlementType.
SETTLEMENT_TYPE_MAP: dict[str, SettlementType] = {
    "city": "city",
    "town": "town",
    "village": "village",
    "hamlet": "hamlet",
    "suburb": "suburb",
    "isolated_dwelling": "isolated_dwelling",
    "farm": "farm",
}

# OSM natural tags mapped to NamedFeatureType.
NAMED_FEATURE_TYPE_MAP: dict[str, NamedFeatureType] = {
    "peak": "peak",
    "ridge": "ridge",
    "valley": "valley",
    "wood": "forest",
    "island": "island",
}

# ── Module-level cache ────────────────────────────────────────────────────────

# GeoDataFrames loaded from the .pbf. Populated lazily on first call to
# extract_context(). Each GDF is reprojected to SWEREF99 TM for spatial ops.
_cache: dict[str, Any] = {}
_pbf_path: Path | None = None


# ── OSM handlers ─────────────────────────────────────────────────────────────
#
# osmium works by passing a handler object over the .pbf file. Each handler
# collects one category of feature into a list of dicts, which is later
# converted into a GeoDataFrame.

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

        # Exclude saltwater.
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

        self.features.append({
            "name": tags.get("name") or tags.get("ref"),
            "road_class": road_class,
            "surface": surface,
            "tracktype": tracktype,
            "vehicle_accessible": accessible,
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
        })


class _RegionHandler(osmium.SimpleHandler):
    """Collect administrative boundary polygons.

    Admin boundaries in OSM are stored as multipolygon relations, which
    require osmium's AreaManager to assemble before the area() callback
    fires. Applied via osmium.apply() with AreaManager in _load_pbf.
    """

    def __init__(self):
        super().__init__()
        self.features: list[dict] = []
        self._wkb = osmium.geom.WKBFactory()

    def area(self, a):
        tags = dict(a.tags)
        if tags.get("boundary") != "administrative":
            return
        admin_level = tags.get("admin_level", "")
        if admin_level not in ("4", "7"):
            return
        try:
            wkb = self._wkb.create_multipolygon(a)
            geom = _load_wkb(wkb)
        except Exception:
            return
        self.features.append({
            "name": tags.get("name"),
            "admin_level": admin_level,
            "geometry": geom,
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
        if not name:
            return
        if natural not in NAMED_FEATURE_TYPE_MAP:
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
        if not name:
            return
        # Named forests as area features.
        if natural not in ("wood",):
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


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _load_wkb(wkb_hex: str):
    """Parse a WKB hex string from osmium into a Shapely geometry."""
    from shapely import wkb as shapely_wkb
    return shapely_wkb.loads(wkb_hex, hex=True)


def _make_point_gdf(lat: float, lon: float) -> gpd.GeoDataFrame:
    """Create a single-row GeoDataFrame for the observation point."""
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(lon, lat)]},
        crs=WGS84,
    )
    return gdf.to_crs(SWEREF99_TM)


def _to_gdf(features: list[dict], crs: str = WGS84) -> gpd.GeoDataFrame:
    """Convert a list of feature dicts to a projected GeoDataFrame."""
    if not features:
        return gpd.GeoDataFrame(columns=["geometry"], crs=SWEREF99_TM)
    gdf = gpd.GeoDataFrame(features, crs=crs)
    return gdf.to_crs(SWEREF99_TM)


# ── Utility functions ─────────────────────────────────────────────────────────

def _degrees_to_compass(degrees: float) -> CompassBearing:
    """Convert a bearing in degrees (0 = N, clockwise) to an 8-point compass."""
    directions: list[CompassBearing] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = round(degrees / 45) % 8
    return directions[index]


def _compute_bearing(from_geom, to_geom) -> CompassBearing:
    """Compute the compass bearing from one projected geometry to another.

    Geometries must be in a projected CRS (SWEREF99 TM) so that coordinate
    differences correspond to north/east offsets in metres.
    """
    from_pt = from_geom.centroid if not from_geom.geom_type == "Point" else from_geom
    to_pt = to_geom.centroid if not to_geom.geom_type == "Point" else to_geom

    dx = to_pt.x - from_pt.x   # east offset in metres
    dy = to_pt.y - from_pt.y   # north offset in metres

    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad) % 360
    return _degrees_to_compass(angle_deg)


def _map_road_class(highway: str) -> RoadClass:
    """Map an OSM highway tag value to the schema RoadClass."""
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

    Rules:
    - Paved road classes (motorway–service, unclassified, residential):
      always True regardless of surface.
    - track with tracktype grade1 or grade2: True.
    - track with tracktype grade3–grade5: False.
    - track without tracktype but with a solid surface tag: True.
    - track without tracktype and without surface: False (conservative).
    - path and unknown: always False.
    """
    if road_class in PAVED_ROAD_CLASSES:
        return True
    if road_class == "track":
        if tracktype in ("grade1", "grade2"):
            return True
        if tracktype in ("grade3", "grade4", "grade5"):
            return False
        # No tracktype: fall back to surface.
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
    """Map OSM landuse/natural tags to a LandCoverType.

    Returns None if the tags don't match any known land-cover category,
    which causes the area to be skipped during loading.
    """
    if natural == "water":
        return "water"
    if natural == "wetland" or landuse == "wetland":
        return "wetland"
    if natural in ("wood", "scrub") or landuse == "forest":
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


def _nan_to_none(val):
    """Convert a pandas NaN to None for optional Pydantic string fields.

    Pandas represents missing values as float NaN in object columns. Pydantic
    rejects NaN where it expects str | None, so all optional string fields
    read from GeoDataFrame rows must be passed through this function.
    """
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
    """Load all feature categories from the .pbf into module-level cache.

    Called once per process. Subsequent calls to extract_context() reuse
    the cached GeoDataFrames without re-reading the file.
    """
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

    region_handler = _RegionHandler()
    osmium.apply(
        osmium.io.Reader(pbf_str, osmium.osm.osm_entity_bits.ALL),
        osmium.NodeLocationsForWays(osmium.index.create_map("flex_mem")),
        osmium.area.AreaManager(),
        region_handler,
    )

    named_feature_handler = _NamedFeatureHandler()
    named_feature_handler.apply_file(pbf_str, locations=True, idx="flex_mem")

    _cache = {
        "water": _to_gdf(water_handler.features),
        "roads": _to_gdf(road_handler.features),
        "settlements": _to_gdf(settlement_handler.features),
        "land_cover": _to_gdf(land_cover_handler.features),
        "regions": _to_gdf(region_handler.features),
        "named_features": _to_gdf(named_feature_handler.features),
    }
    _pbf_path = pbf_path
    print("OSM data loaded.")


def _ensure_loaded(pbf_path: Path) -> None:
    """Load the .pbf if it has not already been loaded."""
    if not _cache or _pbf_path != pbf_path:
        _load_pbf(pbf_path)


# ── Query functions ───────────────────────────────────────────────────────────

def _query_region(point_gdf: gpd.GeoDataFrame) -> Region:
    """Look up the administrative region containing the observation point.

    Performs point-in-polygon against Swedish municipality (admin_level=7)
    and county (admin_level=4) boundaries. Falls back to 'Unknown region'
    if no boundary contains the point (e.g. point is outside the extract).
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

    name = municipality or county or "Unknown region"
    return Region(name=name, admin_area=county if municipality else None)


def _query_land_cover(point_gdf: gpd.GeoDataFrame) -> LandCoverType:
    """Determine the dominant land-cover class at the observation point.

    Performs point-in-polygon against the loaded land-cover GeoDataFrame.
    Priority order when multiple polygons overlap the point:
      water > wetland > urban > forest variants > agricultural > grassland > unknown
    """
    land_cover_gdf = _cache["land_cover"]
    if land_cover_gdf.empty:
        return "unknown"

    point = point_gdf.geometry.iloc[0]
    containing = land_cover_gdf[land_cover_gdf.geometry.contains(point)]

    if containing.empty:
        return "unknown"

    priority: dict[LandCoverType, int] = {
        "water": 0,
        "wetland": 1,
        "urban": 2,
        "snow_or_ice": 3,
        "coniferous_forest": 4,
        "broadleaf_forest": 4,
        "mixed_forest": 4,
        "forest": 5,
        "shrubland": 6,
        "agricultural": 7,
        "grassland": 8,
        "bare_ground": 9,
        "unknown": 10,
    }

    best = min(containing["land_cover"].tolist(), key=lambda c: priority.get(c, 99))
    return best


def _query_water_sources(
    point_gdf: gpd.GeoDataFrame,
    radius_m: float,
) -> list[WaterSource]:
    """Find nearby freshwater sources within radius_m of the observation point.

    Saltwater features are already excluded during loading. For polygon water
    bodies, area is computed in SWEREF99 TM (square metres). For each water
    source, the distance to the nearest vehicle-accessible road is computed
    to indicate operational accessibility.

    Results are sorted by distance and capped at MAX_WATER_SOURCES.
    """
    water_gdf = _cache["water"]
    roads_gdf = _cache["roads"]
    if water_gdf.empty:
        return []

    point = point_gdf.geometry.iloc[0]

    # Filter to within radius.
    water_gdf = water_gdf.copy()
    water_gdf["distance_m"] = water_gdf.geometry.distance(point)
    nearby = water_gdf[water_gdf["distance_m"] <= radius_m].copy()
    nearby = nearby.sort_values("distance_m").head(MAX_WATER_SOURCES)

    if nearby.empty:
        return []

    # Vehicle-accessible roads for accessibility check.
    accessible_roads = (
        roads_gdf[roads_gdf["vehicle_accessible"] == True].copy()  # noqa: E712
        if not roads_gdf.empty else gpd.GeoDataFrame()
    )

    results: list[WaterSource] = []
    for _, row in nearby.iterrows():
        geom = row["geometry"]

        # Area for polygons only.
        area_m2: float | None = None
        if row.get("is_polygon", False):
            area_m2 = float(geom.area)

        # Nearest accessible road distance.
        nearest_road_dist: float | None = None
        if not accessible_roads.empty:
            road_centroid = geom.centroid
            dists = accessible_roads.geometry.distance(road_centroid)
            if not dists.empty:
                nearest_road_dist = float(dists.min())

        bearing = _compute_bearing(point, geom.centroid)

        results.append(WaterSource(
            name=_nan_to_none(row.get("name")),
            source_type=row["source_type"],
            distance_m=float(row["distance_m"]),
            bearing=bearing,
            area_m2=area_m2,
            nearest_road_distance_m=nearest_road_dist,
        ))

    return results


def _query_roads(
    point_gdf: gpd.GeoDataFrame,
    radius_m: float,
) -> Roads:
    """Find roads within radius_m of the observation point.

    Returns a Roads object with:
    - primary_access: the closest vehicle-accessible road.
    - nearby_tracks: up to MAX_TRACKS closest track features.

    Tracks are listed separately because they have different operational
    implications (forward access for crews, not main supply routes).
    """
    roads_gdf = _cache["roads"]
    if roads_gdf.empty:
        return Roads()

    point = point_gdf.geometry.iloc[0]

    roads_gdf = roads_gdf.copy()
    roads_gdf["distance_m"] = roads_gdf.geometry.distance(point)
    nearby = roads_gdf[roads_gdf["distance_m"] <= radius_m].copy()

    if nearby.empty:
        return Roads()

    # Primary access: nearest vehicle-accessible non-track road.
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
            distance_m=float(row["distance_m"]),
            bearing=_compute_bearing(point, row["geometry"].centroid),
        )

    # Nearby tracks.
    tracks = nearby[nearby["road_class"] == "track"].sort_values("distance_m").head(MAX_TRACKS)
    nearby_tracks: list[Road] = []
    for _, row in tracks.iterrows():
        nearby_tracks.append(Road(
            name=_nan_to_none(row.get("name")),
            road_class="track",
            surface=_nan_to_none(row.get("surface")),
            tracktype=_nan_to_none(row.get("tracktype")),
            vehicle_accessible=bool(row["vehicle_accessible"]),
            distance_m=float(row["distance_m"]),
            bearing=_compute_bearing(point, row["geometry"].centroid),
        ))

    return Roads(primary_access=primary_access, nearby_tracks=nearby_tracks)


def _query_settlements(
    point_gdf: gpd.GeoDataFrame,
    radius_m: float,
) -> list[Settlement]:
    """Find nearby settlements within radius_m of the observation point.

    Results are sorted by distance, capped at MAX_SETTLEMENTS.
    """
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
    """Find nearby named natural features within radius_m.

    Results are sorted by distance, capped at MAX_NAMED_FEATURES.
    Named forests already captured in land_cover are included here too
    when they carry a distinct name, since that name aids LLM reasoning.
    """
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


# ── Public API ────────────────────────────────────────────────────────────────

def extract_context(
    lat: float,
    lon: float,
    pbf_path: Path = DEFAULT_PBF,
) -> OperationalContext:
    """Extract a fully populated OperationalContext for the given coordinates.

    Loads the .pbf on first call (slow), then reuses cached GeoDataFrames
    for subsequent calls (fast). Terrain is not populated in the initial
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

    metadata = ExtractionMetadata(
        extracted_at=datetime.now(timezone.utc),
        osm_dataset=pbf_path.name,
        settlement_radius_m=float(SETTLEMENT_RADIUS_M),
        water_source_radius_m=float(WATER_RADIUS_M),
        track_radius_m=float(ROAD_RADIUS_M),
        named_feature_radius_m=float(NAMED_FEATURE_RADIUS_M),
    )

    return OperationalContext(
        coordinates=Coordinates(latitude=lat, longitude=lon),
        region=region,
        land_cover=land_cover,
        terrain=None,
        water_sources=water_sources,
        roads=roads,
        settlements=settlements,
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
        "--pbf",
        type=Path,
        default=DEFAULT_PBF,
        help=f"Path to .osm.pbf file (default: {DEFAULT_PBF})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
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
    print(f"  Region:       {context.region.name}")
    print(f"  Land cover:   {context.land_cover}")
    print(f"  Settlements:  {len(context.settlements)}")
    print(f"  Water sources:{len(context.water_sources)}")
    print(f"  Named features:{len(context.named_features)}")
    primary = context.roads.primary_access
    print(f"  Primary road: {primary.road_class if primary else 'none'} "
          f"({primary.distance_m:.0f}m)" if primary else "  Primary road: none")


if __name__ == "__main__":
    main()