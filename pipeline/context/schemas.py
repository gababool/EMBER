"""Schemas for structured operational wildfire context.

The OperationalContext model defines the structured context passed from GIS/context
extraction into prompt formatting and LLM inference.

Unit conventions:
- Coordinates use WGS84 decimal degrees.
- Distances use metres.
- Areas use square metres.
- Elevation uses metres above mean sea level.
- Slope uses degrees.
- Wind speed uses metres per second.

Important:
- This module only defines and validates the shape of the data.
- It does not perform GIS extraction, accessibility computation,
  land-cover lookup, terrain analysis, or prompt formatting.
- The `extraction_metadata` field is included in the JSON for
  reproducibility but is intentionally excluded by `format_context`
  from the prompt-ready text block sent to the LLM.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


CompassBearing = Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

TerrainAspect = Literal[
    "N", "NE", "E", "SE", "S", "SW", "W", "NW",
    "flat",
    "unknown",
]

SlopeSteepness = Literal[
    "flat",
    "gentle",
    "moderate",
    "steep",
    "very_steep",
    "unknown",
]

LandCoverType = Literal[
    "forest",
    "coniferous_forest",
    "broadleaf_forest",
    "mixed_forest",
    "shrubland",
    "grassland",
    "agricultural",
    "urban",
    "wetland",
    "water",
    "bare_ground",
    "snow_or_ice",
    "unknown",
]

WaterSourceType = Literal[
    "lake",
    "river",
    "stream",
    "reservoir",
    "pond",
    "wetland",
    "unknown",
]

# Note: `sea` and `ocean` are intentionally excluded from WaterSourceType.
# Saltwater is unusable for firefighting; extraction filters these out.

WaterSupplyCategory = Literal[
    "heavy",   # Large lakes (area > 10 000 m²), reservoirs, rivers.
               # Suitable for tanker filling and sustained pumping operations.
    "light",   # Small ponds (area <= 10 000 m²), streams, ditches, wetlands.
               # Suitable for portable pumps only.
    "unknown", # Area or type information is insufficient to classify.
]

RoadClass = Literal[
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service",
    "track",
    "path",
    "unknown",
]

SettlementType = Literal[
    "city",
    "town",
    "village",
    "hamlet",
    "suburb",
    "isolated_dwelling",
    "farm",
    "unknown",
]

NamedFeatureType = Literal[
    "peak",
    "ridge",
    "valley",
    "forest",
    "island",
    "other",
]


class StrictBaseModel(BaseModel):
    """Base model for all operational context schemas.

    Extra fields are forbidden so schema drift between extraction, formatting,
    and LLM inference is caught early.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class Coordinates(StrictBaseModel):
    """Observation coordinates in WGS84 decimal degrees."""

    latitude: float = Field(
        ..., ge=-90, le=90,
        description=(
            "Latitude of the observation point in WGS84 decimal degrees. "
            "Positive values indicate north. Unit: degrees."
        ),
    )
    longitude: float = Field(
        ..., ge=-180, le=180,
        description=(
            "Longitude of the observation point in WGS84 decimal degrees. "
            "Positive values indicate east. Unit: degrees."
        ),
    )


class Region(StrictBaseModel):
    """Human-readable administrative or operational region."""

    name: str = Field(
        ..., min_length=1,
        description=(
            "Primary human-readable region name, typically the Swedish municipality "
            "(kommun). Unit: none."
        ),
    )
    admin_area: str | None = Field(
        default=None,
        description=(
            "Larger administrative area, typically the Swedish county (län). "
            "Unit: none."
        ),
    )


class Terrain(StrictBaseModel):
    """Terrain properties at or near the observation point.

    The entire Terrain object is optional in OperationalContext. It is included
    here for forward compatibility; populating it requires an elevation raster
    which is out of scope for the initial extraction implementation.
    """

    elevation_m: float | None = Field(
        default=None,
        description=(
            "Elevation above mean sea level at the observation point. "
            "Use None if unavailable. Unit: metres."
        ),
    )
    slope_degrees: float | None = Field(
        default=None, ge=0, le=90,
        description=(
            "Terrain slope angle at the observation point, where 0 is flat "
            "and 90 is vertical. Use None if unavailable. Unit: degrees."
        ),
    )
    slope_steepness: SlopeSteepness = Field(
        default="unknown",
        description=(
            "Operational interpretation of terrain steepness. "
            "Unit: categorical label."
        ),
    )
    aspect: TerrainAspect = Field(
        default="unknown",
        description=(
            "Downslope-facing terrain aspect as an 8-point compass bearing. "
            "Use 'flat' if no meaningful aspect, 'unknown' if unavailable. "
            "Unit: categorical compass bearing."
        ),
    )


class WaterSource(StrictBaseModel):
    """A single nearby water source usable for firefighting operations.

    Saltwater (sea, ocean) is excluded by extraction since it is unusable
    for fire suppression. Supply category distinguishes between sources
    suitable for heavy tanker operations versus portable pumps only.
    """

    name: str | None = Field(
        default=None,
        description="Name of the water source if available. Unit: none.",
    )
    source_type: WaterSourceType = Field(
        ...,
        description="Type of water source. Saltwater excluded. Unit: categorical label.",
    )
    supply_category: WaterSupplyCategory = Field(
        ...,
        description=(
            "Operational supply category. 'heavy' for large lakes (area > 10 000 m²), "
            "reservoirs, and rivers — suitable for tanker filling and sustained pumping. "
            "'light' for small ponds, streams, ditches, and wetlands — portable pumps "
            "only. 'unknown' when insufficient data is available. Unit: categorical label."
        ),
    )
    distance_m: float = Field(
        ..., ge=0,
        description=(
            "Shortest horizontal distance from the observation point to the "
            "nearest geometry of this water source. Unit: metres."
        ),
    )
    bearing: CompassBearing | None = Field(
        default=None,
        description=(
            "Compass direction from the observation point toward the water source. "
            "Unit: categorical compass bearing."
        ),
    )
    area_m2: float | None = Field(
        default=None, ge=0,
        description=(
            "Surface area for polygon water bodies (lakes, ponds, reservoirs, "
            "wetlands). None for line features (rivers, streams). Unit: square metres."
        ),
    )
    nearest_road_distance_m: float | None = Field(
        default=None, ge=0,
        description=(
            "Distance from this water source to the nearest vehicle-accessible road. "
            "Indicates whether the source is reachable by ground crews. "
            "None if no accessible road is found within the search radius. Unit: metres."
        ),
    )


class Road(StrictBaseModel):
    """A single road or access route relative to the observation point.

    The `vehicle_accessible` flag is derived during extraction from road class,
    surface, and tracktype tags and indicates whether a fire truck can plausibly
    drive on this road:

    - True for motorway, trunk, primary, secondary, tertiary, unclassified,
      residential, and service (regardless of surface).
    - True for tracks tagged tracktype=grade1 or grade2, or with a solid
      surface tag (asphalt, concrete, paved, compacted, gravel, fine_gravel,
      paving_stones).
    - False for tracks tagged tracktype=grade3–grade5.
    - False for tracks lacking both tracktype and a solid surface tag
      (conservative — no data means we cannot confirm passability).
    - False for path and unknown road classes.

    The `max_weight_tonnes` field reflects the OSM `maxweight` tag when present.
    In practice this tag is rarely populated on Swedish forest roads; None should
    be interpreted as "weight limit unknown, not necessarily unlimited."

    The `has_bridge` flag indicates whether the road segment crosses a bridge
    (OSM `bridge=yes`). Bridge presence is a proxy for culvert or load-bearing
    constraints that may limit heavy vehicle access.
    """

    name: str | None = Field(
        default=None,
        description="Name or reference of the road if available. Unit: none.",
    )
    road_class: RoadClass = Field(
        ...,
        description="OSM highway classification. Unit: categorical label.",
    )
    surface: str | None = Field(
        default=None,
        description=(
            "Raw OSM surface tag (e.g. 'asphalt', 'gravel', 'unpaved'). "
            "None if not tagged. Unit: none."
        ),
    )
    tracktype: str | None = Field(
        default=None,
        description=(
            "OSM tracktype tag for tracks (grade1 = solid, grade5 = soft/impassable). "
            "None for non-track roads or when not tagged. Unit: none."
        ),
    )
    vehicle_accessible: bool = Field(
        ...,
        description=(
            "Derived flag: whether a fire truck can plausibly drive on this road. "
            "Computed from road_class, surface, and tracktype. See class docstring "
            "for derivation rules. Unit: boolean."
        ),
    )
    max_weight_tonnes: float | None = Field(
        default=None, ge=0,
        description=(
            "Maximum vehicle weight permitted on this road from OSM maxweight tag. "
            "None when not tagged — which is the common case for Swedish forest roads. "
            "Unit: metric tonnes."
        ),
    )
    has_bridge: bool = Field(
        default=False,
        description=(
            "True if this road segment is tagged bridge=yes in OSM. Bridge presence "
            "is a proxy for potential load-bearing or culvert constraints. Unit: boolean."
        ),
    )
    distance_m: float = Field(
        ..., ge=0,
        description=(
            "Shortest horizontal distance from the observation point to the road "
            "geometry. Unit: metres."
        ),
    )
    bearing: CompassBearing | None = Field(
        default=None,
        description=(
            "Compass direction from the observation point toward the road. "
            "Unit: categorical compass bearing."
        ),
    )


class Roads(StrictBaseModel):
    """Road context relative to the observation point.

    Distinguishes the primary vehicle-accessible road (most operationally
    relevant for fire response logistics) from nearby tracks (forest roads
    and similar that may or may not be vehicle-accessible).
    """

    primary_access: Road | None = Field(
        default=None,
        description=(
            "Closest vehicle-accessible road suitable for fire-response vehicles. "
            "None if no accessible road exists within the search radius."
        ),
    )
    nearby_tracks: list[Road] = Field(
        default_factory=list,
        description=(
            "Nearby tracks (typically forest roads), including those not confirmed "
            "vehicle-accessible. Useful for forward access or crew movement. "
            "May be empty."
        ),
    )


class Settlement(StrictBaseModel):
    """A single nearby settlement relative to the observation point."""

    name: str | None = Field(
        default=None,
        description="Name of the settlement if available. Unit: none.",
    )
    settlement_type: SettlementType = Field(
        ...,
        description="Type of settlement. Unit: categorical label.",
    )
    distance_m: float = Field(
        ..., ge=0,
        description=(
            "Shortest horizontal distance from the observation point to the "
            "settlement centroid. Unit: metres."
        ),
    )
    bearing: CompassBearing | None = Field(
        default=None,
        description=(
            "Compass direction from the observation point toward the settlement. "
            "Unit: categorical compass bearing."
        ),
    )


class NamedFeature(StrictBaseModel):
    """A named natural feature near the observation point.

    Captures named hills, ridges, valleys, named forests, named islands, or
    other notable named landmarks present in OSM. Coverage is sparse and
    inconsistent; this list is frequently empty.
    """

    name: str = Field(
        ..., min_length=1,
        description="Name of the feature. Unit: none.",
    )
    feature_type: NamedFeatureType = Field(
        ...,
        description=(
            "Category of the named feature. 'other' for features that do not "
            "fit a more specific category. Unit: categorical label."
        ),
    )
    distance_m: float = Field(
        ..., ge=0,
        description="Distance from the observation point to the feature. Unit: metres.",
    )
    bearing: CompassBearing | None = Field(
        default=None,
        description=(
            "Compass direction from the observation point toward the feature. "
            "Unit: categorical compass bearing."
        ),
    )


class AssetsAtRisk(StrictBaseModel):
    """Summary of structures and infrastructure at risk near the observation point.

    Derived from OSM building and infrastructure features within a configurable
    radius. Intended to help the LLM flag evacuation priority and suppression
    strategy when settlements or critical infrastructure are nearby.

    Coverage note: OSM building data is reliable in populated areas but sparse
    in remote or rural zones. Counts should be treated as lower bounds.
    """

    buildings_within_radius: int = Field(
        ..., ge=0,
        description=(
            "Count of OSM building polygons within the configured search radius. "
            "Includes all building types. Unit: count."
        ),
    )
    has_permanent_structures: bool = Field(
        ...,
        description=(
            "True if any building tagged as a permanent residence "
            "(building=house, residential, apartments, detached, terrace) "
            "is present within the search radius. Unit: boolean."
        ),
    )
    power_lines_present: bool = Field(
        ...,
        description=(
            "True if any OSM power=line feature is present within the search radius. "
            "Power line proximity is operationally significant for aerial suppression "
            "and crew safety. Unit: boolean."
        ),
    )
    protected_area: str | None = Field(
        default=None,
        description=(
            "Name of a nature reserve or protected area (OSM leisure=nature_reserve "
            "or boundary=protected_area) if the observation point falls within one. "
            "Protected area designation affects suppression strategy and resource "
            "prioritisation. None if no protected area contains the point. Unit: none."
        ),
    )
    assets_radius_m: float = Field(
        ..., gt=0,
        description=(
            "The search radius used when collecting asset data. "
            "Unit: metres."
        ),
    )


class Wind(StrictBaseModel):
    """Wind conditions near the observation point.

    Wind is optional because weather/wind integration is a stretch feature.
    Wind may be mocked for reproducible scenarios or provided manually.
    """

    direction_degrees: float = Field(
        ..., ge=0, lt=360,
        description=(
            "Meteorological wind direction in degrees — the direction the wind "
            "blows FROM, not toward. Unit: degrees."
        ),
    )
    direction_compass: CompassBearing = Field(
        ...,
        description=(
            "Meteorological wind direction — the direction the wind blows FROM, "
            "not toward. Unit: categorical compass bearing."
        ),
    )
    speed_mps: float = Field(
        ..., ge=0,
        description="Sustained wind speed. Unit: metres per second.",
    )
    source: Literal["mocked", "manual"] = Field(
        ...,
        description="Source of the wind value. Unit: categorical provenance label.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_wind_keys(cls, data):
        if not isinstance(data, dict):
            return data

        migrated = dict(data)
        if "speed_m_s" in migrated and "speed_mps" not in migrated:
            migrated["speed_mps"] = migrated.pop("speed_m_s")
        if "direction_from" in migrated and "direction_compass" not in migrated:
            migrated["direction_compass"] = migrated.pop("direction_from")
        if "direction_compass" in migrated and "direction_degrees" not in migrated:
            compass_to_degrees = {
                "N": 0,
                "NE": 45,
                "E": 90,
                "SE": 135,
                "S": 180,
                "SW": 225,
                "W": 270,
                "NW": 315,
            }
            migrated["direction_degrees"] = compass_to_degrees.get(
                migrated["direction_compass"]
            )
        has_wind_value = "speed_mps" in migrated or "direction_compass" in migrated
        if has_wind_value and "source" not in migrated:
            migrated["source"] = "manual"
        return migrated


class ExtractionMetadata(StrictBaseModel):
    """Provenance information about how this context was produced.

    Included in the on-disk JSON for reproducibility. The formatter is expected
    to exclude this from the prompt-ready text block sent to the LLM.
    """

    extracted_at: datetime = Field(
        ...,
        description="Timestamp at which extraction was run. Unit: datetime.",
    )
    osm_dataset: str = Field(
        ..., min_length=1,
        description="OSM dataset filename used for extraction. Unit: none.",
    )
    settlement_radius_m: float = Field(..., gt=0, description="Settlement search radius. Unit: metres.")
    water_source_radius_m: float = Field(..., gt=0, description="Water source search radius. Unit: metres.")
    track_radius_m: float = Field(..., gt=0, description="Road/track search radius. Unit: metres.")
    named_feature_radius_m: float = Field(..., gt=0, description="Named feature search radius. Unit: metres.")
    assets_radius_m: float = Field(..., gt=0, description="Assets-at-risk search radius. Unit: metres.")
    fire_station_radius_m: float = Field(default=30_000.0, gt=0, description="Fire station search radius. Unit: metres.")


class FireStation(StrictBaseModel):
    """A nearby fire station that could provide crew staging resources."""

    name: str | None = Field(
        default=None,
        description="Name of the fire station if available. Unit: none.",
    )
    distance_m: float = Field(
        ..., ge=0,
        description=(
            "Shortest horizontal distance from the observation point to the fire station. "
            "Unit: metres."
        ),
    )
    bearing: CompassBearing | None = Field(
        default=None,
        description=(
            "Compass direction from the observation point toward the fire station. "
            "Unit: categorical compass bearing."
        ),
    )


class OperationalContext(StrictBaseModel):
    """Structured operational context for wildfire decision support.

    This model is the contract between GIS/context extraction, prompt formatting,
    and LLM inference.
    """

    schema_version: Literal["2.0"] = Field(
        default="2.0",
        description="Version of the operational context schema. Unit: none.",
    )
    coordinates: Coordinates = Field(
        ...,
        description="Observation coordinates. Units documented in Coordinates.",
    )
    region: Region = Field(
        ...,
        description="Administrative region. Units documented in Region.",
    )
    land_cover: LandCoverType = Field(
        ...,
        description="Dominant land-cover class at the observation point. Unit: categorical label.",
    )
    terrain: Terrain | None = Field(
        default=None,
        description=(
            "Terrain context. Optional: None when elevation data is unavailable. "
            "Units documented in Terrain."
        ),
    )
    water_sources: list[WaterSource] = Field(
        default_factory=list,
        description=(
            "Nearby freshwater sources within the configured search radius, sorted "
            "by operational relevance (distance weighted by road accessibility). "
            "Saltwater excluded. May be empty. Units documented in WaterSource."
        ),
    )
    roads: Roads = Field(
        ...,
        description="Road context relative to the observation point. Units documented in Roads.",
    )
    settlements: list[Settlement] = Field(
        default_factory=list,
        description=(
            "Nearby settlements sorted by distance. May be empty. "
            "Units documented in Settlement."
        ),
    )
    fire_stations: list[FireStation] = Field(
        default_factory=list,
        description=(
            "Nearby fire stations within the configured search radius, sorted by distance. "
            "May be empty. Units documented in FireStation."
        ),
    )
    assets_at_risk: AssetsAtRisk = Field(
        ...,
        description=(
            "Summary of structures and infrastructure at risk near the observation point. "
            "Units documented in AssetsAtRisk."
        ),
    )
    named_features: list[NamedFeature] = Field(
        default_factory=list,
        description=(
            "Named natural features near the observation point. Frequently empty "
            "due to sparse OSM coverage. Units documented in NamedFeature."
        ),
    )
    wind: Wind | None = Field(
        default=None,
        description=(
            "Optional wind context. None when wind integration is not available. "
            "Units documented in Wind."
        ),
    )
    extraction_metadata: ExtractionMetadata = Field(
        ...,
        description=(
            "Provenance and parameter information. Included in JSON for reproducibility; "
            "excluded from the LLM-facing prompt by the formatter."
        ),
    )
