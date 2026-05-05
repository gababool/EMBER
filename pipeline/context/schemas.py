"""Schemas for structured operational wildfire context.

The OperationalContext model defines the structured context passed from
GIS/context extraction into prompt formatting and LLM inference.

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

from pydantic import BaseModel, ConfigDict, Field


CompassBearing = Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

TerrainAspect = Literal[
    "N",
    "NE",
    "E",
    "SE",
    "S",
    "SW",
    "W",
    "NW",
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
        ...,
        ge=-90,
        le=90,
        description=(
            "Latitude of the observation point in WGS84 decimal degrees. "
            "Positive values indicate north. Unit: degrees."
        ),
    )

    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description=(
            "Longitude of the observation point in WGS84 decimal degrees. "
            "Positive values indicate east. Unit: degrees."
        ),
    )


class Region(StrictBaseModel):
    """Human-readable administrative or operational region."""

    name: str = Field(
        ...,
        min_length=1,
        description=(
            "Primary human-readable region name, such as municipality, county, "
            "or operational area. Unit: none."
        ),
    )

    admin_area: str | None = Field(
        default=None,
        description=(
            "Optional larger administrative area, such as county, state, or "
            "province. Unit: none."
        ),
    )


class Terrain(StrictBaseModel):
    """Terrain properties at or near the observation point.

    The entire Terrain object is optional in OperationalContext; if elevation
    data is unavailable, the parent field is set to None rather than
    populating Terrain with placeholder values.
    """

    elevation_m: float | None = Field(
        default=None,
        description=(
            "Elevation above mean sea level at the observation point. "
            "Use None if elevation is unavailable. Unit: metres."
        ),
    )

    slope_degrees: float | None = Field(
        default=None,
        ge=0,
        le=90,
        description=(
            "Terrain slope angle at the observation point, where 0 is flat "
            "and 90 is vertical. Use None if unavailable. Unit: degrees."
        ),
    )

    slope_steepness: SlopeSteepness = Field(
        default="unknown",
        description=(
            "Operational interpretation of terrain steepness based on slope. "
            "Allowed values are flat, gentle, moderate, steep, very_steep, "
            "and unknown. Unit: categorical label."
        ),
    )

    aspect: TerrainAspect = Field(
        default="unknown",
        description=(
            "Downslope-facing terrain aspect as an 8-point compass bearing. "
            "Use 'flat' if the slope has no meaningful aspect and 'unknown' "
            "if aspect data is unavailable. Unit: categorical compass bearing."
        ),
    )


class WaterSource(StrictBaseModel):
    """A single nearby water source usable for firefighting operations.

    Saltwater (sea, ocean) is excluded by extraction since it is unusable
    for fire suppression.
    """

    name: str | None = Field(
        default=None,
        description="Name of the water source if available. Unit: none.",
    )

    source_type: WaterSourceType = Field(
        ...,
        description=(
            "Type of water source. Saltwater is excluded from extraction. "
            "Unit: categorical label."
        ),
    )

    distance_m: float = Field(
        ...,
        ge=0,
        description=(
            "Shortest horizontal distance from the observation point to the "
            "nearest geometry of this water source. Unit: metres."
        ),
    )

    bearing: CompassBearing | None = Field(
        default=None,
        description=(
            "Compass direction from the observation point toward the water "
            "source. Use None if unavailable. Unit: categorical compass bearing."
        ),
    )

    area_m2: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Surface area of the water body when it is a polygon feature "
            "(lake, reservoir, pond, wetland). None for line features such "
            "as rivers and streams. Unit: square metres."
        ),
    )

    nearest_road_distance_m: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Distance from this water source to the nearest vehicle-accessible "
            "road. A water source far from any road is unusable in practice "
            "regardless of its size. Use None if no accessible road is found "
            "within the search radius. Unit: metres."
        ),
    )


class Road(StrictBaseModel):
    """A single road or access route relative to the observation point."""

    name: str | None = Field(
        default=None,
        description="Name or identifier of the road if available. Unit: none.",
    )

    road_class: RoadClass = Field(
        ...,
        description=(
            "Raw OSM road class. Unit: categorical label."
        ),
    )

    surface: str | None = Field(
        default=None,
        description=(
            "Surface type from OSM `surface=*` tag (e.g. 'asphalt', 'paved', "
            "'unpaved', 'gravel', 'dirt'). None when the tag is absent. "
            "Unit: none."
        ),
    )

    tracktype: str | None = Field(
        default=None,
        description=(
            "OSM `tracktype=*` value for forest tracks, ranging from 'grade1' "
            "(solid, all-weather drivable) to 'grade5' (soft, often impassable). "
            "Only meaningful when road_class is 'track'. None otherwise. "
            "Unit: none."
        ),
    )

    vehicle_accessible: bool = Field(
        ...,
        description=(
            "Whether a typical fire response vehicle can drive on this road. "
            "Computed by extraction from road_class, surface, and tracktype: "
            "True for motorway through residential and service; True for "
            "tracks with tracktype grade1 or grade2; False for tracks with "
            "grade3 or higher; False for path, footway, and unknown classes. "
            "Unit: boolean."
        ),
    )

    distance_m: float = Field(
        ...,
        ge=0,
        description=(
            "Shortest horizontal distance from the observation point to the "
            "nearest road geometry. Unit: metres."
        ),
    )

    bearing: CompassBearing | None = Field(
        default=None,
        description=(
            "Compass direction from the observation point toward the road. "
            "Use None if unavailable. Unit: categorical compass bearing."
        ),
    )


class Roads(StrictBaseModel):
    """Road context relative to the observation point.

    Distinguishes the primary vehicle-accessible road (most operationally
    relevant for fire response logistics) from a list of nearby tracks
    that may be relevant for forward access or escape routes.
    """

    primary_access: Road | None = Field(
        default=None,
        description=(
            "The closest vehicle-accessible road. None if no accessible road "
            "is found within the search radius."
        ),
    )

    nearby_tracks: list[Road] = Field(
        default_factory=list,
        description=(
            "Nearby roads classified as tracks (typically forest roads). "
            "May be vehicle-accessible or not depending on tracktype. Empty "
            "list if no tracks are within the search radius."
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
        ...,
        ge=0,
        description=(
            "Shortest horizontal distance from the observation point to the "
            "settlement geometry or centroid. Unit: metres."
        ),
    )

    bearing: CompassBearing | None = Field(
        default=None,
        description=(
            "Compass direction from the observation point toward the "
            "settlement. Use None if unavailable. Unit: categorical compass "
            "bearing."
        ),
    )


class NamedFeature(StrictBaseModel):
    """A named natural feature near the observation point.

    Captures named hills, ridges, valleys, named forests, named islands,
    or other notable named landmarks present in OSM. Coverage is sparse
    in OSM, so this list will frequently be empty.
    """

    name: str = Field(
        ...,
        min_length=1,
        description="Name of the feature. Unit: none.",
    )

    feature_type: NamedFeatureType = Field(
        ...,
        description=(
            "Category of the named feature. Use 'other' for named features "
            "that do not fit a more specific category. Unit: categorical label."
        ),
    )

    distance_m: float = Field(
        ...,
        ge=0,
        description=(
            "Shortest horizontal distance from the observation point to the "
            "feature. Unit: metres."
        ),
    )

    bearing: CompassBearing | None = Field(
        default=None,
        description=(
            "Compass direction from the observation point toward the feature. "
            "Use None if unavailable. Unit: categorical compass bearing."
        ),
    )


class Wind(StrictBaseModel):
    """Wind conditions near the observation point."""

    speed_m_s: float = Field(
        ...,
        ge=0,
        description=(
            "Sustained wind speed near the observation point. "
            "Unit: metres per second."
        ),
    )

    direction_from: CompassBearing = Field(
        ...,
        description=(
            "Meteorological wind direction, meaning the direction the wind "
            "blows from, not the direction it blows toward. "
            "Unit: categorical compass bearing."
        ),
    )


class ExtractionMetadata(StrictBaseModel):
    """Provenance information about how this context was produced.

    This data is included in the on-disk JSON for reproducibility but is
    intentionally excluded from the prompt-ready text block sent to the
    LLM. The formatter is responsible for omitting it.
    """

    extracted_at: datetime = Field(
        ...,
        description=(
            "Timestamp at which extraction was run. Prefer timezone-aware "
            "ISO 8601 datetime values. Unit: datetime."
        ),
    )

    osm_dataset: str = Field(
        ...,
        min_length=1,
        description=(
            "Identifier for the OSM dataset used (e.g. file name and "
            "Geofabrik snapshot date). Unit: none."
        ),
    )

    settlement_radius_m: float = Field(
        ...,
        gt=0,
        description=(
            "Search radius used for nearby settlements. Unit: metres."
        ),
    )

    water_source_radius_m: float = Field(
        ...,
        gt=0,
        description=(
            "Search radius used for nearby water sources. Unit: metres."
        ),
    )

    track_radius_m: float = Field(
        ...,
        gt=0,
        description=(
            "Search radius used for nearby tracks. Unit: metres."
        ),
    )

    named_feature_radius_m: float = Field(
        ...,
        gt=0,
        description=(
            "Search radius used for named natural features. Unit: metres."
        ),
    )


class OperationalContext(StrictBaseModel):
    """Structured operational context for wildfire decision support.

    This model is the contract between GIS/context extraction, prompt
    formatting, and LLM inference.
    """

    schema_version: Literal["2.0"] = Field(
        default="2.0",
        description="Version of the operational context schema. Unit: none.",
    )

    coordinates: Coordinates = Field(
        ...,
        description=(
            "Observation coordinates. Units documented in Coordinates."
        ),
    )

    region: Region = Field(
        ...,
        description=(
            "Administrative or operational region. Units documented in Region."
        ),
    )

    land_cover: LandCoverType = Field(
        ...,
        description=(
            "Dominant land-cover class at the observation point. "
            "Unit: categorical label."
        ),
    )

    terrain: Terrain | None = Field(
        default=None,
        description=(
            "Terrain context at or near the observation point. Optional: "
            "set to None when elevation data is unavailable. "
            "Units documented in Terrain."
        ),
    )

    water_sources: list[WaterSource] = Field(
        default_factory=list,
        description=(
            "Nearby water sources usable for firefighting, sorted by "
            "operational relevance. Saltwater is excluded. Empty list if "
            "no usable water sources are within the search radius. "
            "Units documented in WaterSource."
        ),
    )

    roads: Roads = Field(
        ...,
        description=(
            "Road context relative to the observation point. "
            "Units documented in Roads."
        ),
    )

    settlements: list[Settlement] = Field(
        default_factory=list,
        description=(
            "Nearby settlements sorted by distance. Empty list if no "
            "settlements are within the search radius. "
            "Units documented in Settlement."
        ),
    )

    named_features: list[NamedFeature] = Field(
        default_factory=list,
        description=(
            "Optional list of named natural features near the observation "
            "point (named peaks, ridges, valleys, etc.). Frequently empty "
            "due to sparse OSM coverage; handled gracefully when empty. "
            "Units documented in NamedFeature."
        ),
    )

    wind: Wind | None = Field(
        default=None,
        description=(
            "Optional wind context near the observation point. Use None if "
            "wind integration is not available. Units documented in Wind."
        ),
    )

    extraction_metadata: ExtractionMetadata = Field(
        ...,
        description=(
            "Provenance and parameter information about the extraction run. "
            "Included in JSON for reproducibility; the formatter excludes "
            "this from the prompt-ready text sent to the LLM."
        ),
    )