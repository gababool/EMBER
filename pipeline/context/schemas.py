"""Schemas for structured operational wildfire context.

The OperationalContext model defines the structured context passed from GIS/context
extraction into prompt formatting and LLM inference.

Unit conventions:
- Coordinates use WGS84 decimal degrees.
- Distances use metres.
- Elevation uses metres above mean sea level.
- Slope uses degrees.
- Wind speed uses metres per second.

Important:
- This module only defines and validates the shape of the data.
- It does not perform GIS extraction, nearest-road lookup, land-cover lookup,
  terrain analysis, or prompt formatting.
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
    "sea",
    "wetland",
    "unknown",
]

RoadClass = Literal[
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
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
    "isolated_dwelling",
    "unknown",
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

    country_code: str | None = Field(
        default=None,
        pattern=r"^[A-Z]{2}$",
        description=(
            "Optional ISO 3166-1 alpha-2 country code, for example 'SE'. "
            "Unit: none."
        ),
    )

    admin_area: str | None = Field(
        default=None,
        description=(
            "Optional larger administrative area, such as county, state, or province. "
            "Unit: none."
        ),
    )


class Terrain(StrictBaseModel):
    """Terrain properties at or near the observation point."""

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


class NearestWaterSource(StrictBaseModel):
    """Nearest known water source relative to the observation point."""

    name: str | None = Field(
        default=None,
        description="Name of the water source if available. Unit: none.",
    )

    source_type: WaterSourceType = Field(
        ...,
        description="Type of nearest water source. Unit: categorical label.",
    )

    distance_m: float = Field(
        ...,
        ge=0,
        description=(
            "Shortest horizontal distance from the observation point to the "
            "nearest water-source geometry. Unit: metres."
        ),
    )

    bearing: CompassBearing | None = Field(
        default=None,
        description=(
            "Compass direction from the observation point toward the water source. "
            "Use None if unavailable. Unit: categorical compass bearing."
        ),
    )


class NearestRoad(StrictBaseModel):
    """Nearest known road or access route relative to the observation point."""

    name: str | None = Field(
        default=None,
        description="Name or identifier of the road if available. Unit: none.",
    )

    road_class: RoadClass = Field(
        ...,
        description="Class of nearest road or access route. Unit: categorical label.",
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


class NearestSettlement(StrictBaseModel):
    """Nearest known settlement relative to the observation point."""

    name: str | None = Field(
        default=None,
        description="Name of the settlement if available. Unit: none.",
    )

    settlement_type: SettlementType = Field(
        ...,
        description="Type of nearest settlement. Unit: categorical label.",
    )

    distance_m: float = Field(
        ...,
        ge=0,
        description=(
            "Shortest horizontal distance from the observation point to the "
            "nearest settlement geometry or centroid. Unit: metres."
        ),
    )

    bearing: CompassBearing | None = Field(
        default=None,
        description=(
            "Compass direction from the observation point toward the settlement. "
            "Use None if unavailable. Unit: categorical compass bearing."
        ),
    )


class Wind(StrictBaseModel):
    """Optional wind conditions near the observation point.

    Wind is optional right now because weather/wind integration is an optional
    feature that will only be implemented if there is enough time.
    """

    speed_m_s: float = Field(
        ...,
        ge=0,
        description=(
            "Sustained wind speed near the observation point. "
            "Unit: metres per second."
        ),
    )

    gust_m_s: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional wind gust speed near the observation point. "
            "Use None if unavailable. Unit: metres per second."
        ),
    )

    direction_from: CompassBearing = Field(
        ...,
        description=(
            "Meteorological wind direction, meaning the direction the wind blows from, "
            "not the direction it blows toward. Unit: categorical compass bearing."
        ),
    )

    observed_at: datetime | None = Field(
        default=None,
        description=(
            "Timestamp for the wind observation. Prefer timezone-aware ISO 8601 "
            "datetime values. Use None if unavailable. Unit: datetime."
        ),
    )


class OperationalContext(StrictBaseModel):
    """Structured operational context for wildfire decision support.

    This model is the contract between GIS/context extraction, prompt formatting,
    and LLM inference.
    """

    schema_version: Literal["1.0"] = Field(
        default="1.0",
        description="Version of the operational context schema. Unit: none.",
    )

    coordinates: Coordinates = Field(
        ...,
        description="Observation coordinates. Units documented in Coordinates.",
    )

    region: Region = Field(
        ...,
        description="Administrative or operational region. Units documented in Region.",
    )

    land_cover: LandCoverType = Field(
        ...,
        description=(
            "Dominant land-cover class at the observation point. "
            "Unit: categorical label."
        ),
    )

    terrain: Terrain = Field(
        ...,
        description=(
            "Terrain context at or near the observation point. "
            "Units documented in Terrain."
        ),
    )

    nearest_water_source: NearestWaterSource = Field(
        ...,
        description=(
            "Nearest known water source relative to the observation point. "
            "Units documented in NearestWaterSource."
        ),
    )

    nearest_road: NearestRoad = Field(
        ...,
        description=(
            "Nearest known road or access route relative to the observation point. "
            "Units documented in NearestRoad."
        ),
    )

    nearest_settlement: NearestSettlement = Field(
        ...,
        description=(
            "Nearest known settlement relative to the observation point. "
            "Units documented in NearestSettlement."
        ),
    )

    # Wind is intentionally optional for now. It is included in the schema so the
    # interface is ready if weather/wind support is added later, but current GIS
    # context extraction does not need to provide it.
    wind: Wind | None = Field(
        default=None,
        description=(
            "Optional wind context near the observation point. "
            "Use None if wind integration is not available. "
            "Units documented in Wind."
        ),
    )