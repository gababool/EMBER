"""Formatter for operational context passed to the LLM.

The formatter converts an OperationalContext object into a concise plain-text
block that can be included in the LLM user message.

Documented output template:

Operational context:
- Location: <region>, <country> (<lat>, <lon>)
- Land cover: <land_cover>
- Terrain: <elevation>, <slope_steepness> slope (<slope_percent>%), aspect <aspect>
- Water: <type/name> <distance_km> km <bearing>
- Road: <road_class/name> <distance_km> km <bearing>
- Settlement: <type/name> <distance_km> km <bearing>
- Wind: <speed_m_s> m/s from <bearing> | unavailable

The output should stay under roughly 150 tokens so it remains suitable for
LLM prompts.
"""

from __future__ import annotations

import math

from pipeline.context.schemas import OperationalContext


def _humanize_label(value: str | None) -> str:
    """Convert schema labels like 'mixed_forest' into 'mixed forest'."""
    if value is None:
        return "unknown"

    return value.replace("_", " ")


def _format_optional_text(value: str | None) -> str:
    """Format optional text fields without exposing None in the prompt."""
    if value is None or value.strip() == "":
        return "unknown"

    return value


def _format_bearing(value: str | None) -> str:
    """Format optional compass bearings."""
    if value is None:
        return "unknown direction"

    return value


def _metres_to_km(distance_m: float) -> str:
    """Convert metres to kilometres with two decimals."""
    return f"{distance_m / 1000:.2f} km"


def _format_wind_speed(speed_m_s: float) -> str:
    """Format wind speed using metres per second.

    Wind speed is kept in m/s because this is the schema unit and a standard
    meteorological unit in Swedish/European operational contexts.
    """
    return f"{speed_m_s:.1f} m/s"


def _slope_degrees_to_percent(slope_degrees: float | None) -> str:
    """Convert slope degrees to slope percent.

    Slope percent is calculated as tan(slope_angle) * 100.
    """
    if slope_degrees is None:
        return "unknown"

    slope_percent = math.tan(math.radians(slope_degrees)) * 100

    return f"{slope_percent:.0f}%"


def _format_elevation(elevation_m: float | None) -> str:
    """Format elevation with unit suffix."""
    if elevation_m is None:
        return "unknown elevation"

    return f"{elevation_m:.0f} m elevation"


def format_context(context: OperationalContext) -> str:
    """Format operational context as a concise LLM-ready text block.

    Args:
        context: Validated operational context.

    Returns:
        A consistently structured plain-text context block.
    """
    country = context.region.country_code or "unknown country"
    region = _format_optional_text(context.region.name)

    location = (
        f"- Location: {region}, {country} "
        f"({context.coordinates.latitude:.4f}, {context.coordinates.longitude:.4f})"
    )

    land_cover = f"- Land cover: {_humanize_label(context.land_cover)}"

    terrain = (
        "- Terrain: "
        f"{_format_elevation(context.terrain.elevation_m)}, "
        f"{_humanize_label(context.terrain.slope_steepness)} slope "
        f"({_slope_degrees_to_percent(context.terrain.slope_degrees)}), "
        f"aspect {context.terrain.aspect}"
    )

    water_name = (
        f" {_format_optional_text(context.nearest_water_source.name)}"
        if context.nearest_water_source.name
        else ""
    )
    water = (
        "- Water: "
        f"{_humanize_label(context.nearest_water_source.source_type)}{water_name}, "
        f"{_metres_to_km(context.nearest_water_source.distance_m)} "
        f"{_format_bearing(context.nearest_water_source.bearing)}"
    )

    road_name = (
        f" {_format_optional_text(context.nearest_road.name)}"
        if context.nearest_road.name
        else ""
    )
    road = (
        "- Road: "
        f"{_humanize_label(context.nearest_road.road_class)} road{road_name}, "
        f"{_metres_to_km(context.nearest_road.distance_m)} "
        f"{_format_bearing(context.nearest_road.bearing)}"
    )

    settlement_name = (
        f" {_format_optional_text(context.nearest_settlement.name)}"
        if context.nearest_settlement.name
        else ""
    )
    settlement = (
        "- Settlement: "
        f"{_humanize_label(context.nearest_settlement.settlement_type)}{settlement_name}, "
        f"{_metres_to_km(context.nearest_settlement.distance_m)} "
        f"{_format_bearing(context.nearest_settlement.bearing)}"
    )

    if context.wind is None:
        wind = "- Wind: unavailable"
    else:
        wind = (
            "- Wind: "
            f"{_format_wind_speed(context.wind.speed_m_s)} "
            f"from {context.wind.direction_from}"
        )

    return "\n".join(
        [
            "Operational context:",
            location,
            land_cover,
            terrain,
            water,
            road,
            settlement,
            wind,
        ]
    )