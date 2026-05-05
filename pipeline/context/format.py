"""Formatter for operational context passed to the LLM.

The formatter converts an OperationalContext object into a concise plain-text
block that can be included in the LLM user message.

Documented output template:

Operational context:
- Location: <region>, <admin_area> (<lat>, <lon>)
- Land cover: <land_cover>
- Terrain: <elevation>, <slope_steepness> slope (<slope_percent>%), aspect <aspect> | unavailable
- Water: <type/name>, <distance_km> <bearing>, road <distance_km> away | unavailable
- Road: <road_class/name>, <distance_km> <bearing>, accessible <yes/no> | unavailable
- Settlement: <type/name>, <distance_km> <bearing> | unavailable
- Wind: <speed_m_s> m/s from <bearing> | unavailable

Extraction metadata is intentionally excluded from the formatted prompt block.

The output should stay under roughly 150 tokens so it remains suitable for
LLM prompts.
"""

from __future__ import annotations

import math

from pipeline.context.schemas import OperationalContext, Road, Settlement, WaterSource


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


def _metres_to_km(distance_m: float | None) -> str:
    """Convert metres to kilometres with two decimals."""
    if distance_m is None:
        return "unknown km"

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


def _format_water_source(water_source: WaterSource | None) -> str:
    """Format the most operationally relevant water source."""
    if water_source is None:
        return "- Water: unavailable"

    name = (
        f" {_format_optional_text(water_source.name)}"
        if water_source.name
        else ""
    )

    road_access = ""
    if water_source.nearest_road_distance_m is not None:
        road_access = (
            f", road {_metres_to_km(water_source.nearest_road_distance_m)} away"
        )

    return (
        "- Water: "
        f"{_humanize_label(water_source.source_type)}{name}, "
        f"{_metres_to_km(water_source.distance_m)} "
        f"{_format_bearing(water_source.bearing)}"
        f"{road_access}"
    )


def _format_road(road: Road | None) -> str:
    """Format the primary vehicle-accessible road."""
    if road is None:
        return "- Road: unavailable"

    name = f" {_format_optional_text(road.name)}" if road.name else ""
    accessible = "yes" if road.vehicle_accessible else "no"

    return (
        "- Road: "
        f"{_humanize_label(road.road_class)} road{name}, "
        f"{_metres_to_km(road.distance_m)} "
        f"{_format_bearing(road.bearing)}, "
        f"accessible {accessible}"
    )


def _format_settlement(settlement: Settlement | None) -> str:
    """Format the nearest settlement."""
    if settlement is None:
        return "- Settlement: unavailable"

    name = f" {_format_optional_text(settlement.name)}" if settlement.name else ""

    return (
        "- Settlement: "
        f"{_humanize_label(settlement.settlement_type)}{name}, "
        f"{_metres_to_km(settlement.distance_m)} "
        f"{_format_bearing(settlement.bearing)}"
    )


def format_context(context: OperationalContext) -> str:
    """Format operational context as a concise LLM-ready text block.

    Args:
        context: Validated operational context.

    Returns:
        A consistently structured plain-text context block.
    """
    admin_area = context.region.admin_area or "unknown admin area"
    region = _format_optional_text(context.region.name)

    location = (
        f"- Location: {region}, {admin_area} "
        f"({context.coordinates.latitude:.4f}, {context.coordinates.longitude:.4f})"
    )

    land_cover = f"- Land cover: {_humanize_label(context.land_cover)}"

    if context.terrain is None:
        terrain = "- Terrain: unavailable"
    else:
        terrain = (
            "- Terrain: "
            f"{_format_elevation(context.terrain.elevation_m)}, "
            f"{_humanize_label(context.terrain.slope_steepness)} slope "
            f"({_slope_degrees_to_percent(context.terrain.slope_degrees)}), "
            f"aspect {context.terrain.aspect}"
        )

    primary_water_source = context.water_sources[0] if context.water_sources else None
    primary_settlement = context.settlements[0] if context.settlements else None

    water = _format_water_source(primary_water_source)
    road = _format_road(context.roads.primary_access)
    settlement = _format_settlement(primary_settlement)

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