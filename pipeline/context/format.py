"""Formatter for operational context passed to the LLM.

Converts an OperationalContext into a concise plain-text block included in
the LLM user message alongside the fire image. Extraction metadata is
intentionally excluded — it is for reproducibility only and wastes tokens.

Design principles:
- Every field that is absent or null produces a graceful fallback, never
  exposes Python None, and never crashes.
- The output is consistently structured so the LLM can parse it reliably
  across runs.
- The block is kept under roughly 200 tokens to stay within the context
  budget of the smallest thesis-core model (Gemma 4 E2B, 8k tokens).
- New fields from the enriched schema (supply category, assets at risk,
  road weight/bridge, multiple water sources and settlements) are included
  selectively: enough detail to inform tactical reasoning, not so much that
  the block becomes hard to parse for a small model.

Output template:

    OPERATIONAL CONTEXT
    Location: <municipality>, <county>
    Land cover: <land cover>
    Terrain: <elevation and slope> | unavailable

    Water sources (<n> within 10 km):
      - <supply> supply: <type> <name>, <distance> <bearing>, road access <distance>
      ...

    Road access:
      Primary: <class> <name>, <distance> <bearing>[, max <weight>t][, bridge]
      Tracks: <class> <name>, <distance> <bearing>, [accessible / not confirmed]
      ...

    Settlements (<n> within 20 km):
      - <type> <name>, <distance> <bearing>
      ...

    Assets at risk (<radius>):
      Buildings: <count> [(<permanent/no permanent structures>)]
      Power lines: present | none mapped
      Protected area: <name> | none

    Wind: <speed> m/s from <bearing> | unavailable
"""

from __future__ import annotations

import math
from typing import Sequence

from pipeline.context.schemas import (
    AssetsAtRisk,
    OperationalContext,
    Road,
    Settlement,
    WaterSource,
)


# ── Primitive formatters ───────────────────────────────────────────────────────

def _label(value: str | None) -> str:
    """Convert schema snake_case labels to readable text, e.g. 'mixed_forest' → 'mixed forest'."""
    if not value:
        return "unknown"
    return value.replace("_", " ")


def _km(distance_m: float | None, decimals: int = 1) -> str:
    """Format a distance in metres as kilometres."""
    if distance_m is None:
        return "unknown distance"
    return f"{distance_m / 1000:.{decimals}f} km"


def _bearing(value: str | None) -> str:
    if not value:
        return ""
    return value


def _slope_pct(slope_degrees: float | None) -> str:
    if slope_degrees is None:
        return "unknown%"
    return f"{math.tan(math.radians(slope_degrees)) * 100:.0f}%"


# ── Section formatters ─────────────────────────────────────────────────────────

def _format_terrain(context: OperationalContext) -> str:
    t = context.terrain
    if t is None:
        return "Terrain: unavailable"
    parts = []
    if t.elevation_m is not None:
        parts.append(f"{t.elevation_m:.0f} m elevation")
    if t.slope_steepness != "unknown":
        slope_str = _label(t.slope_steepness) + " slope"
        if t.slope_degrees is not None:
            slope_str += f" ({_slope_pct(t.slope_degrees)})"
        parts.append(slope_str)
    if t.aspect not in (None, "unknown", "flat"):
        parts.append(f"{t.aspect}-facing")
    elif t.aspect == "flat":
        parts.append("flat terrain")
    return "Terrain: " + (", ".join(parts) if parts else "unavailable")


def _format_water_sources(sources: Sequence[WaterSource], radius_m: float) -> str:
    radius_km = f"{radius_m / 1000:.0f} km"
    if not sources:
        return f"Water sources (none within {radius_km})"

    lines = [f"Water sources ({len(sources)} within {radius_km}):"]
    for ws in sources:
        name = f" {ws.name}" if ws.name else ""
        supply = _label(ws.supply_category)
        type_str = _label(ws.source_type)
        dist = _km(ws.distance_m)
        brng = _bearing(ws.bearing)
        road = (
            f", road access {_km(ws.nearest_road_distance_m)}"
            if ws.nearest_road_distance_m is not None
            else ""
        )
        lines.append(f"  - {supply} supply: {type_str}{name}, {dist} {brng}{road}")

    return "\n".join(lines)


def _format_road(road: Road) -> str:
    """Format a single road entry for the road access section."""
    name = f" {road.name}" if road.name else ""
    cls = _label(road.road_class) + " road"
    dist = _km(road.distance_m)
    brng = _bearing(road.bearing)

    extras = []
    if road.max_weight_tonnes is not None:
        extras.append(f"max {road.max_weight_tonnes:.0f}t")
    if road.has_bridge:
        extras.append("bridge present")
    extra_str = f" ({', '.join(extras)})" if extras else ""

    return f"{cls}{name}, {dist} {brng}{extra_str}"


def _format_roads(context: OperationalContext) -> str:
    lines = ["Road access:"]
    primary = context.roads.primary_access
    if primary:
        lines.append(f"  Primary: {_format_road(primary)}")
    else:
        lines.append("  Primary: none within search radius")

    tracks = context.roads.nearby_tracks
    if tracks:
        for t in tracks:
            accessible = "accessible" if t.vehicle_accessible else "not confirmed accessible"
            lines.append(f"  Track: {_format_road(t)}, {accessible}")
    else:
        lines.append("  Tracks: none nearby")

    return "\n".join(lines)


def _format_settlements(settlements: Sequence[Settlement], radius_m: float) -> str:
    radius_km = f"{radius_m / 1000:.0f} km"
    if not settlements:
        return f"Settlements (none within {radius_km})"

    lines = [f"Settlements ({len(settlements)} within {radius_km}):"]
    for s in settlements:
        name = f" {s.name}" if s.name else ""
        stype = _label(s.settlement_type)
        dist = _km(s.distance_m)
        brng = _bearing(s.bearing)
        lines.append(f"  - {stype}{name}, {dist} {brng}")

    return "\n".join(lines)


def _format_assets(assets: AssetsAtRisk) -> str:
    radius_km = f"{assets.assets_radius_m / 1000:.1f} km"
    lines = [f"Assets at risk (within {radius_km}):"]

    building_str = str(assets.buildings_within_radius)
    if assets.buildings_within_radius > 0:
        perm = "including permanent structures" if assets.has_permanent_structures else "no permanent structures"
        building_str += f" ({perm})"
    lines.append(f"  Buildings: {building_str}")

    power_str = "present" if assets.power_lines_present else "none mapped"
    lines.append(f"  Power lines: {power_str}")

    protected_str = assets.protected_area if assets.protected_area else "none"
    lines.append(f"  Protected area: {protected_str}")

    return "\n".join(lines)

def _format_named_features(context: OperationalContext) -> list[str]:
    """Return named feature lines, or empty list if none present."""
    if not context.named_features:
        return []
    lines = ["Named features:"]
    for f in context.named_features:
        brng = _bearing(f.bearing)
        lines.append(f"  - {_label(f.feature_type)} {f.name}, {_km(f.distance_m)} {brng}")
    return lines


def _format_wind(context: OperationalContext) -> str:
    if context.wind is None:
        return "Wind: unavailable"
    return f"Wind: {context.wind.speed_m_s:.1f} m/s from {context.wind.direction_from}"


# ── Public API ────────────────────────────────────────────────────────────────

def format_context(context: OperationalContext) -> str:
    """Format an OperationalContext as a concise LLM-ready plain-text block.

    Extraction metadata is excluded. All optional fields degrade gracefully.
    The output is consistently structured so the LLM can reliably parse it.

    Args:
        context: A validated OperationalContext.

    Returns:
        A plain-text context block ready for inclusion in the LLM user message.
    """
    region = context.region.name or "unknown location"
    county = context.region.admin_area or "unknown county"
    location = f"Location: {region}, {county}"

    land_cover = f"Land cover: {_label(context.land_cover)}"

    terrain = _format_terrain(context)

    water_radius = context.extraction_metadata.water_source_radius_m
    settlement_radius = context.extraction_metadata.settlement_radius_m

    water = _format_water_sources(context.water_sources, water_radius)
    roads = _format_roads(context)
    settlements = _format_settlements(context.settlements, settlement_radius)
    assets = _format_assets(context.assets_at_risk)
    wind = _format_wind(context)

    named = _format_named_features(context)

    sections = [
        "OPERATIONAL CONTEXT",
        location,
        land_cover,
        terrain,
        "",
        water,
        "",
        roads,
        "",
        settlements,
        "",
        assets,
    ]

    if named:
        sections += ["", *named]

    sections += ["", wind]

    return "\n".join(sections)