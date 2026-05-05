from copy import deepcopy

from pipeline.context.format import format_context
from pipeline.context.schemas import OperationalContext


SAMPLE_CONTEXT = {
    "schema_version": "2.0",
    "coordinates": {
        "latitude": 57.7089,
        "longitude": 11.9746,
    },
    "region": {
        "name": "Gothenburg Municipality",
        "admin_area": "Västra Götaland County",
    },
    "land_cover": "mixed_forest",
    "terrain": {
        "elevation_m": 82.0,
        "slope_degrees": 18.0,
        "slope_steepness": "steep",
        "aspect": "SW",
    },
    "water_sources": [
        {
            "name": "Delsjön",
            "source_type": "lake",
            "supply_category": "heavy",
            "distance_m": 450.0,
            "bearing": "E",
            "area_m2": 1370000.0,
            "nearest_road_distance_m": 120.0,
        }
    ],
    "roads": {
        "primary_access": {
            "name": "Road 40",
            "road_class": "primary",
            "surface": "asphalt",
            "tracktype": None,
            "vehicle_accessible": True,
            "distance_m": 320.0,
            "bearing": "NW",
        },
        "nearby_tracks": [
            {
                "name": "Forest track",
                "road_class": "track",
                "surface": "gravel",
                "tracktype": "grade2",
                "vehicle_accessible": True,
                "distance_m": 180.0,
                "bearing": "N",
            }
        ],
    },
    "settlements": [
        {
            "name": "Gothenburg",
            "settlement_type": "city",
            "distance_m": 3000.0,
            "bearing": "W",
        }
    ],
    "named_features": [
        {
            "name": "Delsjöområdet",
            "feature_type": "forest",
            "distance_m": 250.0,
            "bearing": "E",
        }
    ],
    "assets_at_risk": {
        "buildings_within_radius": 3,
        "has_permanent_structures": True,
        "power_lines_present": False,
        "protected_area": None,
        "assets_radius_m": 2000.0,
    },
    "wind": None,
    "extraction_metadata": {
        "extracted_at": "2026-05-04T13:00:00+02:00",
        "osm_dataset": "sweden-latest.osm.pbf",
        "settlement_radius_m": 10000.0,
        "water_source_radius_m": 5000.0,
        "track_radius_m": 2000.0,
        "named_feature_radius_m": 5000.0,
        "assets_radius_m": 2000.0,
    },
}


def test_format_context_matches_documented_template():
    context = OperationalContext.model_validate(SAMPLE_CONTEXT)

    formatted = format_context(context)

    expected = "\n".join(
        [
            "OPERATIONAL CONTEXT",
            "Location: Gothenburg Municipality, Västra Götaland County",
            "Land cover: mixed forest",
            "Terrain: 82 m elevation, steep slope (32%), SW-facing",
            "",
            "Water sources (1 within 5 km):",
            "  - heavy supply: lake Delsjön, 0.5 km E, road access 0.1 km",
            "",
            "Road access:",
            "  Primary: primary road Road 40, 0.3 km NW",
            "  Track: track road Forest track, 0.2 km N, accessible",
            "",
            "Settlements (1 within 10 km):",
            "  - city Gothenburg, 3.0 km W",
            "",
            "Assets at risk (within 2.0 km):",
            "  Buildings: 3 (including permanent structures)",
            "  Power lines: none mapped",
            "  Protected area: none",
            "",
            "Named features:",
            "  - forest Delsjöområdet, 0.2 km E",
            "",
            "Wind: unavailable",
        ]
    )

    assert formatted == expected


def test_format_context_stays_under_token_target():
    context = OperationalContext.model_validate(SAMPLE_CONTEXT)

    formatted = format_context(context)

    approximate_token_count = len(formatted.split())

    assert approximate_token_count <= 150


def test_format_context_handles_missing_optional_fields():
    context_data = deepcopy(SAMPLE_CONTEXT)

    context_data["region"]["admin_area"] = None
    context_data["terrain"] = None
    context_data["water_sources"] = []
    context_data["roads"]["primary_access"] = None
    context_data["roads"]["nearby_tracks"] = []
    context_data["settlements"] = []
    context_data["named_features"] = []
    context_data["wind"] = None

    context = OperationalContext.model_validate(context_data)

    formatted = format_context(context)

    assert "Terrain: unavailable" in formatted
    assert "Water sources (none within 5 km)" in formatted
    assert "Primary: none within search radius" in formatted
    assert "Tracks: none nearby" in formatted
    assert "Settlements (none within 10 km)" in formatted
    assert "Wind: unavailable" in formatted
    assert "unknown county" in formatted
    assert "None" not in formatted


def test_format_context_includes_inline_units():
    context = OperationalContext.model_validate(SAMPLE_CONTEXT)

    formatted = format_context(context)

    assert "0.5 km" in formatted
    assert "0.1 km" in formatted
    assert "0.3 km" in formatted
    assert "3.0 km" in formatted
    assert "2.0 km" in formatted
    assert "32%" in formatted
    assert "82 m elevation" in formatted


def test_format_context_formats_optional_wind_when_available():
    context_data = deepcopy(SAMPLE_CONTEXT)
    context_data["wind"] = {
        "speed_m_s": 5.0,
        "direction_from": "SW",
    }

    context = OperationalContext.model_validate(context_data)

    formatted = format_context(context)

    assert "Wind: 5.0 m/s from SW" in formatted


def test_format_context_excludes_extraction_metadata():
    context = OperationalContext.model_validate(SAMPLE_CONTEXT)

    formatted = format_context(context)

    assert "extracted_at" not in formatted
    assert "osm_dataset" not in formatted
    assert "sweden-latest.osm.pbf" not in formatted