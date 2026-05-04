from copy import deepcopy

from pipeline.context.format import format_context
from pipeline.context.schemas import OperationalContext


SAMPLE_CONTEXT = {
    "schema_version": "1.0",
    "coordinates": {
        "latitude": 57.7089,
        "longitude": 11.9746,
    },
    "region": {
        "name": "Gothenburg Municipality",
        "country_code": "SE",
        "admin_area": "Västra Götaland County",
    },
    "land_cover": "mixed_forest",
    "terrain": {
        "elevation_m": 82.0,
        "slope_degrees": 18.0,
        "slope_steepness": "steep",
        "aspect": "SW",
    },
    "nearest_water_source": {
        "name": "Delsjön",
        "source_type": "lake",
        "distance_m": 450.0,
        "bearing": "E",
    },
    "nearest_road": {
        "name": "Road 40",
        "road_class": "primary",
        "distance_m": 320.0,
        "bearing": "NW",
    },
    "nearest_settlement": {
        "name": "Gothenburg",
        "settlement_type": "city",
        "distance_m": 3000.0,
        "bearing": "W",
    },
    "wind": None,
}


def test_format_context_matches_documented_template():
    context = OperationalContext.model_validate(SAMPLE_CONTEXT)

    formatted = format_context(context)

    expected = "\n".join(
        [
            "Operational context:",
            "- Location: Gothenburg Municipality, SE (57.7089, 11.9746)",
            "- Land cover: mixed forest",
            "- Terrain: 82 m elevation, steep slope (32%), aspect SW",
            "- Water: lake Delsjön, 0.45 km E",
            "- Road: primary road Road 40, 0.32 km NW",
            "- Settlement: city Gothenburg, 3.00 km W",
            "- Wind: unavailable",
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

    context_data["region"]["country_code"] = None
    context_data["terrain"]["elevation_m"] = None
    context_data["terrain"]["slope_degrees"] = None
    context_data["nearest_water_source"]["name"] = None
    context_data["nearest_water_source"]["bearing"] = None
    context_data["nearest_road"]["name"] = None
    context_data["nearest_road"]["bearing"] = None
    context_data["nearest_settlement"]["name"] = None
    context_data["nearest_settlement"]["bearing"] = None

    context = OperationalContext.model_validate(context_data)

    formatted = format_context(context)

    assert "unknown country" in formatted
    assert "unknown elevation" in formatted
    assert "unknown%)" not in formatted
    assert "unknown" in formatted
    assert "None" not in formatted


def test_format_context_includes_inline_units():
    context = OperationalContext.model_validate(SAMPLE_CONTEXT)

    formatted = format_context(context)

    assert "0.45 km" in formatted
    assert "0.32 km" in formatted
    assert "3.00 km" in formatted
    assert "32%" in formatted
    assert "82 m elevation" in formatted


def test_format_context_formats_optional_wind_when_available():
    context_data = deepcopy(SAMPLE_CONTEXT)
    context_data["wind"] = {
        "speed_m_s": 5.0,
        "gust_m_s": None,
        "direction_from": "SW",
        "observed_at": None,
    }

    context = OperationalContext.model_validate(context_data)

    formatted = format_context(context)

    assert "- Wind: 5.0 m/s from SW" in formatted