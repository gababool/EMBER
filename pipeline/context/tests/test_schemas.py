from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

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


def test_operational_context_validates_sample_context_dict():
    context = OperationalContext.model_validate(SAMPLE_CONTEXT)

    assert context.schema_version == "2.0"
    assert context.coordinates.latitude == 57.7089
    assert context.coordinates.longitude == 11.9746
    assert context.region.name == "Gothenburg Municipality"
    assert context.region.admin_area == "Västra Götaland County"
    assert context.land_cover == "mixed_forest"
    assert context.water_sources[0].supply_category == "heavy"
    assert context.roads.primary_access.vehicle_accessible is True
    assert context.assets_at_risk.buildings_within_radius == 3
    assert context.extraction_metadata.assets_radius_m == 2000.0


def test_operational_context_serializes_and_deserializes_json_cleanly():
    context = OperationalContext.model_validate(SAMPLE_CONTEXT)

    serialized = context.model_dump_json()
    deserialized = OperationalContext.model_validate_json(serialized)

    assert deserialized == context


def test_sample_context_json_file_validates():
    sample_path = Path("pipeline/context/examples/sample_context.json")

    context = OperationalContext.model_validate_json(sample_path.read_text())

    assert context.schema_version == "2.0"
    assert context.water_sources[0].supply_category == "heavy"
    assert context.assets_at_risk.assets_radius_m == 2000.0


def test_operational_context_json_schema_can_be_generated():
    schema = OperationalContext.model_json_schema()

    assert schema["title"] == "OperationalContext"
    assert "properties" in schema
    assert "coordinates" in schema["properties"]
    assert "terrain" in schema["properties"]
    assert "water_sources" in schema["properties"]
    assert "roads" in schema["properties"]
    assert "assets_at_risk" in schema["properties"]
    assert "extraction_metadata" in schema["properties"]


def test_operational_context_accepts_missing_optional_wind():
    context_data = deepcopy(SAMPLE_CONTEXT)
    context_data.pop("wind")

    context = OperationalContext.model_validate(context_data)

    assert context.wind is None


def test_operational_context_accepts_explicit_null_wind():
    context_data = deepcopy(SAMPLE_CONTEXT)
    context_data["wind"] = None

    context = OperationalContext.model_validate(context_data)

    assert context.wind is None


def test_operational_context_accepts_missing_optional_terrain():
    context_data = deepcopy(SAMPLE_CONTEXT)
    context_data["terrain"] = None

    context = OperationalContext.model_validate(context_data)

    assert context.terrain is None


def test_operational_context_accepts_empty_context_lists():
    context_data = deepcopy(SAMPLE_CONTEXT)
    context_data["water_sources"] = []
    context_data["roads"]["nearby_tracks"] = []
    context_data["settlements"] = []
    context_data["named_features"] = []

    context = OperationalContext.model_validate(context_data)

    assert context.water_sources == []
    assert context.roads.nearby_tracks == []
    assert context.settlements == []
    assert context.named_features == []


def test_operational_context_rejects_invalid_land_cover():
    context_data = deepcopy(SAMPLE_CONTEXT)
    context_data["land_cover"] = "invalid_land_cover"

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(context_data)


def test_operational_context_rejects_invalid_slope_steepness():
    context_data = deepcopy(SAMPLE_CONTEXT)
    context_data["terrain"]["slope_steepness"] = "vertical"

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(context_data)


def test_operational_context_rejects_invalid_coordinates():
    context_data = deepcopy(SAMPLE_CONTEXT)
    context_data["coordinates"]["latitude"] = 120.0

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(context_data)


def test_operational_context_rejects_invalid_slope_degrees():
    context_data = deepcopy(SAMPLE_CONTEXT)
    context_data["terrain"]["slope_degrees"] = -1.0

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(context_data)


def test_operational_context_rejects_negative_distances():
    context_data = deepcopy(SAMPLE_CONTEXT)
    context_data["water_sources"][0]["distance_m"] = -10.0

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(context_data)


def test_operational_context_rejects_missing_required_roads():
    context_data = deepcopy(SAMPLE_CONTEXT)
    context_data.pop("roads")

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(context_data)


def test_operational_context_rejects_missing_extraction_metadata():
    context_data = deepcopy(SAMPLE_CONTEXT)
    context_data.pop("extraction_metadata")

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(context_data)


def test_operational_context_rejects_extra_fields():
    context_data = deepcopy(SAMPLE_CONTEXT)
    context_data["unexpected_field"] = "not allowed"

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(context_data)