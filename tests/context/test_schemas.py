import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

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


def test_operational_context_validates_sample_context_dict():
    context = OperationalContext.model_validate(SAMPLE_CONTEXT)

    assert context.schema_version == "1.0"
    assert context.coordinates.latitude == 57.7089
    assert context.coordinates.longitude == 11.9746
    assert context.land_cover == "mixed_forest"
    assert context.terrain.elevation_m == 82.0
    assert context.terrain.slope_degrees == 18.0
    assert context.terrain.slope_steepness == "steep"
    assert context.terrain.aspect == "SW"


def test_operational_context_serializes_and_deserializes_json_cleanly():
    context = OperationalContext.model_validate(SAMPLE_CONTEXT)

    json_data = context.model_dump_json()
    decoded = json.loads(json_data)

    assert decoded["schema_version"] == "1.0"
    assert decoded["coordinates"]["latitude"] == 57.7089
    assert decoded["terrain"]["slope_steepness"] == "steep"
    assert decoded["nearest_water_source"]["distance_m"] == 450.0

    round_tripped = OperationalContext.model_validate_json(json_data)

    assert round_tripped == context


def test_sample_context_json_file_validates():
    sample_path = Path("pipeline/context/examples/sample_context.json")

    context = OperationalContext.model_validate_json(sample_path.read_text())

    assert context.schema_version == "1.0"
    assert context.land_cover == "mixed_forest"
    assert context.terrain.slope_steepness == "steep"
    assert context.wind is None


def test_operational_context_json_schema_can_be_generated():
    schema = OperationalContext.model_json_schema()

    assert schema["title"] == "OperationalContext"
    assert "properties" in schema
    assert "coordinates" in schema["properties"]
    assert "terrain" in schema["properties"]
    assert "nearest_water_source" in schema["properties"]
    assert "nearest_road" in schema["properties"]
    assert "nearest_settlement" in schema["properties"]


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


def test_operational_context_rejects_invalid_land_cover():
    invalid_context = deepcopy(SAMPLE_CONTEXT)
    invalid_context["land_cover"] = "magic_fire_forest"

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(invalid_context)


def test_operational_context_rejects_invalid_slope_steepness():
    invalid_context = deepcopy(SAMPLE_CONTEXT)
    invalid_context["terrain"]["slope_steepness"] = "kind_of_hilly"

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(invalid_context)


def test_operational_context_rejects_invalid_coordinates():
    invalid_context = deepcopy(SAMPLE_CONTEXT)
    invalid_context["coordinates"]["latitude"] = 120.0

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(invalid_context)


def test_operational_context_rejects_invalid_slope_degrees():
    invalid_context = deepcopy(SAMPLE_CONTEXT)
    invalid_context["terrain"]["slope_degrees"] = 120.0

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(invalid_context)


def test_operational_context_rejects_negative_distances():
    invalid_context = deepcopy(SAMPLE_CONTEXT)
    invalid_context["nearest_road"]["distance_m"] = -10.0

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(invalid_context)


def test_operational_context_rejects_extra_fields():
    invalid_context = deepcopy(SAMPLE_CONTEXT)
    invalid_context["unexpected_field"] = "should not be accepted"

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(invalid_context)