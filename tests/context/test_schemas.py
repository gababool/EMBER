import json
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
    "wind": None,
    "extraction_metadata": {
        "extracted_at": "2026-05-04T13:00:00+02:00",
        "osm_dataset": "sweden-latest.osm.pbf",
        "settlement_radius_m": 10000.0,
        "water_source_radius_m": 5000.0,
        "track_radius_m": 2000.0,
        "named_feature_radius_m": 5000.0,
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

    assert context.terrain is not None
    assert context.terrain.elevation_m == 82.0
    assert context.terrain.slope_degrees == 18.0
    assert context.terrain.slope_steepness == "steep"
    assert context.terrain.aspect == "SW"

    assert len(context.water_sources) == 1
    assert context.water_sources[0].source_type == "lake"

    assert context.roads.primary_access is not None
    assert context.roads.primary_access.road_class == "primary"
    assert context.roads.primary_access.vehicle_accessible is True

    assert len(context.settlements) == 1
    assert context.settlements[0].settlement_type == "city"

    assert len(context.named_features) == 1
    assert context.named_features[0].feature_type == "forest"

    assert context.wind is None
    assert context.extraction_metadata.osm_dataset == "sweden-latest.osm.pbf"


def test_operational_context_serializes_and_deserializes_json_cleanly():
    context = OperationalContext.model_validate(SAMPLE_CONTEXT)

    json_data = context.model_dump_json()
    decoded = json.loads(json_data)

    assert decoded["schema_version"] == "2.0"
    assert decoded["coordinates"]["latitude"] == 57.7089
    assert decoded["terrain"]["slope_steepness"] == "steep"
    assert decoded["water_sources"][0]["distance_m"] == 450.0
    assert decoded["roads"]["primary_access"]["distance_m"] == 320.0
    assert decoded["settlements"][0]["distance_m"] == 3000.0
    assert decoded["extraction_metadata"]["osm_dataset"] == "sweden-latest.osm.pbf"

    round_tripped = OperationalContext.model_validate_json(json_data)

    assert round_tripped == context


def test_sample_context_json_file_validates():
    sample_path = Path("pipeline/context/examples/sample_context.json")

    context = OperationalContext.model_validate_json(sample_path.read_text())

    assert context.schema_version == "2.0"
    assert context.land_cover == "mixed_forest"
    assert context.terrain is not None
    assert context.terrain.slope_steepness == "steep"
    assert context.wind is None
    assert context.extraction_metadata.osm_dataset == "sweden-latest.osm.pbf"


def test_operational_context_json_schema_can_be_generated():
    schema = OperationalContext.model_json_schema()

    assert schema["title"] == "OperationalContext"
    assert "properties" in schema
    assert "coordinates" in schema["properties"]
    assert "terrain" in schema["properties"]
    assert "water_sources" in schema["properties"]
    assert "roads" in schema["properties"]
    assert "settlements" in schema["properties"]
    assert "named_features" in schema["properties"]
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
    invalid_context["roads"]["primary_access"]["distance_m"] = -10.0

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(invalid_context)


def test_operational_context_rejects_missing_required_roads():
    invalid_context = deepcopy(SAMPLE_CONTEXT)
    invalid_context.pop("roads")

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(invalid_context)


def test_operational_context_rejects_missing_extraction_metadata():
    invalid_context = deepcopy(SAMPLE_CONTEXT)
    invalid_context.pop("extraction_metadata")

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(invalid_context)


def test_operational_context_rejects_extra_fields():
    invalid_context = deepcopy(SAMPLE_CONTEXT)
    invalid_context["unexpected_field"] = "should not be accepted"

    with pytest.raises(ValidationError):
        OperationalContext.model_validate(invalid_context)