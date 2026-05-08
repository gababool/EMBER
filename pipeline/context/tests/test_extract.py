from pathlib import Path

from pipeline.context import extract
from pipeline.context.schemas import OperationalContext
from pipeline.context.tests.test_schemas import SAMPLE_CONTEXT
from pipeline.context.wind import build_mock_wind


def _stub_extraction_queries(monkeypatch):
    sample = OperationalContext.model_validate(SAMPLE_CONTEXT)

    monkeypatch.setattr(extract, "_ensure_loaded", lambda pbf_path, dem_path=None: None)
    monkeypatch.setattr(extract, "_make_point_gdf", lambda lat, lon: object())
    monkeypatch.setattr(extract, "_query_region", lambda point_gdf: sample.region)
    monkeypatch.setattr(extract, "_query_land_cover", lambda point_gdf: sample.land_cover)
    monkeypatch.setattr(extract, "_query_water_sources", lambda point_gdf, radius_m: sample.water_sources)
    monkeypatch.setattr(extract, "_query_roads", lambda point_gdf, radius_m: sample.roads)
    monkeypatch.setattr(extract, "_query_settlements", lambda point_gdf, radius_m: sample.settlements)
    monkeypatch.setattr(extract, "_query_named_features", lambda point_gdf, radius_m: sample.named_features)
    monkeypatch.setattr(extract, "_query_assets_at_risk", lambda point_gdf, radius_m: sample.assets_at_risk)


def test_extract_context_keeps_wind_null_by_default(monkeypatch):
    _stub_extraction_queries(monkeypatch)

    context = extract.extract_context(
        lat=57.7089,
        lon=11.9746,
        pbf_path=Path("scenario_01-50km.osm.pbf"),
    )

    assert context.wind is None


def test_extract_context_adds_reproducible_mocked_wind_when_enabled(monkeypatch):
    _stub_extraction_queries(monkeypatch)

    context = extract.extract_context(
        lat=57.7089,
        lon=11.9746,
        pbf_path=Path("scenario_01-50km.osm.pbf"),
        mock_wind=True,
        mock_wind_key="scenario_01",
    )

    assert context.wind is not None
    assert context.wind.direction_compass in {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
    assert context.wind.direction_degrees in {0, 45, 90, 135, 180, 225, 270, 315}
    assert 1.0 <= context.wind.speed_mps <= 10.0
    assert context.wind.source == "mocked"

    repeated_context = extract.extract_context(
        lat=57.7089,
        lon=11.9746,
        pbf_path=Path("scenario_01-50km.osm.pbf"),
        mock_wind=True,
        mock_wind_key="scenario_01",
    )

    assert repeated_context.wind == context.wind


def test_mocked_wind_varies_by_key():
    first = build_mock_wind("scenario_01")
    second = build_mock_wind("scenario_02")

    assert first != second


def test_mocked_wind_can_use_fixed_benchmark_values():
    first = build_mock_wind(
        "scenario_01",
        direction_compass="SW",
        speed_mps=6.5,
    )
    second = build_mock_wind(
        "scenario_02",
        direction_compass="SW",
        speed_mps=6.5,
    )

    assert first == second
    assert first.direction_degrees == 225
    assert first.direction_compass == "SW"
    assert first.speed_mps == 6.5
