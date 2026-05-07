# Operational Context

This module defines and formats the structured operational context passed from GIS extraction to prompt formatting and LLM inference.

## Schema

The schema is implemented with Pydantic and validates:

- coordinates
- administrative/operational region
- land cover
- terrain
- nearby water sources
- road access
- nearby settlements
- named natural features
- optional wind context
- extraction metadata

All distances in the schema are stored in metres.
Areas are stored in square metres.
Coordinates use WGS84 decimal degrees.
Elevation is stored in metres.
Slope is stored in degrees and can also include a categorical steepness label.
Aspect describes local terrain direction around the observation point.
Wind, when available, is stored as meteorological wind direction
(`direction_degrees` and `direction_compass`), speed in metres per second
(`speed_mps`), and a source label (`mocked` or `manual`).
Mocked wind is deterministic per scenario key so benchmark runs are
reproducible while still varying direction and speed between scenarios.
For controlled benchmark comparisons, mocked wind can also be fixed to the
same direction and speed for every generated scenario.

`extraction_metadata` is included in the JSON for reproducibility, but it is intentionally excluded from the prompt-ready text sent to the LLM.

## Formatter

`format_context(context)` converts a validated `OperationalContext` into a concise plain-text block for the LLM user message.

The formatted output follows this structure:

```text
Operational context:
- Location: <region>, <admin_area> (<lat>, <lon>)
- Land cover: <land_cover>
- Terrain: <elevation>, <slope_steepness> slope (<slope_percent>%), aspect <aspect> | unavailable
- Water: <type/name>, <distance_km> <bearing>, road <distance_km> away | unavailable
- Road: <road_class/name>, <distance_km> <bearing>, accessible <yes/no> | unavailable
- Settlement: <type/name>, <distance_km> <bearing> | unavailable
- Wind: <speed_mps> m/s from <bearing> (<degrees> degrees) | omitted when unavailable
```

Distances are formatted in kilometres for readability.
Slope is formatted as a percentage.
Wind is formatted in metres per second and omitted entirely when unavailable.

The output is intentionally short and consistently structured so it can be inserted into the LLM prompt without adding unnecessary noise.

## Extraction wind modes

Use `uv run` so the command uses the project environment and works across
machines with the repo dependencies installed.

```bash
# No wind. This keeps existing behaviour and writes "wind": null.
uv run python -m pipeline.context.extract \
  --lat 59.8 \
  --lon 16.1 \
  --pbf data/gis/osm/scenario_01-50km.osm.pbf \
  --output data/contexts/scenario_01.json

# Mocked semi-random wind. The key makes the value reproducible.
uv run python -m pipeline.context.extract \
  --lat 59.8 \
  --lon 16.1 \
  --pbf data/gis/osm/scenario_01-50km.osm.pbf \
  --output data/contexts/scenario_01.json \
  --mock-wind \
  --mock-wind-key scenario_01

# Fixed benchmark wind. Use this when wind should be controlled across prompts.
uv run python -m pipeline.context.extract \
  --lat 59.8 \
  --lon 16.1 \
  --pbf data/gis/osm/scenario_01-50km.osm.pbf \
  --output data/contexts/scenario_01.json \
  --mock-wind \
  --mock-wind-direction SW \
  --mock-wind-speed-mps 6.5
```

For all predefined scenarios, prefer `scripts/generate_context_scenarios.sh`.
It supports the same backend modes at scenario-batch level:

```bash
# No wind
scripts/generate_context_scenarios.sh --generate-only

# Reproducible semi-random wind per scenario
scripts/generate_context_scenarios.sh --generate-only --mock-wind

# Same fixed mocked wind for every scenario
scripts/generate_context_scenarios.sh --generate-only --benchmark-wind
```

## Example

See `examples/sample_context.json`.

## Run tests

```bash
uv run pytest pipeline/context/tests -v
```
