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
Wind, when available, is stored in metres per second.

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
- Wind: <speed_m_s> m/s from <bearing> | unavailable
```

Distances are formatted in kilometres for readability.
Slope is formatted as a percentage.
Wind is formatted in metres per second.

The output is intentionally short and consistently structured so it can be inserted into the LLM prompt without adding unnecessary noise.

## Example

See `examples/sample_context.json`.

## Run tests

```bash
uv run pytest pipeline/context/tests -v
```