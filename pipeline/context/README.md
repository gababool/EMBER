# Operational Context Schema

This module defines the structured operational context passed from GIS extraction to prompt formatting and LLM inference.

The schema is implemented with Pydantic and validates:
- coordinates
- administrative/operational region
- land cover
- terrain
- nearest water source
- nearest road
- nearest settlement

All distances are in metres.
Coordinates use WGS84 decimal degrees.
Slope and aspect describe local terrain around the observation point.

## Example

See `examples/sample_context.json`.

## Run tests

```bash
uv run pytest tests/context/test_schemas.py