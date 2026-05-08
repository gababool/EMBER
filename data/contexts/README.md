# Scenario Context Files

This folder contains generated operational wildfire scenario contexts used for
benchmarking, qualitative evaluation, and stakeholder interviews.

Each scenario folder contains:
- `scenario_NN.json` — pre-computed GIS context (committed)
- `scenario_NN.jpg` / `scenario_NN.png` — paired wildfire image (committed)

The clipped `.osm.pbf` and `.tif` elevation files are stored under `data/gis/`
and committed. Only the full Sweden download (`sweden-latest.osm.pbf`) is gitignored.

---

## Scenario summaries

| Scenario | Location | Key features | What it tests |
|---|---|---|---|
| `scenario_01` | Acktjärnsåsarna, Västmanland | Wetland forest, protected area, named lakes, limited tracks | Water availability vs difficult access |
| `scenario_02` | Tyresta NP edge, Stockholm | 83 buildings, power lines, Tyresta by 666m, paved road | Wildland-urban interface, asset protection |
| `scenario_03` | Skuleskogen, High Coast | Steep terrain (manually set), named peaks, moderate remoteness | Terrain-driven fire behavior and access difficulty |
| `scenario_04` | Skåne countryside, Kristianstad | 72 buildings, forest, light water supply, rural roads | Dense assets, good access, no heavy water |
| `scenario_05` | Vakö myr, Kronoberg | Protected mire, flat, all light supply water, isolated | Smouldering peat risk, deceptive wet terrain |
| `scenario_06` | Sorsele remote forest, Västerbotten | 588m elevation, named tracks, light water only, sparse settlements | Remote access, minimal infrastructure, sparse OSM |

---

## Regenerating scenarios

From the repository root:

```bash
# Full pipeline: extract OSM + elevation, generate JSONs, validate
scripts/generate_context_scenarios.sh --mock-wind

# One scenario only
scripts/generate_context_scenarios.sh --scenario 03 --mock-wind

# Already have OSM/DEM, just regenerate JSONs
scripts/generate_context_scenarios.sh --generate-only --mock-wind --skip-tests

# Same wind for all scenarios (benchmark reproducibility)
scripts/generate_context_scenarios.sh --generate-only --benchmark-wind --skip-tests
```

Run `scripts/generate_context_scenarios.sh --help` for all options.

### Wind notes

`--mock-wind` adds a deterministic semi-random wind per scenario:

```json
"wind": {
  "direction_degrees": 135,
  "direction_compass": "SE",
  "speed_mps": 8.9,
  "source": "mocked"
}
```

`--benchmark-wind` writes the same value to every scenario (SW, 6.5 m/s) for
controlled benchmark runs. Wind is meteorological — the direction is where the
wind blows *from*, not toward. The formatter computes the spread direction automatically.

### Terrain notes

Terrain is derived from SRTM 30m elevation data via the `elevation` Python package.
Scenario 03 (Skuleskogen) has terrain manually adjusted to reflect known local
topography that SRTM 30m underestimates on the High Coast ridge.

---

## Manual reference — OSM extraction

Bounding box format: `west,south,east,north`

| Scenario | Bounding box | Lat | Lon |
|---|---|---|---|
| 01 | `15.653541,59.575422,16.546459,60.024578` | 59.8000 | 16.1000 |
| 02 | `17.790000,58.990000,18.690000,59.400000` | 59.17372 | 18.24039 |
| 03 | `18.000000,62.850000,18.200000,63.050000` | 62.94232 | 18.04936 |
| 04 | `13.370000,55.650000,14.270000,56.130000` | 55.890 | 13.820 |
| 05 | `13.846719,56.275423,14.660499,56.724579` | 56.500001 | 14.253609 |
| 06 | `16.200000,64.950000,17.200000,65.400000` | 65.16019 | 16.70643 |

Example (single scenario):

```bash
osmium extract \
  -b 15.653541,59.575422,16.546459,60.024578 \
  data/gis/osm/sweden-latest.osm.pbf \
  -o data/gis/osm/scenario_01-50km.osm.pbf \
  --overwrite
```

---

## Manual reference — Elevation download

```bash
eio clip -o data/gis/elevation/scenario_01.tif \
    --bounds 15.653541 59.575422 16.546459 60.024578
```

Note: `eio` expects bounds as space-separated `west south east north`.
GDAL must be installed (`brew install gdal`) for `eio` to work.

---

## Manual reference — Context extraction

```bash
uv run python -m pipeline.context.extract \
  --lat 59.8000 \
  --lon 16.1000 \
  --pbf data/gis/osm/scenario_01-50km.osm.pbf \
  --dem data/gis/elevation/scenario_01.tif \
  --output data/contexts/scenario_01/scenario_01.json
```

---

## Validation

```bash
# Validate JSON
for file in data/contexts/scenario_*/scenario_*.json; do
  python -m json.tool "$file" > /dev/null && echo "valid: $file"
done

# Run context tests
uv run pytest pipeline/context/tests -v

# Inspect a scenario
jq '{region, land_cover, terrain, assets_at_risk}' \
  data/contexts/scenario_01/scenario_01.json
```