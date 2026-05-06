# Scenario Context Files

This folder contains generated operational wildfire scenario contexts.

Each scenario is created in two steps:

1. Extract a local 50 × 50 km OpenStreetMap `.osm.pbf` file from `sweden-latest.osm.pbf`.
2. Run the GIS context extractor on that clipped `.osm.pbf` file to generate `scenario_XX.json`.

The `.osm.pbf` files are local GIS data and must **not** be committed.  
The generated `scenario_XX.json` files should be committed.

---

## Required local input file

Before running the commands, make sure this file exists:

```text
data/gis/osm/sweden-latest.osm.pbf
```

Run all commands from the repository root:

```bash
cd "/Users/robin.carlander/Wildfire BSC Thesis"
mkdir -p data/gis/osm
mkdir -p data/contexts
```

---

## Recommended — Generate everything with one script

From the repository root, run:

```bash
scripts/generate_context_scenarios.sh
```

This will:

1. Extract all seven local 50 × 50 km `.osm.pbf` files into `data/gis/osm/`.
2. Generate all seven `data/contexts/scenario_XX.json` files.
3. Validate JSON formatting and run the context tests.

Useful options:

```bash
# Run only one scenario
scripts/generate_context_scenarios.sh --scenario 03

# Generate JSON from already extracted .osm.pbf files
scripts/generate_context_scenarios.sh --generate-only

# Validate existing scenario JSON files
scripts/generate_context_scenarios.sh --validate-only

# Use a source PBF from another location
scripts/generate_context_scenarios.sh --input /path/to/sweden-latest.osm.pbf
```

Run `scripts/generate_context_scenarios.sh --help` for all options.

---

## Manual reference — Step 1: Extract 50 × 50 km OSM files

The bounding box format for `osmium extract -b` is:

```text
west_lon,south_lat,east_lon,north_lat
```

### Scenario 01 — Acktjärnsåsarna, Västmanland

```bash
osmium extract \
  -b 15.653541,59.575422,16.546459,60.024578 \
  data/gis/osm/sweden-latest.osm.pbf \
  -o data/gis/osm/scenario_01-50km.osm.pbf \
  --overwrite
```

### Scenario 02 — Hälleskogsbrännan, Västmanland

```bash
osmium extract \
  -b 15.747293,59.610830,16.641161,60.059986 \
  data/gis/osm/sweden-latest.osm.pbf \
  -o data/gis/osm/scenario_02-50km.osm.pbf \
  --overwrite
```

### Scenario 03 — Tyresta, Stockholm area

```bash
osmium extract \
  -b 17.837176,58.952206,18.713764,59.401362 \
  data/gis/osm/sweden-latest.osm.pbf \
  -o data/gis/osm/scenario_03-50km.osm.pbf \
  --overwrite
```

### Scenario 04 — Skansberget / Kårböle, Ljusdal

```bash
osmium extract \
  -b 14.854897,61.746117,15.810703,62.195273 \
  data/gis/osm/sweden-latest.osm.pbf \
  -o data/gis/osm/scenario_04-50km.osm.pbf \
  --overwrite
```

### Scenario 05 — Nötbergets norra, Ljusdal

```bash
osmium extract \
  -b 14.783927,61.789042,15.741081,62.238198 \
  data/gis/osm/sweden-latest.osm.pbf \
  -o data/gis/osm/scenario_05-50km.osm.pbf \
  --overwrite
```

### Scenario 06 — Torsburgen, Gotland

```bash
osmium extract \
  -b 18.291195,57.184137,19.125061,57.633293 \
  data/gis/osm/sweden-latest.osm.pbf \
  -o data/gis/osm/scenario_06-50km.osm.pbf \
  --overwrite
```

### Scenario 07 — Vakö myr, Kronoberg/Skåne

```bash
osmium extract \
  -b 13.846719,56.275423,14.660499,56.724579 \
  data/gis/osm/sweden-latest.osm.pbf \
  -o data/gis/osm/scenario_07-50km.osm.pbf \
  --overwrite
```

Check that the clipped files were created:

```bash
ls -lh data/gis/osm/scenario_*-50km.osm.pbf
```

Expected files:

```text
data/gis/osm/scenario_01-50km.osm.pbf
data/gis/osm/scenario_02-50km.osm.pbf
data/gis/osm/scenario_03-50km.osm.pbf
data/gis/osm/scenario_04-50km.osm.pbf
data/gis/osm/scenario_05-50km.osm.pbf
data/gis/osm/scenario_06-50km.osm.pbf
data/gis/osm/scenario_07-50km.osm.pbf
```

---

## Manual reference — Step 2: Generate scenario JSON files

After the `.osm.pbf` files have been created, run the GIS context extractor.

### Scenario 01 — Acktjärnsåsarna, Västmanland

```bash
uv run python -m pipeline.context.extract \
  --lat 59.8000 \
  --lon 16.1000 \
  --pbf data/gis/osm/scenario_01-50km.osm.pbf \
  --output data/contexts/scenario_01.json
```

### Scenario 02 — Hälleskogsbrännan, Västmanland

```bash
uv run python -m pipeline.context.extract \
  --lat 59.835408 \
  --lon 16.194227 \
  --pbf data/gis/osm/scenario_02-50km.osm.pbf \
  --output data/contexts/scenario_02.json
```

### Scenario 03 — Tyresta, Stockholm area

```bash
uv run python -m pipeline.context.extract \
  --lat 59.176784 \
  --lon 18.275470 \
  --pbf data/gis/osm/scenario_03-50km.osm.pbf \
  --output data/contexts/scenario_03.json
```

### Scenario 04 — Skansberget / Kårböle, Ljusdal

```bash
uv run python -m pipeline.context.extract \
  --lat 61.970695 \
  --lon 15.332800 \
  --pbf data/gis/osm/scenario_04-50km.osm.pbf \
  --output data/contexts/scenario_04.json
```

### Scenario 05 — Nötbergets norra, Ljusdal

```bash
uv run python -m pipeline.context.extract \
  --lat 62.013620 \
  --lon 15.262504 \
  --pbf data/gis/osm/scenario_05-50km.osm.pbf \
  --output data/contexts/scenario_05.json
```

### Scenario 06 — Torsburgen, Gotland

```bash
uv run python -m pipeline.context.extract \
  --lat 57.408715 \
  --lon 18.708128 \
  --pbf data/gis/osm/scenario_06-50km.osm.pbf \
  --output data/contexts/scenario_06.json
```

### Scenario 07 — Vakö myr, Kronoberg/Skåne

```bash
uv run python -m pipeline.context.extract \
  --lat 56.500001 \
  --lon 14.253609 \
  --pbf data/gis/osm/scenario_07-50km.osm.pbf \
  --output data/contexts/scenario_07.json
```

Check that the JSON files were created:

```bash
ls -lh data/contexts/scenario_*.json
```

Expected files:

```text
data/contexts/scenario_01.json
data/contexts/scenario_02.json
data/contexts/scenario_03.json
data/contexts/scenario_04.json
data/contexts/scenario_05.json
data/contexts/scenario_06.json
data/contexts/scenario_07.json
```

---

## Manual reference — Step 3: Validate generated files

Validate JSON formatting:

```bash
for file in data/contexts/scenario_*.json; do
  python -m json.tool "$file" > /dev/null
  echo "valid JSON: $file"
done
```

Run context tests:

```bash
uv run pytest pipeline/context/tests -v
```

Inspect one generated context:

```bash
jq '{
  coordinates,
  region,
  land_cover,
  water_sources,
  roads,
  settlements,
  assets_at_risk,
  named_features
}' data/contexts/scenario_01.json
```

---

## Scenario summaries

| Scenario | Location | Why it was chosen |
|---|---|---|
| `scenario_01` | Acktjärnsåsarna, Västmanland | Wetland forest with nearby lakes. Tests whether the model can reason about water availability while still considering difficult access. |
| `scenario_02` | Hälleskogsbrännan, Västmanland | Historical 2014 wildfire area. Tests post-fire landscape reasoning, burn scars, dead wood, access, and protected-area context. |
| `scenario_03` | Tyresta, Stockholm area | Wildland-urban interface. Tests public safety, visitors, nearby buildings, trails, roads, and protected forest. |
| `scenario_04` | Skansberget / Kårböle, Ljusdal | Slope and settlement-edge scenario. Tests terrain difficulty combined with nearby settlement exposure. |
| `scenario_05` | Nötbergets norra, Ljusdal | Remote post-fire forest. Tests limited access, sparse settlement exposure, and long approach distances. |
| `scenario_06` | Torsburgen, Gotland | Dry island and cliff terrain. Tests dry fuels, exposed terrain, cliffs, wind exposure, and limited water availability. |
| `scenario_07` | Vakö myr, Kronoberg/Skåne | Peat/mire scenario. Tests whether the model handles wetland and smouldering-fire risk instead of assuming wet ground means low risk. |

---
