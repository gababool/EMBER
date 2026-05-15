#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_SOURCE_PBF="data/gis/osm/sweden-latest.osm.pbf"
readonly OSM_DIR="data/gis/osm"
readonly ELEVATION_DIR="data/gis/elevation"
readonly CONTEXT_DIR="data/contexts"

source_pbf="$DEFAULT_SOURCE_PBF"
do_extract=1
do_generate=1
do_validate=1
run_tests=1
scenario_filter=""
mock_wind=0
benchmark_wind=0

usage() {
  cat <<'USAGE'
Generate wildfire scenario context files.

Usage:
  scripts/generate_context_scenarios.sh [options]

Options:
  --input FILE       Source Sweden OSM PBF file.
                     Default: data/gis/osm/sweden-latest.osm.pbf
  --scenario NN      Run only one scenario, for example 01 or 7.
  --extract-only     Only create OSM and elevation files.
  --generate-only    Only generate context JSON files (requires existing OSM/DEM).
  --validate-only    Only validate existing JSON files and run context tests.
  --mock-wind        Add deterministic semi-random regional wind to generated JSON.
  --benchmark-wind   Add the same mocked wind to every generated JSON (SW, 6.5 m/s).
  --skip-tests       Validate JSON but skip pytest.
  -h, --help         Show this help text.

Examples:
  scripts/generate_context_scenarios.sh
  scripts/generate_context_scenarios.sh --scenario 03
  scripts/generate_context_scenarios.sh --generate-only --skip-tests
  scripts/generate_context_scenarios.sh --extract-only
USAGE
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
}

python_cmd() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    echo "uv run python"
  fi
}

project_python_cmd() {
  if [[ -x ".venv/bin/python" ]]; then
    echo ".venv/bin/python"
  else
    echo "uv run python"
  fi
}

pytest_cmd() {
  if [[ -x ".venv/bin/pytest" ]]; then
    echo ".venv/bin/pytest"
  else
    echo "uv run pytest"
  fi
}

normalize_scenario_id() {
  local raw_id="$1"
  if [[ "$raw_id" =~ ^[0-9]$ ]]; then
    printf "0%s" "$raw_id"
  elif [[ "$raw_id" =~ ^[0-9][0-9]$ ]]; then
    printf "%s" "$raw_id"
  else
    echo "Invalid scenario id: $raw_id" >&2
    exit 1
  fi
}

# Convert "west,south,east,north" (osmium format) to "west south east north" (eio format).
bbox_to_eio() {
  echo "$1" | tr ',' ' '
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      if [[ $# -lt 2 ]]; then
        echo "--input requires a file path" >&2
        exit 1
      fi
      source_pbf="$2"
      shift 2
      ;;
    --scenario)
      if [[ $# -lt 2 ]]; then
        echo "--scenario requires an id" >&2
        exit 1
      fi
      scenario_filter="$(normalize_scenario_id "$2")"
      shift 2
      ;;
    --extract-only)
      do_extract=1
      do_generate=0
      do_validate=0
      shift
      ;;
    --generate-only)
      do_extract=0
      do_generate=1
      do_validate=1
      shift
      ;;
    --validate-only)
      do_extract=0
      do_generate=0
      do_validate=1
      shift
      ;;
    --skip-tests)
      run_tests=0
      shift
      ;;
    --mock-wind)
      mock_wind=1
      shift
      ;;
    --benchmark-wind)
      mock_wind=1
      benchmark_wind=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Scenario definitions: id|name|bbox (west,south,east,north)|lat|lon
#
# Scenario selection rationale:
#   01 — Wetland forest, protected area, lakes nearby but limited road access
#   02 — Wildland-urban interface, national park edge, buildings, paved roads
#   03 — Steep forested slope, High Coast, named peaks, remote
#   04 — Agricultural/rural Skane, dense settlement, good road access, flat
#   05 — Peat mire, protected area, flat terrain, light supply water only
#   06 — Remote northern forest, sparse OSM, minimal roads and settlements

readonly SCENARIOS=(
  "01|Acktjarnsasarna, Vastmanland|15.653541,59.575422,16.546459,60.024578|59.8000|16.1000"
  "02|Tyresta National Park edge, Stockholm|17.790000,58.990000,18.690000,59.400000|59.17372|18.24039"
  "03|Skuleskogen, High Coast, Vasternorrland|17.605638,62.725180,18.594362,63.174820|62.950035|18.050492"
  "04|Skane countryside near Hassleholm|13.370000,55.650000,14.270000,56.130000|55.890|13.820"
  "05|Vako myr, Kronoberg/Skane|13.846719,56.275423,14.660499,56.724579|56.500001|14.253609"
  "06|Sorsele remote forest, Vasterbotten|16.200000,64.950000,17.200000,65.400000|65.16019|16.70643"
)

mkdir -p "$OSM_DIR" "$ELEVATION_DIR" "$CONTEXT_DIR"

# ── Pre-flight checks ─────────────────────────────────────────────────────────

if [[ "$do_extract" -eq 1 ]]; then
  require_command osmium
  require_command eio
  if [[ ! -f "$source_pbf" ]]; then
    echo "Missing source OSM PBF file: $source_pbf" >&2
    echo "Place sweden-latest.osm.pbf there or pass --input /path/to/file.osm.pbf" >&2
    exit 1
  fi
fi

if [[ "$do_generate" -eq 1 ]]; then
  project_python="$(project_python_cmd)"
  if [[ "$project_python" == "uv run python" ]]; then
    require_command uv
  fi
fi

if [[ "$run_tests" -eq 1 ]]; then
  context_pytest="$(pytest_cmd)"
  if [[ "$context_pytest" == "uv run pytest" ]]; then
    require_command uv
  fi
fi

if [[ "$do_validate" -eq 1 ]]; then
  json_python="$(python_cmd)"
  if [[ "$json_python" == "uv run python" ]]; then
    require_command uv
  fi
fi

# ── Main loop ─────────────────────────────────────────────────────────────────

selected_count=0

for scenario in "${SCENARIOS[@]}"; do
  IFS="|" read -r id name bbox lat lon <<< "$scenario"

  if [[ -n "$scenario_filter" && "$id" != "$scenario_filter" ]]; then
    continue
  fi

  selected_count=$((selected_count + 1))

  clipped_pbf="$OSM_DIR/scenario_${id}-50km.osm.pbf"
  dem_file="$ELEVATION_DIR/scenario_${id}.tif"
  output_json="$CONTEXT_DIR/scenario_${id}/scenario_${id}.json"

  # ── Extract OSM and elevation ───────────────────────────────────────────────

  if [[ "$do_extract" -eq 1 ]]; then
    echo ""
    echo "=== Scenario ${id}: ${name} ==="

    echo "  Clipping OSM extract..."
    osmium extract \
      -b "$bbox" \
      "$source_pbf" \
      -o "$clipped_pbf" \
      --overwrite
    echo "  OSM extract: ${clipped_pbf}"

    echo "  Downloading SRTM elevation tiles..."
    eio_bounds="$(bbox_to_eio "$bbox")"
    # Clean any corrupted cache entries before downloading.
    eio clean 2>/dev/null || true
    eio clip -o "$dem_file" --bounds $eio_bounds
    echo "  Elevation raster: ${dem_file}"
  fi

  # ── Generate context JSON ───────────────────────────────────────────────────

  if [[ "$do_generate" -eq 1 ]]; then
    if [[ ! -f "$clipped_pbf" ]]; then
      echo "Missing clipped OSM PBF for scenario ${id}: $clipped_pbf" >&2
      echo "Run without --generate-only first." >&2
      exit 1
    fi

    if [[ ! -f "$dem_file" ]]; then
      echo "  Warning: elevation raster not found for scenario ${id}: $dem_file" >&2
      echo "  Terrain will be null. Run without --generate-only to download it." >&2
      dem_arg="--dem none"
    else
      dem_arg="--dem $dem_file"
    fi

    mkdir -p "$CONTEXT_DIR/scenario_${id}"
    echo "  Generating context JSON: ${output_json}"

    if [[ "$mock_wind" -eq 1 ]]; then
      if [[ "$benchmark_wind" -eq 1 ]]; then
        $project_python -m pipeline.context.extract \
          --lat "$lat" \
          --lon "$lon" \
          --pbf "$clipped_pbf" \
          $dem_arg \
          --output "$output_json" \
          --mock-wind \
          --mock-wind-direction SW \
          --mock-wind-speed-mps 6.5
      else
        $project_python -m pipeline.context.extract \
          --lat "$lat" \
          --lon "$lon" \
          --pbf "$clipped_pbf" \
          $dem_arg \
          --output "$output_json" \
          --mock-wind \
          --mock-wind-key "scenario_${id}"
      fi
    else
      $project_python -m pipeline.context.extract \
        --lat "$lat" \
        --lon "$lon" \
        --pbf "$clipped_pbf" \
        $dem_arg \
        --output "$output_json"
    fi

    echo "  Done: ${output_json}"
  fi

done

if [[ "$selected_count" -eq 0 ]]; then
  echo "No matching scenario found for id: $scenario_filter" >&2
  exit 1
fi

# ── Validate ──────────────────────────────────────────────────────────────────

if [[ "$do_validate" -eq 1 ]]; then
  echo ""
  echo "=== Validating JSON files ==="

  if [[ -n "$scenario_filter" ]]; then
    json_files=("$CONTEXT_DIR/scenario_${scenario_filter}/scenario_${scenario_filter}.json")
  else
    json_files=("$CONTEXT_DIR"/scenario_*/scenario_*.json)
  fi

  all_valid=1
  for file in "${json_files[@]}"; do
    if [[ ! -f "$file" ]]; then
      echo "  Missing: $file" >&2
      all_valid=0
      continue
    fi
    if $json_python -m json.tool "$file" >/dev/null 2>&1; then
      # Check terrain is populated (not null).
      terrain=$(grep -o '"terrain": null' "$file" || true)
      if [[ -n "$terrain" ]]; then
        echo "  Warning: terrain is null in $file — was the DEM available?"
      fi
      echo "  valid: $file"
    else
      echo "  INVALID JSON: $file" >&2
      all_valid=0
    fi
  done

  if [[ "$all_valid" -eq 0 ]]; then
    echo "One or more JSON files failed validation." >&2
    exit 1
  fi

  if [[ "$run_tests" -eq 1 ]]; then
    echo ""
    echo "=== Running context tests ==="
    $context_pytest pipeline/context/tests -v
  fi
fi

echo ""
echo "All done."