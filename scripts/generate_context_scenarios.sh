#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_SOURCE_PBF="data/gis/osm/sweden-latest.osm.pbf"
readonly OSM_DIR="data/gis/osm"
readonly CONTEXT_DIR="data/contexts"

source_pbf="$DEFAULT_SOURCE_PBF"
do_extract=1
do_generate=1
do_validate=1
run_tests=1
scenario_filter=""

usage() {
  cat <<'USAGE'
Generate wildfire scenario context files.

Usage:
  scripts/generate_context_scenarios.sh [options]

Options:
  --input FILE       Source Sweden OSM PBF file.
                     Default: data/gis/osm/sweden-latest.osm.pbf
  --scenario NN      Run only one scenario, for example 01 or 7.
  --extract-only     Only create data/gis/osm/scenario_XX-50km.osm.pbf files.
  --generate-only    Only generate data/contexts/scenario_XX.json files.
  --validate-only    Only validate existing JSON files and run context tests.
  --skip-tests       Validate JSON but skip pytest.
  -h, --help         Show this help text.

Examples:
  scripts/generate_context_scenarios.sh
  scripts/generate_context_scenarios.sh --scenario 03
  scripts/generate_context_scenarios.sh --generate-only --skip-tests
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

readonly SCENARIOS=(
  "01|Acktjarnsasarna, Vastmanland|15.653541,59.575422,16.546459,60.024578|59.8000|16.1000"
  "02|Halleskogsbrannan, Vastmanland|15.747293,59.610830,16.641161,60.059986|59.835408|16.194227"
  "03|Tyresta, Stockholm area|17.837176,58.952206,18.713764,59.401362|59.176784|18.275470"
  "04|Skansberget / Karbole, Ljusdal|14.854897,61.746117,15.810703,62.195273|61.970695|15.332800"
  "05|Notbergets norra, Ljusdal|14.783927,61.789042,15.741081,62.238198|62.013620|15.262504"
  "06|Torsburgen, Gotland|18.291195,57.184137,19.125061,57.633293|57.408715|18.708128"
  "07|Vako myr, Kronoberg/Skane|13.846719,56.275423,14.660499,56.724579|56.500001|14.253609"
)

mkdir -p "$OSM_DIR" "$CONTEXT_DIR"

if [[ "$do_extract" -eq 1 ]]; then
  require_command osmium
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

selected_count=0

for scenario in "${SCENARIOS[@]}"; do
  IFS="|" read -r id name bbox lat lon <<< "$scenario"

  if [[ -n "$scenario_filter" && "$id" != "$scenario_filter" ]]; then
    continue
  fi

  selected_count=$((selected_count + 1))
  clipped_pbf="$OSM_DIR/scenario_${id}-50km.osm.pbf"
  output_json="$CONTEXT_DIR/scenario_${id}.json"

  if [[ "$do_extract" -eq 1 ]]; then
    echo "Extracting scenario ${id}: ${name}"
    osmium extract \
      -b "$bbox" \
      "$source_pbf" \
      -o "$clipped_pbf" \
      --overwrite
  fi

  if [[ "$do_generate" -eq 1 ]]; then
    if [[ ! -f "$clipped_pbf" ]]; then
      echo "Missing clipped OSM PBF file for scenario ${id}: $clipped_pbf" >&2
      echo "Run without --generate-only first, or create the clipped file manually." >&2
      exit 1
    fi

    echo "Generating scenario ${id}: ${output_json}"
    $project_python -m pipeline.context.extract \
      --lat "$lat" \
      --lon "$lon" \
      --pbf "$clipped_pbf" \
      --output "$output_json"
  fi
done

if [[ "$selected_count" -eq 0 ]]; then
  echo "No matching scenario found for id: $scenario_filter" >&2
  exit 1
fi

if [[ "$do_validate" -eq 1 ]]; then
  if [[ -n "$scenario_filter" ]]; then
    json_files=("$CONTEXT_DIR/scenario_${scenario_filter}.json")
  else
    json_files=("$CONTEXT_DIR"/scenario_*.json)
  fi

  echo "Validating JSON files"
  for file in "${json_files[@]}"; do
    if [[ ! -f "$file" ]]; then
      echo "Missing JSON file: $file" >&2
      exit 1
    fi
    $json_python -m json.tool "$file" >/dev/null
    echo "valid JSON: $file"
  done

  if [[ "$run_tests" -eq 1 ]]; then
    echo "Running context tests"
    $context_pytest pipeline/context/tests -v
  fi
fi
