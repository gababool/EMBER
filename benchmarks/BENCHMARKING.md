# Benchmarking

Benchmarks multimodal LLMs on wildfire detection images using Ollama, with optional YOLO preprocessing and GIS context. The benchmark invokes the LLM with Ollama's structured-output feature, enforcing JSON output matching the pipeline's Cycle 2 schema (`classification`, `reasoning`, `recommendation`, `situation_brief`, `tactical_priority`).

## Prerequisites

- [Ollama](https://ollama.com) running (`ollama serve`)
- Models pulled (e.g. `ollama pull ministral-3:3b`)
- `uv add ollama pydantic matplotlib`
- For YOLO preprocessing: `ultralytics` and `opencv-python` (installed as part of the object detection module)
- For GIS context: scenario JSON files under `data/contexts/` (generated with `pipeline.context.extract`)

## Models

The three thesis-core models are used by default. All models listed can be selected explicitly with `--models`.

| Model | Tag | Size | Used by default |
|---|---|---|---|
| Ministral (Mistral) | `ministral-3:3b` | 3.0GB | yes |
| Ministral (Mistral) | `ministral-3:8b` | 6.0GB | no |
| Qwen3-VL (Alibaba) | `qwen3-vl:4b` | 3.3GB | yes |
| Qwen3-VL (Alibaba) | `qwen3-vl:8b` | 6.1GB | no |
| Gemma 4 (Google) | `gemma4:e2b` | 7.2GB | yes |
| Gemma 4 (Google) | `gemma4:e4b` | 9.6GB | no |

## Dataset

Images are expected in a directory with `fire/` and `nofire/` subfolders (ground truth is inferred from folder name). Filename prefixes (`fire_*.jpg` / `nofire_*.jpg`) are supported as a fallback. Subdirectories above `fire/` and `nofire/` (e.g. `test/`, `train/`, `val/`) are traversed automatically.

## Prompts

The benchmark loads a system prompt from a text file. The default points to the canonical Cycle 2 prompt:

```
pipeline/llm/prompts/c2v4prompt.txt
```

Override with `--system-prompt`. The prompt must instruct the model to return JSON matching the Cycle 2 schema fields. `classification` must be one of `fire_detected` or `no_fire_detected` — `uncertain` is not a valid output. Ollama's structured-output enforcement guarantees schema-valid JSON at the API level.

## Quick Test

Run one model on one image:

```bash
uv run python quick_test.py ../dataset/wildfire-dataset/the_wildfire_dataset_2n_version/test/fire/some_image.jpg
uv run python quick_test.py ../dataset/wildfire-dataset/the_wildfire_dataset_2n_version/test/fire/some_image.jpg --model qwen3-vl:4b
```

## Benchmarking

Two modes: `accuracy` and `performance`. Both accept `--dry-run` to preview, `--seed` for reproducible sampling, `--system-prompt` to override the default prompt, `--with-yolo` to enable YOLO preprocessing, and `--with-context` to enable GIS context.

### Accuracy

All (or sampled) images, no cooldown. Measures classification correctness.

```bash
# All images, default models, no YOLO, no context (Cycle 1 equivalent)
uv run python run_benchmark.py accuracy --images ../dataset

# Balanced sample of 100 images
uv run python run_benchmark.py accuracy --images ../dataset --num-images 100

# With YOLO preprocessing
uv run python run_benchmark.py accuracy --images ../dataset --with-yolo

# With YOLO metadata as LLM text context instead of annotated image
uv run python run_benchmark.py accuracy --images ../dataset --with-yolo \
    --yolo-input-mode context_summary

uv run python run_benchmark.py accuracy --images ../dataset --with-yolo \
    --yolo-input-mode context_locations

# With GIS context (uses scenario_01/scenario_01.json by default)
uv run python run_benchmark.py accuracy --images ../dataset --with-context

# Full Cycle 2: YOLO + context
uv run python run_benchmark.py accuracy --images ../dataset --with-yolo --with-context

# Different scenario
uv run python run_benchmark.py accuracy --images ../dataset --with-context \
    --scenario scenario_02/scenario_02.json

# Specific models only
uv run python run_benchmark.py accuracy --images ../dataset --models ministral-3:3b qwen3-vl:4b
```

Results → `results/accuracy/accuracy_{model}_{yolo_tag}_{context_tag}.csv`.

CSV filename tags:
- `_noyolo` / `_yolo` — whether YOLO preprocessing was used
- `_ctx` — whether GIS context was enabled (absent when context is off)

Existing files are not overwritten — delete them manually to rerun.

### Performance

Balanced random sample with cooldowns to prevent thermal throttling. Measures inference speed, tokens/sec, memory.

```bash
# Default: 3 images, 10s between images, 90s between models
uv run python run_benchmark.py performance --images ../dataset

# Larger sample, longer cooldowns for fanless machines
uv run python run_benchmark.py performance --images ../dataset --num-images 5 --cooldown 0 --model-cooldown 180

# Full Cycle 2: YOLO + context
uv run python run_benchmark.py performance --images ../dataset --with-yolo --with-context
```

Each run produces two files per model:

- `results/performance/performance_{model}_{yolo_tag}_{context_tag}_{timestamp}.csv` — per-inference measurements
- `results/performance/performance_{model}_{yolo_tag}_{context_tag}_{timestamp}.json` — companion file with hardware specs, run configuration, and prompt file used

Run CSVs are timestamped so multiple runs accumulate for better statistics.

### YOLO Preprocessing

When `--with-yolo` is set, each image passes through the YOLO object detection stage before reaching the LLM. `--yolo-model` selects which weights to use (default: `best`).

`--yolo-input-mode` controls how YOLO output reaches the LLM:

- `disabled` — no YOLO preprocessing; the LLM receives the original image.
- `annotated_image` — the current/default behavior; the LLM receives the annotated image with fire/smoke boxes.
- `context_summary` — the LLM receives the original image plus detection count, labels, and confidence values as text context.
- `context_locations` — the LLM receives the original image plus detection count, labels, confidence values, and bounding box coordinates as text context.

YOLO mode, timing, and detection count are logged as separate CSV columns (`yolo_input_mode`, `yolo_duration_s`, `yolo_detection_count`) so YOLO cost and input style are not lumped into LLM timing metrics.

### GIS Context

When `--with-context` is set, a pre-computed scenario JSON is loaded, formatted, and passed to the LLM alongside the image. A single scenario is used for all images in a run so that context is not a variable between images.

Scenario JSONs are generated with the context extraction pipeline:

```bash
uv run python -m pipeline.context.extract \
    --lat 59.8 --lon 16.1 \
    --output data/contexts/scenario_01/scenario_01.json
```

The `context_enabled` and `context_scenario` columns in the CSV record whether context was active and which scenario was used.

### Scenario Evaluation

A separate script evaluates all models against hand-picked scenario image+context pairings. Unlike the accuracy benchmark (many images, one scenario), this runs one image per scenario to support qualitative analysis and stakeholder interviews.

Scenario folder structure:

```
data/contexts/
    scenario_01/
        scenario_01.json
        scenario_01.jpg
    scenario_02/
        scenario_02.json
        scenario_02.png
```

```bash
# Dry run
uv run python run_scenario_eval.py --dry-run

# All scenarios, no YOLO
uv run python run_scenario_eval.py

# All scenarios, with YOLO
uv run python run_scenario_eval.py --with-yolo

# Specific scenarios only
uv run python run_scenario_eval.py --scenarios scenario_01 scenario_03
```

Results → `results/scenarios/scenario_eval_{yolo_tag}_{timestamp}.csv` (combined, one row per model×scenario) + companion JSON with run metadata and hardware specs.

### Invalid Output Handling

The benchmark validates each LLM response against the schema. If a model returns output that fails validation, the benchmark logs an error for that image and continues. No partial row is written to the CSV.

## Analysis

Generates PNG charts from benchmark CSVs:

```bash
# Default: no-YOLO CSVs only
uv run python analyze.py

# Compare no-YOLO vs YOLO side by side
uv run python analyze.py --compare-yolo
```

Output → `charts/`

| Chart | Source | Description |
|---|---|---|
| `model_sizes.png` | static | Model disk size |
| `accuracy_by_model.png` | accuracy | Classification accuracy per model |
| `precision_recall_f1.png` | accuracy | Precision, recall, F1 (fire = positive class) |
| `mcc.png` | accuracy | Matthews Correlation Coefficient per model |
| `confusion_matrix.png` | accuracy | TP, FP, TN, FN counts per model |
| `response_length.png` | accuracy | Word count distribution per model |
| `tokens_per_sec.png` | performance | Generation speed per model |
| `inference_breakdown.png` | performance | Prompt eval vs generation vs overhead |
| `total_inference_time.png` | performance | Total inference time per model |
| `memory_usage.png` | performance | Disk size vs runtime memory |

Also prints a summary table with accuracy, precision, recall, F1, MCC, tokens/sec, total inference time, and memory usage. In `--compare-yolo` mode the table shows paired rows (no YOLO / with YOLO) per model.

## Directory Structure

```
benchmarks/
├── src/
│   ├── data_loader.py
│   ├── hardware.py
│   ├── llm_inference.py
│   ├── logger.py
│   └── metrics.py
├── run_benchmark.py
├── run_scenario_eval.py
├── quick_test.py
├── analyze.py
├── results/               # gitignored
│   ├── accuracy/
│   ├── performance/
│   └── scenarios/
└── charts/                # gitignored
```

Scenario data lives under `data/contexts/` in the repo root (gitignored for large files). The system prompt file lives under `pipeline/llm/prompts/` and is shared between the pipeline and the benchmark.

## Platform Notes

Full hardware specs are collected on macOS. On Linux and Windows, benchmarks still run but some hardware fields (chip, RAM) may be recorded as `null` in the JSON companion file. Memory tracking during inference also currently relies on macOS/Unix tooling.
