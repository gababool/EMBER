# Benchmarking

Benchmarks multimodal LLMs on wildfire detection images using Ollama, with optional YOLO preprocessing.

## Prerequisites

- [Ollama](https://ollama.com) running (`ollama serve`)
- Models pulled (e.g. `ollama pull ministral-3:3b`)
- `uv add ollama matplotlib`
- For YOLO preprocessing: `ultralytics` and `opencv-python` (installed as part of the object detection module)

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

## Quick Test

Run one model on one image:

```bash
uv run python quick_test.py ../dataset/wildfire-dataset/the_wildfire_dataset_2n_version/test/fire/some_image.jpg
uv run python quick_test.py ../dataset/wildfire-dataset/the_wildfire_dataset_2n_version/test/fire/some_image.jpg --model qwen3-vl:4b
```

## Benchmarking

Two modes, both accept `--dry-run` to preview, `--seed` for reproducible sampling, and `--with-yolo` to enable YOLO preprocessing before LLM inference.

### Accuracy

All (or sampled) images, no cooldown. Measures classification correctness.

```bash
# All images in the dataset, default three models
uv run python run_benchmark.py accuracy --images ../dataset --prompt prompts/prompt.txt

# Random balanced sample of 100 images
uv run python run_benchmark.py accuracy --images ../dataset --prompt prompts/prompt.txt --num-images 100

# Specific models only (any from the models table above)
uv run python run_benchmark.py accuracy --images ../dataset --prompt prompts/prompt.txt --models ministral-3:8b qwen3-vl:8b

# With YOLO preprocessing
uv run python run_benchmark.py accuracy --images ../dataset --prompt prompts/prompt.txt --with-yolo
```

Results → `results/accuracy/accuracy_{model}.csv`. Existing files are not overwritten — delete them manually to rerun.

### Performance

Balanced random sample with cooldowns to prevent thermal throttling. Measures inference speed, tokens/sec, memory.

```bash
# Default: 3 images, 10s between images, 90s between models
uv run python run_benchmark.py performance --images ../dataset --prompt prompts/prompt.txt

# Larger sample, longer cooldowns for fanless machines
uv run python run_benchmark.py performance --images ../dataset --prompt prompts/prompt.txt --num-images 5 --cooldown 0 --model-cooldown 180

# With YOLO preprocessing
uv run python run_benchmark.py performance --images ../dataset --prompt prompts/prompt.txt --with-yolo
```

Each run produces two files per model:

- `results/performance/performance_{model}_{timestamp}.csv` — per-inference measurements
- `results/performance/performance_{model}_{timestamp}.json` — companion file with hardware specs (chip, RAM, OS, Python, Ollama version) and the run configuration

Run CSVs are timestamped so multiple runs accumulate for better statistics. The JSON companion allows comparing results across different laptops.

### YOLO Preprocessing

When `--with-yolo` is set, each image is passed through the YOLO object detection stage before reaching the LLM. The annotated image (with bounding boxes for fire/smoke) becomes the LLM input. `--yolo-model` selects which weights to use (default: `best`).

YOLO timing and detection count are logged as separate CSV columns (`yolo_duration_s`, `yolo_detection_count`) so YOLO cost does not get lumped into LLM timing metrics. This allows analysis of whether YOLO preprocessing affects LLM inference time, accuracy, or both.

## Analysis

Generates PNG charts from benchmark CSVs:

```bash
uv run python analyze.py
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

Also prints a summary table with accuracy, precision, recall, F1, MCC, tokens/sec, total inference time, and memory usage.

## Directory Structure

```
benchmarks/
├── prompts/prompt.txt
├── src/
│   ├── data_loader.py
│   ├── hardware.py
│   ├── llm_inference.py
│   ├── logger.py
│   └── metrics.py
├── run_benchmark.py
├── quick_test.py
├── analyze.py
├── results/          # gitignored
│   ├── accuracy/
│   └── performance/
└── charts/           # gitignored
```

## Platform Notes

Full hardware specs are collected on macOS. On Linux and Windows, benchmarks still run but some hardware fields (chip, RAM) may be recorded as `null` in the JSON companion file. Memory tracking during inference also currently relies on macOS/Unix tooling.