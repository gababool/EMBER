# Benchmarking

Benchmarks multimodal LLMs on wildfire detection images using Ollama.

## Prerequisites

- [Ollama](https://ollama.com) running (`ollama serve`)
- Models pulled (e.g. `ollama pull ministral-3:3b`)
- `uv add ollama matplotlib`

## Models

| Model | Tag | Size |
|---|---|---|
| Ministral (Mistral) | `ministral-3:3b`, `ministral-3:8b` | 3.0GB, 6.0GB |
| Qwen3-VL (Alibaba) | `qwen3-vl:2b`, `qwen3-vl:4b`, `qwen3-vl:8b` | 1.9GB, 3.3GB, 6.1GB |
| Gemma 4 (Google) | `gemma4:e2b`, `gemma4:e4b` | 7.2GB, 9.6GB |

## Dataset

Images are expected in a directory with `fire/` and `nofire/` subfolders (ground truth is inferred from folder name). Filename prefixes (`fire_*.jpg` / `nofire_*.jpg`) are supported as a fallback.

## Quick Test

Run one model on one image:

```bash
uv run python quick_test.py test_images/fire/fire_004.jpg
uv run python quick_test.py test_images/fire/fire_004.jpg --model qwen3-vl:4b
```

## Benchmarking

Two modes, both accept `--dry-run` to preview and `--seed` for reproducible sampling.

### Accuracy

All (or sampled) images, no cooldown. Measures classification correctness.

```bash
# All images in the dataset
uv run python run_benchmark.py accuracy --images ../dataset --prompt prompts/prompt.txt

# Random balanced sample of 100 images
uv run python run_benchmark.py accuracy --images ../dataset --prompt prompts/prompt.txt --num-images 100

# Specific models only
uv run python run_benchmark.py accuracy --images ../dataset --prompt prompts/prompt.txt --models ministral-3:3b qwen3-vl:4b
```

Results → `results/accuracy/accuracy_{model}.csv`. Existing files are not overwritten — delete them manually to rerun.

### Performance

Balanced random sample with cooldowns to prevent thermal throttling. Measures inference speed, tokens/sec, memory.

```bash
# Default: 3 images, 10s between images, 90s between models
uv run python run_benchmark.py performance --images ../dataset --prompt prompts/prompt.txt

# Larger sample, longer cooldowns for fanless machines
uv run python run_benchmark.py performance --images ../dataset --prompt prompts/prompt.txt --num-images 5 --cooldown 0 --model-cooldown 180
```

Results → `results/performance/performance_{model}_{timestamp}.csv`. Each run gets its own timestamped file so multiple runs accumulate for better statistics.

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