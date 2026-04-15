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

## Quick Test

Run one model on one image:

```bash
uv run python quick_test.py test_images/fire/fire_004.jpg
uv run python quick_test.py test_images/fire/fire_004.jpg --model qwen3-vl:4b
```

## Benchmarking

Two modes, both accept `--dry-run` to preview before running.

### Accuracy

All images, no cooldown. Measures classification correctness.

```bash
uv run python run_benchmark.py accuracy --images test_images --prompt prompts/prompt.txt
uv run python run_benchmark.py accuracy --images test_images --prompt prompts/prompt.txt --models ministral-3:3b qwen3-vl:4b
```

Results → `results/accuracy/`

### Performance

Small image subset with cooldowns to prevent thermal throttling. Measures inference speed, tokens/sec, memory.

```bash
uv run python run_benchmark.py performance --images test_images --prompt prompts/prompt.txt
uv run python run_benchmark.py performance --images test_images --prompt prompts/prompt.txt --num-images 5 --cooldown 15 --model-cooldown 120
```

Defaults: 3 images, 10s between images, 90s between models. Results → `results/performance/`

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
| `classification_breakdown.png` | accuracy | Correct/incorrect/ambiguous breakdown |
| `response_length.png` | accuracy | Word count distribution per model |
| `tokens_per_sec.png` | performance | Generation speed per model |
| `inference_breakdown.png` | performance | Prompt eval vs generation vs overhead |
| `total_inference_time.png` | performance | Total inference time per model |
| `memory_usage.png` | performance | Disk size vs runtime memory |

Also prints a summary table to the terminal.

## Directory Structure

```
benchmarks/
├── prompts/prompt.txt
├── test_images/
│   ├── fire/fire_001.jpg ...
│   └── nofire/nofire_001.jpg ...
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