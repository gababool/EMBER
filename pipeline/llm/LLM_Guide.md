# Wildfire MLLM Inference

Runs multimodal LLM inference on an input image using Ollama.

## Setup

**1. Install and start Ollama**
```bash
brew install ollama
ollama serve
```

**2. Pull models** (in a new terminal)
```bash
ollama pull ministral-3:3b
ollama pull qwen3-vl:4b
ollama pull gemma4:e2b
```

**3. Install dependencies**
```bash
uv sync
```

## Usage

The script can be run from anywhere.
The default system prompt is loaded from `pipeline/llm/prompts/c1v1prompt.txt`.

## Run model on one image

```bash
uv run python pipeline/llm/inference.py \
  --image "pipeline/llm/yolo_annotated_test.jpg" \
  --model ministral
```

## Run one model with optional context

```bash
uv run python pipeline/llm/inference.py \
  --image "pipeline/llm/yolo_annotated_test.jpg" \
  --model ministral \
  --context-file pipeline/llm/context.txt
```

## Run all models

```bash
uv run python pipeline/llm/inference.py \
  --image "pipeline/llm/yolo_annotated_test.jpg" \
  --model all \
  --context-file pipeline/llm/context.txt
```

## Save JSON output

```bash
uv run python pipeline/llm/inference.py \
  --image "pipeline/llm/yolo_annotated_test.jpg" \
  --model all \
  --context-file pipeline/llm/context.txt \
  --output-json results.json
```

## Use a custom prompt file

```bash
uv run python pipeline/llm/inference.py \
  --image "pipeline/llm/yolo_annotated_test.jpg" \
  --model ministral \
  --prompt-file pipeline/llm/prompts/c1v1prompt.txt
```