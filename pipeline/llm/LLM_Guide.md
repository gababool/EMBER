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

```bash
# Default prompt
uv run python chat_ollama.py image.jpg

# Custom prompt
uv run python chat_ollama.py image.jpg "Is there smoke or fire in this image?"
```

## Run model on one image

```bash
uv run python pipeline/llm/inference.py \
  --image "pipeline/llm/yolo_annotated_test.jpg" \
  --model ministral

## Run one model with optional context

```bash 
uv run python pipeline/llm/inference.py \
  --image "pipeline/llm/yolo_annotated_test.jpg" \
  --model ministral \
  --context-file pipeline/llm/context.txt

  ## Run all models
  ```bash
  uv run python pipeline/llm/inference.py \
  --image "pipeline/llm/yolo_annotated_test.jpg" \
  --model all \
  --context-file pipeline/llm/context.txt

  ## Save JSON output
  ```bash
  uv run python pipeline/llm/inference.py \
  --image "pipeline/llm/yolo_annotated_test.jpg" \
  --model all \
  --context-file pipeline/llm/context.txt \
  --output-json outputs/llm_inference/result.json