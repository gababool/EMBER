# Wildfire MLLM Inference

Runs multimodal LLM inference.

## Setup

**1. Install and start Ollama**
```bash
brew install ollama
ollama serve
```

**2. Pull models** (in a new terminal)
```bash
ollama pull ministral-3:3b
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