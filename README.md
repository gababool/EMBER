# EMBER: Edge-Deployed Multimodal LLMs for Wildfire Decision Support in Critical Operations

Bachelor's thesis (Software Engineering and Management) — University of Gothenburg / Chalmers University of Technology, 2026.

An edge-deployable pipeline that combines **YOLOv8-based fire/smoke detection** with **multimodal large language models (MLLMs)** to turn aerial wildfire imagery into structured, geographically-grounded tactical decision support — fully offline, on consumer-grade hardware.

Conducted in collaboration with the Försvarets Materielverk FMV (Swedish Defence Materiel Administration) and evaluated with Swedish wildfire response professionals.

## Why

Wildfire reconnaissance often happens where network connectivity is unreliable or absent, yet existing detection systems output raw bounding boxes with no interpretation or tactical guidance. This project asks whether small, locally-run MLLMs can bridge that gap — reasoning over both an image and pre-loaded geographic context (roads, water sources, terrain, wind) to produce recommendations an operator can actually act on.

## What it does

1. **Quality screening** — OpenCV-based checks (resolution, brightness, contrast, a tile-based sharpness measure) filter out unusable imagery before it reaches detection.
2. **Object detection** — a YOLOv8 model fine-tuned for fire/smoke produces bounding boxes and confidence scores.
3. **Geographic context extraction** — offline OSM and SRTM data is queried for a given coordinate to build a structured `OperationalContext` (roads, water sources with supply category, settlements, terrain slope/aspect, assets at risk, fire stations, wind-derived spread direction).
4. **MLLM reasoning** — a locally-served multimodal LLM (via Ollama) combines the image and context to produce structured JSON output: classification, reasoning, a tactical recommendation, a radio-ready situation brief, and key operational constraints.
5. **Operator UI** — a PySide6 desktop app shows the pipeline output alongside a scenario map.

## Models evaluated

Three architectures, chosen for parameter comparability and geographic/organizational diversity:

| Model | Origin | Effective params | Architecture |
|---|---|---|---|
| Ministral 3 (3B) | Mistral AI | 3B | Dense |
| Qwen3-VL (4B) | Alibaba | 4B | Dense |
| Gemma 4 E2B | Google DeepMind | 2.3B | Per-Layer Embedding |

All benchmarked on identical Apple Silicon (M2, 16 GB) hardware via Ollama.

## Methodology

Two Design Science Research cycles:
- **Cycle 1** — minimum viable pipeline (image → YOLO → LLM classification), baseline quantitative benchmarking, requirements elicitation via practitioner interviews.
- **Cycle 2** — added the geographic context pipeline, structured output schema, prompt engineering to address model-specific failure modes, and a full benchmark plus qualitative evaluation interviews with wildfire response professionals.

## Key findings

- Small MLLMs can classify fire presence competently and run within the memory budget of a consumer laptop, but **richer prompts and geographic context degraded classification reliability** in all three models — in different, architecture-specific ways (Qwen3-VL: frequent malformed/empty output under load; Ministral: collapse toward always predicting "fire"; Gemma: increasingly conservative, shallow context use).
- **Specificity, not mere explainability, drives decision-support value.** Practitioners rejected generic advice and confidence scores as noise, but responded positively to recommendations grounded in named locations, distances, and resource categories.
- No single model suited every role — pointing toward a **multi-pass architecture** (e.g., one model for classification, another for reasoning/communication) as future work.

Full results, benchmarks, and the qualitative interview analysis are in the thesis report.

## Authors

Martin Lidgren, Erik Nisbet, Edvin Sanfridsson, Love Carlander Strandäng, Johannes Borg

Supervisor: Hans-Martin Heyn · Examiner: Christian Berger

## Citation

If you reference this work, please cite the thesis:

> Lidgren, M., Nisbet, E., Sanfridsson, E., Carlander Strandäng, L., Borg, J. (2026). *Edge-Deployed Multimodal LLMs for Wildfire Decision Support in Critical Operations*. Bachelor's thesis, University of Gothenburg / Chalmers University of Technology.
