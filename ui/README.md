# EMBER Frontend

PySide6 frontend for demoing the wildfire detection pipeline on Apple Silicon.

## Run

From the repository root:

```bash
uv run python ui/main_window.py
```

For Apple Silicon GPU percentage in the top bar, run with administrator privileges:

```bash
sudo uv run python ui/main_window.py
```

CPU and RAM percentages work without `sudo`. GPU percentage uses macOS `powermetrics`, which normal user processes may not be allowed to read.

## Demo Flow

Press `Test Image` to select one random image from the wildfire test dataset, either `fire` or `nofire`, and run it through:

- quality screening
- YOLO object detection
- selected LLM reasoning model

If quality screening fails, the app immediately loads another random image. The failed image is still added to the in-memory history strip with a red failed-quality card.

## Controls

- `Test Image`: runs one random dataset image through the pipeline.
- model dropdown: selects the LLM model for the next run.
- theme button: toggles light/dark mode.
- history strip: click a recent processed image to reload its stored outputs for the current app session.

Available LLM models:

- `ministral-3:3b`
- `qwen3-vl:4b`
- `gemma4:e2b`

## Output Panels

- Quality Screening: checklist for resolution, brightness, contrast, and sharpness.
- Processed Image: annotated object-detection output with bounding boxes.
- LLM Reasoning: model classification, correctness badge, reasoning, and recommendation.

History is not persisted between app launches. Processed history images are copied into a temporary directory during the session so that later YOLO runs do not overwrite earlier history thumbnails.

## Apple Silicon GPU Monitor

You can test GPU access directly with:

```bash
sudo powermetrics --samplers gpu_power -i 200 -n 1
```

If the app is not run with `sudo`, the GPU meter may show `N/A`. This is expected.

## Notes

- The full demo depends on local YOLO weights and local Ollama models.
- The frontend uses the reusable pipeline service in `pipeline/service.py`.
- The UI runs the pipeline in a Qt worker thread so the window remains responsive while the demo is running.
