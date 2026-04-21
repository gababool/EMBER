# How to run the quality screening pipeline

All commands below are run from the repository root (`wildfire-bsc-thesis/`). They are portable — they do not depend on anyone's local machine path.

## Run the quality screening stage on an image

The stage can be run directly as a module. It accepts a path to the image as a positional argument. The path may be absolute or relative to your current working directory.

```bash
uv run python -m pipeline.quality_screening.screening <path/to/image>
```

**Example (image anywhere on disk):**

```bash
uv run python -m pipeline.quality_screening.screening ./pipeline/test_image.jpg
```

**Stop on the first failing check:**

```bash
uv run python -m pipeline.quality_screening.screening <path/to/image> --stop-on-first-failure
```

The command prints a JSON result to stdout and exits with code `0` when the image passes and `1` when it fails.

## Run the quality screening tests

```bash
uv run pytest pipeline/quality_screening/tests -v
```

## What this runs

The quality screening stage runs these checks in order:

1. Resolution
2. Brightness
3. Contrast
4. Sharpness

## Expected output

The command returns a structured JSON result showing:

- Whether the image passed overall
- The result of each individual check
- Any failed checks
- The image path used

## Using the stage from Python

If you prefer to call the stage from your own Python code instead of the CLI:

```python
from pipeline.quality_screening import run_quality_screening_from_path

result = run_quality_screening_from_path("path/to/image.jpg")
print(result)
```
