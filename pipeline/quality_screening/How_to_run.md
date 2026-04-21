## How to run

### Run the quality screening tests

From the repository root:

```bash
uv run pytest pipeline/quality_screening/tests -v
```

### Run the full quality screening stage on an image

You can run the quality screening stage on a single image from the repository root with:

```bash
uv run python -c 'from pipeline.quality_screening import run_quality_screening_from_path; print(run_quality_screening_from_path("/absolute/path/to/image.jpg"))'
```

**Example:**

```bash
uv run python -c 'from pipeline.quality_screening import run_quality_screening_from_path; print(run_quality_screening_from_path("/Users/robin.carlander/Wildfire BSC Thesis/pipeline/test_image.jpg"))'
```

### What this runs

The quality screening stage runs these checks in order:

1. Resolution
2. Brightness
3. Contrast
4. Sharpness

### Expected output

The command returns a structured result showing:

- Whether the image passed overall
- The result of each individual check
- Any failed checks
- The image path used