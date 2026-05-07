# Quality screening

All commands below are run from the repository root (`wildfire-bsc-thesis/`).

## Run on one image

The stage can be run directly as a module. It accepts a path to the image as a positional argument. The path may be absolute or relative to your current working directory.

```bash
uv run python -m pipeline.quality_screening.screening <path/to/image>
```

Example:

```bash
uv run python -m pipeline.quality_screening.screening ./pipeline/object_detection/test_image.jpg
```

Stop after the first failing check:

```bash
uv run python -m pipeline.quality_screening.screening <path/to/image> --stop-on-first-failure
```

The command prints a JSON result to stdout and exits with code `0` when the image passes and `1` when it fails.

## Run tests

```bash
uv run pytest pipeline/quality_screening/tests -v
```

Run one test file:

```bash
uv run pytest pipeline/quality_screening/tests/test_brightness.py -v
uv run pytest pipeline/quality_screening/tests/test_sharpness.py -v
```

## Checks

The quality screening stage runs these checks in order:

1. `resolution`
2. `brightness`
3. `contrast`
4. `sharpness`

### Resolution

Rejects images smaller than the configured minimum size.

Pipeline defaults:

- `min_width = 300`
- `min_height = 300`

Result fields include:

- `resolution`

### Brightness

Brightness uses grayscale intensity. It passes when either:

- the average brightness is between `min_brightness` and `max_brightness`, or
- the image is dark on average but has enough bright pixels above `bright_pixel_threshold`

The bright-pixel fallback is meant for night wildfire images where the frame can be mostly dark while the fire itself is visible.

Pipeline defaults:

- `min_brightness = 40.0`
- `max_brightness = 220.0`
- `bright_pixel_threshold = 100.0`
- `min_bright_pixel_fraction = 0.01`

Failure reasons:

- `underexposed`
- `overexposed`

Result fields include:

- `brightness`
- `bright_pixel_fraction`
- `bright_pixel_threshold`
- `min_bright_pixel_fraction`

### Contrast

Contrast currently uses global grayscale standard deviation. It rejects images whose contrast is below `min_contrast` or above `max_contrast`.

Pipeline defaults:

- `min_contrast = 10.0`
- `max_contrast = 120.0`

Failure reasons:

- `low_contrast`
- `high_contrast`

Result fields include:

- `contrast`

### Sharpness

Sharpness uses tile-based Tenengrad scoring with texture filtering.

The image is split into non-overlapping tiles. Very small edge tiles are skipped. Tiles with too little texture are ignored because smoke, sky, and smooth haze do not provide enough structure for a useful blur decision. The remaining tiles are scored using Tenengrad sharpness.

The check passes only when:

- enough tiles contain texture,
- the configured sharpness percentile is above `min_sharpness`, and
- enough valid tiles are individually sharp.

Pipeline defaults:

- `min_sharpness = 100.0`
- `min_texture = 8.0`
- `tile_size = 128`
- `min_valid_tile_fraction = 0.1`
- `min_sharp_tile_fraction = 0.5`
- `sharpness_percentile = 30.0`

Failure reasons:

- `invalid_image`
- `insufficient_textured_regions`
- `low_sharpness`
- `too_few_sharp_tiles`

Result fields include:

- `sharpness`
- `valid_tiles`
- `total_tiles`
- `sharp_tiles`
- `texture_threshold`
- `sharpness_threshold`
- `sharpness_percentile`

## Expected output

The command returns a structured JSON result showing:

- Whether the image passed overall
- The result of each individual check
- Any failed checks
- The image path used

Example shape:

```json
{
  "passed": true,
  "checks": {
    "resolution": {"passed": true, "reason": "", "resolution": [640, 480]},
    "brightness": {"passed": true, "reason": "", "brightness": 95.2},
    "contrast": {"passed": true, "reason": "", "contrast": 42.6},
    "sharpness": {"passed": true, "reason": "", "sharpness": 312.4}
  },
  "failed_checks": [],
  "image_path": "path/to/image.jpg"
}
```

## Using the stage from Python

If you prefer to call the stage from your own Python code instead of the CLI:

```python
from pipeline.quality_screening import run_quality_screening_from_path

result = run_quality_screening_from_path("path/to/image.jpg")
print(result)
```

To override thresholds, pass a `QualityScreeningConfig`:

```python
from pipeline.quality_screening import (
    QualityScreeningConfig,
    run_quality_screening_from_path,
)

config = QualityScreeningConfig(
    min_width=300,
    min_height=300,
    bright_pixel_threshold=100.0,
    min_bright_pixel_fraction=0.01,
    min_sharpness=100.0,
)

result = run_quality_screening_from_path("path/to/image.jpg", config=config)
print(result)
```
