import numpy as np

# Run tests with: uv run pytest tests/ -v
from quality_screening.contrast import check_contrast


def test_normal_image_passes():
    # Half dark-gray (80), half light-gray (180) gives standard deviance around 50, within [30, 120]
    image = np.full((480, 640, 3), 80, dtype=np.uint8)
    image[240:, :] = 180
    result = check_contrast(image)
    assert result["passed"] is True
