import cv2
import numpy as np

from quality_screening.sharpness import check_sharpness


def test_sharp_image_passes():
    # Image with strong edges produces high Laplacian variance
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (590, 430), (255, 255, 255), thickness=2)
    cv2.line(image, (0, 0), (640, 480), (255, 255, 255), thickness=2)
    result = check_sharpness(image)
    assert result["passed"] is True
