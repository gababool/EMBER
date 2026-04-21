from __future__ import annotations

import cv2
import numpy as np


def validate_image_array(image: np.ndarray) -> None:
    if image is None:
        raise ValueError("image must not be None.")
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy ndarray.")
    if image.size == 0:
        raise ValueError("image must not be empty.")
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D grayscale image or a 3D image.")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError("image must have 1, 3, or 4 channels.")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("image dtype must be numeric.")
    if not np.isfinite(image).all():
        raise ValueError("image must not contain NaN or infinite values.")


def validate_non_negative_number(name: str, value: float | int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if value < 0:
        raise ValueError(f"{name} must not be negative.")


def validate_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must not be negative.")


def validate_min_max(
    min_name: str,
    min_value: float | int,
    max_name: str,
    max_value: float | int,
) -> None:
    validate_non_negative_number(min_name, min_value)
    validate_non_negative_number(max_name, max_value)
    if min_value > max_value:
        raise ValueError(f"{min_name} must not be greater than {max_name}.")


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    validate_image_array(image)

    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.integer):
        min_value = int(np.min(image))
        max_value = int(np.max(image))

        if min_value < 0:
            raise ValueError("integer image values must not be negative.")

        if max_value <= 255:
            return image.astype(np.uint8)

        dtype_info = np.iinfo(image.dtype)
        scaled = image.astype(np.float32) / float(dtype_info.max) * 255.0
        return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)

    if np.issubdtype(image.dtype, np.floating):
        min_value = float(np.min(image))
        max_value = float(np.max(image))

        if min_value < 0:
            raise ValueError("floating-point image values must not be negative.")

        if max_value <= 1.0:
            scaled = image * 255.0
        elif max_value <= 255.0:
            scaled = image
        else:
            raise ValueError(
                "floating-point image values must be in the range 0-1 or 0-255."
            )

        return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)

    raise TypeError("Unsupported image dtype.")


def to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    image_uint8 = to_uint8_image(image)

    if image_uint8.ndim == 2:
        return image_uint8

    channels = image_uint8.shape[2]

    if channels == 1:
        return image_uint8[:, :, 0]
    if channels == 3:
        return cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(image_uint8, cv2.COLOR_BGRA2GRAY)

    raise ValueError("image must have 1, 3, or 4 channels.")