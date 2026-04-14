"""
Data Loader Module
Handles loading test images and extracting ground truth from filenames.
"""

import os
from pathlib import Path


def get_ground_truth(filepath):
    """
    Extract ground truth label from filename convention.
    """
    filename = os.path.basename(filepath)
    if filename.startswith("fire_"):
        return "fire"
    elif filename.startswith("nofire_"):
        return "no_fire"
    else:
        raise ValueError(f"Filename {filename} does not follow expected naming convention.")


def get_all_test_images(directory):
    """
    Recursively find all .jpg images in a directory.
    Returns sorted list for reproducibility.
    """
    path = Path(directory)
    images = list(path.glob('**/*.jpg'))
    return sorted(images)