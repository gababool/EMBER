"""
Data Loader Module
Handles loading test images and extracting ground truth.

Ground truth is inferred from either:
  1. Parent folder name ('fire' or 'nofire')
  2. Filename prefix ('fire_' or 'nofire_') as fallback
"""

import os
from pathlib import Path


def get_ground_truth(filepath):
    """
    Extract ground truth label.
    Checks parent folder first, then falls back to filename prefix.
    """
    path = Path(filepath)
    parent_folder = path.parent.name.lower()
    filename = path.name.lower()

    # Check parent folder first
    if parent_folder == 'fire':
        return 'fire'
    elif parent_folder in ('nofire', 'no_fire'):
        return 'no_fire'

    # Fall back to filename prefix
    if filename.startswith('fire_'):
        return 'fire'
    elif filename.startswith('nofire_'):
        return 'no_fire'

    raise ValueError(
        f"Cannot determine ground truth for {filepath} — "
        f"parent folder must be 'fire'/'nofire' or filename must start with 'fire_'/'nofire_'"
    )


def get_all_test_images(directory):
    """
    Recursively find all .jpg and .jpeg images in a directory.
    Returns sorted list for reproducibility.
    """
    path = Path(directory)
    images = list(path.glob('**/*.jpg')) + list(path.glob('**/*.jpeg'))
    return sorted(images)


def get_balanced_sample(images, num_images, seed=42):
    """
    Select a balanced random sample of fire/nofire images.

    Args:
        images: list of image Path objects
        num_images: total images to return
        seed: random seed for reproducibility

    Returns:
        sorted list of selected Path objects
    """
    import random

    fire = [img for img in images if get_ground_truth(img) == 'fire']
    nofire = [img for img in images if get_ground_truth(img) == 'no_fire']

    n_fire = num_images // 2
    n_nofire = num_images - n_fire

    rng = random.Random(seed)
    selected = rng.sample(fire, min(n_fire, len(fire))) + \
               rng.sample(nofire, min(n_nofire, len(nofire)))

    return sorted(selected)