import os
from pathlib import Path

def get_ground_truth(filepath):
    filename = os.path.basename(filepath) 

    if filename.startswith("fire_"):
        return "fire"
    elif filename.startswith("nofire_"):
        return "nofire"
    else:
        raise ValueError(f"Filename {filename} does not follow expected naming convention.")
    
def get_all_test_images(directory):
    path = Path(directory)
    return list(path.glob('**/*.jpg'))

