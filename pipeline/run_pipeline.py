# To run, from repo root:
# uv run python -m pipeline/run_pipeline.py <image_path> --yolo-model best --llm-model ministral

# Example:
# uv run python -m pipeline.run_pipeline pipeline/object_detection/test_image.jpg --yolo-model best --llm-model ministral


import sys
from pathlib import Path

# make both stage modules importable
PIPELINE_DIR = Path(__file__).resolve().parent

import argparse

from pipeline.object_detection.config import ANNOTATED_OUTPUT_DIR, ANNOTATED_OUTPUT_FILENAME
from pipeline.object_detection.run import run as run_detection
from pipeline.llm.inference import run_llm_inference, load_system_prompt

DEFAULT_PROMPT_FILE = PIPELINE_DIR / "llm" / "prompt.txt"


def run_pipeline(image_path: str, yolo_model: str, llm_model: str):

    #gonna add stage 1: Quality Screening as soon as that module is done



    print("STAGE 2: OBJECT DETECTION")
    run_detection(image_path, yolo_model, show=False)

    annotated_image_path = ANNOTATED_OUTPUT_DIR / ANNOTATED_OUTPUT_FILENAME

    if not annotated_image_path.exists():
        print("[PIPELINE ERROR] Annotated image not found — object detection may have failed.")
        return

    print("\nSTAGE 3: LLM INFERENCE")
    system_prompt = load_system_prompt(str(DEFAULT_PROMPT_FILE))
    result = run_llm_inference(
        model_name=llm_model,
        image_path=str(annotated_image_path),
        system_prompt=system_prompt,
    )

    import json
    print(json.dumps(result["parsed"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--yolo-model", default="best")
    parser.add_argument("--llm-model", default="ministral")
    args = parser.parse_args()

    run_pipeline(args.image_path, args.yolo_model, args.llm_model)
