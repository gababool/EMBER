'''
To run:

uv run python -m pipeline.run_pipeline pipeline/llm/image.png \
    --yolo-model best \
    --llm-model ministral \
    --prompt-file pipeline/llm/prompts/c2v4prompt.txt \
    --context-json data/contexts/example_scenario.json
    
'''

import argparse
import json
from pathlib import Path
from pipeline.service import create_default_pipeline
from pipeline.object_detection.detections import YOLO_INPUT_MODES

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = SCRIPT_DIR / "pipeline" / "llm" / "prompts" / "c2v4prompt.txt"


def run_pipeline(image_path: str, yolo_model: str, llm_model: str,
prompt_file: str, context_json: str | None = None,
yolo_input_mode: str = "annotated_image"):
    runner = create_default_pipeline(
        yolo_model=yolo_model,
        llm_model=llm_model,
        prompt_file=prompt_file,
        context_file=context_json,
        yolo_input_mode=yolo_input_mode,
    )
    stage_results = []

    for event in runner.iter_events(image_path):
        if event.event_type == "stage_started":
            print(f"\nSTAGE: {event.label.upper()}")
            continue

        if event.result is None:
            continue

        result = event.result
        stage_results.append(result)

        if result.passed:
            print(f"[OK] {result.label}")
        else:
            print(f"[PIPELINE ERROR] {result.error or result.label}")

        if result.stage_name == "quality_screening":
            print(json.dumps(result.output.get("checks", {}), indent=2, default=str))

        if result.stage_name == "object_detection":
            print(json.dumps(result.output.get("detections", []), indent=2, default=str))
            print(f"Annotated image: {result.output.get('annotated_image_path')}")

        if result.stage_name == "llm_reasoning":
            print(json.dumps(result.output.get("parsed", {}), indent=2, default=str))

    return {
        "image_path": str(Path(image_path).resolve()),
        "passed": all(result.passed for result in stage_results),
        "stages": [
            {
                "stage_name": result.stage_name,
                "label": result.label,
                "passed": result.passed,
                "error": result.error,
                "output": result.output,
            }
            for result in stage_results
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--yolo-model", default="best")
    parser.add_argument("--llm-model", default="ministral")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT))
    parser.add_argument("--context-json", default=None)
    parser.add_argument(
        "--yolo-input-mode",
        choices=YOLO_INPUT_MODES,
        default="annotated_image",
        help=(
            "How YOLO output is provided to the LLM: annotated image, "
            "summary text context, or text context with bounding box locations."
        ),
    )
    args = parser.parse_args()
    run_pipeline(args.image_path, args.yolo_model, args.llm_model,
                 args.prompt_file, args.context_json, args.yolo_input_mode)
