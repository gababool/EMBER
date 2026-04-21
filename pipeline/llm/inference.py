import argparse
import base64
import json
from pathlib import Path
from typing import Literal

import ollama
from pydantic import BaseModel, ValidationError

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = SCRIPT_DIR / "prompt.txt"
DEFAULT_CONTEXT_FILE = SCRIPT_DIR / "context.txt"

MODELS = {
    "ministral": "ministral-3:3b",
    "qwen3": "qwen3-vl:4b",
    "gemma4": "gemma4:e2b",
}


class LLMInferenceResult(BaseModel):
    classification: Literal["fire_detected", "no_fire_detected", "uncertain"]
    reasoning: str
    recommendation: str

def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()

def load_system_prompt(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found at {file_path}")
    return path.read_text(encoding="utf-8")


def load_image(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_user_prompt(additional_context: str | None = None) -> str:
    base_prompt = "Analyze this YOLO-annotated wildfire reconnaissance image."

    if additional_context and additional_context.strip():
        return (
            f"{base_prompt}\n\n"
            f"Additional operational context:\n"
            f"{additional_context.strip()}"
        )
    return base_prompt


def run_llm_inference(
    model_name: str,
    image_path: str,
    system_prompt: str,
    additional_context: str | None = None,
) -> dict:
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Valid options: {list(MODELS.keys())}"
        )

    model_tag = MODELS[model_name]
    image_b64 = load_image(image_path)
    user_prompt = build_user_prompt(additional_context)

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
            "images": [image_b64],
        },
    ]

    response = ollama.chat(
        model=model_tag,
        messages=messages,
        format=LLMInferenceResult.model_json_schema(),
        options={"temperature": 0},
    )

    raw_content = response.message.content

    try:
        parsed = LLMInferenceResult.model_validate_json(raw_content)
    except ValidationError as e:
        raise ValueError(
            f"Model returned invalid structured output.\n"
            f"Raw output:\n{raw_content}\n\n"
            f"Validation error:\n{e}"
        ) from e

    return {
        "model_name": model_name,
        "model_tag": model_tag,
        "image_path": image_path,
        "parsed": parsed.model_dump(),
        "raw_response": raw_content,
    }


def save_result_json(result: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multimodal LLM inference on a YOLO-annotated image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the YOLO-annotated image.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()) + ["all"],
        help="Which model to run.",
    )
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_FILE),
        help="Path to the system prompt file.",
    )
    parser.add_argument(
        "--context-file",
        help="Optional path to additional context text.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save the result as JSON. When --model all is used, "
        "the model name will be appended before the file suffix.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    system_prompt = load_system_prompt(args.prompt_file)
    additional_context = None

    if args.context_file:
        context_path = resolve_path(args.context_file)
        if not context_path.exists():
            raise FileNotFoundError(f"Context file not found at {context_path}")
        additional_context = context_path.read_text(encoding="utf-8")

    model_names = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_name in model_names:
        print(f"\n=== {model_name} ({MODELS[model_name]}) ===")

        result = run_llm_inference(
            model_name=model_name,
            image_path=args.image,
            system_prompt=system_prompt,
            additional_context=additional_context,
        )

        print(json.dumps(result["parsed"], indent=2))

        if args.output_json:
            output_path = Path(args.output_json)
            if args.model == "all":
                stem = output_path.stem
                suffix = output_path.suffix or ".json"
                model_output = output_path.with_name(f"{stem}_{model_name}{suffix}")
                save_result_json(result, str(model_output))
                print(f"Saved JSON to {model_output}")
            else:
                save_result_json(result, args.output_json)
                print(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()