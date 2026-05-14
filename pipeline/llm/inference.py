import argparse
import base64
import json
from pathlib import Path
import time
from typing import Callable, Literal

import ollama
from pydantic import BaseModel, ValidationError

from pipeline.context.format import format_context
from pipeline.context.schemas import OperationalContext

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_PROMPT_FILE = PROMPTS_DIR / "latest.txt"
DEFAULT_SYSTEM_CONTENT_FILE = PROMPTS_DIR / "system_content" / "system_content_latest.txt"

MODELS = {
    "ministral": "ministral-3:3b",
    "qwen3": "qwen3-vl:4b",
    "gemma4": "gemma4:e2b",
    "ministral-3:3b": "ministral-3:3b",
    "qwen3-vl:4b": "qwen3-vl:4b",
    "gemma4:e2b": "gemma4:e2b",
}

BENCHMARK_MODEL_TAGS = [
    "ministral-3:3b",
    "qwen3-vl:4b",
    "gemma4:e2b",
]


class LLMInferenceResult(BaseModel):
    classification: Literal["fire_detected", "no_fire_detected"]
    reasoning: str
    recommendation: str
    situation_brief: str | None = None
    key_constraints: list[str] | None = None


class LLMInferenceCancelled(RuntimeError):
    """Raised when an in-flight LLM request is cancelled by the caller."""


class LLMInferenceTimedOut(RuntimeError):
    """Raised when an in-flight LLM request exceeds its timeout."""


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def load_text_file(file_path: str, label: str) -> str:
    path = resolve_path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at {path}")
    return path.read_text(encoding="utf-8")

def load_system_prompt(file_path: str) -> str:
    return load_text_file(file_path, "Prompt file")

def load_system_content(file_path: str) -> str:
    return load_text_file(file_path, "System content file")

def load_context(file_path: str) -> str:
    return load_text_file(file_path, "Context file")


def load_image(image_path: str) -> str:
    path = resolve_path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_user_content(
    additional_context: str | None = None,
    operational_context: OperationalContext | None = None,
) -> str:
    parts = []
    if operational_context is not None:
        parts.append(format_context(operational_context))
    if additional_context and additional_context.strip():
        parts.append(f"Additional operational context:\n{additional_context.strip()}")
    return "\n\n".join(parts)


def run_llm_inference(
    model_name: str,
    image_path: str,
    system_prompt: str,
    system_content: str,
    prompt_file: str,
    additional_context: str | None = None,
    context_file: str | None = None,
    operational_context: OperationalContext | None = None,
    should_cancel: Callable[[], bool] | None = None,
    timeout_seconds: float | None = None,
) -> dict:
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Valid options: {list(MODELS.keys())}"
        )

    model_tag = MODELS[model_name]
    image_path_resolved = str(resolve_path(image_path))
    image_b64 = load_image(image_path)
    deadline = time.monotonic() + timeout_seconds if timeout_seconds else None

    if operational_context is None and context_file is not None:
        context_path = resolve_path(context_file)
        if context_path.exists():
            import json
            from pipeline.context.schemas import OperationalContext
            data = json.loads(context_path.read_text(encoding="utf-8"))
            operational_context = OperationalContext(**data)

    user_content = build_user_content(additional_context, operational_context)
    if should_cancel and should_cancel():
        raise LLMInferenceCancelled("LLM inference cancelled.")
    if deadline is not None and time.monotonic() >= deadline:
        raise LLMInferenceTimedOut("LLM inference timed out.")

    messages = [
        {
            "role": "system",
            "content": f"{system_content}"
        },
        {
            "role": "user",
            "content": f"{system_prompt}\n\n{user_content}",
            "images": [image_b64],
        },
    ]

    response_stream = ollama.chat(
        model=model_tag,
        messages=messages,
        format=LLMInferenceResult.model_json_schema(),
        options={"temperature": 0},
        stream=True,
    )

    raw_parts = []
    for chunk in response_stream:
        if should_cancel and should_cancel():
            close = getattr(response_stream, "close", None)
            if close is not None:
                close()
            raise LLMInferenceCancelled("LLM inference cancelled.")
        if deadline is not None and time.monotonic() >= deadline:
            close = getattr(response_stream, "close", None)
            if close is not None:
                close()
            raise LLMInferenceTimedOut("LLM inference timed out.")

        message = getattr(chunk, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content:
            raw_parts.append(content)

    raw_content = "".join(raw_parts)

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
        "image_path": image_path_resolved,
        "prompt_file": str(resolve_path(prompt_file)),
        "context_file": str(resolve_path(context_file)) if context_file else None,
        "parsed": parsed.model_dump(),
        "raw_response": raw_content,
    }


def save_result_json(result: dict, output_path: str) -> None:
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multimodal LLM inference on an image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
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
        "--context-json",
        help="Optional path to an OperationalContext JSON file (formatted automatically).",
    )
    parser.add_argument(
        "--output-json",
        help=(
            "Optional path to save the result as JSON. "
            "When --model all is used, the model name will be appended before the file suffix."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prompt_path = resolve_path(args.prompt_file)
    system_prompt = load_system_prompt(str(prompt_path))
    system_content = load_system_content(str(DEFAULT_SYSTEM_CONTENT_FILE))

    additional_context = None
    context_path = None
    operational_context = None

    if args.context_file:
        context_path = resolve_path(args.context_file)
        additional_context = load_context(str(context_path))

    if args.context_json:
        json_path = resolve_path(args.context_json)
        operational_context = OperationalContext.model_validate_json(
            json_path.read_text(encoding="utf-8")
        )

    model_names = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_name in model_names:
        print(f"\n=== {model_name} ({MODELS[model_name]}) ===")

        result = run_llm_inference(
            model_name=model_name,
            image_path=args.image,
            system_prompt=system_prompt,
            system_content=system_content,
            prompt_file=str(prompt_path),
            additional_context=additional_context,
            context_file=str(context_path) if context_path else None,
            operational_context=operational_context,
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
