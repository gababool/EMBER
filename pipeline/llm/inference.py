import ollama
import base64
import sys
from typing import Optional, Literal
from pydantic import BaseModel, ValidationError
from pathlib import Path

MODELS = {
    "ministral": "ministral-3:3b",
    "qwen3": "qwen3-vl:4b",
    "gemma4": "gemma4:e4b",
}

class LLMInferenceResult(BaseModel):
    classification: Literal["fire_detected", "no_fire_detected", "uncertain"]
    reasoning: str
    recommendation: str


def load_image(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_system_prompt(prompt_file: str) -> str:
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found at {prompt_file}")
    return path.read_text(encoding='utf-8')

def build_user_prompt(base_prompt: str, additional_context: Optional[str] = None) -> str:
    if additional_context and additional_context.strip():
        return (
            f"{base_prompt}\n\n"
            f"Additional operational context:\n"
            f"{additional_context.strip()}"
        )
    return base_prompt


def query_model(
    model_name: str,
    image_path: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    additional_context: Optional[str] = None,
) -> dict:
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Valid options: {list(MODELS.keys())}"
        )

    model_tag = MODELS[model_name]
    image_b64 = load_image(image_path)
    user_prompt = build_user_prompt(prompt, additional_context)

    print(f"\n=== {model_name} ({model_tag}) ===")

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt,
        })

    messages.append({
        "role": "user",
        "content": user_prompt,
        "images": [image_b64],
    })

    response = ollama.chat(
        model=model_tag,
        messages=messages,
        format=LLMInferenceResult.model_json_schema(),
        options={"temperature": 0},
    )

    eval_count = response.get("eval_count", 0)
    eval_duration = response.get("eval_duration", 0)
    prompt_eval = response.get("prompt_eval_duration", 0)
    total = response.get("total_duration", 0)

    print(f"Prompt eval: {prompt_eval / 1e9:.2f}s")
    print(f"Eval: {eval_duration / 1e9:.2f}s ({eval_count} tokens)")
    print(f"Total: {total / 1e9:.2f}s")

    raw_content = response.message.content

    try:
        parsed = LLMInferenceResult.model_validate_json(raw_content)
    except ValidationError as e:
        raise ValueError(
            f"Model returned invalid structured output.\nRaw output:\n{raw_content}\n\nValidation error:\n{e}"
        ) from e

    return {
        "model_name": model_name,
        "model_tag": model_tag,
        "raw_response": raw_content,
        "parsed": parsed.model_dump(),
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration,
        "prompt_eval_duration_ns": prompt_eval,
        "total_duration_ns": total,
    }


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    prompt = sys.argv[2] if len(sys.argv) > 2 else (
        "Analyze this YOLO-annotated wildfire reconnaissance image and return "
        "classification, reasoning, and recommendation."
    )

    system_prompt = None
    additional_context = None

    if "--prompt-file" in sys.argv:
        idx = sys.argv.index("--prompt-file")
        system_prompt = load_system_prompt(sys.argv[idx + 1])

    if "--context-file" in sys.argv:
        idx = sys.argv.index("--context-file")
        additional_context = Path(sys.argv[idx + 1]).read_text(encoding="utf-8")

    if not Path(image_path).exists():
        print(f"Error: image not found at {image_path}")
        sys.exit(1)

    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")

    for model_name in MODELS:
        result = query_model(
            model_name=model_name,
            image_path=image_path,
            prompt=prompt,
            system_prompt=system_prompt,
            additional_context=additional_context,
        )
        print(result["parsed"])


if __name__ == "__main__":
    main()