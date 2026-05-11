"""
LLM Inference Module
Handles calling Ollama API for multimodal LLM inference with structured
JSON output enforced via Pydantic schema. Returns parsed output plus
timing metadata for benchmarking.

The benchmark maintains its own invocation code rather than importing
from the pipeline. The schema and prompts are kept aligned by discipline
so that benchmark results reflect production behavior.
"""
import base64
import json
from pathlib import Path
from typing import Literal

import ollama
from pydantic import BaseModel, ValidationError

_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
_PROMPTS_DIR = _BENCHMARKS_DIR.parent / "pipeline" / "llm" / "prompts"
_DEFAULT_SYSTEM_CONTENT_FILE = _PROMPTS_DIR / "system_content" / "system_content_latest.txt"


class LLMInferenceResult(BaseModel):
    """ Schema for the LLM's structured JSON output."""
    classification: Literal["fire_detected", "no_fire_detected"]
    reasoning: str
    recommendation: str
    situation_brief: str


def load_prompt_file(filepath):
    """Load a prompt from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def load_system_prompt(filepath):
    return load_prompt_file(filepath)

def load_image_b64(image_path):
    """Read an image file and return its base64-encoded string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def load_operational_context(context_file):
    """Load and format an OperationalContext JSON file as a prompt-ready text block.

    Returns None if context_file is None or does not exist.
    """
    if context_file is None:
        return None
    path = Path(context_file)
    if not path.exists():
        return None

    from pipeline.context.schemas import OperationalContext
    from pipeline.context.format import format_context

    data = json.loads(path.read_text(encoding='utf-8'))
    context = OperationalContext(**data)
    return format_context(context)


def _load_system_content() -> str:
    """Load system role content from system_content.txt.

    Falls back to a minimal inline string if the file is not found, so the
    benchmark can run even if the prompts directory is not on the path.
    """
    if _DEFAULT_SYSTEM_CONTENT_FILE.exists():
        return _DEFAULT_SYSTEM_CONTENT_FILE.read_text(encoding='utf-8').strip()
    return "You are a wildfire analyst. You must respond strictly in JSON format."


def call_llm(model_name, image_path, system_prompt, context_file=None,
             additional_context=None):
    """
    Call Ollama with structured JSON output enforced via Pydantic schema.

    The system role carries a concise role definition loaded from
    pipeline/llm/prompts/system_content.txt. The full task prompt and
    operational context are passed in the user role alongside the image,
    which reduces context exhaustion failures observed with Qwen3-VL when
    long instructions are placed in the system role.

    Args:
        model_name:         Ollama model tag (e.g. 'ministral-3:3b')
        image_path:         Path to the image to analyze
        system_prompt:      Full task prompt text (already loaded from file)
        context_file:       Optional path to a scenario JSON file. When
                            provided, the formatted operational context is
                            prepended to the user message.
        additional_context: Optional extra text appended after the context
                            block (e.g. structured YOLO detection metadata).

    Returns:
        dict with keys:
            - response_text: JSON string of the parsed output (for CSV logging)
            - parsed: dict with all schema fields
            - classification: str, fire_detected or no_fire_detected
            - eval_count, eval_duration_ns, prompt_eval_count,
              prompt_eval_duration_ns, load_duration_ns, total_duration_ns,
              tokens_per_sec
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_b64 = load_image_b64(image_path)

    # Build user message: operational context + any additional context.
    user_content_parts = []
    operational_context_text = load_operational_context(context_file)
    if operational_context_text:
        user_content_parts.append(operational_context_text)
    if additional_context and additional_context.strip():
        user_content_parts.append(additional_context.strip())
    user_content = "\n\n".join(user_content_parts)

    system_content = _load_system_content()

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": f"{system_prompt}\n\n{user_content}",
                "images": [image_b64],
            },
        ],
        format=LLMInferenceResult.model_json_schema(),
        options={"temperature": 0},
    )

    raw_content = response['message']['content']

    if not raw_content.strip():
        raise ValueError(
            "Model returned empty response. This may indicate context window "
            "exhaustion due to thinking mode (observed with qwen3-vl on complex "
            "images with long prompts). Consider a simpler image or shorter prompt."
        )

    try:
        parsed = LLMInferenceResult.model_validate_json(raw_content)
    except ValidationError as e:
        raise ValueError(
            f"Model returned invalid structured output.\n"
            f"Raw output:\n{raw_content}\n\n"
            f"Validation error:\n{e}"
        ) from e

    parsed_dict = parsed.model_dump()

    eval_count = response.get('eval_count', 0)
    eval_duration = response.get('eval_duration', 0)
    prompt_eval_count = response.get('prompt_eval_count', 0)
    prompt_eval_duration = response.get('prompt_eval_duration', 0)
    load_duration = response.get('load_duration', 0)
    total_duration = response.get('total_duration', 0)
    tokens_per_sec = (eval_count / (eval_duration / 1e9)) if eval_duration > 0 else 0.0

    return {
        'response_text': json.dumps(parsed_dict),
        'parsed': parsed_dict,
        'classification': parsed_dict['classification'],
        'eval_count': eval_count,
        'eval_duration_ns': eval_duration,
        'prompt_eval_count': prompt_eval_count,
        'prompt_eval_duration_ns': prompt_eval_duration,
        'load_duration_ns': load_duration,
        'total_duration_ns': total_duration,
        'tokens_per_sec': tokens_per_sec,
    }