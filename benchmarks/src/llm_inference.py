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


class LLMInferenceResult(BaseModel):
    """Schema for the LLM's structured JSON output.

    Kept in sync with the pipeline's corresponding schema.
    Binary classification only — 'uncertain' is not a valid output.
    """
    classification: Literal["fire_detected", "no_fire_detected"]
    reasoning: str
    recommendation: str
    situation_brief: str
    tactical_priority: Literal["suppress", "contain", "evacuate", "monitor"]


def load_prompt_file(filepath):
    """Load a prompt from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


# Kept for backward compatibility with existing callers.
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

    # Import here to avoid circular dependency issues and keep the benchmark
    # module self-contained where possible.
    from pipeline.context.schemas import OperationalContext
    from pipeline.context.format import format_context

    data = json.loads(path.read_text(encoding='utf-8'))
    context = OperationalContext(**data)
    return format_context(context)


def call_llm(model_name, image_path, system_prompt, context_file=None):
    """
    Call Ollama with structured JSON output enforced via Pydantic schema.

    Args:
        model_name:    Ollama model tag (e.g. 'ministral-3:3b')
        image_path:    Path to the image to analyze
        system_prompt: System prompt text (already loaded)
        context_file:  Optional path to a scenario JSON file. When provided,
                       the formatted operational context is appended to the
                       user message before the image is sent to the LLM.

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

    # Build user message content. Operational context is prepended as text
    # when a context file is provided, so the model sees the context block
    # before the image content.
    user_content = ""
    operational_context_text = load_operational_context(context_file)
    if operational_context_text:
        user_content = operational_context_text

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content, "images": [image_b64]},
        ],
        format=LLMInferenceResult.model_json_schema(),
        options={"temperature": 0},
    )

    raw_content = response['message']['content']
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