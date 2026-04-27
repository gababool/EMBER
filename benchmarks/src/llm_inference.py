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
    """
    classification: Literal["fire_detected", "no_fire_detected", "uncertain"]
    reasoning: str
    recommendation: str


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


def call_llm(model_name, image_path, system_prompt):
    """
    Call Ollama with structured JSON output enforced via Pydantic schema.

    Args:
        model_name: Ollama model tag (e.g. 'ministral-3:3b')
        image_path: Path to the image to analyze
        system_prompt: System prompt text (already loaded)

    Returns:
        dict with keys:
            - response_text: JSON string of the parsed output (for CSV logging)
            - parsed: dict with classification, reasoning, recommendation
            - classification: str, one of fire_detected / no_fire_detected / uncertain
            - eval_count, eval_duration_ns, prompt_eval_count,
              prompt_eval_duration_ns, load_duration_ns, total_duration_ns,
              tokens_per_sec
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_b64 = load_image_b64(image_path)

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "", "images": [image_b64]},
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

    # Extract timing metadata from the Ollama response
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