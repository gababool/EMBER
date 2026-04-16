"""
LLM Inference Module
Handles calling Ollama API for multimodal LLM inference.
Returns both the response text and timing metadata for benchmarking.
"""

import base64
import ollama
from pathlib import Path


def load_system_prompt(filepath):
    """Load system prompt from text file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


def load_image_b64(image_path):
    """Read image file and return base64-encoded string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def call_llm(model_name, image_path, system_prompt):
    """
    Call Ollama LLM with image and return response + metadata.

    Args:
        model_name: Ollama model tag (e.g., 'gemma4:e4b')
        image_path: Path to image file
        system_prompt: System prompt text

    Returns:
        dict with keys:
            - response_text: str, the LLM's text output
            - eval_count: int, tokens generated
            - eval_duration_ns: int, generation time in nanoseconds
            - prompt_eval_count: int, prompt tokens processed
            - prompt_eval_duration_ns: int, prompt processing time in ns
            - load_duration_ns: int, model load time in ns
            - total_duration_ns: int, total request time in ns
            - tokens_per_sec: float, generation speed
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_b64 = load_image_b64(image_path)

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": "Analyze this wildfire reconnaissance image.",
                "images": [image_b64]
            }
        ]
    )

    # Extract timing metadata from Ollama response
    eval_count = response.get('eval_count', 0)
    eval_duration = response.get('eval_duration', 0)
    prompt_eval_count = response.get('prompt_eval_count', 0)
    prompt_eval_duration = response.get('prompt_eval_duration', 0)
    load_duration = response.get('load_duration', 0)
    total_duration = response.get('total_duration', 0)

    tokens_per_sec = (eval_count / (eval_duration / 1e9)) if eval_duration > 0 else 0.0

    return {
        'response_text': response['message']['content'],
        'eval_count': eval_count,
        'eval_duration_ns': eval_duration,
        'prompt_eval_count': prompt_eval_count,
        'prompt_eval_duration_ns': prompt_eval_duration,
        'load_duration_ns': load_duration,
        'total_duration_ns': total_duration,
        'tokens_per_sec': tokens_per_sec,
    }