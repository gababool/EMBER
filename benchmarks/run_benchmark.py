"""
Benchmark Runner
Runs all models against all test images and logs results.

Usage:
    uv run python run_benchmark.py --images data/test_images --prompt prompts/system_prompt.txt
    uv run python run_benchmark.py --images data/test_images --prompt prompts/system_prompt.txt --models ministral-3:3b qwen3-vl:4b
    uv run python run_benchmark.py --images data/test_images --prompt prompts/system_prompt.txt --dry-run
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from src.data_loader import get_all_test_images, get_ground_truth
from src.llm_inference import load_system_prompt, call_llm
from src.logger import init_csv, log_result, log_progress, RESULT_COLUMNS
from src.metrics import (
    track_memory_usage,
    get_model_size,
    count_words,
    evaluate_accuracy,
)

# All models available for benchmarking
ALL_MODELS = [
    'ministral-3:3b',
    'ministral-3:8b',
    'qwen3-vl:2b',
    'qwen3-vl:4b',
    'qwen3-vl:8b',
    'gemma4:e2b',
    'gemma4:e4b',
]


def run_single_inference(model_name, image_path, system_prompt):
    """
    Run a single model on a single image and return the result dict.
    """
    ground_truth = get_ground_truth(image_path)

    # Run inference
    result = call_llm(model_name, str(image_path), system_prompt)

    # Evaluate
    response_text = result['response_text']
    correct = evaluate_accuracy(response_text, ground_truth)

    # Determine classification for logging
    response_upper = response_text.upper()
    if '[FIRE_DETECTED]' in response_upper and '[NO_FIRE_DETECTED]' not in response_upper:
        llm_classification = 'fire'
    elif '[NO_FIRE_DETECTED]' in response_upper:
        llm_classification = 'no_fire'
    else:
        llm_classification = 'ambiguous'

    # Memory snapshot (taken right after inference while model is still loaded)
    memory_gb = track_memory_usage()

    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': model_name,
        'image_path': str(image_path),
        'ground_truth': ground_truth,
        'llm_classification': llm_classification,
        'correct': correct,
        'response_text': response_text,
        'word_count': count_words(response_text),
        'eval_count': result['eval_count'],
        'tokens_per_sec': round(result['tokens_per_sec'], 2),
        'prompt_eval_duration_s': round(result['prompt_eval_duration_ns'] / 1e9, 3),
        'eval_duration_s': round(result['eval_duration_ns'] / 1e9, 3),
        'total_duration_s': round(result['total_duration_ns'] / 1e9, 3),
        'load_duration_s': round(result['load_duration_ns'] / 1e9, 3),
        'model_size_gb': get_model_size(model_name),
        'memory_usage_gb': memory_gb,
    }


def warmup_model(model_name, system_prompt, warmup_image=None):
    """
    Send a throwaway request to load the model into memory.
    Prevents cold-start from skewing the first real inference.
    """
    log_progress(f"Warming up {model_name}...")
    try:
        # Use a minimal prompt if no warmup image is available
        import ollama
        ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}]
        )
        log_progress(f"  {model_name} loaded and ready.")
    except Exception as e:
        log_progress(f"  Warmup failed for {model_name}: {e}")


def run_benchmark(image_dir, prompt_file, models, output_dir='results', dry_run=False):
    """
    Main benchmark loop.

    For each model, iterates over all images, runs inference,
    and logs results to a per-model CSV file.
    """
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt(prompt_file)
    images = get_all_test_images(image_dir)

    if not images:
        log_progress(f"No images found in {image_dir}")
        return

    log_progress(f"Found {len(images)} images in {image_dir}")
    log_progress(f"Models to benchmark: {models}")
    log_progress(f"Output directory: {output_dir}")

    if dry_run:
        log_progress("DRY RUN — listing what would be executed:")
        for model in models:
            log_progress(f"  {model}: {len(images)} images")
        log_progress(f"  Total inferences: {len(models) * len(images)}")
        return

    # Run each model
    for model_name in models:
        csv_file = output_path / f"benchmark_{model_name.replace(':', '_')}.csv"
        init_csv(str(csv_file))

        log_progress(f"=== Starting {model_name} ({len(images)} images) ===")

        # Warmup to avoid cold-start bias on first image
        warmup_model(model_name, system_prompt)

        correct_count = 0
        error_count = 0
        ambiguous_count = 0

        for i, image_path in enumerate(images):
            try:
                result = run_single_inference(model_name, image_path, system_prompt)
                log_result(str(csv_file), result)

                # Track running stats
                if result['correct'] is True:
                    correct_count += 1
                elif result['correct'] is None:
                    ambiguous_count += 1

                log_progress(
                    f"  [{i+1}/{len(images)}] {image_path.name} -> "
                    f"{result['llm_classification']} "
                    f"({'✓' if result['correct'] else '✗' if result['correct'] is False else '?'}) "
                    f"| {result['tokens_per_sec']} tok/s "
                    f"| {result['total_duration_s']}s"
                )

            except Exception as e:
                error_count += 1
                log_progress(f"  [{i+1}/{len(images)}] ERROR on {image_path.name}: {e}")

        # Summary for this model
        total = len(images) - error_count
        accuracy = (correct_count / total * 100) if total > 0 else 0
        log_progress(
            f"=== {model_name} done === "
            f"Accuracy: {correct_count}/{total} ({accuracy:.1f}%) | "
            f"Ambiguous: {ambiguous_count} | Errors: {error_count}"
        )

    log_progress("Benchmark complete.")


def main():
    parser = argparse.ArgumentParser(description='Run wildfire detection benchmark')
    parser.add_argument(
        '--images', required=True,
        help='Directory containing test images (fire_*.jpg, nofire_*.jpg)'
    )
    parser.add_argument(
        '--prompt', required=True,
        help='Path to system prompt text file'
    )
    parser.add_argument(
        '--models', nargs='+', default=ALL_MODELS,
        help=f'Models to benchmark (default: all). Options: {ALL_MODELS}'
    )
    parser.add_argument(
        '--output', default='results',
        help='Output directory for CSV results (default: results/)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be executed without running inference'
    )

    args = parser.parse_args()

    run_benchmark(
        image_dir=args.images,
        prompt_file=args.prompt,
        models=args.models,
        output_dir=args.output,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()