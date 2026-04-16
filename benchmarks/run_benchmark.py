"""
Benchmark Runner
Two modes:
  accuracy     — All (or sampled) images, no cooldown. Measures classification correctness.
  performance  — Small balanced sample, cooldown between images and between models.
                 Measures inference speed, tokens/sec, memory under fair thermal conditions.

Usage:
    uv run python run_benchmark.py accuracy --images ../dataset --prompt prompts/prompt.txt
    uv run python run_benchmark.py accuracy --images ../dataset --prompt prompts/prompt.txt --num-images 100
    uv run python run_benchmark.py performance --images ../dataset --prompt prompts/prompt.txt
    uv run python run_benchmark.py performance --images ../dataset --prompt prompts/prompt.txt --num-images 5 --cooldown 0 --model-cooldown 180

    Add --dry-run to either mode to preview what would be executed.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from src.data_loader import get_all_test_images, get_ground_truth, get_balanced_sample
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
    'qwen3-vl:4b',
    'qwen3-vl:8b',
    'gemma4:e2b',
    'gemma4:e4b',
]


def run_single_inference(model_name, image_path, system_prompt):
    """Run a single model on a single image and return the result dict."""
    ground_truth = get_ground_truth(image_path)

    result = call_llm(model_name, str(image_path), system_prompt)

    response_text = result['response_text']
    correct = evaluate_accuracy(response_text, ground_truth)

    response_upper = response_text.upper()
    if '[FIRE_DETECTED]' in response_upper and '[NO_FIRE_DETECTED]' not in response_upper:
        llm_classification = 'fire'
    elif '[NO_FIRE_DETECTED]' in response_upper:
        llm_classification = 'no_fire'
    else:
        llm_classification = 'ambiguous'

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


def warmup_model(model_name, system_prompt):
    """Send a throwaway request to load the model into memory."""
    log_progress(f"Warming up {model_name}...")
    try:
        import ollama
        ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}]
        )
        log_progress(f"  {model_name} loaded and ready.")
    except Exception as e:
        log_progress(f"  Warmup failed for {model_name}: {e}")


# ── Accuracy Mode ────────────────────────────────────────────────────

def run_accuracy(image_dir, prompt_file, models, output_dir, dry_run,
                 num_images=None, seed=42):
    """
    All images (or balanced sample), no cooldown. Measures classification correctness.
    """
    output_path = Path(output_dir) / 'accuracy'
    output_path.mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt(prompt_file)
    all_images = get_all_test_images(image_dir)

    if not all_images:
        log_progress(f"No images found in {image_dir}")
        return

    if num_images:
        images = get_balanced_sample(all_images, num_images, seed=seed)
        log_progress(
            f"[ACCURACY MODE] Sampled {len(images)} images from {len(all_images)} "
            f"(seed={seed}), {len(models)} models, no cooldown"
        )
    else:
        images = all_images
        log_progress(
            f"[ACCURACY MODE] {len(images)} images (all), {len(models)} models, no cooldown"
        )

    if dry_run:
        for model in models:
            log_progress(f"  {model}: {len(images)} images")
        total = len(models) * len(images)
        est_minutes = total * 25 / 60
        log_progress(f"  Total inferences: {total}")
        log_progress(f"  Estimated time: ~{est_minutes:.0f} minutes")
        return

    for model_name in models:
        csv_file = output_path / f"accuracy_{model_name.replace(':', '_')}.csv"
        if csv_file.exists() and csv_file.stat().st_size > 0:
            log_progress(f"Skipping {model_name} — {csv_file.name} already exists")
            continue
        init_csv(str(csv_file))

        log_progress(f"=== {model_name} — accuracy run ({len(images)} images) ===")
        warmup_model(model_name, system_prompt)

        correct_count = 0
        error_count = 0
        ambiguous_count = 0

        for i, image_path in enumerate(images):
            try:
                result = run_single_inference(model_name, image_path, system_prompt)
                log_result(str(csv_file), result)

                if result['correct'] is True:
                    correct_count += 1
                elif result['correct'] is None:
                    ambiguous_count += 1

                log_progress(
                    f"  [{i+1}/{len(images)}] {image_path.name} -> "
                    f"{result['llm_classification']} "
                    f"({'✓' if result['correct'] else '✗' if result['correct'] is False else '?'})"
                )

            except Exception as e:
                error_count += 1
                log_progress(f"  [{i+1}/{len(images)}] ERROR on {image_path.name}: {e}")

        total = len(images) - error_count
        accuracy = (correct_count / total * 100) if total > 0 else 0
        log_progress(
            f"=== {model_name} done === "
            f"Accuracy: {correct_count}/{total} ({accuracy:.1f}%) | "
            f"Ambiguous: {ambiguous_count} | Errors: {error_count}"
        )

    log_progress("Accuracy benchmark complete.")


# ── Performance Mode ─────────────────────────────────────────────────

def run_performance(image_dir, prompt_file, models, output_dir, dry_run,
                    num_images=3, cooldown=10, model_cooldown=90, seed=42):
    """
    Balanced random sample of images with cooldowns. All models run the same images.
    Each run gets a timestamped CSV so multiple runs accumulate.
    """
    output_path = Path(output_dir) / 'performance'
    output_path.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    system_prompt = load_system_prompt(prompt_file)
    all_images = get_all_test_images(image_dir)

    if not all_images:
        log_progress(f"No images found in {image_dir}")
        return

    images = get_balanced_sample(all_images, num_images, seed=seed)

    log_progress(
        f"[PERFORMANCE MODE] {len(images)} images from {len(all_images)} (seed={seed}), "
        f"{len(models)} models, {cooldown}s between images, {model_cooldown}s between models"
    )
    log_progress(f"  Run ID: {run_id}")
    log_progress(f"  Selected images: {[img.name for img in images]}")

    if dry_run:
        for model in models:
            log_progress(f"  {model}: {len(images)} images")
        total = len(models) * len(images)
        est_seconds = (total * (25 + cooldown)) + ((len(models) - 1) * model_cooldown)
        log_progress(f"  Total inferences: {total}")
        log_progress(f"  Estimated time: ~{est_seconds / 60:.0f} minutes")
        return

    for mi, model_name in enumerate(models):
        csv_file = output_path / f"performance_{model_name.replace(':', '_')}_{run_id}.csv"
        init_csv(str(csv_file))

        log_progress(f"=== {model_name} — performance run ({len(images)} images) ===")
        warmup_model(model_name, system_prompt)

        for i, image_path in enumerate(images):
            try:
                result = run_single_inference(model_name, image_path, system_prompt)
                log_result(str(csv_file), result)

                log_progress(
                    f"  [{i+1}/{len(images)}] {image_path.name} -> "
                    f"{result['tokens_per_sec']} tok/s | "
                    f"prompt: {result['prompt_eval_duration_s']}s | "
                    f"eval: {result['eval_duration_s']}s | "
                    f"total: {result['total_duration_s']}s"
                )

            except Exception as e:
                log_progress(f"  [{i+1}/{len(images)}] ERROR on {image_path.name}: {e}")

            if cooldown > 0 and i < len(images) - 1:
                log_progress(f"  Cooling down {cooldown}s...")
                time.sleep(cooldown)

        log_progress(f"=== {model_name} done ===")

        if model_cooldown > 0 and mi < len(models) - 1:
            log_progress(f"  Model cooldown {model_cooldown}s before next model...")
            time.sleep(model_cooldown)

    log_progress("Performance benchmark complete.")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Wildfire detection benchmark')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--images', required=True, help='Directory containing test images')
    shared.add_argument('--prompt', required=True, help='Path to system prompt text file')
    shared.add_argument('--models', nargs='+', default=ALL_MODELS, help='Models to benchmark')
    shared.add_argument('--output', default='results', help='Output directory (default: results/)')
    shared.add_argument('--dry-run', action='store_true', help='Preview without running')
    shared.add_argument('--seed', type=int, default=42, help='Random seed for sampling (default: 42)')

    # Accuracy
    acc = subparsers.add_parser('accuracy', parents=[shared],
                                help='All or sampled images, no cooldown.')
    acc.add_argument('--num-images', type=int, default=None,
                     help='Balanced random sample size (default: all images)')

    # Performance
    perf = subparsers.add_parser('performance', parents=[shared],
                                 help='Balanced sample with cooldowns. Measures inference speed.')
    perf.add_argument('--num-images', type=int, default=3,
                      help='Number of images per model (default: 3)')
    perf.add_argument('--cooldown', type=int, default=10,
                      help='Seconds between images (default: 10)')
    perf.add_argument('--model-cooldown', type=int, default=90,
                      help='Seconds between models (default: 90)')

    args = parser.parse_args()

    if args.mode == 'accuracy':
        run_accuracy(
            image_dir=args.images,
            prompt_file=args.prompt,
            models=args.models,
            output_dir=args.output,
            dry_run=args.dry_run,
            num_images=args.num_images,
            seed=args.seed,
        )
    elif args.mode == 'performance':
        run_performance(
            image_dir=args.images,
            prompt_file=args.prompt,
            models=args.models,
            output_dir=args.output,
            dry_run=args.dry_run,
            num_images=args.num_images,
            cooldown=args.cooldown,
            model_cooldown=args.model_cooldown,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()