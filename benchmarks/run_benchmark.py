"""
Benchmark Runner
Two modes:
  accuracy     — All (or sampled) images, no cooldown. Measures classification correctness.
  performance  — Small balanced sample, cooldown between images and between models.
                 Measures inference speed, tokens/sec, memory under fair thermal conditions.

The benchmark invokes the LLM with Ollama's structured-output feature,
enforcing JSON output matching the pipeline's schema. The system prompt
is loaded from a modular prompt file so that different prompt versions
can be evaluated without changes to the code.

Optional YOLO preprocessing adds bounding box annotations to images before
they reach the LLM. YOLO timing is tracked separately from LLM timing.
Runs with and without YOLO produce separate CSV files, allowing the two
conditions to be compared during analysis.

Usage:
    uv run python run_benchmark.py accuracy --images ../dataset
    uv run python run_benchmark.py accuracy --images ../dataset --with-yolo
    uv run python run_benchmark.py performance --images ../dataset --with-yolo

    Override the prompt explicitly:
    uv run python run_benchmark.py accuracy --images ../dataset \\
        --system-prompt ../pipeline/llm/prompts/c1v2prompt.txt

    Add --dry-run to either mode to preview what would be executed.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow importing the object_detection module from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import get_all_test_images, get_ground_truth, get_balanced_sample
from src.llm_inference import load_prompt_file, call_llm
from src.logger import init_csv, log_result, log_progress, RESULT_COLUMNS
from src.metrics import (
    track_memory_usage,
    get_model_size,
    count_words,
    evaluate_accuracy,
)
from src.hardware import get_hardware_specs


# Thesis-core models. Used by default when --models is not specified.
DEFAULT_MODELS = [
    'ministral-3:3b',
    'qwen3-vl:4b',
    'gemma4:e2b',
]

# Full set of models available for explicit override via --models.
ALL_MODELS = [
    'ministral-3:3b',
    'ministral-3:8b',
    'qwen3-vl:4b',
    'qwen3-vl:8b',
    'gemma4:e2b',
    'gemma4:e4b',
]

# Default system prompt location (Cycle 1 v1)
DEFAULT_SYSTEM_PROMPT = REPO_ROOT / 'pipeline' / 'llm' / 'prompts' / 'c1v1prompt.txt'


def _yolo_tag(with_yolo):
    """Short filename suffix indicating whether YOLO preprocessing was used."""
    return '_yolo' if with_yolo else '_noyolo'


# ── YOLO Integration ─────────────────────────────────────────────────

def _run_yolo(image_path, yolo_model_name):
    """
    Run YOLO inference on an image. Returns a dict with the annotated image
    path, detection count, and wall-clock inference duration.

    Importing YOLO is deferred to this function so benchmarks without
    --with-yolo don't require ultralytics to be installed.
    """
    from pipeline.object_detection.model_handler import get_model
    from pipeline.object_detection.config import (
        ANNOTATED_OUTPUT_DIR,
        ANNOTATED_OUTPUT_FILENAME,
    )
    import cv2

    model = get_model(yolo_model_name)

    start = time.perf_counter()
    results = model(str(image_path))
    duration_s = time.perf_counter() - start

    result = results[0]
    detection_count = len(result.boxes)

    ANNOTATED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ANNOTATED_OUTPUT_DIR / ANNOTATED_OUTPUT_FILENAME
    annotated = result.plot()
    cv2.imwrite(str(output_path), annotated)

    return {
        'annotated_path': output_path,
        'detection_count': detection_count,
        'duration_s': round(duration_s, 3),
    }


# ── Inference ────────────────────────────────────────────────────────

def run_single_inference(model_name, image_path, system_prompt,
                         with_yolo=False, yolo_model='best'):
    """Run one model on one image. Optionally runs YOLO preprocessing first."""
    ground_truth = get_ground_truth(image_path)

    # Optional YOLO preprocessing
    yolo_duration_s = None
    yolo_detection_count = None
    llm_input_path = image_path

    if with_yolo:
        yolo_result = _run_yolo(image_path, yolo_model)
        yolo_duration_s = yolo_result['duration_s']
        yolo_detection_count = yolo_result['detection_count']
        llm_input_path = yolo_result['annotated_path']

    # LLM inference (structured output enforced via schema)
    result = call_llm(model_name, str(llm_input_path), system_prompt)

    classification_raw = result['classification']
    correct = evaluate_accuracy(classification_raw, ground_truth)

    # Map schema classification to the benchmark's existing label space.
    if classification_raw == 'fire_detected':
        llm_classification = 'fire'
    elif classification_raw == 'no_fire_detected':
        llm_classification = 'no_fire'
    else:
        llm_classification = 'ambiguous'  # 'uncertain' or unknown

    response_text = result['response_text']
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
        'yolo_enabled': with_yolo,
        'yolo_model': yolo_model if with_yolo else None,
        'yolo_duration_s': yolo_duration_s,
        'yolo_detection_count': yolo_detection_count,
    }


def warmup_model(model_name):
    """Send a throwaway request to load the model into memory."""
    log_progress(f"Warming up {model_name}...")
    try:
        import ollama
        ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
        )
        log_progress(f"  {model_name} loaded and ready.")
    except Exception as e:
        log_progress(f"  Warmup failed for {model_name}: {e}")


def write_hardware_json(csv_path, run_id, mode, config):
    """Write a companion JSON file next to a performance CSV."""
    json_path = csv_path.with_suffix('.json')
    payload = {
        'run_id': run_id,
        'mode': mode,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': config,
        'hardware': get_hardware_specs(),
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


# ── Accuracy Mode ────────────────────────────────────────────────────

def run_accuracy(image_dir, system_prompt_file, models, output_dir, dry_run,
                 num_images=None, seed=42, with_yolo=False, yolo_model='best'):
    """All images (or balanced sample), no cooldown. Measures classification correctness."""
    output_path = Path(output_dir) / 'accuracy'
    output_path.mkdir(parents=True, exist_ok=True)

    system_prompt = load_prompt_file(system_prompt_file)
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

    log_progress(f"  System prompt: {system_prompt_file}")
    if with_yolo:
        log_progress(f"  YOLO preprocessing enabled (model={yolo_model})")

    if dry_run:
        for model in models:
            log_progress(f"  {model}: {len(images)} images")
        total = len(models) * len(images)
        est_minutes = total * 25 / 60
        log_progress(f"  Total inferences: {total}")
        log_progress(f"  Estimated time: ~{est_minutes:.0f} minutes")
        return

    yolo_tag = _yolo_tag(with_yolo)

    for model_name in models:
        csv_file = output_path / f"accuracy_{model_name.replace(':', '_')}{yolo_tag}.csv"
        if csv_file.exists() and csv_file.stat().st_size > 0:
            log_progress(f"Skipping {model_name} — {csv_file.name} already exists")
            continue
        init_csv(str(csv_file))

        log_progress(f"=== {model_name} — accuracy run ({len(images)} images) ===")
        warmup_model(model_name)

        correct_count = 0
        error_count = 0
        ambiguous_count = 0

        for i, image_path in enumerate(images):
            try:
                result = run_single_inference(
                    model_name, image_path, system_prompt,
                    with_yolo=with_yolo, yolo_model=yolo_model,
                )
                log_result(str(csv_file), result)

                if result['correct'] is True:
                    correct_count += 1
                elif result['correct'] is None:
                    ambiguous_count += 1

                yolo_suffix = ''
                if with_yolo:
                    yolo_suffix = f" | yolo: {result['yolo_detection_count']} box(es), {result['yolo_duration_s']}s"

                log_progress(
                    f"  [{i+1}/{len(images)}] {image_path.name} -> "
                    f"{result['llm_classification']} "
                    f"({'✓' if result['correct'] else '✗' if result['correct'] is False else '?'})"
                    f"{yolo_suffix}"
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

def run_performance(image_dir, system_prompt_file, models, output_dir, dry_run,
                    num_images=3, cooldown=10, model_cooldown=90, seed=42,
                    with_yolo=False, yolo_model='best'):
    """Balanced random sample with cooldowns. Each run gets a timestamped CSV+JSON."""
    output_path = Path(output_dir) / 'performance'
    output_path.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    system_prompt = load_prompt_file(system_prompt_file)
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
    log_progress(f"  System prompt: {system_prompt_file}")
    log_progress(f"  Selected images: {[img.name for img in images]}")

    if with_yolo:
        log_progress(f"  YOLO preprocessing enabled (model={yolo_model})")

    if dry_run:
        for model in models:
            log_progress(f"  {model}: {len(images)} images")
        total = len(models) * len(images)
        est_seconds = (total * (25 + cooldown)) + ((len(models) - 1) * model_cooldown)
        log_progress(f"  Total inferences: {total}")
        log_progress(f"  Estimated time: ~{est_seconds / 60:.0f} minutes")
        return

    # Config snapshot saved alongside each CSV for reproducibility
    config_snapshot = {
        'models': models,
        'num_images': num_images,
        'cooldown': cooldown,
        'model_cooldown': model_cooldown,
        'seed': seed,
        'with_yolo': with_yolo,
        'yolo_model': yolo_model if with_yolo else None,
        'system_prompt_file': str(system_prompt_file),
        'selected_images': [img.name for img in images],
    }

    yolo_tag = _yolo_tag(with_yolo)

    for mi, model_name in enumerate(models):
        csv_file = output_path / f"performance_{model_name.replace(':', '_')}{yolo_tag}_{run_id}.csv"
        init_csv(str(csv_file))
        write_hardware_json(csv_file, run_id, 'performance', config_snapshot)

        log_progress(f"=== {model_name} — performance run ({len(images)} images) ===")
        warmup_model(model_name)

        for i, image_path in enumerate(images):
            try:
                result = run_single_inference(
                    model_name, image_path, system_prompt,
                    with_yolo=with_yolo, yolo_model=yolo_model,
                )
                log_result(str(csv_file), result)

                yolo_suffix = ''
                if with_yolo:
                    yolo_suffix = f" | yolo: {result['yolo_duration_s']}s ({result['yolo_detection_count']} box)"

                log_progress(
                    f"  [{i+1}/{len(images)}] {image_path.name} -> "
                    f"{result['tokens_per_sec']} tok/s | "
                    f"prompt: {result['prompt_eval_duration_s']}s | "
                    f"eval: {result['eval_duration_s']}s | "
                    f"total: {result['total_duration_s']}s"
                    f"{yolo_suffix}"
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
    shared.add_argument(
        '--system-prompt', default=str(DEFAULT_SYSTEM_PROMPT),
        help=f'Path to system prompt file (default: {DEFAULT_SYSTEM_PROMPT})',
    )
    shared.add_argument(
        '--models', nargs='+', default=DEFAULT_MODELS,
        help=f'Models to benchmark. Default: {DEFAULT_MODELS}. '
             f'All available: {ALL_MODELS}',
    )
    shared.add_argument('--output', default='results', help='Output directory (default: results/)')
    shared.add_argument('--dry-run', action='store_true', help='Preview without running')
    shared.add_argument('--seed', type=int, default=42, help='Random seed for sampling (default: 42)')
    shared.add_argument('--with-yolo', action='store_true',
                        help='Run YOLO preprocessing before LLM inference (default: off)')
    shared.add_argument('--yolo-model', default='best',
                        help='YOLO model weights to use when --with-yolo is set (default: best)')

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
            system_prompt_file=args.system_prompt,
            models=args.models,
            output_dir=args.output,
            dry_run=args.dry_run,
            num_images=args.num_images,
            seed=args.seed,
            with_yolo=args.with_yolo,
            yolo_model=args.yolo_model,
        )
    elif args.mode == 'performance':
        run_performance(
            image_dir=args.images,
            system_prompt_file=args.system_prompt,
            models=args.models,
            output_dir=args.output,
            dry_run=args.dry_run,
            num_images=args.num_images,
            cooldown=args.cooldown,
            model_cooldown=args.model_cooldown,
            seed=args.seed,
            with_yolo=args.with_yolo,
            yolo_model=args.yolo_model,
        )


if __name__ == '__main__':
    main()