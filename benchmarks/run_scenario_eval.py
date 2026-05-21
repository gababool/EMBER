"""
Scenario Evaluation Runner

Evaluates all thesis-core models against a set of hand-picked scenario
image+context pairings. Each scenario consists of a pre-computed GIS context
JSON and a corresponding image stored together in a scenario folder.

Unlike the main accuracy benchmark — which runs many images against one
scenario to measure statistical accuracy — this script runs one image per
scenario against all models to evaluate qualitative reasoning quality. It
is intended for use during stakeholder interviews and qualitative analysis.

Results are written to a single combined CSV in results/scenarios/ with one
row per model×scenario combination.

Folder structure expected under --context-dir:

    data/contexts/
        scenario_01/
            scenario_01.json
            scenario_01.jpg   (or .png)
        scenario_02/
            scenario_02.json
            scenario_02.png
        ...

Usage:
    # Run all scenarios, all default models, no YOLO
    uv run python run_scenario_eval.py

    # Run all scenarios with YOLO preprocessing
    uv run python run_scenario_eval.py --with-yolo

    # Run specific scenarios only
    uv run python run_scenario_eval.py --scenarios scenario_01 scenario_03

    # Override prompt or models
    uv run python run_scenario_eval.py \\
        --system-prompt ../pipeline/llm/prompts/latest.txt \\
        --models ministral-3:3b qwen3-vl:4b

    # Preview without running
    uv run python run_scenario_eval.py --dry-run
"""

import argparse
import concurrent.futures
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.llm_inference import load_prompt_file, call_llm
from src.logger import log_progress
from src.metrics import track_memory_usage, get_model_size, count_words
from src.hardware import get_hardware_specs
from src.yolo import run_yolo_for_llm, yolo_mode_tag
from pipeline.object_detection.detections import YOLO_INPUT_MODES, resolve_yolo_input_mode


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    'ministral-3:3b',
    'qwen3-vl:4b',
    'gemma4:e2b',
]

DEFAULT_PROMPT_FILE_NAME = 'latest.txt'
DEFAULT_SYSTEM_PROMPT = REPO_ROOT / 'pipeline' / 'llm' / 'prompts' / DEFAULT_PROMPT_FILE_NAME
DEFAULT_CONTEXT_DIR = REPO_ROOT / 'data' / 'contexts'
DEFAULT_OUTPUT_DIR = Path('results') / 'scenarios'

LLM_INFERENCE_TIMEOUT = 240

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# Combined CSV columns for the scenario evaluation output.
SCENARIO_RESULT_COLUMNS = [
    'timestamp',
    'scenario_id',
    'model_name',
    'image_path',
    'context_file',
    'yolo_enabled',
    'yolo_model',
    'yolo_input_mode',
    'yolo_detection_count',
    'yolo_duration_s',
    'classification',
    'reasoning',
    'recommendation',
    'situation_brief',
    'key_constraints',
    'word_count',
    'tokens_per_sec',
    'prompt_eval_duration_s',
    'prompt_eval_count',
    'eval_duration_s',
    'total_duration_s',
    'load_duration_s',
    'model_size_gb',
    'memory_usage_gb',
]


# ── Scenario discovery ────────────────────────────────────────────────────────

def discover_scenarios(context_dir: Path, names: list[str] | None = None) -> list[dict]:
    """Discover valid scenario folders under context_dir.

    A valid scenario folder contains exactly one JSON file and at least one
    image file with a supported extension. The JSON and image must share the
    same stem as the folder name (e.g. scenario_01/scenario_01.json +
    scenario_01.jpg).

    Args:
        context_dir: Root directory to search for scenario folders.
        names:       Optional list of folder names to restrict to. If None,
                     all valid scenario folders are returned.

    Returns:
        List of dicts with keys: scenario_id, json_path, image_path.
    """
    scenarios = []

    if not context_dir.exists():
        log_progress(f"Context directory not found: {context_dir}")
        return scenarios

    candidates = sorted(
        [d for d in context_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    for folder in candidates:
        if names and folder.name not in names:
            continue

        # Look for JSON matching folder name.
        json_path = folder / f"{folder.name}.json"
        if not json_path.exists():
            log_progress(f"  Skipping {folder.name}: no matching JSON ({json_path.name})")
            continue

        # Look for image matching folder name.
        image_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = folder / f"{folder.name}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            log_progress(
                f"  Skipping {folder.name}: no image found "
                f"(expected {folder.name}.<jpg|png|...>)"
            )
            continue

        scenarios.append({
            'scenario_id': folder.name,
            'json_path': json_path,
            'image_path': image_path,
        })

    return scenarios


# ── CSV helpers ───────────────────────────────────────────────────────────────

def init_scenario_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SCENARIO_RESULT_COLUMNS)
        writer.writeheader()


def append_scenario_result(csv_path: Path, row: dict) -> None:
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SCENARIO_RESULT_COLUMNS, extrasaction='ignore')
        writer.writerow(row)


# ── Model warmup ──────────────────────────────────────────────────────────────

def warmup_model(model_name: str) -> None:
    log_progress(f"  Warming up {model_name}...")
    try:
        import ollama
        ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
        )
        log_progress(f"  {model_name} ready.")
    except Exception as e:
        log_progress(f"  Warmup failed for {model_name}: {e}")


# ── Core evaluation ───────────────────────────────────────────────────────────

def run_scenario_eval(
    context_dir: Path,
    system_prompt_file: Path,
    models: list[str],
    output_dir: Path,
    with_yolo: bool,
    yolo_model: str,
    yolo_input_mode: str,
    scenario_names: list[str] | None,
    dry_run: bool,
) -> None:
    """Run all models over all discovered scenarios and write a combined CSV."""

    scenarios = discover_scenarios(context_dir, names=scenario_names)

    if not scenarios:
        log_progress("No valid scenarios found. Check that each scenario folder contains "
                     "a matching JSON and image file.")
        return

    system_prompt = load_prompt_file(str(system_prompt_file))

    log_progress(f"[SCENARIO EVAL] {len(scenarios)} scenario(s), {len(models)} model(s)")
    log_progress(f"  System prompt: {system_prompt_file.name}")
    resolved_mode = resolve_yolo_input_mode(yolo_input_mode)
    yolo_status = "disabled"
    if with_yolo and resolved_mode != "disabled":
        yolo_status = f"enabled (model={yolo_model}, mode={resolved_mode})"
    elif with_yolo:
        yolo_status = "disabled (mode=disabled)"
    log_progress(f"  YOLO: {yolo_status}")
    log_progress(f"  Scenarios:")
    for s in scenarios:
        log_progress(f"    {s['scenario_id']}: {s['image_path'].name}")

    if dry_run:
        total = len(scenarios) * len(models)
        log_progress(f"  Total inferences: {total}")
        log_progress(f"  Estimated time: ~{total * 25 / 60:.0f} minutes")
        return

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    yolo_tag = yolo_mode_tag(with_yolo, yolo_input_mode)
    csv_path = output_dir / f"scenario_eval{yolo_tag}_{run_id}.csv"
    init_scenario_csv(csv_path)
    log_progress(f"  Output: {csv_path}")

    # Write a companion JSON with run metadata.
    meta_path = csv_path.with_suffix('.json')
    meta = {
        'run_id': run_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': models,
        'with_yolo': with_yolo,
        'yolo_model': yolo_model if with_yolo else None,
        'yolo_input_mode': yolo_input_mode if with_yolo else None,
        'system_prompt_file': str(system_prompt_file),
        'scenarios': [
            {'id': s['scenario_id'], 'image': s['image_path'].name}
            for s in scenarios
        ],
        'hardware': get_hardware_specs(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    for model_name in models:
        log_progress(f"\n=== {model_name} ===")
        warmup_model(model_name)

        for scenario in scenarios:
            scenario_id = scenario['scenario_id']
            image_path = scenario['image_path']
            json_path = scenario['json_path']

            yolo_duration_s = None
            yolo_detection_count = None
            llm_input_path = image_path
            yolo_context = None
            resolved_yolo_input_mode = resolve_yolo_input_mode(yolo_input_mode)
            yolo_enabled = with_yolo and resolved_yolo_input_mode != "disabled"

            try:
                # Optional YOLO preprocessing.
                if yolo_enabled:
                    yolo_result = run_yolo_for_llm(image_path, yolo_model, yolo_input_mode)
                    yolo_duration_s = yolo_result['duration_s']
                    yolo_detection_count = yolo_result['detection_count']
                    llm_input_path = yolo_result['llm_input_path']
                    yolo_context = yolo_result['additional_context']
                    resolved_yolo_input_mode = yolo_result['yolo_input_mode']

                # LLM inference with context and timeout.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        call_llm,
                        model_name,
                        str(llm_input_path),
                        system_prompt,
                        context_file=str(json_path),
                        additional_context=yolo_context,
                    )
                    try:
                        result = future.result(timeout=LLM_INFERENCE_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        future.cancel()
                        raise TimeoutError(
                            f"LLM inference timed out after {LLM_INFERENCE_TIMEOUT}s "
                            f"(model={model_name}, scenario={scenario_id})"
                        )

                parsed = result['parsed']
                memory_gb = track_memory_usage()

                row = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'scenario_id': scenario_id,
                    'model_name': model_name,
                    'image_path': str(image_path),
                    'context_file': str(json_path),
                    'yolo_enabled': yolo_enabled,
                    'yolo_model': yolo_model if yolo_enabled else None,
                    'yolo_input_mode': resolved_yolo_input_mode if with_yolo else None,
                    'yolo_detection_count': yolo_detection_count,
                    'yolo_duration_s': yolo_duration_s,
                    'classification': parsed.get('classification'),
                    'reasoning': parsed.get('reasoning'),
                    'recommendation': parsed.get('recommendation'),
                    'situation_brief': parsed.get('situation_brief'),
                    'key_constraints': json.dumps(parsed.get('key_constraints')) if parsed.get('key_constraints') else None,
                    'word_count': count_words(result['response_text']),
                    'tokens_per_sec': round(result['tokens_per_sec'], 2),
                    'prompt_eval_duration_s': round(result['prompt_eval_duration_ns'] / 1e9, 3),
                    'eval_duration_s': round(result['eval_duration_ns'] / 1e9, 3),
                    'total_duration_s': round(result['total_duration_ns'] / 1e9, 3),
                    'load_duration_s': round(result['load_duration_ns'] / 1e9, 3),
                    'model_size_gb': get_model_size(model_name),
                    'memory_usage_gb': memory_gb,
                }

                append_scenario_result(csv_path, row)

                log_progress(
                    f"  {scenario_id} -> {parsed.get('classification')} | "
                    f"{result['tokens_per_sec']:.1f} tok/s"
                )

            except Exception as e:
                log_progress(f"  {scenario_id} -> ERROR: {e}")
                append_scenario_result(csv_path, {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'scenario_id': scenario_id,
                    'model_name': model_name,
                    'image_path': str(image_path),
                    'context_file': str(json_path),
                    'yolo_enabled': yolo_enabled,
                    'yolo_model': yolo_model if yolo_enabled else '',
                    'yolo_input_mode': resolved_yolo_input_mode if with_yolo else '',
                    'yolo_detection_count': '',
                    'yolo_duration_s': '',
                    'classification': '',
                    'reasoning': str(e)[:200],
                    'recommendation': '',
                    'situation_brief': '',
                    'key_constraints': '',
                    'word_count': 0,
                    'tokens_per_sec': 0,
                    'prompt_eval_duration_s': 0,
                    'prompt_eval_count': 0,
                    'eval_duration_s': 0,
                    'total_duration_s': 0,
                    'load_duration_s': 0,
                    'model_size_gb': get_model_size(model_name),
                    'memory_usage_gb': '',
                })

    log_progress(f"\nScenario evaluation complete. Results: {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run all models over hand-picked scenario image+context pairings.'
    )
    parser.add_argument(
        '--context-dir', type=Path, default=DEFAULT_CONTEXT_DIR,
        help=f'Root directory containing scenario folders (default: {DEFAULT_CONTEXT_DIR})',
    )
    parser.add_argument(
        '--system-prompt', type=Path, default=DEFAULT_SYSTEM_PROMPT,
        help=f'Path to system prompt file (default: {DEFAULT_PROMPT_FILE_NAME})',
    )
    parser.add_argument(
        '--models', nargs='+', default=DEFAULT_MODELS,
        help=f'Models to evaluate (default: {DEFAULT_MODELS})',
    )
    parser.add_argument(
        '--output', type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})',
    )
    parser.add_argument(
        '--with-yolo', action='store_true',
        help='Run YOLO preprocessing before LLM inference',
    )
    parser.add_argument(
        '--yolo-model', default='best',
        help='YOLO model weights to use when --with-yolo is set (default: best)',
    )
    parser.add_argument(
        '--yolo-input-mode',
        choices=YOLO_INPUT_MODES,
        default='annotated_image',
        help=(
            'How YOLO output reaches the LLM when --with-yolo is set '
            '(default: annotated_image)'
        ),
    )
    parser.add_argument(
        '--scenarios', nargs='+', default=None,
        help='Specific scenario folder names to run (default: all discovered)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview what would be run without executing',
    )

    args = parser.parse_args()

    run_scenario_eval(
        context_dir=args.context_dir,
        system_prompt_file=args.system_prompt,
        models=args.models,
        output_dir=args.output,
        with_yolo=args.with_yolo,
        yolo_model=args.yolo_model,
        yolo_input_mode=args.yolo_input_mode,
        scenario_names=args.scenarios,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
