"""
Benchmark Analysis — generates thesis-ready charts from benchmark CSVs.

Usage:
    uv run python analyze.py
    uv run python analyze.py --accuracy-dir results/accuracy --performance-dir results/performance
    uv run python analyze.py --output charts/

Reads all CSV files from accuracy and performance result directories,
produces PNG charts ready for Overleaf.
"""

import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ── Data Loading ─────────────────────────────────────────────────────

def load_csvs(directory):
    """Load all CSV files from a directory into a list of dicts."""
    rows = []
    path = Path(directory)
    if not path.exists():
        return rows
    for csv_file in sorted(path.glob('*.csv')):
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def group_by_model(rows):
    """Group rows by model_name."""
    groups = {}
    for row in rows:
        model = row['model_name']
        if model not in groups:
            groups[model] = []
        groups[model].append(row)
    return groups


# ── Chart Helpers ────────────────────────────────────────────────────

MODEL_COLORS = {
    'ministral-3:3b': '#2196F3',
    'ministral-3:8b': '#1565C0',
    'qwen3-vl:2b': '#4CAF50',
    'qwen3-vl:4b': '#2E7D32',
    'qwen3-vl:8b': '#1B5E20',
    'gemma4:e2b': '#FF9800',
    'gemma4:e4b': '#E65100',
}

def get_color(model):
    return MODEL_COLORS.get(model, '#888888')


def save_chart(fig, output_dir, filename):
    path = Path(output_dir) / filename
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {path}")


# ── Classification Metrics ───────────────────────────────────────────

def compute_metrics(rows):
    """
    Compute TP, FP, TN, FN, precision, recall, F1.
    'fire' is the positive class.
    Ambiguous responses count as incorrect (FN if ground truth is fire, FP if no_fire).
    """
    tp = fp = tn = fn = 0
    for r in rows:
        gt = r['ground_truth']
        pred = r['llm_classification']

        if gt == 'fire' and pred == 'fire':
            tp += 1
        elif gt == 'no_fire' and pred == 'fire':
            fp += 1
        elif gt == 'no_fire' and pred == 'no_fire':
            tn += 1
        elif gt == 'fire' and pred == 'no_fire':
            fn += 1
        elif gt == 'fire':  # ambiguous → treated as missed fire
            fn += 1
        else:  # gt == 'no_fire', pred == 'ambiguous'
            fp += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Matthews Correlation Coefficient — range [-1, 1], robust to class imbalance
    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
    }


# ── Accuracy Charts ──────────────────────────────────────────────────

def chart_accuracy(rows, output_dir):
    """Bar chart: accuracy % per model."""
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    accuracies = []
    ambiguous_rates = []
    for model in models:
        model_rows = groups[model]
        total = len(model_rows)
        correct = sum(1 for r in model_rows if r['correct'] == 'True')
        ambiguous = sum(1 for r in model_rows if r['correct'] == 'None')
        accuracies.append(correct / total * 100 if total > 0 else 0)
        ambiguous_rates.append(ambiguous / total * 100 if total > 0 else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, accuracies, 0.5, color=[get_color(m) for m in models])

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classification Accuracy by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.set_ylim(0, 105)

    for bar, acc, amb in zip(bars, accuracies, ambiguous_rates):
        label = f'{acc:.1f}%'
        if amb > 0:
            label += f'\n({amb:.0f}% ambig.)'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                label, ha='center', va='bottom', fontsize=9)

    save_chart(fig, output_dir, 'accuracy_by_model.png')


def chart_precision_recall_f1(rows, output_dir):
    """Grouped bar chart: precision, recall, F1 per model."""
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    metrics = {m: compute_metrics(groups[m]) for m in models}

    precisions = [metrics[m]['precision'] * 100 for m in models]
    recalls = [metrics[m]['recall'] * 100 for m in models]
    f1s = [metrics[m]['f1'] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, precisions, width, label='Precision', color='#2196F3')
    ax.bar(x, recalls, width, label='Recall', color='#4CAF50')
    ax.bar(x + width, f1s, width, label='F1', color='#FF9800')

    ax.set_ylabel('Score (%)')
    ax.set_title('Precision, Recall, and F1 by Model (positive class = fire)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right')

    # Value labels
    for i, m in enumerate(models):
        ax.text(i - width, precisions[i] + 1, f'{precisions[i]:.0f}',
                ha='center', va='bottom', fontsize=8)
        ax.text(i, recalls[i] + 1, f'{recalls[i]:.0f}',
                ha='center', va='bottom', fontsize=8)
        ax.text(i + width, f1s[i] + 1, f'{f1s[i]:.0f}',
                ha='center', va='bottom', fontsize=8)

    save_chart(fig, output_dir, 'precision_recall_f1.png')


def chart_mcc(rows, output_dir):
    """Bar chart: Matthews Correlation Coefficient per model."""
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    mccs = [compute_metrics(groups[m])['mcc'] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, mccs, 0.5, color=[get_color(m) for m in models])

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('MCC')
    ax.set_title('Matthews Correlation Coefficient by Model\n(range: -1 to +1, higher is better, 0 = random)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.set_ylim(-1.05, 1.05)

    for bar, mcc in zip(bars, mccs):
        y_offset = 0.03 if mcc >= 0 else -0.08
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
                f'{mcc:.2f}', ha='center', va='bottom' if mcc >= 0 else 'top', fontsize=9)

    save_chart(fig, output_dir, 'mcc.png')


def chart_confusion(rows, output_dir):
    """Per-model: TP, FP, TN, FN counts."""
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    metrics = {m: compute_metrics(groups[m]) for m in models}

    tps = [metrics[m]['tp'] for m in models]
    fps = [metrics[m]['fp'] for m in models]
    tns = [metrics[m]['tn'] for m in models]
    fns = [metrics[m]['fn'] for m in models]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(models))
    width = 0.2

    ax.bar(x - 1.5 * width, tps, width, label='TP (correct fire)', color='#4CAF50')
    ax.bar(x - 0.5 * width, tns, width, label='TN (correct no-fire)', color='#81C784')
    ax.bar(x + 0.5 * width, fps, width, label='FP (false alarm)', color='#FFC107')
    ax.bar(x + 1.5 * width, fns, width, label='FN (missed fire)', color='#F44336')

    ax.set_ylabel('Count')
    ax.set_title('Confusion Matrix Counts by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.legend(loc='upper right', fontsize=8)

    save_chart(fig, output_dir, 'confusion_matrix.png')


def chart_response_length(rows, output_dir):
    """Box plot: word count distribution per model."""
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    data = []
    for model in models:
        words = [int(r['word_count']) for r in groups[model]]
        data.append(words)

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=models, patch_artist=True)

    for patch, model in zip(bp['boxes'], models):
        patch.set_facecolor(get_color(model))
        patch.set_alpha(0.7)

    ax.set_ylabel('Word Count')
    ax.set_title('Response Length by Model')
    plt.xticks(rotation=25, ha='right')

    save_chart(fig, output_dir, 'response_length.png')


# ── Performance Charts ───────────────────────────────────────────────

def chart_tokens_per_sec(rows, output_dir):
    """Bar chart with error bars: median tokens/sec per model."""
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    medians = []
    stds = []
    for model in models:
        vals = [float(r['tokens_per_sec']) for r in groups[model]]
        medians.append(np.median(vals))
        stds.append(np.std(vals))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, medians, 0.5, yerr=stds, capsize=4,
                  color=[get_color(m) for m in models])

    ax.set_ylabel('Tokens/sec')
    ax.set_title('Generation Speed by Model (median ± std)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')

    for bar, med in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{med:.1f}', ha='center', va='bottom', fontsize=9)

    save_chart(fig, output_dir, 'tokens_per_sec.png')


def chart_inference_breakdown(rows, output_dir):
    """Stacked bar: prompt eval vs generation vs overhead per model."""
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    prompt_evals = []
    eval_times = []
    overheads = []

    for model in models:
        prompt = np.median([float(r['prompt_eval_duration_s']) for r in groups[model]])
        gen = np.median([float(r['eval_duration_s']) for r in groups[model]])
        total = np.median([float(r['total_duration_s']) for r in groups[model]])
        overhead = max(0, total - prompt - gen)

        prompt_evals.append(prompt)
        eval_times.append(gen)
        overheads.append(overhead)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.5

    ax.bar(x, prompt_evals, width, label='Image/Prompt Processing', color='#2196F3')
    ax.bar(x, eval_times, width, bottom=prompt_evals, label='Text Generation', color='#4CAF50')
    ax.bar(x, overheads, width,
           bottom=[p + e for p, e in zip(prompt_evals, eval_times)],
           label='Ollama Overhead', color='#BDBDBD')

    ax.set_ylabel('Time (seconds)')
    ax.set_title('Inference Time Breakdown by Model (median)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.legend()

    save_chart(fig, output_dir, 'inference_breakdown.png')


def chart_total_inference(rows, output_dir):
    """Bar chart: median total inference time per model."""
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    medians = []
    stds = []
    for model in models:
        vals = [float(r['total_duration_s']) for r in groups[model]]
        medians.append(np.median(vals))
        stds.append(np.std(vals))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, medians, 0.5, yerr=stds, capsize=4,
                  color=[get_color(m) for m in models])

    ax.set_ylabel('Time (seconds)')
    ax.set_title('Total Inference Time by Model (median ± std)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')

    for bar, med in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{med:.1f}s', ha='center', va='bottom', fontsize=9)

    save_chart(fig, output_dir, 'total_inference_time.png')


def chart_memory_usage(rows, output_dir):
    """Bar chart: median runtime memory vs model disk size."""
    groups = group_by_model(rows)
    models = sorted(groups.keys())

    medians = []
    model_sizes = []
    for model in models:
        vals = [float(r['memory_usage_gb']) for r in groups[model] if r['memory_usage_gb']]
        medians.append(np.median(vals) if vals else 0)
        size = groups[model][0].get('model_size_gb', '')
        model_sizes.append(float(size) if size else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, model_sizes, width, label='Model Size (disk)', color='#BDBDBD')
    ax.bar(x + width/2, medians, width, label='RSS Memory (runtime)',
           color=[get_color(m) for m in models])

    ax.axhline(y=16, color='red', linestyle='--', alpha=0.5, label='16GB RAM limit')
    ax.set_ylabel('GB')
    ax.set_title('Model Size vs Runtime Memory')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.legend()

    save_chart(fig, output_dir, 'memory_usage.png')


# ── Static Charts ────────────────────────────────────────────────────

MODEL_SIZES = {
    'ministral-3:3b': 3.0,
    'ministral-3:8b': 6.0,
    'qwen3-vl:2b': 1.9,
    'qwen3-vl:4b': 3.3,
    'qwen3-vl:8b': 6.1,
    'gemma4:e2b': 7.2,
    'gemma4:e4b': 9.6,
}


def chart_model_sizes(output_dir, models=None):
    """Bar chart: model file size on disk."""
    if models is None:
        models = sorted(MODEL_SIZES.keys())
    else:
        models = sorted(m for m in models if m in MODEL_SIZES)

    sizes = [MODEL_SIZES[m] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, sizes, 0.5, color=[get_color(m) for m in models])

    ax.set_ylabel('Size (GB)')
    ax.set_title('Model Size on Disk')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')

    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f'{size}GB', ha='center', va='bottom', fontsize=9)

    save_chart(fig, output_dir, 'model_sizes.png')


# ── Summary Table ────────────────────────────────────────────────────

def print_summary(accuracy_rows, performance_rows):
    """Print a summary table to stdout."""
    acc_groups = group_by_model(accuracy_rows) if accuracy_rows else {}
    perf_groups = group_by_model(performance_rows) if performance_rows else {}

    all_models = sorted(set(list(acc_groups.keys()) + list(perf_groups.keys())))

    print(f"\n{'Model':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'MCC':>6} {'Tok/s':>8} {'Total(s)':>10} {'Mem(GB)':>8}")
    print('─' * 89)

    for model in all_models:
        if model in acc_groups:
            m = compute_metrics(acc_groups[model])
            acc_str = f"{m['accuracy']*100:.0f}%"
            prec_str = f"{m['precision']*100:.0f}%"
            rec_str = f"{m['recall']*100:.0f}%"
            f1_str = f"{m['f1']*100:.0f}%"
            mcc_str = f"{m['mcc']:.2f}"
        else:
            acc_str = prec_str = rec_str = f1_str = mcc_str = "—"

        if model in perf_groups:
            tok = np.median([float(r['tokens_per_sec']) for r in perf_groups[model]])
            total_t = np.median([float(r['total_duration_s']) for r in perf_groups[model]])
            mem_vals = [float(r['memory_usage_gb']) for r in perf_groups[model] if r['memory_usage_gb']]
            mem = np.median(mem_vals) if mem_vals else 0
            tok_str = f"{tok:.1f}"
            total_str = f"{total_t:.1f}"
            mem_str = f"{mem:.1f}"
        else:
            tok_str = total_str = mem_str = "—"

        print(f"{model:<20} {acc_str:>6} {prec_str:>6} {rec_str:>6} {f1_str:>6} {mcc_str:>6} {tok_str:>8} {total_str:>10} {mem_str:>8}")

    print()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--accuracy-dir', default='results/accuracy', help='Accuracy CSV directory')
    parser.add_argument('--performance-dir', default='results/performance', help='Performance CSV directory')
    parser.add_argument('--output', default='charts', help='Output directory for PNGs')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_rows = load_csvs(args.accuracy_dir)
    performance_rows = load_csvs(args.performance_dir)

    print(f"Loaded {len(accuracy_rows)} accuracy rows, {len(performance_rows)} performance rows")

    # Always generate model size chart
    print("\nGenerating model size chart...")
    all_models = set()
    for r in accuracy_rows + performance_rows:
        all_models.add(r['model_name'])
    chart_model_sizes(output_dir, list(all_models) if all_models else None)

    if accuracy_rows:
        print("\nGenerating accuracy charts...")
        chart_accuracy(accuracy_rows, output_dir)
        chart_precision_recall_f1(accuracy_rows, output_dir)
        chart_mcc(accuracy_rows, output_dir)
        chart_confusion(accuracy_rows, output_dir)
        chart_response_length(accuracy_rows, output_dir)

    if performance_rows:
        print("\nGenerating performance charts...")
        chart_tokens_per_sec(performance_rows, output_dir)
        chart_inference_breakdown(performance_rows, output_dir)
        chart_total_inference(performance_rows, output_dir)
        chart_memory_usage(performance_rows, output_dir)

    if accuracy_rows or performance_rows:
        print_summary(accuracy_rows, performance_rows)

    print(f"Done. Charts saved to {output_dir}/")


if __name__ == '__main__':
    main()