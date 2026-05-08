"""
Benchmark Analysis — generates thesis-ready charts from benchmark CSVs.

Usage:
    uv run python analyze.py
    uv run python analyze.py --compare-yolo
    uv run python analyze.py --accuracy-dir results/accuracy --performance-dir results/performance
    uv run python analyze.py --output charts/

Reads CSV files from accuracy and performance result directories,
produces PNG charts ready for Overleaf.

Modes
-----
Default            : Load only no-YOLO CSVs (filename contains '_noyolo').
                     Charts show one bar per model, the standard view.
--compare-yolo     : Load both no-YOLO and YOLO CSVs. Charts add a second
                     bar per model showing the YOLO-preprocessing variant
                     alongside the no-YOLO baseline. Requires both CSV
                     types to be present; errors out otherwise.

Classification
--------------
Binary classification only. Every prediction is either correct or incorrect.
Ambiguous/uncertain output is not part of the schema and is not handled here.
"""

import argparse
import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ── Data Loading ─────────────────────────────────────────────────────

def load_csvs(directory, compare_yolo=False):
    """
    Load CSV files from a directory.

    Returns:
        (rows_noyolo, rows_yolo)

    Default mode (compare_yolo=False): only loads files whose names contain
    '_noyolo'. The yolo list is empty.

    Compare mode (compare_yolo=True): loads both '_noyolo' and '_yolo' files
    into separate lists.
    """
    rows_noyolo = []
    rows_yolo = []

    path = Path(directory)
    if not path.exists():
        return rows_noyolo, rows_yolo

    for csv_file in sorted(path.glob('*.csv')):
        name = csv_file.name
        is_yolo = '_yolo' in name and '_noyolo' not in name
        is_noyolo = '_noyolo' in name

        # Skip files that don't match either tag (legacy or unrelated).
        if not (is_yolo or is_noyolo):
            continue

        # In default mode, skip YOLO files entirely.
        if not compare_yolo and is_yolo:
            continue

        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip rows where inference failed (empty classification).
                if not row.get('llm_classification', '').strip():
                    continue
                if is_yolo:
                    rows_yolo.append(row)
                else:
                    rows_noyolo.append(row)

    return rows_noyolo, rows_yolo


def count_errors(directory):
    """Count rows with missing classification (inference failures).

    Returns (total_rows, error_rows) across all CSV files in the directory.
    Used to report how many inference failures were skipped during analysis.
    """
    total = 0
    errors = 0
    path = Path(directory)
    if not path.exists():
        return 0, 0
    for csv_file in sorted(path.glob('*.csv')):
        name = csv_file.name
        if '_noyolo' not in name and '_yolo' not in name:
            continue
        with open(csv_file, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                total += 1
                if not row.get('llm_classification', '').strip():
                    errors += 1
    return total, errors


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

CORRECT_COLOR = '#4CAF50'
INCORRECT_COLOR = '#F44336'


def get_color(model):
    return MODEL_COLORS.get(model, '#888888')


def save_chart(fig, output_dir, filename):
    path = Path(output_dir) / filename
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {path}")


def hatch_pattern_for_yolo():
    """Hatch pattern used for the YOLO variant in comparison charts."""
    return '///'


# ── Classification Metrics ───────────────────────────────────────────

def compute_metrics(rows):
    """
    Compute confusion counts and binary classification metrics.

    'fire' is the positive class.
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

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Matthews Correlation Coefficient — range [-1, 1], robust to class imbalance.
    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'total': total,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
    }


# ── Accuracy Charts ──────────────────────────────────────────────────

def chart_accuracy(rows_noyolo, rows_yolo, output_dir, compare_yolo):
    """Bar chart: accuracy per model.

    In default mode, one bar per model.
    In compare-yolo mode, two bars per model (no-YOLO and YOLO).
    """
    groups_noyolo = group_by_model(rows_noyolo)
    models = sorted(groups_noyolo.keys())

    fig, ax = plt.subplots(figsize=(11, 5) if compare_yolo else (10, 5))
    x = np.arange(len(models))

    if compare_yolo:
        groups_yolo = group_by_model(rows_yolo)
        width = 0.35
        accs_noyolo = [compute_metrics(groups_noyolo.get(m, []))['accuracy'] * 100 for m in models]
        accs_yolo = [compute_metrics(groups_yolo.get(m, []))['accuracy'] * 100 for m in models]

        ax.bar(x - width / 2, accs_noyolo, width, label='no YOLO',
               color=[get_color(m) for m in models],
               edgecolor='black', linewidth=0.5)
        ax.bar(x + width / 2, accs_yolo, width, label='with YOLO',
               color=[get_color(m) for m in models],
               hatch=hatch_pattern_for_yolo(),
               edgecolor='black', linewidth=0.5)

        for i, (a_no, a_yes) in enumerate(zip(accs_noyolo, accs_yolo)):
            ax.text(i - width / 2, a_no + 1, f'{a_no:.0f}%',
                    ha='center', va='bottom', fontsize=8)
            ax.text(i + width / 2, a_yes + 1, f'{a_yes:.0f}%',
                    ha='center', va='bottom', fontsize=8)
    else:
        width = 0.5
        accs = [compute_metrics(groups_noyolo[m])['accuracy'] * 100 for m in models]
        bars = ax.bar(x, accs, width, color=[get_color(m) for m in models])

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Accuracy (%)')
    title = 'Classification Accuracy by Model'
    if compare_yolo:
        title += ' (no YOLO vs. with YOLO)'
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.set_ylim(0, 110)
    if compare_yolo:
        ax.legend(loc='lower right')

    save_chart(fig, output_dir, 'accuracy_by_model.png')


def _grouped_bar_chart(metric_fn, rows_noyolo, rows_yolo, output_dir,
                       compare_yolo, filename, ylabel, title,
                       value_fmt='{:.0f}', ylim=None,
                       show_value_labels=True):
    """
    Helper: draws a grouped bar chart of `metric_fn(rows) -> float` per model,
    optionally split by YOLO variant.
    """
    groups_noyolo = group_by_model(rows_noyolo)
    models = sorted(groups_noyolo.keys())

    if compare_yolo:
        groups_yolo = group_by_model(rows_yolo)

    fig, ax = plt.subplots(figsize=(11, 5) if compare_yolo else (10, 5))
    x = np.arange(len(models))

    if compare_yolo:
        width = 0.35
        for offset, label, groups, hatch in [
            (-width / 2, 'no YOLO', groups_noyolo, ''),
            (width / 2, 'with YOLO', groups_yolo, hatch_pattern_for_yolo()),
        ]:
            values = [metric_fn(groups.get(m, [])) for m in models]
            colors = [get_color(m) for m in models]
            ax.bar(x + offset, values, width, label=label,
                   color=colors, hatch=hatch, edgecolor='black', linewidth=0.5)
            if show_value_labels:
                for i, v in enumerate(values):
                    ax.text(i + offset, v, value_fmt.format(v),
                            ha='center', va='bottom', fontsize=7)
    else:
        width = 0.5
        values = [metric_fn(groups_noyolo.get(m, [])) for m in models]
        colors = [get_color(m) for m in models]
        ax.bar(x, values, width, color=colors)
        if show_value_labels:
            for i, v in enumerate(values):
                ax.text(i, v, value_fmt.format(v),
                        ha='center', va='bottom', fontsize=9)

    ax.set_ylabel(ylabel)
    if compare_yolo:
        title = title + ' (no YOLO vs. with YOLO)'
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    if ylim:
        ax.set_ylim(*ylim)
    if compare_yolo:
        ax.legend(loc='lower right')

    save_chart(fig, output_dir, filename)


def chart_precision_recall_f1(rows_noyolo, rows_yolo, output_dir, compare_yolo):
    """Grouped bar chart: precision, recall, F1 per model.

    In compare-yolo mode, shows precision/recall/F1 for both variants.
    """
    groups_noyolo = group_by_model(rows_noyolo)
    models = sorted(groups_noyolo.keys())

    if compare_yolo:
        groups_yolo = group_by_model(rows_yolo)

    fig, ax = plt.subplots(figsize=(13, 5) if compare_yolo else (11, 5))
    x = np.arange(len(models))

    if compare_yolo:
        width = 0.13
        positions = [-2.5 * width, -1.5 * width, -0.5 * width,
                     0.5 * width, 1.5 * width, 2.5 * width]
        m_noyolo = {m: compute_metrics(groups_noyolo.get(m, [])) for m in models}
        m_yolo = {m: compute_metrics(groups_yolo.get(m, [])) for m in models}

        ax.bar(x + positions[0], [m_noyolo[m]['precision'] * 100 for m in models],
               width, label='Precision (no YOLO)', color='#2196F3',
               edgecolor='black', linewidth=0.5)
        ax.bar(x + positions[1], [m_noyolo[m]['recall'] * 100 for m in models],
               width, label='Recall (no YOLO)', color='#4CAF50',
               edgecolor='black', linewidth=0.5)
        ax.bar(x + positions[2], [m_noyolo[m]['f1'] * 100 for m in models],
               width, label='F1 (no YOLO)', color='#FF9800',
               edgecolor='black', linewidth=0.5)
        ax.bar(x + positions[3], [m_yolo[m]['precision'] * 100 for m in models],
               width, label='Precision (with YOLO)', color='#2196F3',
               hatch=hatch_pattern_for_yolo(), edgecolor='black', linewidth=0.5)
        ax.bar(x + positions[4], [m_yolo[m]['recall'] * 100 for m in models],
               width, label='Recall (with YOLO)', color='#4CAF50',
               hatch=hatch_pattern_for_yolo(), edgecolor='black', linewidth=0.5)
        ax.bar(x + positions[5], [m_yolo[m]['f1'] * 100 for m in models],
               width, label='F1 (with YOLO)', color='#FF9800',
               hatch=hatch_pattern_for_yolo(), edgecolor='black', linewidth=0.5)
    else:
        width = 0.25
        metrics = {m: compute_metrics(groups_noyolo[m]) for m in models}
        precisions = [metrics[m]['precision'] * 100 for m in models]
        recalls = [metrics[m]['recall'] * 100 for m in models]
        f1s = [metrics[m]['f1'] * 100 for m in models]
        ax.bar(x - width, precisions, width, label='Precision', color='#2196F3')
        ax.bar(x, recalls, width, label='Recall', color='#4CAF50')
        ax.bar(x + width, f1s, width, label='F1', color='#FF9800')

        for i, m in enumerate(models):
            ax.text(i - width, precisions[i] + 1, f'{precisions[i]:.0f}',
                    ha='center', va='bottom', fontsize=8)
            ax.text(i, recalls[i] + 1, f'{recalls[i]:.0f}',
                    ha='center', va='bottom', fontsize=8)
            ax.text(i + width, f1s[i] + 1, f'{f1s[i]:.0f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score (%)')
    title = 'Precision, Recall, and F1 by Model (positive class = fire)'
    if compare_yolo:
        title = ('Precision, Recall, and F1 by Model — no YOLO vs. with YOLO\n'
                 '(positive class = fire)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right', fontsize=7 if compare_yolo else 9, ncol=2 if compare_yolo else 1)

    save_chart(fig, output_dir, 'precision_recall_f1.png')


def chart_mcc(rows_noyolo, rows_yolo, output_dir, compare_yolo):
    """Bar chart: Matthews Correlation Coefficient per model."""
    _grouped_bar_chart(
        metric_fn=lambda rows: compute_metrics(rows)['mcc'] if rows else 0,
        rows_noyolo=rows_noyolo,
        rows_yolo=rows_yolo,
        output_dir=output_dir,
        compare_yolo=compare_yolo,
        filename='mcc.png',
        ylabel='MCC',
        title=('Matthews Correlation Coefficient by Model\n'
               '(range: -1 to +1, higher is better, 0 = random)'),
        value_fmt='{:.2f}',
        ylim=(-1.05, 1.05),
    )


def chart_confusion(rows_noyolo, rows_yolo, output_dir, compare_yolo):
    """TP, TN, FP, FN counts per model.

    In compare-yolo mode, two charts side by side — one per variant.
    """
    if compare_yolo:
        groups_noyolo = group_by_model(rows_noyolo)
        groups_yolo = group_by_model(rows_yolo)
        models = sorted(set(list(groups_noyolo.keys()) + list(groups_yolo.keys())))

        fig, axes = plt.subplots(1, 2, figsize=(20, 5), sharey=True)
        for ax, (groups, variant) in zip(
            axes,
            [(groups_noyolo, 'no YOLO'), (groups_yolo, 'with YOLO')],
        ):
            metrics = {m: compute_metrics(groups.get(m, [])) for m in models}
            x = np.arange(len(models))
            width = 0.2

            tps = [metrics[m]['tp'] for m in models]
            tns = [metrics[m]['tn'] for m in models]
            fps = [metrics[m]['fp'] for m in models]
            fns = [metrics[m]['fn'] for m in models]

            ax.bar(x - 1.5 * width, tps, width, label='TP (correct fire)',
                   color=CORRECT_COLOR)
            ax.bar(x - 0.5 * width, tns, width, label='TN (correct no-fire)',
                   color='#81C784')
            ax.bar(x + 0.5 * width, fps, width, label='FP (false alarm)',
                   color='#FFC107')
            ax.bar(x + 1.5 * width, fns, width, label='FN (missed fire)',
                   color=INCORRECT_COLOR)

            ax.set_title(f'Prediction Outcomes — {variant}')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=25, ha='right')
            ax.set_ylabel('Count')

        axes[0].legend(loc='upper right', fontsize=8)
        save_chart(fig, output_dir, 'confusion_matrix.png')
        return

    groups = group_by_model(rows_noyolo)
    models = sorted(groups.keys())
    metrics = {m: compute_metrics(groups[m]) for m in models}

    tps = [metrics[m]['tp'] for m in models]
    fps = [metrics[m]['fp'] for m in models]
    tns = [metrics[m]['tn'] for m in models]
    fns = [metrics[m]['fn'] for m in models]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(models))
    width = 0.2

    ax.bar(x - 1.5 * width, tps, width, label='TP (correct fire)', color=CORRECT_COLOR)
    ax.bar(x - 0.5 * width, tns, width, label='TN (correct no-fire)', color='#81C784')
    ax.bar(x + 0.5 * width, fps, width, label='FP (false alarm)', color='#FFC107')
    ax.bar(x + 1.5 * width, fns, width, label='FN (missed fire)', color=INCORRECT_COLOR)

    ax.set_ylabel('Count')
    ax.set_title('Prediction Outcomes by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.legend(loc='upper right', fontsize=8)

    save_chart(fig, output_dir, 'confusion_matrix.png')


def chart_response_length(rows_noyolo, rows_yolo, output_dir, compare_yolo):
    """Box plot: word count distribution per model."""
    groups_noyolo = group_by_model(rows_noyolo)
    models = sorted(groups_noyolo.keys())

    if compare_yolo:
        groups_yolo = group_by_model(rows_yolo)

        fig, ax = plt.subplots(figsize=(13, 5))
        positions = []
        data = []
        labels = []
        colors = []
        hatches = []

        spacing = 1.0
        for i, model in enumerate(models):
            base = i * spacing * 2
            positions.extend([base, base + 0.7])
            data.append([int(r['word_count']) for r in groups_noyolo.get(model, [])])
            data.append([int(r['word_count']) for r in groups_yolo.get(model, [])])
            labels.extend([f'{model}\nno YOLO', f'{model}\nwith YOLO'])
            colors.extend([get_color(model), get_color(model)])
            hatches.extend(['', hatch_pattern_for_yolo()])

        bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True)
        for patch, color, hatch in zip(bp['boxes'], colors, hatches):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            if hatch:
                patch.set_hatch(hatch)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('Word Count')
        ax.set_title('Response Length by Model (no YOLO vs. with YOLO)')

        save_chart(fig, output_dir, 'response_length.png')
        return

    data = []
    for model in models:
        words = [int(r['word_count']) for r in groups_noyolo[model]]
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

def chart_tokens_per_sec(rows_noyolo, rows_yolo, output_dir, compare_yolo):
    """Bar chart: median tokens/sec per model."""
    _grouped_bar_chart(
        metric_fn=lambda rows: float(np.median([float(r['tokens_per_sec']) for r in rows])) if rows else 0,
        rows_noyolo=rows_noyolo,
        rows_yolo=rows_yolo,
        output_dir=output_dir,
        compare_yolo=compare_yolo,
        filename='tokens_per_sec.png',
        ylabel='Tokens/sec',
        title='Generation Speed by Model (median)',
        value_fmt='{:.1f}',
    )


def chart_inference_breakdown(rows_noyolo, rows_yolo, output_dir, compare_yolo):
    """Stacked bar: prompt eval vs generation vs overhead per model."""
    groups_noyolo = group_by_model(rows_noyolo)
    models = sorted(groups_noyolo.keys())

    def medians(rows):
        if not rows:
            return 0, 0, 0
        prompt = np.median([float(r['prompt_eval_duration_s']) for r in rows])
        gen = np.median([float(r['eval_duration_s']) for r in rows])
        total = np.median([float(r['total_duration_s']) for r in rows])
        overhead = max(0, total - prompt - gen)
        return prompt, gen, overhead

    fig, ax = plt.subplots(figsize=(11, 5) if compare_yolo else (10, 5))
    x = np.arange(len(models))

    if compare_yolo:
        groups_yolo = group_by_model(rows_yolo)
        width = 0.35
        for offset, label, groups, hatch in [
            (-width / 2, 'no YOLO', groups_noyolo, ''),
            (width / 2, 'with YOLO', groups_yolo, hatch_pattern_for_yolo()),
        ]:
            prompts, gens, overheads = [], [], []
            for m in models:
                p, g, o = medians(groups.get(m, []))
                prompts.append(p)
                gens.append(g)
                overheads.append(o)

            ax.bar(x + offset, prompts, width, color='#2196F3',
                   hatch=hatch, edgecolor='black', linewidth=0.5,
                   label='Image/Prompt Processing' if offset < 0 else None)
            ax.bar(x + offset, gens, width, bottom=prompts, color='#4CAF50',
                   hatch=hatch, edgecolor='black', linewidth=0.5,
                   label='Text Generation' if offset < 0 else None)
            ax.bar(x + offset, overheads, width,
                   bottom=[p + g for p, g in zip(prompts, gens)],
                   color='#BDBDBD', hatch=hatch, edgecolor='black', linewidth=0.5,
                   label='Ollama Overhead' if offset < 0 else None)

            for i in range(len(models)):
                ax.text(i + offset, -0.5, label, ha='center', va='top',
                        fontsize=7, color='gray')
    else:
        width = 0.5
        prompts, gens, overheads = [], [], []
        for m in models:
            p, g, o = medians(groups_noyolo[m])
            prompts.append(p)
            gens.append(g)
            overheads.append(o)

        ax.bar(x, prompts, width, label='Image/Prompt Processing', color='#2196F3')
        ax.bar(x, gens, width, bottom=prompts, label='Text Generation', color='#4CAF50')
        ax.bar(x, overheads, width,
               bottom=[p + g for p, g in zip(prompts, gens)],
               label='Ollama Overhead', color='#BDBDBD')

    ax.set_ylabel('Time (seconds)')
    title = 'Inference Time Breakdown by Model (median)'
    if compare_yolo:
        title += ' — no YOLO vs. with YOLO'
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.legend()

    save_chart(fig, output_dir, 'inference_breakdown.png')


def chart_total_inference(rows_noyolo, rows_yolo, output_dir, compare_yolo):
    """Bar chart: median total inference time per model."""
    _grouped_bar_chart(
        metric_fn=lambda rows: float(np.median([float(r['total_duration_s']) for r in rows])) if rows else 0,
        rows_noyolo=rows_noyolo,
        rows_yolo=rows_yolo,
        output_dir=output_dir,
        compare_yolo=compare_yolo,
        filename='total_inference_time.png',
        ylabel='Time (seconds)',
        title='Total LLM Inference Time by Model (median)',
        value_fmt='{:.1f}s',
    )


def chart_memory_usage(rows_noyolo, rows_yolo, output_dir, compare_yolo):
    """Bar chart: median runtime memory vs model disk size."""
    groups_noyolo = group_by_model(rows_noyolo)
    models = sorted(groups_noyolo.keys())

    def median_mem(rows):
        vals = [float(r['memory_usage_gb']) for r in rows if r['memory_usage_gb']]
        return np.median(vals) if vals else 0

    def model_size(rows):
        if not rows:
            return 0
        size = rows[0].get('model_size_gb', '')
        return float(size) if size else 0

    fig, ax = plt.subplots(figsize=(11, 5) if compare_yolo else (10, 5))
    x = np.arange(len(models))

    if compare_yolo:
        groups_yolo = group_by_model(rows_yolo)
        width = 0.22
        sizes = [model_size(groups_noyolo[m]) for m in models]
        ax.bar(x - 1.5 * width, sizes, width, label='Model Size (disk)',
               color='#BDBDBD')

        mem_noyolo = [median_mem(groups_noyolo.get(m, [])) for m in models]
        mem_yolo = [median_mem(groups_yolo.get(m, [])) for m in models]

        ax.bar(x - 0.5 * width, mem_noyolo, width, label='RSS (no YOLO)',
               color='#FF9800', edgecolor='black', linewidth=0.5)
        ax.bar(x + 0.5 * width, mem_yolo, width, label='RSS (with YOLO)',
               color='#FF9800', hatch=hatch_pattern_for_yolo(),
               edgecolor='black', linewidth=0.5)
    else:
        width = 0.35
        sizes = [model_size(groups_noyolo[m]) for m in models]
        mems = [median_mem(groups_noyolo[m]) for m in models]
        ax.bar(x - width / 2, sizes, width, label='Model Size (disk)',
               color='#BDBDBD')
        ax.bar(x + width / 2, mems, width, label='RSS Memory (runtime)',
               color='#FF9800')

    ax.axhline(y=16, color='red', linestyle='--', alpha=0.5, label='16GB RAM limit')
    ax.set_ylabel('GB')
    title = 'Model Size vs Runtime Memory'
    if compare_yolo:
        title += ' (no YOLO vs. with YOLO)'
    ax.set_title(title)
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

def print_summary(acc_noyolo, acc_yolo, perf_noyolo, perf_yolo, compare_yolo):
    """Print a summary table to stdout. Paired rows when compare_yolo is True."""
    acc_groups_noyolo = group_by_model(acc_noyolo) if acc_noyolo else {}
    perf_groups_noyolo = group_by_model(perf_noyolo) if perf_noyolo else {}
    acc_groups_yolo = group_by_model(acc_yolo) if (compare_yolo and acc_yolo) else {}
    perf_groups_yolo = group_by_model(perf_yolo) if (compare_yolo and perf_yolo) else {}

    all_models = sorted(set(
        list(acc_groups_noyolo.keys())
        + list(perf_groups_noyolo.keys())
        + list(acc_groups_yolo.keys())
        + list(perf_groups_yolo.keys())
    ))

    if compare_yolo:
        header = (
            f"{'Model':<20} {'YOLO':<6} "
            f"{'Acc':>6} "
            f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'MCC':>6} "
            f"{'Tok/s':>8} {'Total(s)':>10} {'Mem(GB)':>8}"
        )
    else:
        header = (
            f"{'Model':<20} "
            f"{'Acc':>6} "
            f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'MCC':>6} "
            f"{'Tok/s':>8} {'Total(s)':>10} {'Mem(GB)':>8}"
        )
    print()
    print(header)
    print('─' * len(header))

    def row_for(model, acc_groups, perf_groups, yolo_label=None):
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

        if yolo_label is not None:
            return (
                f"{model:<20} {yolo_label:<6} "
                f"{acc_str:>6} "
                f"{prec_str:>6} {rec_str:>6} {f1_str:>6} {mcc_str:>6} "
                f"{tok_str:>8} {total_str:>10} {mem_str:>8}"
            )
        return (
            f"{model:<20} "
            f"{acc_str:>6} "
            f"{prec_str:>6} {rec_str:>6} {f1_str:>6} {mcc_str:>6} "
            f"{tok_str:>8} {total_str:>10} {mem_str:>8}"
        )

    for model in all_models:
        if compare_yolo:
            print(row_for(model, acc_groups_noyolo, perf_groups_noyolo, 'no'))
            print(row_for(model, acc_groups_yolo, perf_groups_yolo, 'yes'))
        else:
            print(row_for(model, acc_groups_noyolo, perf_groups_noyolo))

    print()
    print("  Acc       = classification accuracy")
    print("  Prec/Rec/F1/MCC computed with fire as the positive class.")
    if compare_yolo:
        print("  YOLO column indicates whether YOLO preprocessing was applied to the input.")
    print()


# ── Validation ───────────────────────────────────────────────────────

def validate_compare_yolo(acc_noyolo, acc_yolo, perf_noyolo, perf_yolo):
    """
    For --compare-yolo, ensure that if a result type (accuracy or performance)
    has any data at all, both YOLO and no-YOLO variants are present. The
    comparison is only meaningful when both sides exist.

    Exits with an error message if validation fails.
    """
    issues = []

    has_acc_noyolo = bool(acc_noyolo)
    has_acc_yolo = bool(acc_yolo)
    has_perf_noyolo = bool(perf_noyolo)
    has_perf_yolo = bool(perf_yolo)

    if (has_acc_noyolo or has_acc_yolo) and not (has_acc_noyolo and has_acc_yolo):
        missing = 'YOLO' if has_acc_noyolo else 'no-YOLO'
        issues.append(f"  Accuracy: {missing} CSVs are missing.")

    if (has_perf_noyolo or has_perf_yolo) and not (has_perf_noyolo and has_perf_yolo):
        missing = 'YOLO' if has_perf_noyolo else 'no-YOLO'
        issues.append(f"  Performance: {missing} CSVs are missing.")

    if not (has_acc_noyolo or has_acc_yolo or has_perf_noyolo or has_perf_yolo):
        issues.append("  No CSV files found in either accuracy or performance directory.")

    if issues:
        print("Cannot run --compare-yolo: comparison requires both no-YOLO and "
              "YOLO results to be present.")
        for issue in issues:
            print(issue)
        print("\nGenerate the missing runs and try again, or run without "
              "--compare-yolo for the standard view.")
        sys.exit(1)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--accuracy-dir', default='results/accuracy',
                        help='Accuracy CSV directory')
    parser.add_argument('--performance-dir', default='results/performance',
                        help='Performance CSV directory')
    parser.add_argument('--output', default='charts',
                        help='Output directory for PNGs')
    parser.add_argument('--compare-yolo', action='store_true',
                        help='Compare no-YOLO vs YOLO runs side by side. '
                             'Requires both CSV types to be present.')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    acc_noyolo, acc_yolo = load_csvs(args.accuracy_dir, compare_yolo=args.compare_yolo)
    perf_noyolo, perf_yolo = load_csvs(args.performance_dir, compare_yolo=args.compare_yolo)

    if args.compare_yolo:
        validate_compare_yolo(acc_noyolo, acc_yolo, perf_noyolo, perf_yolo)

    print(f"Loaded {len(acc_noyolo)} accuracy rows (no YOLO), "
          f"{len(acc_yolo)} accuracy rows (with YOLO)")
    print(f"Loaded {len(perf_noyolo)} performance rows (no YOLO), "
          f"{len(perf_yolo)} performance rows (with YOLO)")

    acc_total, acc_errors = count_errors(args.accuracy_dir)
    perf_total, perf_errors = count_errors(args.performance_dir)
    if acc_errors:
        print(f"  Warning: {acc_errors}/{acc_total} accuracy rows skipped "
              f"(inference failures — empty classification)")
    if perf_errors:
        print(f"  Warning: {perf_errors}/{perf_total} performance rows skipped "
              f"(inference failures — empty classification)")

    # Always generate model size chart
    print("\nGenerating model size chart...")
    all_models = set()
    for r in acc_noyolo + acc_yolo + perf_noyolo + perf_yolo:
        all_models.add(r['model_name'])
    chart_model_sizes(output_dir, list(all_models) if all_models else None)

    if acc_noyolo or acc_yolo:
        print("\nGenerating accuracy charts...")
        chart_accuracy(acc_noyolo, acc_yolo, output_dir, args.compare_yolo)
        chart_precision_recall_f1(acc_noyolo, acc_yolo, output_dir, args.compare_yolo)
        chart_mcc(acc_noyolo, acc_yolo, output_dir, args.compare_yolo)
        chart_confusion(acc_noyolo, acc_yolo, output_dir, args.compare_yolo)
        chart_response_length(acc_noyolo, acc_yolo, output_dir, args.compare_yolo)

    if perf_noyolo or perf_yolo:
        print("\nGenerating performance charts...")
        chart_tokens_per_sec(perf_noyolo, perf_yolo, output_dir, args.compare_yolo)
        chart_inference_breakdown(perf_noyolo, perf_yolo, output_dir, args.compare_yolo)
        chart_total_inference(perf_noyolo, perf_yolo, output_dir, args.compare_yolo)
        chart_memory_usage(perf_noyolo, perf_yolo, output_dir, args.compare_yolo)

    if acc_noyolo or acc_yolo or perf_noyolo or perf_yolo:
        print_summary(acc_noyolo, acc_yolo, perf_noyolo, perf_yolo, args.compare_yolo)

    print(f"Done. Charts saved to {output_dir}/")


if __name__ == '__main__':
    main()