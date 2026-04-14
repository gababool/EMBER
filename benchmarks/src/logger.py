"""
Logger Module
Handles saving benchmark results to CSV and progress logging.
"""

import csv
from pathlib import Path
from datetime import datetime


# CSV columns for benchmark results
RESULT_COLUMNS = [
    'timestamp',
    'model_name',
    'image_path',
    'ground_truth',
    'llm_classification',
    'correct',
    'response_text',
    'word_count',
    'eval_count',
    'tokens_per_sec',
    'prompt_eval_duration_s',
    'eval_duration_s',
    'total_duration_s',
    'load_duration_s',
    'model_size_gb',
    'memory_usage_gb',
]


def init_csv(filepath, columns=None):
    """
    Create CSV file with headers.
    Uses RESULT_COLUMNS by default.
    """
    if columns is None:
        columns = RESULT_COLUMNS

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def log_result(filepath, data_dict, columns=None):
    """
    Append one row to CSV.
    """
    if columns is None:
        columns = RESULT_COLUMNS

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(data_dict)


def log_progress(message, logfile='results/benchmark_progress.log'):
    """Write timestamped progress message to log file."""
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")
    # Also print to stdout for live monitoring
    print(f"[{timestamp}] {message}")