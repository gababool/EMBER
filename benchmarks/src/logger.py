"""
Logger Module
Handles saving benchmark results to CSV and progress logging
"""

import csv
from pathlib import Path
from datetime import datetime


def init_csv(filepath, columns):
    """
    Create CSV file with headers
    
    Args:
        filepath: Path to CSV file
        columns: List of column names
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Write header row
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def log_result(filepath, data_dict):
    """
    Append one row to CSV
    
    Args:
        filepath: Path to CSV file
        data_dict: Dictionary with data to log (keys should match columns)
    """
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        # Get column names from first line
        f.seek(0)
        reader = csv.DictReader(open(filepath, 'r'))
        fieldnames = reader.fieldnames
        
        # Write the row
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(data_dict)


def log_progress(message, logfile='results/benchmark_progress.log'):
    """
    Write progress message to log file with timestamp
    
    Args:
        message: Progress message
        logfile: Path to log file
    """
    # Create directory if needed
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    
    # Append message with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")