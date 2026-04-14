"""
Metrics Module
Calculates performance metrics for benchmarking.
"""

import subprocess


# Model sizes in GB (from `ollama list`). Static lookup.
MODEL_SIZES = {
    'gemma4:e2b': 7.2,
    'gemma4:e4b': 9.6,
    'qwen3-vl:4b': 3.3,
    'qwen3-vl:8b': 6.1,
    'ministral-3:3b': 3.0,
    'ministral-3:8b': 6.0,
}


def track_memory_usage():
    """
    Get total RSS memory used by all Ollama processes (macOS specific).
    Returns total in GB, or None if no Ollama processes found.
    """
    try:
        result = subprocess.run(
            ['ps', '-e', '-o', 'pid,rss,comm'],
            capture_output=True, text=True
        )

        total_rss_kb = 0
        found = False

        for line in result.stdout.strip().split('\n'):
            if 'ollama' in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        total_rss_kb += int(parts[1])
                        found = True
                    except ValueError:
                        continue

        if found:
            return round(total_rss_kb / (1024 * 1024), 2)
        return None

    except Exception as e:
        print(f"Error tracking memory: {e}")
        return None


def get_model_size(model_name):
    """Get model file size in GB from static lookup."""
    return MODEL_SIZES.get(model_name, None)


def count_words(text):
    """Count words in text."""
    return len(text.split())


def evaluate_accuracy(llm_response, ground_truth):
    """
    Check if LLM correctly identified fire/no_fire.

    Returns:
        True if correct, False if incorrect, None if ambiguous/missing classification.
    """
    response_upper = llm_response.upper()

    llm_says_fire = '[FIRE_DETECTED]' in response_upper
    llm_says_nofire = '[NO_FIRE_DETECTED]' in response_upper

    if llm_says_fire and not llm_says_nofire:
        llm_classification = 'fire'
    elif llm_says_nofire and not llm_says_fire:
        llm_classification = 'no_fire'
    else:
        return None  # Ambiguous — needs manual review

    return llm_classification == ground_truth