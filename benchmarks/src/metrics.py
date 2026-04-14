"""
Metrics Module
Calculates performance metrics for benchmarking
"""

import psutil
import subprocess
import json

# Model sizes (from `ollama list`). These are static.
MODEL_SIZES = {
    'gemma4:e2b': 7.2,
    'gemma4:e4b': 9.6,
    'qwen3-vl:4b': 3.3,
    'qwen3-vl:8b': 6.1,
    'ministral-3:3b': 3.0,
    'ministral-3:8b': 6.0
}

def track_memory_usage():
    """
    Get memory using macOS ps command (works with unified memory)
    """
    try:
        # Get all processes
        result = subprocess.run(
            ['ps', '-e', '-o', 'pid,rss,comm'],
            capture_output=True,
            text=True
        )
        
        max_memory_gb = 0
        # Filter for Ollama processes and sum their RSS memory usage
        for line in result.stdout.strip().split('\n'):
            if 'ollama' in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    # RSS is in KB, convert to GB
                    try:
                        rss_kb = int(parts[1])
                        memory_gb = rss_kb / (1024 * 1024)
                        max_memory_gb = max(max_memory_gb, memory_gb)
                    except ValueError:
                        continue
        
        # Return max memory in GB, or None if no valid memory usage found
        return max_memory_gb if max_memory_gb > 0 else None
        
    except Exception as e:
        print(f"Error tracking memory: {e}")
        return None


def get_model_size(model_name):
    """
    Get model file size from Ollama
    
    Args:
        model_name: Name of model
        
    Returns:
        float: Model size in GB
    """
    return MODEL_SIZES.get(model_name, None)


def count_words(text):
    """
    Count words in text
    
    Args:
        text: Text string
        
    Returns:
        int: Word count
    """
    return len(text.split())

    
def evaluate_accuracy(llm_response, ground_truth):
    """
    Check if LLM correctly identified fire/no_fire
    
    Args:
        llm_response: LLM response text
        ground_truth: 'fire' or 'no_fire'
        
    Returns:
        bool: True if correct, False otherwise
    """
    response_upper = llm_response.upper()
    
    # System prompt designed to contain these exact phrases for classification 
    # [IMPORTANT: DO NOT CHANGE THESE PHRASES IN THE PROMPT!]
    llm_says_fire = '[FIRE_DETECTED]' in response_upper
    llm_says_nofire = '[NO_FIRE_DETECTED]' in response_upper
    
    if llm_says_fire and not llm_says_nofire:
        llm_classification = 'fire'
    elif llm_says_nofire and not llm_says_fire:
        llm_classification = 'no_fire'
    else:
        # Ambiguous or missing classification - return None for manual review
        return None
    
    return llm_classification == ground_truth