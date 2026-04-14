"""
Main Benchmark Script
Runs LLM benchmarking on all models and test images
"""

import time
from pathlib import Path
from src.data_loader import get_ground_truth, get_all_test_images
from src.llm_inference import load_system_prompt, call_llm
from src.metrics import get_model_size, track_memory_usage, count_words, evaluate_accuracy
from src.logger import init_csv, log_result, log_progress


# Configuration
MODELS = [
    'qwen3-vl:4b',
    'qwen3-vl:8b',
    'gemma4:e2b',
    'gemma4:e4b',
    'ministral-3:3b',
    'ministral-3:8b'
]

TEST_IMAGES_DIR = 'data/test_images'
SYSTEM_PROMPT_FILE = 'prompts/placeholder_prompt.txt'
RESULTS_CSV = 'results/benchmark_results.csv'


def main():
    """
    Main benchmarking loop
    """
    # Setup
    log_progress("Starting benchmark")
    
    # Load system prompt
    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
    
    # Get all test images
    test_images = get_all_test_images(TEST_IMAGES_DIR)
    log_progress(f"Found {len(test_images)} test images")
    
    # Initialize CSV
    columns = [
        'model', 'image', 'ground_truth',
        'llm_time', 'memory_gb', 'model_size_gb',
        'word_count', 'correct', 'llm_response'
    ]
    init_csv(RESULTS_CSV, columns)
    log_progress(f"Initialized results CSV: {RESULTS_CSV}")
    
    # Main benchmarking loop
    total_runs = len(MODELS) * len(test_images)
    current_run = 0
    
    for model in MODELS:
        log_progress(f"\n{'='*50}")
        log_progress(f"Testing model: {model}")
        log_progress(f"{'='*50}")
        
        # Get model size once per model
        model_size = get_model_size(model)
        
        for image_path in test_images:
            current_run += 1
            image_name = image_path.name
            ground_truth = get_ground_truth(image_path)
            
            log_progress(f"[{current_run}/{total_runs}] {model} on {image_name}")
            
            try:
                # Track memory before
                memory_before = track_memory_usage()
                
                # LLM inference
                llm_start = time.time()
                llm_response = call_llm(model, str(image_path), system_prompt)
                llm_time = time.time() - llm_start
                
                # Track memory after
                memory_after = track_memory_usage()
                peak_memory = max(memory_before or 0, memory_after or 0)
                
                # Calculate metrics
                word_count = count_words(llm_response)
                is_correct = evaluate_accuracy(llm_response, ground_truth)
                
                # Log result
                log_result(RESULTS_CSV, {
                    'model': model,
                    'image': image_name,
                    'ground_truth': ground_truth,
                    'llm_time': round(llm_time, 2),
                    'memory_gb': round(peak_memory, 2) if peak_memory else None,
                    'model_size_gb': model_size,
                    'word_count': word_count,
                    'correct': is_correct,
                    'llm_response': llm_response
                })
                
                log_progress(f"  ✓ Completed in {llm_time:.2f}s, {word_count} words")
                
            except Exception as e:
                log_progress(f"  ✗ ERROR: {e}")
                # Log error row
                log_result(RESULTS_CSV, {
                    'model': model,
                    'image': image_name,
                    'ground_truth': ground_truth,
                    'llm_time': None,
                    'memory_gb': None,
                    'model_size_gb': model_size,
                    'word_count': None,
                    'correct': None,
                    'llm_response': f"ERROR: {e}"
                })
    
    log_progress("\n" + "="*50)
    log_progress("Benchmark complete!")
    log_progress(f"Results saved to: {RESULTS_CSV}")
    log_progress("="*50)


if __name__ == "__main__":
    main()