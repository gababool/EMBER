"""
Quick test script — run one model on one image and see the result.

Usage:
    uv run python quick_test.py test_images/fire/fire_004.jpg
    uv run python quick_test.py test_images/fire/fire_004.jpg --model qwen3-vl:4b
    uv run python quick_test.py test_images/fire/fire_004.jpg --model gemma4:e2b --prompt prompts/custom.txt
"""

import argparse
from src.llm_inference import call_llm, load_system_prompt
from src.metrics import evaluate_accuracy, count_words, get_model_size
from src.data_loader import get_ground_truth


def main():
    parser = argparse.ArgumentParser(description='Quick single-image LLM test')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--model', default='ministral-3:3b', help='Ollama model tag (default: ministral-3:3b)')
    parser.add_argument('--prompt', default='prompts/prompt.txt', help='System prompt file (default: prompts/prompt.txt)')
    args = parser.parse_args()

    system_prompt = load_system_prompt(args.prompt)
    result = call_llm(args.model, args.image, system_prompt)

    # Try to get ground truth (won't work if filename doesn't follow convention)
    try:
        ground_truth = get_ground_truth(args.image)
        correct = evaluate_accuracy(result['response_text'], ground_truth)
        gt_str = f"Ground truth: {ground_truth} | Correct: {'✓' if correct else '✗' if correct is False else '?'}"
    except ValueError:
        gt_str = "Ground truth: unknown (filename doesn't follow convention)"

    print(f"\n{'=' * 60}")
    print(f"Model: {args.model} ({get_model_size(args.model) or '?'}GB)")
    print(f"Image: {args.image}")
    print(f"{gt_str}")
    print(f"{'=' * 60}")
    print(f"\n{result['response_text']}\n")
    print(f"{'─' * 60}")
    print(f"Tokens: {result['eval_count']} ({count_words(result['response_text'])} words)")
    print(f"Tokens/sec: {result['tokens_per_sec']:.1f}")
    print(f"Prompt eval: {result['prompt_eval_duration_ns'] / 1e9:.2f}s")
    print(f"Generation:  {result['eval_duration_ns'] / 1e9:.2f}s")
    print(f"Total:       {result['total_duration_ns'] / 1e9:.2f}s")


if __name__ == '__main__':
    main()