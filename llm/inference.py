import ollama
import base64
import sys
from pathlib import Path

MODELS = {
    "ministral": "ministral-3:3b",
    #"qwen3": "qwen3-vl:4b",
    #"gemma4": "gemma4:e4b",
}


def load_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_model(model_name: str, image_path: str, prompt: str, system_prompt: str = None) -> str:
    model_tag = MODELS[model_name]
    image_b64 = load_image(image_path)

    print(f"\n=== {model_name} ({model_tag}) ===")

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": prompt,
        "images": [image_b64]
    })

    response = ollama.chat(model=model_tag, messages=messages)

    # Print timing info
    eval_count = response.get('eval_count', 0)
    eval_duration = response.get('eval_duration', 0)
    prompt_eval = response.get('prompt_eval_duration', 0)
    total = response.get('total_duration', 0)

    print(f"Prompt eval: {prompt_eval / 1e9:.2f}s")
    print(f"Eval: {eval_duration / 1e9:.2f}s ({eval_count} tokens)")
    print(f"Total: {total / 1e9:.2f}s")

    return response.message.content


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image."

    # Optional: pass --prompt-file prompts/prompt.txt as 3rd and 4th arg
    system_prompt = None
    if "--prompt-file" in sys.argv:
        idx = sys.argv.index("--prompt-file")
        prompt_file = sys.argv[idx + 1]
        system_prompt = Path(prompt_file).read_text(encoding='utf-8')
        print(f"System prompt: {prompt_file} ({len(system_prompt)} chars)")

    if not Path(image_path).exists():
        print(f"Error: image not found at {image_path}")
        sys.exit(1)

    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")

    for model_name in MODELS:
        result = query_model(model_name, image_path, prompt, system_prompt)
        print(result)


if __name__ == "__main__":
    main()