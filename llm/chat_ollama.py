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

def query_model(model_name: str, image_path: str, prompt: str) -> str:
    model_tag = MODELS[model_name]
    image_b64 = load_image(image_path)

    print(f"\n=== {model_name} ({model_tag}) ===")
    response = ollama.chat(
        model=model_tag,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_b64]
        }]
    )
    return response.message.content

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image."

    if not Path(image_path).exists():
        print(f"Error: image not found at {image_path}")
        sys.exit(1)

    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")

    for model_name in MODELS:
        result = query_model(model_name, image_path, prompt)
        print(result)

if __name__ == "__main__":
    main()