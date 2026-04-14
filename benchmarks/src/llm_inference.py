"""
LLM Inference Module
Handles calling Ollama API for multimodal LLM inference
"""

import ollama
from pathlib import Path


def load_system_prompt(filepath):
    """
    Load system prompt from text file
    
    Args:
        filepath: Path to prompt file
        
    Returns:
        str: Prompt text
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


def call_llm(model_name, image_path, system_prompt):
    """
    Call Ollama LLM with image
    
    Args:
        model_name: Name of Ollama model (e.g., 'gemma4:e4b')
        image_path: Path to image file
        system_prompt: System prompt text
        
    Returns:
        str: LLM response text
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": "Analyze this wildfire reconnaissance image.",
                "images": [str(image_path)]
            }
        ]
    )
    
    return response['message']['content']