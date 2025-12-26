"""
Main entry point for VisionVerse.
This script orchestrates the complete workflow from image capture to art generation.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.photo_to_poetry import (
    init_resources,
    add_word_to_bank,
    capture_image,
    generate_blip_prompt,
    get_semantic_words,
    compose_poetry_instruction,
    generate_with_ollama,
    poem_to_diffusion_prompt,
    generate_sd_image,
    save_results,
    OLLAMA_MODEL,
    word_bank
)

if __name__ == "__main__":
    init_resources()
    add_word_to_bank() 
    try:
        # 1. Capture and describe image
        image_path = capture_image()
        caption = generate_blip_prompt(image_path)
        print(f"Image Caption: {caption}")
        
        # 2. Generate poetry
        keywords = get_semantic_words(caption, word_bank, k=4)
        print(f"Semantically Matched Keywords: {keywords}")
        poem = generate_with_ollama(OLLAMA_MODEL, compose_poetry_instruction(caption, keywords))
        print(f"\nPoem:\n{poem}")
        
        # 3. Convert to prompt
        sd_prompt = poem_to_diffusion_prompt(poem)
        print(f"\nPrompt:\n{sd_prompt}")
        
        # 4. Generate image using Stable Diffusion
        art_path = generate_sd_image(sd_prompt)
        
        # 5. Save results
        save_results(image_path, caption, keywords, poem, sd_prompt, art_path)
        print(f"\nResults saved to output directory")
        
    except Exception as e:
        print(f"Error: {e}")
        raise
