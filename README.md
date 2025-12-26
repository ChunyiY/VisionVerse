# VisionVerse

An AI-powered art generation pipeline that transforms captured images into artistic creations through poetry and Stable Diffusion.

## Overview

This project creates a complete workflow from image capture to AI-generated art:

1. **Image Capture**: Captures images from your camera
2. **Image Understanding**: Uses BLIP (Bootstrapping Language-Image Pre-training) to generate captions
3. **Poetry Generation**: Creates poetry based on the image using Ollama
4. **Art Style Selection**: Selects from 100+ historical art styles
5. **Art Generation**: Generates artistic images using Stable Diffusion

## Features

- 🎨 **100+ Art Styles**: From Prehistoric Cave Art to Contemporary Digital Art
- 📸 **Camera Integration**: Direct image capture from webcam
- 🧠 **Semantic Matching**: Uses CLIP to find semantically relevant keywords
- ✍️ **Poetry Generation**: Creates contextual poetry using Ollama
- 🖼️ **AI Art Generation**: High-quality image generation with Stable Diffusion
- 💾 **Result Logging**: Saves all generations with timestamps

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or Apple Silicon (MPS) or CPU
- [Ollama](https://ollama.ai/) installed with the `mistral` model

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd VisionVerse
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama and download the mistral model:
```bash
# Install Ollama from https://ollama.ai/
ollama pull mistral
```

5. Download required models (will be downloaded automatically on first run):
   - BLIP image captioning model
   - CLIP text encoder
   - Stable Diffusion v1.5

## Usage

### Basic Usage

Run the main script (either entry point works):
```bash
python main.py
```
or
```bash
python photo_to_poetry.py
```

The script will:
1. Prompt you to optionally add a word to the word bank
2. Capture an image from your camera
3. Generate a caption using BLIP
4. Find semantically relevant keywords
5. Generate poetry using Ollama
6. Convert poetry to a Stable Diffusion prompt
7. Generate the final artwork
8. Save all results to the `output/` directory

### Configuration

Edit `config.py` to customize:
- Output directory
- Device (cuda/mps/cpu)
- Model ID

### File Structure

```
VisionVerse/
├── photo_to_poetry.py    # Main script
├── config.py              # Configuration file
├── download_success_test.py  # BLIP model test script
├── random_words.txt       # Word bank for semantic matching
├── art_style.txt          # List of art styles
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── LICENSE               # License file
├── .gitignore            # Git ignore rules
└── output/               # Generated images and logs
```

## Art Styles

The project includes 100+ art styles spanning from prehistoric art to contemporary digital art, including:
- Classical: Ancient Greek, Roman, Byzantine
- Renaissance: Early, High, Northern Renaissance
- Modern: Impressionism, Cubism, Surrealism
- Contemporary: Digital Art, AI Art, Cyberpunk Aesthetic

See `art_style.txt` for the complete list.

## Output

All generated content is saved in the `output/` directory:
- `art_<timestamp>.png`: Generated artwork images
- `ai_art_prompt_<timestamp>.txt`: Complete generation details
- `ai_art_history.log`: Summary log of all generations

## Troubleshooting

### Camera Issues
- Ensure your camera is not being used by another application
- The script tries cameras 0-2, adjust the range in `capture_image()` if needed

### Model Download Issues
- Models are downloaded from HuggingFace on first run
- For regions with access restrictions, uncomment the mirror endpoint in `photo_to_poetry.py`

### Memory Issues
- Reduce image resolution in `generate_sd_image()` (currently 768x768)
- Use CPU mode if GPU memory is insufficient
- Enable attention slicing (already enabled) to reduce memory usage

### Ollama Issues
- Ensure Ollama is installed and running
- Verify the mistral model is downloaded: `ollama list`

## License

This project uses several open-source models:
- Stable Diffusion: [CreativeML Open RAIL-M License](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
- BLIP: Apache 2.0
- CLIP: MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) by Stability AI
- [BLIP](https://github.com/salesforce/BLIP) by Salesforce
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [Ollama](https://ollama.ai/) for local LLM inference

