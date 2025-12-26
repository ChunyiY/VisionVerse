# VisionVerse

A production-ready AI pipeline that transforms visual input into artistic creations through multi-modal understanding, poetry generation, and diffusion-based image synthesis.

## Overview

VisionVerse implements an end-to-end workflow that bridges computer vision, natural language processing, and generative AI. The system processes captured images through multiple stages: visual understanding via BLIP, semantic keyword extraction using CLIP embeddings, poetry generation with Ollama, and final artwork creation using Stable Diffusion with historical art style conditioning.

## Architecture

The pipeline consists of four primary stages:

1. **Image Capture & Understanding**: Camera-based image acquisition with BLIP-based caption generation
2. **Semantic Analysis**: CLIP-encoded semantic matching for keyword extraction from a curated word bank
3. **Poetry Generation**: Context-aware poetry synthesis using Ollama's language models
4. **Art Generation**: Stable Diffusion-based image synthesis with 100+ historical art style conditioning

## Features

- **Multi-Modal Processing**: Integrates vision-language models (BLIP) and text encoders (CLIP) for semantic understanding
- **Historical Art Styles**: Comprehensive library of 100+ art styles spanning from prehistoric to contemporary digital art
- **Semantic Keyword Matching**: CLIP-based cosine similarity for contextually relevant keyword selection
- **Configurable Pipeline**: Modular architecture supporting custom word banks, art styles, and model configurations
- **Production Logging**: Timestamped output logging with complete generation metadata

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended), Apple Silicon (MPS), or CPU
- [Ollama](https://ollama.ai/) runtime with Mistral model installed
- Minimum 8GB GPU memory for Stable Diffusion inference

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/ChunyiY/VisionVerse.git
cd VisionVerse
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama and Models

```bash
# Install Ollama from https://ollama.ai/
ollama pull mistral
```

### 5. Model Downloads

The following models are automatically downloaded on first execution:
- BLIP Image Captioning (Salesforce/blip-image-captioning-base)
- CLIP Text Encoder (openai/clip-vit-base-patch32)
- Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)

## Usage

### Basic Execution

```bash
python src/main.py
```

Alternatively:

```bash
python -m src.photo_to_poetry
```

### Execution Flow

1. Optional word bank expansion via interactive prompt
2. Camera image capture (attempts cameras 0-2)
3. BLIP-based image caption generation
4. CLIP semantic matching for keyword extraction
5. Ollama poetry generation with contextual keywords
6. Art style selection and prompt construction
7. Stable Diffusion image synthesis
8. Results saved to `output/` directory with timestamped metadata

### Configuration

Modify `config/config.py` for:
- Output directory paths
- Device selection (cuda/mps/cpu)
- Model identifiers and versions

## Project Structure

```
VisionVerse/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   └── photo_to_poetry.py      # Core pipeline implementation
├── config/
│   └── config.py               # Configuration management
├── data/
│   ├── art_style.txt           # Art style definitions
│   └── random_words.txt        # Semantic word bank
├── output/                     # Generated artifacts (gitignored)
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## Art Style Library

The system includes 100+ art styles organized by historical periods:

- **Classical**: Ancient Greek, Roman, Byzantine, Islamic
- **Medieval**: Romanesque, Gothic, Byzantine
- **Renaissance**: Early, High, Northern Renaissance, Mannerism
- **Modern**: Impressionism, Post-Impressionism, Cubism, Surrealism
- **Contemporary**: Digital Art, AI Art, Cyberpunk, Vaporwave

Complete style definitions available in `data/art_style.txt`.

## Output Format

Generated artifacts are saved with timestamped filenames:

- `output/art_<timestamp>.png`: Synthesized artwork images (768x768)
- `output/ai_art_prompt_<timestamp>.txt`: Complete generation metadata including:
  - Original image path
  - BLIP-generated caption
  - Semantic keywords
  - Generated poetry
  - Stable Diffusion prompt
  - Output image path
- `output/ai_art_history.log`: Aggregated generation log

## Technical Details

### Models and Frameworks

- **BLIP**: Bootstrapping Language-Image Pre-training for image captioning
- **CLIP**: Contrastive Language-Image Pre-training for semantic embeddings
- **Stable Diffusion**: Latent diffusion model for high-quality image synthesis
- **Ollama**: Local LLM inference for poetry generation

### Performance Considerations

- **Memory**: Enable attention slicing for reduced VRAM usage (default enabled)
- **Device Selection**: Automatic device detection (CUDA > MPS > CPU)
- **Inference**: 40-step diffusion process with guidance scale 8.5

## Troubleshooting

### Camera Access Issues
- Ensure camera permissions are granted
- Verify no other applications are accessing the camera
- Adjust camera ID range in `capture_image()` if needed

### Model Download Failures
- Models download from HuggingFace on first run
- For restricted regions, uncomment mirror endpoint in `src/photo_to_poetry.py`
- Verify network connectivity and HuggingFace access

### Memory Constraints
- Reduce image resolution in `generate_sd_image()` (default: 768x768)
- Use CPU mode for systems with limited GPU memory
- Adjust `num_inference_steps` for faster generation (lower quality)

### Ollama Integration
- Verify Ollama service is running: `ollama list`
- Ensure Mistral model is installed: `ollama pull mistral`
- Check Ollama API accessibility

## License

This project is licensed under the MIT License. See LICENSE file for details.

Note: This project uses third-party models with their respective licenses:
- Stable Diffusion: CreativeML Open RAIL-M License
- BLIP: Apache 2.0
- CLIP: MIT License

Refer to individual model repositories for complete license terms.

## Contributing

Contributions are welcome. Please ensure code follows existing style conventions and includes appropriate documentation.

## Acknowledgments

- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) by Stability AI
- [BLIP](https://github.com/salesforce/BLIP) by Salesforce Research
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [Ollama](https://ollama.ai/) for local LLM inference infrastructure
