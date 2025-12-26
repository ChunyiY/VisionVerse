# config.py
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Device configuration (auto-detect)
# Options: "cuda", "mps" (for Apple Silicon), "cpu"
DEVICE = "mps"  # For M2 Mac, change to "cuda" for NVIDIA GPU or "cpu" for CPU

# Model configuration
MODEL_ID = "runwayml/stable-diffusion-v1-5"
