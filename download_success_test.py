from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# 加载模型和处理器
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
).to("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备

# Load image
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "sample_image.jpg")
image = Image.open(image_path).convert("RGB")

# Generate caption (unconditional)
inputs = processor(image, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=50)
print("Caption:", processor.decode(out[0], skip_special_tokens=True))

# Generate caption (with prompt)
prompt = "This image shows key features such as"
inputs = processor(image, text=prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=50)  # Ensure sufficient length
print("Prompt-guided caption:", processor.decode(out[0], skip_special_tokens=True))