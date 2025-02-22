import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Define Aesthetic Predictor
class AestheticPredictor(nn.Module):
    def __init__(self, clip_model):
        super(AestheticPredictor, self).__init__()
        self.clip = clip_model
        self.fc = nn.Linear(512, 1)  # Regression head
        self.sigmoid = nn.Sigmoid()  # Ensure output is between 0-1

    def forward(self, image):
        image_features = self.clip.get_image_features(image)
        score = self.fc(image_features)
        score = self.sigmoid(score) * 9 + 1  # Scale to 1-10 range
        return score.squeeze()

# Initialize the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AestheticPredictor(clip_model).to(device)
model.eval()  # Set model to evaluation mode

# Load and preprocess image
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def predict_aesthetic_score(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        score = model(inputs["pixel_values"])
        
    if score <= 3:
        return f"{score:.1f} - ❌ Poor aesthetics. The image may be blurry, unbalanced, or have bad lighting."
    elif score <= 6:
        return f"{score:.1f} - 😐 Average aesthetics. The image is decent but lacks impact or refinement."
    elif score <= 8:
        return f"{score:.1f} - 👍 Good aesthetics! The composition is strong, and the image is visually appealing."
    else:
        return f"{score:.1f} - 🌟 Excellent aesthetics! This image looks professional and highly attractive."
