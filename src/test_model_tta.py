import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CHECKPOINT_PATH = "results/checkpoints/resnet_finetuned_best.pth"
IMAGE_PATH = "data/testskin.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NUM_CLASSES = 14

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print("Model loaded successfully.")

CLASSES = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
           'Chickenpox', 'Cowpox', 'Dermatofibroma', 'HFMD', 'Healthy', 'Measles',
           'Melanocytic nevi', 'Melanoma', 'Monkeypox', 'Squamous cell carcinoma',
           'Vascular lesions']

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

image = Image.open(IMAGE_PATH).convert('RGB')
input_tensor = eval_transform(image).unsqueeze(0).to(device)

print(f"Image loaded and preprocessed. Tensor shape: {input_tensor.shape}")

with torch.no_grad():
    # Version 1: the normal image
    out_normal = model(input_tensor)

    # Version 2: flipped left-right
    out_flip = model(torch.flip(input_tensor, dims=[3]))

    # Average the two sets of probabilities (this is the TTA step)
    probabilities = (torch.softmax(out_normal, dim=1) + torch.softmax(out_flip, dim=1)) / 2
    confidence, predicted_idx = torch.max(probabilities, 1)

predicted_class = CLASSES[predicted_idx.item()]
confidence_pct = confidence.item() * 100

print(f"\nPrediction (with TTA): {predicted_class}")
print(f"Confidence: {confidence_pct:.2f}%")