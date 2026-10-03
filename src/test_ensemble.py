import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CKPT_DIR = "results/checkpoints"
IMAGE_PATH = "data/testskin.jpg"
NUM_CLASSES = 14
CLASSES = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
           'Chickenpox', 'Cowpox', 'Dermatofibroma', 'HFMD', 'Healthy', 'Measles',
           'Melanocytic nevi', 'Melanoma', 'Monkeypox', 'Squamous cell carcinoma',
           'Vascular lesions']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

def load_model(arch_ctor, ckpt_name):
    model = arch_ctor().to(device)
    ckpt = torch.load(f"{CKPT_DIR}/{ckpt_name}", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

model_cnn = load_model(lambda: CustomCNN(NUM_CLASSES), "cnn_scratch_best.pth")

def make_resnet():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

model_frozen = load_model(make_resnet, "resnet_frozen_best.pth")
model_finetuned = load_model(make_resnet, "resnet_finetuned_best.pth")

print("All 3 models loaded successfully.")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

image = Image.open(IMAGE_PATH).convert('RGB')
input_tensor = eval_transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    probs_cnn = torch.softmax(model_cnn(input_tensor), dim=1)
    probs_frozen = torch.softmax(model_frozen(input_tensor), dim=1)
    probs_finetuned = torch.softmax(model_finetuned(input_tensor), dim=1)

# Individual predictions (for comparison)
for name, probs in [("CNN scratch", probs_cnn), ("ResNet frozen", probs_frozen), ("ResNet fine-tuned", probs_finetuned)]:
    conf, idx = torch.max(probs, 1)
    print(f"{name}: {CLASSES[idx.item()]} ({conf.item()*100:.2f}%)")

# Simple ensemble: average all 3 probability distributions
ensemble_probs = (probs_cnn + probs_frozen + probs_finetuned) / 3
conf, idx = torch.max(ensemble_probs, 1)
print(f"\nEnsemble (average of all 3): {CLASSES[idx.item()]} ({conf.item()*100:.2f}%)")