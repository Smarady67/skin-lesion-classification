import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

CHECKPOINT_PATH = "results/checkpoints/resnet_finetuned_best.pth"
NUM_CLASSES = 14
CLASSES = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
           'Chickenpox', 'Cowpox', 'Dermatofibroma', 'HFMD', 'Healthy', 'Measles',
           'Melanocytic nevi', 'Melanoma', 'Monkeypox', 'Squamous cell carcinoma',
           'Vascular lesions']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print("Model loaded successfully. Ready to build the interface.")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

def predict(image):
    image = image.convert('RGB')
    input_tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    results = {CLASSES[i]: float(probabilities[i]) for i in range(NUM_CLASSES)}
    return results

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a skin lesion image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Skin Lesion Classifier",
    description="Upload a photo of a skin lesion to classify it into one of 14 conditions. (ResNet-18, fine-tuned)"
)

demo.launch()