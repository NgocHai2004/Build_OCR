import cv2
import yaml
import torch
import os
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

# ===== Load config =====
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

folder_path = config["folder_path"]       # Thư mục chứa ảnh
model_path = config["model_path"]
img_size = config["img_size"]

# ===== Load model =====
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 36  

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

def label2char(label):
    if 0 <= label <= 9:
        return str(label)
    elif 10 <= label <= 35:
        return chr(label - 10 + ord('A'))
    else:
        return '?'

image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
plt.figure(figsize=(15, 6))

for i, img_name in enumerate(image_files[:10]):
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = test_transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = output.max(1)

    pred_char = label2char(pred.item())

    plt.subplot(2, 5, i + 1)
    plt.imshow(img_rgb)
    plt.title(f"Ảnh {i}\nPred: {pred_char}")
    plt.axis('off')

plt.tight_layout()
plt.show()
