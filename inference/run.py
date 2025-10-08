import os
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

def label2char(label):
    if 0 <= label <= 9:
        return str(label)
    elif 10 <= label <= 35:
        return chr(label - 10 + ord('A'))
    return '?'

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def Run_inference(model, folder_path, folder_output, device="cpu", max_images=10):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    os.makedirs(folder_output, exist_ok=True)

    plt.figure(figsize=(15, 6))

    for i, img_name in enumerate(image_files[:max_images]):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = test_transform(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            _, pred = output.max(1)

        pred_char = label2char(pred.item())

        output_path = os.path.join(folder_output, f"pred_{pred_char}_{img_name}")
        cv2.imwrite(output_path, img)

        plt.subplot(2, 5, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"Ảnh {i}\nPred: {pred_char}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
