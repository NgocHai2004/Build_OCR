import os
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

class OCR_Data(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        
        self.path = os.path.join(self.root, "train" if train else "test")
        self.images = []
        self.labels = []

        for label in os.listdir(self.path):
            label_path = os.path.join(self.path, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(label) 

        print(f"Found {len(self.images)} images in {self.path}")
    
    def __len__(self):
        return len(self.labels)

    @staticmethod
    def char2label(c):
        if c.isdigit(): 
            return ord(c) - ord('0')
        elif c.isupper():  
            return ord(c) - ord('A') + 10
        else:
            raise ValueError(f"Invalid label: {c}")
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_char = self.labels[idx]
        label = self.char2label(label_char)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        else:
            img = cv2.resize(img, (224,224))
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.485,0.456,0.406],
                                       std=[0.229,0.224,0.225])(img)

        return img, label

# if __name__ == "__main__":
#     dataset_path = r"G:\Build_OCR\dataset"
#     data = OCR_Data(root=dataset_path, train=True, transform=None)
    
#     image, label = data[2000]
#     print("Image shape:", image.shape)
#     print("Label:", label)

#     img_show = image.permute(1,2,0).numpy() 
#     img_show = img_show * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])  # de-normalize
#     img_show = np.clip(img_show, 0, 1)
    
#     plt.imshow(img_show)
#     plt.title(f"Label: {label}")
#     plt.axis('off')
#     plt.show()
