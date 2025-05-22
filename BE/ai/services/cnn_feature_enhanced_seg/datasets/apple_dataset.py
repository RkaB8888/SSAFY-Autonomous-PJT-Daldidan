import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class AppleDataset(Dataset):
    def __init__(self, image_dir, json_files, manual_features, labels, transform=None, device='cpu'):
        self.image_dir = image_dir
        self.json_files = json_files
        self.manual_features = manual_features
        self.labels = labels
        self.transform = transform
        self.device = device  # ✅ device 추가

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        json_filename = os.path.basename(json_path)
        img_filename = os.path.splitext(json_filename)[0] + '.jpg'
        img_path = os.path.join(self.image_dir, img_filename)

        try:
            image_pil = Image.open(img_path).convert("RGB")
        except Exception:
            print(f"[WARNING] Image load failed: {img_path}")
            return None

        image_tensor = self.transform(image_pil) if self.transform else transforms.ToTensor()(image_pil)
        manual_feat = torch.tensor(self.manual_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # ✅ GPU로 전송
        image_tensor = image_tensor.to(self.device)
        manual_feat = manual_feat.to(self.device)
        label = label.to(self.device)

        return image_tensor, manual_feat, label
