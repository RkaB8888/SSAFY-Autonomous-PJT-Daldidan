import os
import json
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from services.model_jmk3.features.extract_features3 import extract_features_extended
import joblib

class AppleDataset(Dataset):
    def __init__(self, image_dir, json_files, transform=None):
        self.image_dir = image_dir
        self.json_files = json_files
        self.transform = transform
        self.scaler = joblib.load("/home/j-k12e206/jmk/S12P31E206/BE/ai/services/model_jmk3/me/scaler3.pkl")

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        json_filename = os.path.basename(json_path)
        img_filename = os.path.splitext(json_filename)[0] + '.jpg'
        img_path = os.path.join(self.image_dir, img_filename)

        image = cv2.imread(img_path)
        if image is None:
            return None

        points = np.array(data['annotations']['segmentation']).reshape((-1,2)).astype(np.int32)
        mask = np.zeros((data['images']['img_height'], data['images']['img_width']), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        manual_features_raw = extract_features_extended(image, mask)
        manual_features_scaled = self.scaler.transform([manual_features_raw])[0]
        manual_features = torch.tensor(manual_features_scaled, dtype=torch.float32)

        image_pil = Image.open(img_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = transforms.ToTensor()(image_pil)

        label = float(data['collection'].get('sugar_content_nir', 0))
        return image_tensor, manual_features, torch.tensor(label, dtype=torch.float32)
