import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.metrics import r2_score
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from models.fusion_model import FusionModel

# ✅ 경로 설정
IMG_DIR = "/home/j-k12e206/ai-hub/Fuji/train/images"
JSON_DIR = "/home/j-k12e206/ai-hub/Fuji/train/jsons"
# SAVE_DIR = "/home/j-k12e206/jmk/S12P31E206/BE/ai/services/model_jmk2/meme/checkpoints"
SAVE_DIR = "/home/j-k12e206/jmk/S12P31E206/BE/ai/services/cnn_feature_enhanced_seg/me/checkpoints"
FEATURE_DIR = "/home/j-k12e206/jmk/S12P31E206/BE/ai/services/cnn_feature_enhanced_seg/me"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 학습 설정
manual_feature_dim = 126
batch_size = 64
num_epochs = 20
lr = 0.001

# ✅ transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Dataset 정의
class AppleDataset(Dataset):
    def __init__(self, image_dir, json_files, manual_features, labels, transform=None):
        self.image_dir = image_dir
        self.json_files = json_files
        self.manual_features = manual_features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        image_name = os.path.splitext(os.path.basename(json_path))[0] + ".jpg"
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)

        manual_feat = torch.tensor(self.manual_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image_tensor, manual_feat, label

# ✅ 데이터 불러오기
manual_features = np.load(os.path.join(FEATURE_DIR, "manual_features.npy"))
labels = np.load(os.path.join(FEATURE_DIR, "labels.npy"))
json_files = sorted([os.path.join(JSON_DIR, f) for f in os.listdir(JSON_DIR) if f.endswith('.json')])

dataset = AppleDataset(IMG_DIR, json_files, manual_features, labels, transform=transform)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=48, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=48, pin_memory=True)

# ✅ 모델 및 학습 세팅
model = FusionModel(manual_feature_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scaler = GradScaler()

best_val_r2 = -float('inf')

# ✅ 학습 루프
for epoch in range(num_epochs):
    model.train()
    train_loss, all_preds, all_labels = 0.0, [], []
    for images, feats, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        feats = feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images, feats).squeeze()
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    train_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    print(f"✅ Epoch {epoch+1} | Loss: {train_loss:.4f} | MAE: {train_mae:.4f}")

    # ✅ 검증
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for images, feats, labels in val_loader:
            images = images.to(device)
            feats = feats.to(device)
            labels = labels.to(device)

            outputs = model(images, feats).squeeze()
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_r2 = r2_score(val_labels, val_preds)
    val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_labels)))
    print(f"🔍 Validation → MAE: {val_mae:.4f} | R²: {val_r2:.4f}")

    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        print(f"📌 [Epoch {epoch+1}] Best R² 갱신 → 저장 완료")

print(f"\n🎉 학습 완료 | Best Validation R²: {best_val_r2:.4f}")
