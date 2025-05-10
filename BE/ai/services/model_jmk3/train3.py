import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import r2_score
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from services.model_jmk3.datasets.apple_dataset3 import AppleDataset
from services.model_jmk3.models.fusion_model3 import FusionModel

torch.backends.cudnn.benchmark = True

# 경로 설정
train_img_dir = "/home/j-k12e206/ai-hub/Fuji/train/images"
train_json_dir = "/home/j-k12e206/ai-hub/Fuji/train/jsons"
valid_img_dir = "/home/j-k12e206/ai-hub/Fuji/valid/images"
valid_json_dir = "/home/j-k12e206/ai-hub/Fuji/valid/jsons"

manual_feature_dim = 10
batch_size = 64
num_epochs = 10
lr = 0.001

def custom_collate(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch)

# JSON 파일 리스트
train_json_files = [os.path.join(train_json_dir, f) for f in os.listdir(train_json_dir) if f.endswith('.json')]
valid_json_files = [os.path.join(valid_json_dir, f) for f in os.listdir(valid_json_dir) if f.endswith('.json')]

print(f"✅ train json_files 개수: {len(train_json_files)}개")
print(f"✅ valid json_files 개수: {len(valid_json_files)}개")

# 데이터 증강
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.05),
    transforms.RandomRotation(10),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 데이터셋 객체 생성
train_dataset = AppleDataset(train_img_dir, train_json_files, transform=transform)
val_dataset = AppleDataset(valid_img_dir, valid_json_files, transform=transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True, collate_fn=custom_collate)

# 모델, 손실함수, 옵티마이저
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionModel(manual_feature_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
scaler = GradScaler()

SAVE_DIR = "/home/j-k12e206/jmk/S12P31E206/BE/ai/services/model_jmk3/me/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

best_val_loss = float('inf')
patience = 3
no_improve_epoch = 0

for epoch in range(num_epochs):
    model.train()
    running_loss, all_preds, all_labels = 0, [], []

    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
    for images, manual_features, labels in train_bar:
        images = images.to(device)
        manual_features = manual_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images, manual_features).squeeze()
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    train_loss = running_loss / len(train_loader)
    train_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))

    model.eval()
    val_loss, val_preds, val_labels = 0, [], []
    with torch.no_grad():
        for images, manual_features, labels in val_loader:
            images = images.to(device)
            manual_features = manual_features.to(device)
            labels = labels.to(device)

            outputs = model(images, manual_features).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_labels)))
    val_r2 = r2_score(val_labels, val_preds)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epoch = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_val_loss.pth"))
    else:
        no_improve_epoch += 1
        if no_improve_epoch >= patience:
            print("Early stopping triggered.")
            break

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} MAE: {train_mae:.4f} | Val Loss: {val_loss:.4f} Val MAE: {val_mae:.4f} Val R²: {val_r2:.4f}")

print(f"\n✅ 학습 종료 → Best Val Loss: {best_val_loss:.4f}")
