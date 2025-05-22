# from sklearn.model_selection import KFold
# from sklearn.metrics import r2_score, mean_absolute_error
# import numpy as np
# import torch
# import os

# # 설정
# manual_feature_dim = 126
# k = 5
# num_epochs = 3
# batch_size = 16
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # SAVE_DIR = r"C:\Users\SSAFY\Desktop\emodel_result"
# # 서버용 경로
# SAVE_DIR = r"/home/j-k12e206/jmk/S12P31E206/BE/ai/services/cnn_feature_enhanced_seg/me/checkpoints/fusion_model_kfold.pth"
# os.makedirs(SAVE_DIR, exist_ok=True)

# best_fold_r2 = -float('inf')  # 가장 좋은 모델 저장용
# best_model_path = ""

# r2_scores, mae_scores = [], []

# for fold, (train_idx, val_idx) in enumerate(kf.split(sampled_files)):
#     print(f"\n🌀 Fold {fold + 1}/{k}")

#     train_json_files = [os.path.join(local_json_dir, sampled_files[i] + ".json") for i in train_idx]
#     val_json_files = [os.path.join(local_json_dir, sampled_files[i] + ".json") for i in val_idx]

#     train_dataset = AppleDataset(local_img_dir, train_json_files, transform=transform)
#     val_dataset = AppleDataset(local_img_dir, val_json_files, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=...)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=...)

#     model = FusionModel(manual_feature_dim).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.MSELoss()
#     scaler = GradScaler()

#     # === 학습 (epoch 루프 생략) ===
#      # === 학습 ===
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         all_preds, all_labels = [], []

#         for images, manual_features, labels in train_loader:
#             images = images.to(device)
#             manual_features = manual_features.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()
#             with autocast():
#                 outputs = model(images, manual_features).squeeze()
#                 loss = criterion(outputs, labels)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             running_loss += loss.item()
#             all_preds.extend(outputs.detach().cpu().numpy())
#             all_labels.extend(labels.detach().cpu().numpy())

#         train_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
#         print(f"🟢 Epoch {epoch+1}/{num_epochs} | Train Loss: {running_loss:.4f} | Train MAE: {train_mae:.4f}")
#     # === 평가 ===
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for images, features, labels in val_loader:
#             outputs = model(images.to(device), features.to(device)).squeeze()
#             all_preds.extend(outputs.cpu().numpy())
#             all_labels.extend(labels.numpy())

#     fold_r2 = r2_score(all_labels, all_preds)
#     fold_mae = mean_absolute_error(all_labels, all_preds)
#     r2_scores.append(fold_r2)
#     mae_scores.append(fold_mae)

#     print(f"✅ Fold {fold+1} R²: {fold_r2:.4f}, MAE: {fold_mae:.4f}")

#     # ✅ 가장 좋은 R²를 갖는 모델 저장
#     if fold_r2 > best_fold_r2:
#         best_fold_r2 = fold_r2
#         best_model_path = os.path.join(SAVE_DIR, f"best_model_fold{fold+1}_r2_{fold_r2:.4f}.pth")
#         torch.save(model.state_dict(), best_model_path)
#         print(f"📌 현재까지 최고 모델 저장 → {best_model_path}")

# # === 평균 결과 출력 ===
# print(f"\n📊 평균 R²: {np.mean(r2_scores):.4f} | 평균 MAE: {np.mean(mae_scores):.4f}")
# print(f"🎯 최고 성능 모델 위치 → {best_model_path}")

## 서버용 학습, 검증 데이터 분리 로직
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import r2_score
from torch.cuda.amp import autocast, GradScaler
from datasets.apple_dataset import AppleDataset
from models.fusion_model import FusionModel

# === 경로 설정 ===
TRAIN_IMG_DIR = "/home/j-k12e206/ai-hub/Fuji/train/images"
TRAIN_JSON_DIR = "/home/j-k12e206/ai-hub/Fuji/train/jsons"
VALID_IMG_DIR = "/home/j-k12e206/ai-hub/Fuji/valid/images"
VALID_JSON_DIR = "/home/j-k12e206/ai-hub/Fuji/valid/jsons"
SAVE_PATH = "/home/j-k12e206/jmk/S12P31E206/BE/ai/services/cnn_feature_enhanced_seg/me/checkpoints/fusion_model_valsplit.pth"

# === 설정 ===
manual_feature_dim = 126
batch_size = 32
num_epochs = 20
lr = 0.001
torch.backends.cudnn.benchmark = True

# === transform 정의 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === 파일 리스트 ===
train_json_files = [os.path.join(TRAIN_JSON_DIR, f) for f in os.listdir(TRAIN_JSON_DIR) if f.endswith('.json')]
val_json_files = [os.path.join(VALID_JSON_DIR, f) for f in os.listdir(VALID_JSON_DIR) if f.endswith('.json')]

# === 데이터셋/로더 정의 ===
train_dataset = AppleDataset(TRAIN_IMG_DIR, train_json_files, transform=transform)
val_dataset = AppleDataset(VALID_IMG_DIR, val_json_files, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                          collate_fn=lambda b: torch.utils.data.dataloader.default_collate([x for x in b if x]))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                        collate_fn=lambda b: torch.utils.data.dataloader.default_collate([x for x in b if x]))

# === 모델 정의 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionModel(manual_feature_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scaler = GradScaler()
best_val_r2 = -float('inf')

print(f"\n✅ 학습 샘플 수: {len(train_dataset)} | 검증 샘플 수: {len(val_dataset)}")

# === 학습 루프 ===
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, manual_features, labels in pbar:
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

    train_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    print(f"✅ Epoch {epoch+1} | Train Loss: {running_loss:.4f} | Train MAE: {train_mae:.4f}")

    # === 검증 ===
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for images, manual_features, labels in val_loader:
            images = images.to(device)
            manual_features = manual_features.to(device)
            labels = labels.to(device)

            outputs = model(images, manual_features).squeeze()
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_r2 = r2_score(val_labels, val_preds)
    val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_labels)))
    print(f"🔍 Validation → MAE: {val_mae:.4f} | R²: {val_r2:.4f}")

    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"📌 [Epoch {epoch+1}] Best R² 갱신 → 저장 완료 (R²: {val_r2:.4f})")

print(f"\n🎉 학습 완료 | Best Validation R²: {best_val_r2:.4f} | 저장 위치: {SAVE_PATH}")
