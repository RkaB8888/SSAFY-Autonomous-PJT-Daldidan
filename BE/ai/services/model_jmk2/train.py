# import torch
# from torch.utils.data import DataLoader
# from models.fusion_model import FusionModel
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import r2_score
# from torch.utils.data import DataLoader, Subset
# import random
# import os
# from datasets.apple_dataset import AppleDataset, custom_collate
# from torch.cuda.amp import autocast, GradScaler


# # === 경로 ===
# IMAGE_DIR = r"/home/j-k12e206/ai-hub/Fuji/train/images"
# # FEATURE_CSV = "manual_features.csv"
# manual_feature_dim = 6
# batch_size = 16
# num_epochs = 20
# lr = 0.001
# JSON_DIR = r"/home/j-k12e206/ai-hub/Fuji/train/jsons"

# json_files = []
# for root, dirs, files in os.walk(JSON_DIR):
#     for file in files:
#         if file.endswith('.json'):
#             json_files.append(os.path.join(root, file))

# random.shuffle(json_files)
# # json_files = json_files[:1000]
# ## 여기랑 아무 상관없음.

# print(f"✅ json_files 개수: {len(json_files)}개")  # 개수 확인용
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
# ])

# dataset = AppleDataset(IMAGE_DIR, json_files, transform=transform)

# val_split = 0.2
# val_size = int(len(dataset) * val_split)
# train_size = len(dataset) - val_size
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True, collate_fn=custom_collate)
# train_loader = tqdm(train_loader, desc="데이터 로딩 진행 상황")
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=custom_collate)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = FusionModel(manual_feature_dim).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)



# def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
#     best_val_loss = float('inf')
#     best_val_r2 = -float('inf')
#     for epoch in range(num_epochs):
#         # === TRAIN ===
#         model.train()
#         running_loss = 0.0
#         all_preds = []
#         all_labels = []
        
#         train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        
#         scaler = GradScaler()

#         for images, manual_features, labels in train_loader:
#             images = images.to(device)
#             manual_features = manual_features.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()
#             with autocast():
#                 # 이미지 입력이 빠지니까 → model 입력: manual_features만
#                 outputs = model(images, manual_features).squeeze()
#                 # None 대신 dummy or 빼기
#                 loss = criterion(outputs, labels)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             running_loss += loss.item()
#             all_preds.extend(outputs.detach().cpu().numpy())
#             all_labels.extend(labels.detach().cpu().numpy())

#         train_loss = running_loss / len(train_loader)
#         train_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))

#         # === VALIDATION ===
#         model.eval()
#         val_loss = 0.0
#         val_preds = []
#         val_labels = []

#         with torch.no_grad():
#             for images, manual_features, labels in val_loader:
#                 images = images.to(device)
#                 manual_features = manual_features.to(device)
#                 labels = labels.to(device)

#                 outputs = model(images, manual_features).squeeze()

#                 loss = criterion(outputs, labels)

#                        # === 👇 print debug 추가 (여기!) ===
#                 print(f"[DEBUG] outputs: {outputs.detach().cpu().numpy()}")
#                 print(f"[DEBUG] labels: {labels.detach().cpu().numpy()}")

#                 val_loss += loss.item()
#                 val_preds.extend(outputs.cpu().numpy())
#                 val_labels.extend(labels.cpu().numpy())

#         val_loss /= len(val_loader)
#         val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_labels)))
#         val_r2 = r2_score(val_labels, val_preds)

#         SAVE_DIR = "/home/j-k12e206/jmk/S12P31E206/BE/ai/services/model_jmk2"
#         os.makedirs(SAVE_DIR, exist_ok=True)
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_val_loss.pth"))
        
#         if val_r2 > best_val_r2:
#             best_val_r2 = val_r2
#             torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_val_r2.pth"))



#         # === 로그 출력 ===
#         print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} MAE: {train_mae:.4f} | Val Loss: {val_loss:.4f} Val MAE: {val_mae:.4f} Val R²: {val_r2:.4f}")
#     print(f"\n✅ 학습 종료 → Best Val Loss: {best_val_loss:.4f} | Best Val R²: {best_val_r2:.4f}")
#         # === 모델 저장 ===
#         # torch.save(model.state_dict(), f"outputs/checkpoints/model_epoch{epoch+1}.pth")

# train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs)
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import r2_score
from torch.cuda.amp import autocast, GradScaler

from datasets.apple_dataset import AppleDataset
from models.fusion_model import FusionModel
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True
# === 경로 설정 ===
IMG_DIR = r"/home/j-k12e206/ai-hub/Fuji/train/images"
JSON_DIR = r"/home/j-k12e206/ai-hub/Fuji/train/jsons"
manual_feature_dim = 6
batch_size = 64
num_epochs = 20
lr = 0.001

json_files = [os.path.join(JSON_DIR, f) for f in os.listdir(JSON_DIR) if f.endswith('.json')]
print(f"✅ json_files 개수: {len(json_files)}개")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

dataset = AppleDataset(IMG_DIR, json_files, transform=transform)
val_split = 0.2
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionModel(manual_feature_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scaler = GradScaler()

torch.backends.cudnn.benchmark = True

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

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

        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
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

        SAVE_DIR = r"/home/j-k12e206/jmk/S12P31E206/BE/ai/services/model_jmk2/meme/checkpoints"
        os.makedirs(SAVE_DIR, exist_ok=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_val_loss.pth"))

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_val_r2.pth"))

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} MAE: {train_mae:.4f} | Val Loss: {val_loss:.4f} Val MAE: {val_mae:.4f} Val R²: {val_r2:.4f}")

    print(f"\n✅ 학습 종료 → Best Val Loss: {best_val_loss:.4f} | Best Val R²: {best_val_r2:.4f}")

train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs)
