# import os
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from sklearn.metrics import r2_score
# from torch.cuda.amp import autocast, GradScaler
# import numpy as np
# from BE.ai.services.cnn_feature_enhanced_seg.datasets.apple_dataset import AppleDataset
# from BE.ai.services.cnn_feature_enhanced_seg.models.fusion_model import FusionModel

# torch.backends.cudnn.benchmark = True

# # === 경로 설정 ===
# train_img_dir = "/home/j-k12e206/ai-hub/Fuji/train/images"
# train_json_dir = "/home/j-k12e206/ai-hub/Fuji/train/jsons"
# valid_img_dir = "/home/j-k12e206/ai-hub/Fuji/valid/images"
# valid_json_dir = "/home/j-k12e206/ai-hub/Fuji/valid/jsons"

# manual_feature_dim = 126
# batch_size = 64
# num_epochs = 20
# lr = 0.001

# def custom_collate(batch):
#     batch = [b for b in batch if b is not None]
#     return torch.utils.data.dataloader.default_collate(batch)

# # === 파일 리스트
# train_json_files = [os.path.join(train_json_dir, f) for f in os.listdir(train_json_dir) if f.endswith('.json')]
# valid_json_files = [os.path.join(valid_json_dir, f) for f in os.listdir(valid_json_dir) if f.endswith('.json')]

# print(f"✅ train json_files 개수: {len(train_json_files)}개")
# print(f"✅ valid json_files 개수: {len(valid_json_files)}개")

# # === transform
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # === 데이터셋
# train_dataset = AppleDataset(train_img_dir, train_json_files, transform=transform)
# val_dataset = AppleDataset(valid_img_dir, valid_json_files, transform=transform)

# # === DataLoader
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                           num_workers=32, pin_memory=True, collate_fn=custom_collate)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=32, pin_memory=True, collate_fn=custom_collate)

# # === 모델 & 손실 & 옵티마이저
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FusionModel(manual_feature_dim).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scaler = GradScaler()

# SAVE_DIR = "/home/j-k12e206/jmk/S12P31E206/BE/ai/services/cnn_feature_enhanced_seg/me/checkpoints"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # === 학습 루프
# best_val_loss = float('inf')
# best_val_r2 = -float('inf')

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     all_preds = []
#     all_labels = []

#     train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
#     for images, manual_features, labels in train_bar:
#         images = images.to(device)
#         manual_features = manual_features.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         with autocast():
#             outputs = model(images, manual_features).squeeze()
#             loss = criterion(outputs, labels)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         running_loss += loss.item()
#         all_preds.extend(outputs.detach().cpu().numpy())
#         all_labels.extend(labels.detach().cpu().numpy())

#     train_loss = running_loss / len(train_loader)
#     train_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))

#     # === 검증
#     model.eval()
#     val_loss = 0.0
#     val_preds = []
#     val_labels = []
#     with torch.no_grad():
#         for images, manual_features, labels in val_loader:
#             images = images.to(device)
#             manual_features = manual_features.to(device)
#             labels = labels.to(device)

#             outputs = model(images, manual_features).squeeze()
#             loss = criterion(outputs, labels)

#             val_loss += loss.item()
#             val_preds.extend(outputs.cpu().numpy())
#             val_labels.extend(labels.cpu().numpy())

#     val_loss /= len(val_loader)
#     val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_labels)))
#     val_r2 = r2_score(val_labels, val_preds)

#     # === 모델 저장
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_val_loss.pth"))
#     if val_r2 > best_val_r2:
#         best_val_r2 = val_r2
#         torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_val_r2.pth"))

#     print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} MAE: {train_mae:.4f} | "
#           f"Val Loss: {val_loss:.4f} Val MAE: {val_mae:.4f} Val R²: {val_r2:.4f}")

# print(f"\n✅ 학습 종료 → Best Val Loss: {best_val_loss:.4f} | Best Val R²: {best_val_r2:.4f}")




# 로컬용(미완)
# import os
# import json
# import random
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from sklearn.metrics import r2_score
# from torch.cuda.amp import autocast, GradScaler
# from BE.ai.services.cnn_feature_enhanced_seg.datasets.apple_dataset import AppleDataset
# from BE.ai.services.cnn_feature_enhanced_seg.models.fusion_model import FusionModel

# # === 설정 ===
# torch.backends.cudnn.benchmark = True
# manual_feature_dim = 126  # ✅ 확장된 feature 수
# batch_size = 64
# num_epochs = 20
# lr = 0.001

# # === 경로 설정 ===
# train_img_dir = "/home/j-k12e206/ai-hub/Fuji/train/images"
# train_json_dir = "/home/j-k12e206/ai-hub/Fuji/train/jsons"
# valid_img_dir = "/home/j-k12e206/ai-hub/Fuji/valid/images"
# valid_json_dir = "/home/j-k12e206/ai-hub/Fuji/valid/jsons"

# # === JSON 파일 리스트 ===
# train_json_files = [os.path.join(train_json_dir, f) for f in os.listdir(train_json_dir) if f.endswith('.json')]
# valid_json_files = [os.path.join(valid_json_dir, f) for f in os.listdir(valid_json_dir) if f.endswith('.json')]

# # ✅ 로컬 테스트용 1000장 샘플링
# train_json_files = random.sample(train_json_files, min(1000, len(train_json_files)))

# print(f"✅ train json_files 개수: {len(train_json_files)}개")
# print(f"✅ valid json_files 개수: {len(valid_json_files)}개")

# # === Transform 정의 ===
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # === Dataset 로딩 ===
# train_dataset = AppleDataset(train_img_dir, train_json_files, transform=transform)
# val_dataset = AppleDataset(valid_img_dir, valid_json_files, transform=transform)

# def custom_collate(batch):
#     batch = [b for b in batch if b is not None]
#     return torch.utils.data.dataloader.default_collate(batch)

# # === DataLoader 정의 ===
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                           num_workers=8, pin_memory=True, collate_fn=custom_collate)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=8, pin_memory=True, collate_fn=custom_collate)

# # === 모델 & 학습 구성 ===
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FusionModel(manual_feature_dim).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scaler = GradScaler()

# SAVE_DIR = "/home/j-k12e206/jmk/S12P31E206/BE/ai/services/cnn_feature_enhanced_seg/me/checkpoints"
# os.makedirs(SAVE_DIR, exist_ok=True)

# best_val_loss = float('inf')
# best_val_r2 = -float('inf')

# # === 학습 루프 시작 ===
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     all_preds = []
#     all_labels = []

#     train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
#     for images, manual_features, labels in train_bar:
#         images = images.to(device)
#         manual_features = manual_features.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         with autocast():
#             outputs = model(images, manual_features).squeeze()
#             loss = criterion(outputs, labels)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         running_loss += loss.item()
#         all_preds.extend(outputs.detach().cpu().numpy())
#         all_labels.extend(labels.detach().cpu().numpy())

#     train_loss = running_loss / len(train_loader)
#     train_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))

#     # === 검증 ===
#     model.eval()
#     val_loss = 0.0
#     val_preds = []
#     val_labels = []
#     with torch.no_grad():
#         for images, manual_features, labels in val_loader:
#             images = images.to(device)
#             manual_features = manual_features.to(device)
#             labels = labels.to(device)

#             outputs = model(images, manual_features).squeeze()
#             loss = criterion(outputs, labels)

#             val_loss += loss.item()
#             val_preds.extend(outputs.cpu().numpy())
#             val_labels.extend(labels.cpu().numpy())

#     val_loss /= len(val_loader)
#     val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_labels)))
#     val_r2 = r2_score(val_labels, val_preds)

#     # === 체크포인트 저장 ===
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_val_loss.pth"))
#     if val_r2 > best_val_r2:
#         best_val_r2 = val_r2
#         torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_val_r2.pth"))

#     print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} MAE: {train_mae:.4f} | "
#           f"Val Loss: {val_loss:.4f} Val MAE: {val_mae:.4f} Val R²: {val_r2:.4f}")

# print(f"\n✅ 학습 종료 → Best Val Loss: {best_val_loss:.4f} | Best Val R²: {best_val_r2:.4f}")


# 로컬용 최종
# import os
# import json
# import random
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from sklearn.metrics import r2_score
# from torch.cuda.amp import autocast, GradScaler
# from BE.ai.services.cnn_feature_enhanced_seg.datasets.apple_dataset import AppleDataset
# from BE.ai.services.cnn_feature_enhanced_seg.models.fusion_model import FusionModel

# # === 로컬 데이터 경로 ===
# local_img_dir = r"C:\Users\SSAFY\Downloads\146.전북 장수 사과 당도 품질 데이터\01.데이터\1.Training\원천데이터\후지1\당도A등급"
# local_json_dir = r"C:\Users\SSAFY\Downloads\146.전북 장수 사과 당도 품질 데이터\01.데이터\1.Training\라벨링데이터_230525_add\후지1\당도A등급"

# # === 공통 파일명 기준 1000개 매칭 ===
# image_files = set(os.path.splitext(f)[0] for f in os.listdir(local_img_dir) if f.endswith('.jpg'))
# json_files = set(os.path.splitext(f)[0] for f in os.listdir(local_json_dir) if f.endswith('.json'))
# common_files = list(image_files & json_files)
# random.shuffle(common_files)
# sampled_files = common_files[:1000]

# train_json_files = [os.path.join(local_json_dir, f + ".json") for f in sampled_files]

# # === 학습 설정 ===
# manual_feature_dim = 126
# batch_size = 16
# num_epochs = 3
# lr = 0.001
# torch.backends.cudnn.benchmark = True

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# train_dataset = AppleDataset(local_img_dir, train_json_files, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                           num_workers=0, collate_fn=lambda b: torch.utils.data.dataloader.default_collate([x for x in b if x]))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FusionModel(manual_feature_dim).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scaler = GradScaler()

# print(f"✅ 학습 샘플 개수: {len(train_dataset)}")

# # === 학습 루프 ===
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     all_preds, all_labels = [], []

#     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
#     for images, manual_features, labels in pbar:
#         images = images.to(device)
#         manual_features = manual_features.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         with autocast():
#             outputs = model(images, manual_features).squeeze()
#             loss = criterion(outputs, labels)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         running_loss += loss.item()
#         all_preds.extend(outputs.detach().cpu().numpy())
#         all_labels.extend(labels.detach().cpu().numpy())

#     mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
#     r2 = r2_score(all_labels, all_preds)
#     print(f"✅ Epoch {epoch+1} | Loss: {running_loss:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

#     # ✅ 학습 종료 후 모델 저장
#     SAVE_PATH = r"C:\Users\SSAFY\Desktop\emodel_result\fusion_model_local.pth"
#     torch.save(model.state_dict(), SAVE_PATH)
#     print(f"\n✅ 학습 종료 → 모델 저장 완료: {SAVE_PATH}")


# ## 로컬용 가장 좋은 값만 저장
# import os
# import json
# import random
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from sklearn.metrics import r2_score
# from torch.cuda.amp import autocast, GradScaler
# from datasets.apple_dataset import AppleDataset
# from models.fusion_model import FusionModel

# # === 로컬 경로 설정 ===
# # === 로컬 경로 설정 ===
# # local_img_dir = r"C:\Users\SSAFY\Downloads\146.전북 장수 사과 당도 품질 데이터\01.데이터\1.Training\원천데이터\후지1\당도A등급"
# # local_json_dir = r"C:\Users\SSAFY\Downloads\146.전북 장수 사과 당도 품질 데이터\01.데이터\1.Training\라벨링데이터_230525_add\후지1\당도A등급"

# # 서버용 경로
# IMG_DIR = r"/home/j-k12e206/ai-hub/Fuji/train/images"
# JSON_DIR = r"/home/j-k12e206/ai-hub/Fuji/train/jsons"


# # === 1000개 랜덤 추출 후 8:2 분리 ===
# image_files = set(os.path.splitext(f)[0] for f in os.listdir(local_img_dir) if f.endswith('.jpg'))
# json_files = set(os.path.splitext(f)[0] for f in os.listdir(local_json_dir) if f.endswith('.json'))
# common_files = list(image_files & json_files)
# random.shuffle(common_files)
# sampled_files = common_files[:1000]

# split_idx = int(0.8 * len(sampled_files))
# train_files = sampled_files[:split_idx]
# val_files = sampled_files[split_idx:]

# train_json_files = [os.path.join(local_json_dir, f + ".json") for f in train_files]
# val_json_files = [os.path.join(local_json_dir, f + ".json") for f in val_files]


# # === 학습 설정 ===
# manual_feature_dim = 126
# batch_size = 32
# num_epochs = 20
# lr = 0.001
# # SAVE_PATH = r"C:\Users\SSAFY\Desktop\emodel_result\fusion_model_local.pth"
# #  서버용경로
# SAVE_PATH = r"/home/j-k12e206/jmk/S12P31E206/BE/ai/services/cnn_feature_enhanced_seg/me/checkpoints/fusion_model.pth"

# # === 전처리 ===
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # === 데이터셋/로더 ===
# train_dataset = AppleDataset(local_img_dir, train_json_files, transform=transform)
# val_dataset = AppleDataset(local_img_dir, val_json_files, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                           num_workers=0, collate_fn=lambda b: torch.utils.data.dataloader.default_collate([x for x in b if x]))
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=0, collate_fn=lambda b: torch.utils.data.dataloader.default_collate([x for x in b if x]))

# # === 모델 및 학습 준비 ===
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FusionModel(manual_feature_dim).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scaler = GradScaler()
# best_val_r2 = -float('inf')

# print(f"✅ 학습 샘플 수: {len(train_dataset)} | 검증 샘플 수: {len(val_dataset)}")

# # === 학습 루프 ===
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     all_preds, all_labels = [], []

#     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
#     for images, manual_features, labels in pbar:
#         images = images.to(device)
#         manual_features = manual_features.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         with autocast():
#             outputs = model(images, manual_features).squeeze()
#             loss = criterion(outputs, labels)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         running_loss += loss.item()
#         all_preds.extend(outputs.detach().cpu().numpy())
#         all_labels.extend(labels.detach().cpu().numpy())

#     train_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
#     print(f"✅ Epoch {epoch+1} | Train Loss: {running_loss:.4f} | MAE: {train_mae:.4f}")

#     # === 검증 ===
#     model.eval()
#     val_preds, val_labels = [], []
#     with torch.no_grad():
#         for images, manual_features, labels in val_loader:
#             images = images.to(device)
#             manual_features = manual_features.to(device)
#             labels = labels.to(device)

#             outputs = model(images, manual_features).squeeze()
#             val_preds.extend(outputs.cpu().numpy())
#             val_labels.extend(labels.cpu().numpy())

#     val_r2 = r2_score(val_labels, val_preds)
#     val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_labels)))
#     print(f"🔍 Validation → MAE: {val_mae:.4f} | R²: {val_r2:.4f}")

#     # === 가장 높은 R² 기준 저장 ===
#     if val_r2 > best_val_r2:
#         best_val_r2 = val_r2
#         torch.save(model.state_dict(), SAVE_PATH)
#         print(f"📌 [Epoch {epoch+1}] Best R² 갱신 → 저장 완료 (R²: {val_r2:.4f})")

# print(f"\n🎉 학습 완료 | Best Validation R²: {best_val_r2:.4f} | 저장 위치: {SAVE_PATH}")

## 서버용 학습 및 검증 데이터 분리 학습 코드
# import os
# import json
# import random
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from sklearn.metrics import r2_score
# from torch.cuda.amp import autocast, GradScaler
# from datasets.apple_dataset import AppleDataset
# from models.fusion_model import FusionModel

# # === 서버 경로 설정 ===
# train_img_dir = "/home/j-k12e206/ai-hub/Fuji/train/images"
# train_json_dir = "/home/j-k12e206/ai-hub/Fuji/train/jsons"
# valid_img_dir = "/home/j-k12e206/ai-hub/Fuji/valid/images"
# valid_json_dir = "/home/j-k12e206/ai-hub/Fuji/valid/jsons"

# # === JSON 파일 리스트 ===
# train_json_files = [os.path.join(train_json_dir, f) for f in os.listdir(train_json_dir) if f.endswith('.json')]
# val_json_files = [os.path.join(valid_json_dir, f) for f in os.listdir(valid_json_dir) if f.endswith('.json')]

# # === 학습 설정 ===
# manual_feature_dim = 126
# batch_size = 32
# num_epochs = 20
# lr = 0.001
# SAVE_PATH = "/home/j-k12e206/jmk/S12P31E206/BE/ai/services/cnn_feature_enhanced_seg/me/checkpoints/fusion_model.pth"

# # === 전처리 ===
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # === 데이터셋/로더 ===
# train_dataset = AppleDataset(train_img_dir, train_json_files, transform=transform)
# val_dataset = AppleDataset(valid_img_dir, val_json_files, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                           num_workers=0, collate_fn=lambda b: torch.utils.data.dataloader.default_collate([x for x in b if x]))
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=0, collate_fn=lambda b: torch.utils.data.dataloader.default_collate([x for x in b if x]))

# # === 모델 및 학습 준비 ===
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FusionModel(manual_feature_dim).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scaler = GradScaler()
# best_val_r2 = -float('inf')

# print(f"✅ 학습 샘플 수: {len(train_dataset)} | 검증 샘플 수: {len(val_dataset)}")

# # === 학습 루프 ===
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     all_preds, all_labels = [], []

#     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
#     for images, manual_features, labels in pbar:
#         images = images.to(device)
#         manual_features = manual_features.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         with autocast():
#             outputs = model(images, manual_features).squeeze()
#             loss = criterion(outputs, labels)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         running_loss += loss.item()
#         all_preds.extend(outputs.detach().cpu().numpy())
#         all_labels.extend(labels.detach().cpu().numpy())

#     train_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
#     print(f"✅ Epoch {epoch+1} | Train Loss: {running_loss:.4f} | MAE: {train_mae:.4f}")

#     # === 검증 ===
#     model.eval()
#     val_preds, val_labels = [], []
#     with torch.no_grad():
#         for images, manual_features, labels in val_loader:
#             images = images.to(device)
#             manual_features = manual_features.to(device)
#             labels = labels.to(device)

#             outputs = model(images, manual_features).squeeze()
#             val_preds.extend(outputs.cpu().numpy())
#             val_labels.extend(labels.cpu().numpy())

#     val_r2 = r2_score(val_labels, val_preds)
#     val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_labels)))
#     print(f"🔍 Validation → MAE: {val_mae:.4f} | R²: {val_r2:.4f}")

#     # === 가장 높은 R² 기준 저장 ===
#     if val_r2 > best_val_r2:
#         best_val_r2 = val_r2
#         torch.save(model.state_dict(), SAVE_PATH)
#         print(f"📌 [Epoch {epoch+1}] Best R² 갱신 → 저장 완료 (R²: {val_r2:.4f})")

# print(f"\n🎉 학습 완료 | Best Validation R²: {best_val_r2:.4f} | 저장 위치: {SAVE_PATH}")


import os
import cv2
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
from models.fusion_model import FusionModel

# ✅ 경로 설정
IMG_DIR = "/home/j-k12e206/ai-hub/Fuji/train/images"
JSON_DIR = "/home/j-k12e206/ai-hub/Fuji/train/jsons"
SAVE_DIR = "/home/j-k12e206/jmk/S12P31E206/BE/ai/services/model_jmk2/meme/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 학습 설정
manual_feature_dim = 6
batch_size = 64
num_epochs = 20
lr = 0.001

# ✅ transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ feature 추출 함수 (속도 최적화)
def extract_features(image, mask):
    x, y, w, h = cv2.boundingRect(mask)
    roi = image[y:y+h, x:x+w]
    roi = cv2.resize(roi, (64, 64))

    R, G, B = roi[:, :, 2], roi[:, :, 1], roi[:, :, 0]
    sum_RGB = R + G + B + 1e-5
    Rn = np.mean(R / sum_RGB)
    C = np.mean(1 - R / 255.0)

    YCbCr = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    Cb, Cr = YCbCr[:, :, 1], YCbCr[:, :, 2]
    ycbcr_diff = np.mean(Cb) - np.mean(Cr)
    ycbcr_norm = np.mean(Cb) / (np.mean(Cb) + np.mean(Cr) + 1e-5)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])

    return np.array([Rn, C, ycbcr_diff, ycbcr_norm, h_mean, s_mean], dtype=np.float32)

# ✅ custom dataset
class AppleDataset(Dataset):
    def __init__(self, image_dir, json_files, transform=None):
        self.image_dir = image_dir
        self.json_files = json_files
        self.transform = transform

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_name = os.path.splitext(os.path.basename(json_path))[0] + ".jpg"
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            return None

        points = np.array(data['annotations']['segmentation']).reshape((-1, 2)).astype(np.int32)
        mask = np.zeros((data['images']['img_height'], data['images']['img_width']), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        manual_feats = extract_features(image, mask)
        manual_feats = torch.tensor(manual_feats, dtype=torch.float32).to(device)

        image_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = transforms.ToPILImage()(image_pil)
        if self.transform:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = transforms.ToTensor()(image_pil)

        label = float(data['collection'].get('sugar_content_nir', 0))
        return image_tensor.to(device), manual_feats, torch.tensor(label, dtype=torch.float32).to(device)

# ✅ collate_fn
def custom_collate(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch)

# ✅ 데이터 준비
json_files = [os.path.join(JSON_DIR, f) for f in os.listdir(JSON_DIR) if f.endswith('.json')]
dataset = AppleDataset(IMG_DIR, json_files, transform=transform)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, collate_fn=custom_collate)

# ✅ 모델
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
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

    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for images, feats, labels in val_loader:
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
