# train.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset.apple_dataset import AppleDataset
from models.cnn_model import AppleSugarRegressor
from utils import calculate_mae
from tqdm import tqdm

# 버전 선택
fuji_version = ""    # 후지1, 후지2, 후지3, 후지4 중 선택
grade_version = ""  # 당도A등급, 당도B등급, 당도C등급 중 선택

# 경로 설정 (실제 경로로 수정)
IMG_DIR = r""
JSON_DIR = r""

# Dataset & DataLoader
dataset = AppleDataset(IMG_DIR, JSON_DIR)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
model = AppleSugarRegressor()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")
model.to(device)

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    preds = []
    targets = []

    # #버전1
    # for images, sugars in dataloader:
    #     images, sugars = images.to(device), sugars.to(device)

    #     optimizer.zero_grad()
    #     outputs = model(images)
    #     loss = criterion(outputs, sugars)
    #     loss.backward()
    #     optimizer.step()

    #     epoch_loss += loss.item()
    #     preds.extend(outputs.detach().cpu().numpy())
    #     targets.extend(sugars.cpu().numpy())
    # #---------------------------------------------------------
        #버전2
    for images, features, sugars  in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", total=len(dataloader)):  # ✅ 3개 받아야 함
        images, features, sugars = images.to(device), features.to(device), sugars.to(device)

        optimizer.zero_grad()
        outputs = model(images, features)  # ✅ model에 features도 같이 전달
        loss = criterion(outputs, sugars)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds.extend(outputs.detach().cpu().numpy())
        targets.extend(sugars.cpu().numpy())
    #---------------------------------------------------------

    mae = calculate_mae(preds, targets)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss/len(dataloader):.4f} MAE: {mae:.4f}")

torch.save(model.state_dict(), "apple_model.pth")
