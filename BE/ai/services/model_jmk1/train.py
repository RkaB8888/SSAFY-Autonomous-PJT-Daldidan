# train.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset.apple_dataset import AppleDataset
from models.cnn_model import AppleSugarRegressor
from utils import calculate_mae

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

    #버전1
    for images, sugars in dataloader:
        images, sugars = images.to(device), sugars.to(device)

        optimizer.zero_grad()
        outputs = model(images)
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
