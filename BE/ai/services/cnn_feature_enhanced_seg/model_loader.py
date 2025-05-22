import os
import torch
import joblib
from torchvision import transforms
from .models.fusion_model import FusionModel  # 절대경로로 정확히!

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "me", "checkpoints", "best_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "me", "checkpoints", "scaler.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

manual_feature_dim = 126

# 🔽 이 부분 반드시 함수 밖, 전역에 선언되어야 함
model = FusionModel(manual_feature_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

scaler = joblib.load(SCALER_PATH)

transform = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
