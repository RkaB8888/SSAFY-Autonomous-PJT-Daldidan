import torch
import cv2
import numpy as np
from torchvision import transforms
from models.fusion_model import FusionModel
from features.extract_fast import extract_fast_features
import joblib
import time

# 경로 설정
MODEL_PATH = "services/cnn_feature_seg/meme/checkpoints/best_val_r2.pth"
SCALER_PATH = "services/cnn_feature_seg/meme/scaler.pkl"

# 모델 및 장치
manual_feature_dim = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FusionModel(manual_feature_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Scaler
scaler = joblib.load(SCALER_PATH)

# transform (OpenCV 이미지 기준)
transform = transforms.Compose([
    transforms.ToPILImage(),  # ✅ OpenCV → PIL 변환 한 번만
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def predict_sugar(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"이미지 로드 실패: {img_path}")

    h, w = image.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255  # 전체 마스크

    manual_feat = extract_fast_features(image, mask)
    manual_feat = scaler.transform([manual_feat])[0]
    manual_feat_tensor = torch.tensor(manual_feat, dtype=torch.float32).unsqueeze(0).to(device)

    image_tensor = transform(image).unsqueeze(0).to(device)  # OpenCV → PIL → Tensor

    start = time.time()
    with torch.no_grad():
        output = model(image_tensor, manual_feat_tensor).squeeze().item()
    end = time.time()

    print(f"🔥 추론 시간: {end - start:.2f}초")
    return round(output, 2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    args = parser.parse_args()

    result = predict_sugar(args.img_path)
    print(f"🍎 예측된 당도: {result:.2f} Brix")
