import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from models.fusion_model import FusionModel
from features.extract_features import extract_features
import joblib

# 경로 설정
MODEL_PATH = "services/model_jmk2/meme/checkpoints/best_val_r2.pth"
SCALER_PATH = "services/model_jmk2/meme/scaler.pkl"

# 모델 준비
manual_feature_dim = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FusionModel(manual_feature_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# scaler
scaler = joblib.load(SCALER_PATH)

# 예측 함수
def predict_sugar(img_path, polygon_or_bbox):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"이미지 로드 실패: {img_path}")

    # 마스크 생성
    if len(polygon_or_bbox) == 4:  # (xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = polygon_or_bbox
        polygon = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], np.int32)
    else:  # polygon 형태
        polygon = np.array(polygon_or_bbox, np.int32).reshape((-1, 2))

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # feature 추출
    manual_feat = extract_features(image, mask)
    manual_feat = scaler.transform([manual_feat])[0]
    manual_feat_tensor = torch.tensor(manual_feat, dtype=torch.float32).unsqueeze(0).to(device)

    # 이미지 전처리
    image_pil = Image.open(img_path).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # 예측
    with torch.no_grad():
        output = model(image_tensor, manual_feat_tensor).squeeze().item()
    return round(output, 2)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--coords", nargs=4, type=int, required=True)  # xmin ymin xmax ymax
    args = parser.parse_args()

    prediction = predict_sugar(args.img_path, args.coords)
    print(f"📣 예측 당도: {prediction:.2f} Brix")