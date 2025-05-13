import numpy as np
import cv2
import torch
from PIL import Image
from .model_loader import model, scaler, transform

# .features.extract_features에서 가져오는 모델 수정
# from .features.extract_features import extract_fast_features
from .features.extract_features import extract_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_bytes(image_bytes: bytes) -> float:
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("이미지 디코딩 실패")

        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # .features.extract_features에서 가져오는 모델 수정
        # manual_feat = extract_fast_features(img, mask)
        manual_feat = extract_features(img, mask)

        manual_feat = scaler.transform([manual_feat])[0]
        manual_feat_tensor = torch.tensor(manual_feat, dtype=torch.float32).unsqueeze(0).to(device)
        image_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image_tensor, manual_feat_tensor).squeeze().item()

        return round(pred, 2)

    except Exception as e:
        print(f"[ERROR] predict_bytes 실패: {e}")
        raise e
