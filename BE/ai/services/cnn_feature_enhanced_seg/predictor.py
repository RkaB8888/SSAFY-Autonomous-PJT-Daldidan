# # predictor.py

# import cv2
# import torch
# import numpy as np
# from torchvision import transforms
# from PIL import Image
# from model_loader import load_model, load_scaler
# from features.extract_features import extract_features

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 전처리
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # 예측 함수
# def predict(image_path: str, json_path: str) -> float:
#     # 이미지 로딩
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"이미지 로딩 실패: {image_path}")

#     # JSON에서 segmentation 읽기
#     import json
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     img_h = data['images']['img_height']
#     img_w = data['images']['img_width']
#     points = np.array(data['annotations']['segmentation']).reshape((-1, 2)).astype(np.int32)
#     mask = np.zeros((img_h, img_w), dtype=np.uint8)
#     cv2.fillPoly(mask, [points], 255)

#     # 특징 추출 및 스케일링
#     feature = extract_features(image, mask).reshape(1, -1)
#     scaler = load_scaler()
#     scaled_feat = scaler.transform(feature)

#     # 이미지 전처리 (PIL 변환 후 transform 적용)
#     image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     image_tensor = transform(image_pil).unsqueeze(0).to(device)
#     feat_tensor = torch.tensor(scaled_feat, dtype=torch.float32).to(device)

#     # 모델 예측
#     model = load_model(device)
#     with torch.no_grad():
#         pred = model(image_tensor, feat_tensor).item()

#     return pred

# services/cnn_feature_enhanced_seg/predictor.py

import numpy as np
import cv2
import torch
from PIL import Image
from .model_loader import model, scaler, transform
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

        manual_feat = extract_features(img, mask)
        manual_feat = scaler.transform([manual_feat])[0]
        manual_feat_tensor = torch.tensor(manual_feat, dtype=torch.float32).unsqueeze(0).to(device)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        image_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image_tensor, manual_feat_tensor).squeeze().item()

        return round(pred, 2)

    except Exception as e:
        print(f"[ERROR] predict_bytes 실패: {e}")
        raise e
