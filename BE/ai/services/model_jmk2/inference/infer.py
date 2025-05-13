import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from models.fusion_model import FusionModel
from features.extract_features import extract_features  # 기존 feature extractor 함수
import warnings

warnings.filterwarnings("ignore")

# === 경로 설정 ===
test_folder = r"/home/j-k12e206/ai-hub/Fuji/valid/jsons"
json_folder = r"/home/j-k12e206/ai-hub/Fuji/valid/jsons"
model_path = r"/home/j-k12e206/jmk/S12P31E206/BE/ai/services/model_jmk2/outputs/checkpoints/best_val_loss.pth"  # 저장된 모델 weight

# === transform (학습때와 동일) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# === 모델 불러오기 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manual_feature_dim = 6
model = FusionModel(manual_feature_dim)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === inference loop ===
for json_file in os.listdir(json_folder):
    if not json_file.endswith('.json'):
        continue
    
    json_path = os.path.join(json_folder, json_file)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_file_name = data['images']['img_file_name']
    image_path = os.path.join(test_folder, img_file_name)

    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일 없음: {image_path}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        continue

    # === segmentation → mask 생성 ===
    points = np.array(data['annotations']['segmentation']).reshape((-1, 2)).astype(np.int32)
    img_h = data['images']['img_height']
    img_w = data['images']['img_width']
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    # === manual feature 추출 ===
    manual_features = extract_features(image, mask)  # numpy array (shape: (6,))
    manual_features = torch.tensor(manual_features, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 6)

    # === image transform ===
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # === inference ===
    with torch.no_grad():
        prediction = model(image_tensor, manual_features).squeeze().cpu().item()

    print(f"✅ {img_file_name} → Predicted Sugar Content: {prediction:.2f} Brix")
