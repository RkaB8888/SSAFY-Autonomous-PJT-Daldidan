# predict.py

#버전1
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from cnn_model import AppleSugarRegressor
from utils import extract_color_features, extract_texture_features
import os
import glob
import re
from pathlib import Path


# 모델 경로 탐색
SAVE_DIR = "/home/j-k12e206/jmk/S12P31E206/BE/ai/services/model_jmk1"
BASENAME = "apple_model"
EXTENSION = ".pth"

# 폴더 안에 기존 파일 검색
existing_files = os.listdir(SAVE_DIR)

# apple_model_숫자.pth 패턴 매칭
pattern = re.compile(rf"{BASENAME}_(\d+){EXTENSION}")

max_index = 0
latest_model_file = None
for filename in existing_files:
    match = pattern.match(filename)
    if match:
        index = int(match.group(1))
        if index > max_index:
            max_index = index
            latest_model_file = filename

if latest_model_file is None:
    raise FileNotFoundError("No saved model found in directory.")

MODEL_LOAD_PATH = os.path.join(SAVE_DIR, latest_model_file)
print(f"최신 모델 로드 경로: {MODEL_LOAD_PATH}")


# Load model
model = AppleSugarRegressor()
model.load_state_dict(torch.load(MODEL_LOAD_PATH))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 검증 데이터 경로
VALID_ROOT = Path("/home/j-k12e206/ai-hub/Fuji/valid")
VALID_IMAGES_DIR = VALID_ROOT / "images"

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    image_pil = Image.open(image_path).convert('RGB')
    image_np = np.array(image_pil)

    # feature 추출
    color_feat = extract_color_features(image_np)
    texture_feat = extract_texture_features(image_np)
    combined_feat = np.concatenate([color_feat, texture_feat])

    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    features_tensor = torch.tensor(combined_feat, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor, features_tensor)
    return output.item()

if __name__ == "__main__":
    test_folder = str(VALID_IMAGES_DIR)
    image_paths = glob.glob(os.path.join(test_folder, "*.jpg"))

    if not image_paths:
        print("No jpg files found in test folder.")
    else:
        for img_path in image_paths:
            try:
                prediction = predict_image(img_path)
                print(f"{os.path.basename(img_path)} → Predicted Sugar Content: {prediction:.2f} Brix")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

