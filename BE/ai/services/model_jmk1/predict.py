# predict.py

#버전1
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from models.cnn_model import AppleSugarRegressor
from utils import extract_color_features, extract_texture_features
import os
import glob

# Load model
model = AppleSugarRegressor()
model.load_state_dict(torch.load("apple_model.pth"))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

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
    test_folder = r""
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

