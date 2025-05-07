import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from utils import extract_color_features, extract_texture_features
import numpy as np
import random
import cv2

class AppleDataset(Dataset):
    def __init__(self, img_root, json_root, transform=None):
        self.samples = []  # (img_path, json_path)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        print(f"json_root: {json_root}")
        print(f"img_root: {img_root}")

        json_folders = [f for f in os.listdir(json_root) if os.path.isdir(os.path.join(json_root, f))]
        img_folders = [f for f in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, f))]

        print(f"json_folders: {json_folders}")
        print(f"img_folders: {img_folders}")
        print(f"여기까지1")

        # [버전1,2]공통경로.
        for json_file in os.listdir(json_root):
            if not json_file.endswith('.json'):
                continue
            print(f"여기까지2")

            json_path = os.path.join(json_root, json_file)
            img_filename = json_file.replace('.json', '.jpg')
            img_path = os.path.join(img_root, img_filename)

            print(f"검사 img_path: {img_path} → 존재 여부: {os.path.exists(img_path)}")

            if os.path.exists(img_path):
                self.samples.append((img_path, json_path))
                print(f"✅ 매칭 성공 → {img_path}")
            else:
                print(f"⚠️ 매칭 실패 → {img_filename}")
        # -------------------------------

        print(f"\n총 {len(self.samples)}개의 데이터 로드 완료 (이미지 + json 매칭된 것만).")

        # 추가: 처음 ?000개만 사용
        # if len(self.samples) > 2000:
        #     random.shuffle(self.samples) 

        #     self.samples = random.sample(self.samples, 2000)
        #     print(f"✅ 임시로 {len(self.samples)}만 샘플로 사용 (총 {len(self.samples)}개)")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def extract_apple_roi(image_np):
        # HSV 변환
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

        lower_red1 = np.array([0, 50, 20])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 20])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            roi = image_np[y:y+h, x:x+w]
            return roi
        else:
            return image_np  # fallback: ROI 못찾으면 원본

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]
        image_pil = Image.open(img_path).convert('RGB')
        image_np = np.array(image_pil)

        # OpenCV로 사과 ROI 추출
        roi_np = self.extract_apple_roi(image_np)

        # numpy → PIL
        roi_pil = Image.fromarray(roi_np)

        # feature 추출도 ROI로
        color_feat = extract_color_features(roi_np)
        texture_feat = extract_texture_features(roi_np)
        combined_feat = np.concatenate([color_feat, texture_feat])

        # transform 적용
        if self.transform:
            image_tensor = self.transform(roi_pil)
        else:
            image_tensor = transforms.ToTensor()(roi_pil)

        # sugar 값
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            sugar = json_data['collection']['sugar_content_nir']

        return (
            image_tensor,
            torch.tensor(combined_feat, dtype=torch.float32),
            torch.tensor([sugar], dtype=torch.float32)
        )

