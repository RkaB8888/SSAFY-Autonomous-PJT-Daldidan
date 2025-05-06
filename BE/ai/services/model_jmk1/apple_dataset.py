import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from utils import extract_color_features, extract_texture_features
import numpy as np
import random

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
            img_path = os.path.join(img_root, 'A', img_filename)  # A 폴더 직접 지정

            print(f"검사 img_path: {img_path} → 존재 여부: {os.path.exists(img_path)}")

            if os.path.exists(img_path):
                self.samples.append((img_path, json_path))
                print(f"✅ 매칭 성공 → {img_path}")
            else:
                print(f"⚠️ 매칭 실패 → {img_filename}")
        # -------------------------------

        print(f"\n총 {len(self.samples)}개의 데이터 로드 완료 (이미지 + json 매칭된 것만).")

        # 추가: 처음 ?000개만 사용
        if len(self.samples) > 2000:
            random.shuffle(self.samples) 

            self.samples = random.sample(self.samples, 2000)
            print(f"✅ 임시로 {len(self.samples)}만 샘플로 사용 (총 {len(self.samples)}개)")

    def __len__(self):
        return len(self.samples)

    #이미지만CNN에 넣고 sugar을 반환
    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            sugar = json_data['collection']['sugar_content_nir']  # JSON key 확인 필요

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([sugar], dtype=torch.float32)


