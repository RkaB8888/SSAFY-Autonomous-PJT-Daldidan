import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
from features.extract_features import extract_features  # manual feature 추출 함수
import random

def custom_collate(batch):
    # batch 안에 None 데이터 제거
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # 모든 batch가 None일 경우 처리
    return torch.utils.data.dataloader.default_collate(batch)

class AppleDataset(Dataset):
    def __init__(self, image_dir, json_files, transform=None):
        self.image_dir = image_dir
        self.json_files = json_files
        self.transform = transform


        # json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]
        # self.json_files = random.sample(json_files, 1000)  # ✅ 처음부터 1000장만 랜덤 선택

        # json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]
        # random.shuffle(json_files)
        # self.json_files = json_files[:1000]  # ✅ 1000개만 선택

    def __len__(self):
        return len(self.json_files)



    def __getitem__(self, idx):
        json_path = self.json_files[idx]

        # json 파일 읽기
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_filename = data['images']['img_file_name']
        img_filename = os.path.splitext(img_filename)[0] + '.jpg'  # ✅ 확장자 수정
        img_path = os.path.join(self.image_dir, img_filename)

        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARNING] Image not found: {img_path}")
            # return None, None, None 대신 dummy tensor 리턴
            dummy_image = torch.zeros(3, 224, 224)  # 224x224 크기
            dummy_manual = torch.zeros(6)           # manual feature 6개
            dummy_label = torch.tensor(0.0)         # 임시 라벨
            return dummy_image, dummy_manual, dummy_label
        




        # segmentation → mask
        points = np.array(data['annotations']['segmentation']).reshape((-1, 2)).astype(np.int32)
        img_h = data['images']['img_height']
        img_w = data['images']['img_width']
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        # manual feature 추출
        manual_features = extract_features(image, mask)  # shape (6,)
        manual_features = torch.tensor(manual_features, dtype=torch.float32)

        # transform 이미지
        image_pil = Image.open(img_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = transforms.ToTensor()(image_pil)

        # label 불러오기 (json에 sugar_grade나 값이 있다면 → 아니면 dummy 0 리턴)
        label = float(data['collection'].get('sugar_content_nir', 0))

        # 10장마다 한 번씩 진행 상황 출력
        if idx % 100 == 0:  # 10장마다 출력
            print(f"[INFO] {idx}/{len(self)}번째 데이터 feature 추출 완료")
        # print(f"[DEBUG] __getitem__ called at index {idx}")  # 이거 추가
        return image_tensor, manual_features, torch.tensor(label, dtype=torch.float32)
