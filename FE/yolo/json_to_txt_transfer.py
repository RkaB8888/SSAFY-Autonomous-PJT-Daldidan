import os
import json
from PIL import Image
from shutil import copy2

# 폴더 경로 설정
json_dir = "./json/"             # JSON 300개 있는 폴더
image_dir = "./raw_img/"      # 원본 이미지 폴더
yolo_image_dir = "./images/train"
yolo_label_dir = "./labels/train"

# YOLO 학습 폴더 구성
os.makedirs(yolo_image_dir, exist_ok=True)
os.makedirs(yolo_label_dir, exist_ok=True)

# JSON 파일 300개만 처리
json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])[:300]

for jf in json_files:
    json_path = os.path.join(json_dir, jf)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 이미지 파일명
    img_name = data['images']['img_file_name']
    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        print(f"이미지 없음: {img_path}")
        continue

    # 이미지 사이즈
    img_w = data['images']['img_width']
    img_h = data['images']['img_height']

    # bbox 추출 및 YOLO 포맷 변환
    x, y, w, h = data['annotations']['bbox']
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h

    # YOLO .txt 파일 저장
    txt_name = img_name.replace(".jpg", ".txt")
    txt_path = os.path.join(yolo_label_dir, txt_name)
    with open(txt_path, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    # 이미지도 YOLO images/train 디렉토리로 복사
    copy2(img_path, os.path.join(yolo_image_dir, img_name))
