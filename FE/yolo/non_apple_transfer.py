import os
from shutil import copy2

# 원본 비사과 이미지 폴더
non_apple_image_dir = "./non_apple_raw/"
# YOLO 학습용 폴더
yolo_image_dir = "./images/train"
yolo_label_dir = "./labels/train"

# 폴더 생성
os.makedirs(yolo_image_dir, exist_ok=True)
os.makedirs(yolo_label_dir, exist_ok=True)

# 이미지 파일 불러오기 (JPG 기준)
image_files = sorted([f for f in os.listdir(non_apple_image_dir) if f.endswith(".jpg")])[:1000]

# 시작 인덱스는 apple 파일 이후 이어서 (예: apple501부터 시작)
start_index = 501  # 조정 가능

for i, img_file in enumerate(image_files, start=start_index):
    new_name = f"notapple{i:03d}"
    new_img_name = new_name + ".jpg"
    new_txt_name = new_name + ".txt"

    # 이미지 복사
    src_img_path = os.path.join(non_apple_image_dir, img_file)
    dst_img_path = os.path.join(yolo_image_dir, new_img_name)
    copy2(src_img_path, dst_img_path)

    # 빈 라벨(txt) 생성
    txt_path = os.path.join(yolo_label_dir, new_txt_name)
    with open(txt_path, "w") as f:
        pass  # 내용 없음
