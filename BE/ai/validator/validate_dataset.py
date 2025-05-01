import os
import json
import cv2
import numpy as np
import random
import csv
from pathlib import Path
import matplotlib.pyplot as plt


# bbox = [x, y, width, height] 형태
# 이미지 내 유효한 사각형인지 확인
def validate_bbox(bbox, width, height):
    x, y, w, h = bbox
    return (
        0 <= x < width
        and 0 <= y < height
        and x + w <= width
        and y + h <= height
        and w > 0
        and h > 0
    )


# 최소한 3개의 점(6개의 숫자)이 있어야 polygon 형성 가능
# 각 좌표가 이미지 내부에 있는지 확인
def validate_segmentation(seg, width, height):
    if len(seg) < 6:  # at least 3 points
        return False
    for px, py in zip(seg[::2], seg[1::2]):
        if not (0 <= px < width and 0 <= py < height):
            return False
    return True


def visualize(img_path, bbox, seg_points, save_path=None):
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[ERROR] 이미지 로딩 실패: {img_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis_img = image_rgb.copy()
    # draw bbox
    x, y, w, h = bbox
    cv2.rectangle(vis_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    # draw segmentation
    if seg_points:
        pts = [np.array(seg_points, dtype=np.int32)]
        cv2.polylines(vis_img, pts, isClosed=True, color=(255, 0, 0), thickness=2)
    if save_path:
        cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    else:
        plt.imshow(vis_img)
        plt.axis("off")
        plt.show()


def validate_dataset(
    img_dir, json_dir, random_check=0, save_bad_dir=None, save_csv=None
):
    img_dir = Path(img_dir)
    json_dir = Path(json_dir)
    errors = []

    matched_files = [
        f.stem
        for f in json_dir.glob("*.json")
        if (img_dir / (f.stem + ".jpg")).exists()
    ]

    if random_check > 0:
        matched_files = random.sample(
            matched_files, min(random_check, len(matched_files))
        )

    for name in matched_files:
        json_path = json_dir / f"{name}.json"
        img_path = img_dir / f"{name}.jpg"

        with open(json_path, "r") as f:
            data = json.load(f)

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[ERROR] 이미지 없음: {img_path}")
            continue
        height, width = image.shape[:2]

        bbox = data["annotations"]["bbox"]
        seg = data["annotations"]["segmentation"]
        seg_points = list(zip(seg[::2], seg[1::2]))

        valid_bbox = validate_bbox(bbox, width, height)
        valid_seg = validate_segmentation(seg, width, height)

        if not valid_bbox or not valid_seg:
            error_type = []
            if not valid_bbox:
                error_type.append("bbox")
            if not valid_seg:
                error_type.append("seg")
            errors.append([name, ",".join(error_type)])

            if save_bad_dir:
                save_path = Path(save_bad_dir) / f"{name}.jpg"
                visualize(img_path, bbox, seg_points, save_path)

        elif random_check > 0:
            # 시각화만
            visualize(img_path, bbox, seg_points)

    if save_csv and errors:
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "error_type"])
            writer.writerows(errors)

    print(f"총 파일 수: {len(matched_files)}, 오류 파일 수: {len(errors)}")
    if save_csv:
        print(f"오류 로그 저장: {save_csv}")
    if save_bad_dir:
        print(f"오류 이미지 저장: {save_bad_dir}")


# 예시 실행 (사용자 환경에 맞게 경로 수정)
# validate_dataset("./images", "./jsons", random_check=10, save_bad_dir="./bad_cases", save_csv="error_log.csv")
