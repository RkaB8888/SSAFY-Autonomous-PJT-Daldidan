import json
import joblib
from pathlib import Path

import lightgbm as lgb
import numpy as np
from tqdm import tqdm

from common_utils.image_cropper import crop_bbox_from_json
from services.model_jhg1.utils.feature_extractors import extract_features

"""
기능: 사과 이미지 원본 + JSON에서 crop 이미지 생성 후 특징 추출하여 LightGBM 학습
입력: dataset/images/, dataset/jsons/ 디렉토리 내 파일들
출력: weights/lightgbm_model.txt
"""


def load_dataset(images_dir: Path, jsons_dir: Path):
    X, y = [], []

    image_files = sorted(images_dir.glob("*.jpg"))
    for image_path in tqdm(image_files, desc="Extracting features"):
        json_path = jsons_dir / (image_path.stem + ".json")
        if not json_path.exists():
            continue

        try:
            crop_img, _ = crop_bbox_from_json(image_path, json_path)
            features, _ = extract_features(crop_img)

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sugar = data["annotations"].get("sugar_content")
            # 학습에 사용할 sugar_content는 실험실에서 파괴 측정된 정답 값입니다.
            # sugar_content가 없는 경우 (예: 수확 전 NIR 측정) 학습에서 제외합니다.
            if sugar is None:
                continue

            X.append(features)
            y.append(sugar)

        except Exception as e:
            print(f"[Error] {image_path.name}: {e}")

    print(f"✅ 유효 샘플 수: {len(X)} / 전체 이미지: {len(image_files)}")
    return np.array(X), np.array(y)


def train_lightgbm(X: np.ndarray, y: np.ndarray, save_path: Path):
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42
    )
    model.fit(X, y)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"\n✅ LightGBM 모델 저장 완료: {save_path}")


def main():
    images_dir = Path("dataset/images")
    jsons_dir = Path("dataset/jsons")
    save_path = Path("services/model_jhg1/weights/lightgbm_model.pkl")

    X, y = load_dataset(images_dir, jsons_dir)
    print(f"총 샘플 수: {len(X)} | 특징 차원: {X.shape[1]}")

    train_lightgbm(X, y, save_path)


if __name__ == "__main__":
    main()
