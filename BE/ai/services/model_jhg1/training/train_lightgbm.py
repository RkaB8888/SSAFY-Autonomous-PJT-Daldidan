# services/model_jhg1/training/train_lightgbm.py
import json
import joblib
from pathlib import Path

import lightgbm as lgb
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold

from common_utils.image_cropper import crop_bbox_from_json
from services.model_jhg1.utils.feature_extractors import extract_features


def load_dataset(images_dir: Path, jsons_dir: Path):
    X, y = [], []
    feature_names: List[str] = None  # 한번만 채워둘 리스트

    image_files = sorted(images_dir.glob("*.jpg"))
    for image_path in tqdm(image_files, desc="Extracting features"):
        json_path = jsons_dir / (image_path.stem + ".json")
        if not json_path.exists():
            continue

        try:
            crop_img, _ = crop_bbox_from_json(image_path, json_path)
            if crop_img is None:
                tqdm.write(f"[무시] 손상된 이미지: {image_path.name}")
                continue

            feats, names = extract_features(crop_img)
            # 첫 번째 정상 샘플에서만 피처 이름을 캡처해 둡니다.
            if feature_names is None:
                feature_names = names

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            collection = data.get("collection", {})
            sugar = collection.get("sugar_content")
            if sugar is None:
                sugar = collection.get("sugar_content_nir")

            if sugar is None:
                tqdm.write(f"[무시] 당도 정보 없음: {image_path.name}")
                continue

            X.append(feats)
            y.append(sugar)

        except Exception as e:
            tqdm.write(f"[Error] {image_path.name}: {e}")

    print(f"✅ 유효 샘플 수: {len(X)} / 전체: {len(image_files)}")
    return np.array(X), np.array(y), feature_names


def train_lightgbm(
    X: np.ndarray, y: np.ndarray, feature_names: List[str], save_path: Path
):
    # -- 상수 피처 제거 --
    selector = VarianceThreshold(threshold=0.0)
    X_reduced = selector.fit_transform(X)

    # -- 학습 --
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        device="gpu",  # <== GPU 사용 설정
        gpu_use_dp=True,  # (선택) 더 안정적인 처리
        random_state=42,
    )
    model.fit(X_reduced, y)

    # -- 모델 + selector + 원본 feature_names 한꺼번에 덤프 --
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "selector": selector, "feature_names": feature_names},
        save_path,
    )
    print(f"✅ LightGBM + meta 저장 완료: {save_path}")


def main():
    images_dir = Path("dataset/images")
    jsons_dir = Path("dataset/jsons")
    save_path = Path("services/model_jhg1/weights/lightgbm_model.pkl")

    X, y, feature_names = load_dataset(images_dir, jsons_dir)
    print(f"▶ 샘플: {X.shape}, y 분포: {np.unique(y)}")
    train_lightgbm(X, y, feature_names, save_path)


if __name__ == "__main__":
    main()
