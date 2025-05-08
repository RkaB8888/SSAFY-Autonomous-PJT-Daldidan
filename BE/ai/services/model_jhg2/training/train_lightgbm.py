# services/model_jhg2/training/train_lightgbm.py
import json
import joblib
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold

from common_utils.image_cropper import crop_bbox_from_json
from services.model_jhg2.utils.cnn_feature_extractor import (
    extract_batch,
)  # CNN extractor
from services.model_jhg2.config import (
    IMAGES_DIR,
    JSONS_DIR,
    MODEL_SAVE_PATH,
)


# ---------- 🔄 CNN 추출기는 feature 이름이 없으므로 임의 생성 ----------
def _make_feature_names(dim: int) -> List[str]:
    """1280‑차원 벡터 ⇒ ['cnn_0', 'cnn_1', …] 리스트 반환"""
    return [f"cnn_{i}" for i in range(dim)]


def load_dataset(
    images_dir: Path, jsons_dir: Path
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y = [], []
    feature_names: List[str] = None  # 한번만 채워둘 리스트

    image_files = sorted(images_dir.glob("*.jpg"))
    for image_path in tqdm(image_files, desc="Extracting CNN features"):
        json_path = jsons_dir / (image_path.stem + ".json")
        if not json_path.exists():
            continue

        try:
            crop_img, _ = crop_bbox_from_json(image_path, json_path)
            if crop_img is None:
                tqdm.write(f"[무시] 손상된 이미지: {image_path.name}")
                continue

            # ─────── 🔄 CNN 특징 벡터 추출 ───────
            feats = extract_batch(np.expand_dims(crop_img, axis=0))[0]

            # CNN은 고정 길이이므로 한 번만 feature_names 생성
            if feature_names is None:
                feature_names = _make_feature_names(len(feats))

            # ─────── 라벨(당도) 추출 ───────
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            coll = data.get("collection", {})
            sugar = coll.get("sugar_content") or coll.get("sugar_content_nir")

            if sugar is None:
                tqdm.write(f"[무시] 당도 정보 없음: {image_path.name}")
                continue

            X.append(feats)
            y.append(sugar)

        except Exception as e:
            tqdm.write(f"[Error] {image_path.name}: {e}")

    print(f"✅ 유효 샘플 수: {len(X)} / 전체: {len(image_files)}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), feature_names


def train_lightgbm(
    X: np.ndarray, y: np.ndarray, feature_names: List[str], save_path: Path
):
    # ─────── 상수 특성 제거 (CNN 벡터라 거의 변화 없지만 안전 차원) ───────
    selector = VarianceThreshold(threshold=0.0)
    X_reduced = selector.fit_transform(X)

    # ─────── LightGBM 설정 ───────
    model = lgb.LGBMRegressor(
        n_estimators=500,  # 🔄 배치 학습 빠르므로 tree 수↑
        learning_rate=0.05,
        max_depth=-1,  # 🔄 CNN 벡터는 복잡해 depth 제한 해제
        device="gpu",
        gpu_use_dp=True,
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
    X, y, feature_names = load_dataset(IMAGES_DIR, JSONS_DIR)
    print(f"▶ 데이터 형태: {X.shape}, y 분포 (min~max): {y.min():.1f}~{y.max():.1f}")
    train_lightgbm(X, y, feature_names, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
