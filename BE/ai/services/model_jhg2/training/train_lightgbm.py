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
from services.model_jhg2.utils.cnn_feature_extractor import extract_batch
from services.model_jhg2.config import (
    IMAGES_DIR,
    JSONS_DIR,
    MODEL_SAVE_PATH,
    CACHE_DIR,
)
from services.model_jhg2.extract_embeddings import build_and_cache_embeddings


def _make_feature_names(dim: int) -> List[str]:
    return [f"cnn_{i}" for i in range(dim)]


def load_dataset(
    images_dir: Path, jsons_dir: Path
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feat_cache = CACHE_DIR / "train_embeddings.npy"
    label_cache = CACHE_DIR / "train_labels.npy"

    # 1) 캐시가 없으면 extract_embeddings 로직 함수 호출
    if not (feat_cache.exists() and label_cache.exists()):
        print("🚀 캐시가 없으므로 build_and_cache_embeddings() 를 실행합니다…")
        build_and_cache_embeddings(img_dir=images_dir, json_dir=jsons_dir)
        print("✅ 임베딩 캐시 생성 완료.")

    # 2) 캐시에서 바로 로드
    X = np.memmap(feat_cache, dtype=np.float32, mode="r").reshape(-1, 1280)
    y = np.memmap(label_cache, dtype=np.float32, mode="r", shape=(X.shape[0],))

    print(f"✅ Loaded cached train set: {len(X)} samples")

    # feature_names는 CNN 차원에 맞춰 생성
    feature_names = _make_feature_names(X.shape[1])
    return X, y, feature_names


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
