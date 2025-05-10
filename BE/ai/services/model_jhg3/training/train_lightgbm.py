# services/model_jhg3/training/train_lightgbm.py
import joblib
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import numpy as np
from sklearn.feature_selection import VarianceThreshold

from services.model_jhg3.config import (
    IMAGES_DIR,
    JSONS_DIR,
    VALID_IMAGES_DIR,
    VALID_JSONS_DIR,
    MODEL_SAVE_PATH,
    CACHE_DIR,
)
from services.model_jhg3.extract_embeddings import build_and_cache_embeddings
from services.model_jhg3.extract_valid_embeddings import (
    build_and_cache_embeddings as build_valid_cache,
)


def _make_feature_names(dim: int) -> List[str]:
    return [f"cnn_{i}" for i in range(dim)]


def r2_eval(y_pred: np.ndarray, data: lgb.Dataset):
    y_true = data.get_label()
    # 1 - SSR/SST
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return "r2", 1 - ss_res / ss_tot, True


def load_dataset(
    images_dir: Path, jsons_dir: Path, prefix: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feat_cache = CACHE_DIR / f"{prefix}_embeddings.npy"
    label_cache = CACHE_DIR / f"{prefix}_labels.npy"

    # 캐시 없으면 생성
    if not (feat_cache.exists() and label_cache.exists()):
        print(f"🚀 캐시가 없으므로 {prefix}용 build_and_cache_embeddings() 실행…")
        if prefix == "train":
            build_and_cache_embeddings(images_dir, jsons_dir)
        elif prefix == "valid":
            build_valid_cache(images_dir, jsons_dir)
        print("✅ 캐시 생성 완료.")

    # 캐시 불러오기
    X = np.memmap(feat_cache, dtype=np.float32, mode="r").reshape(-1, 1280)
    y = np.memmap(label_cache, dtype=np.float32, mode="r", shape=(X.shape[0],))

    print(f"✅ Loaded cached {prefix} set: {len(X)} samples")
    feature_names = _make_feature_names(X.shape[1])
    return X, y, feature_names


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    save_path: Path,
):
    # ─────── feature selector 상수 특성 제거 (CNN 벡터라 거의 변화 없지만 안전 차원) ───────
    selector = VarianceThreshold(threshold=0.0)
    X_train_sel = selector.fit_transform(X_train)
    X_val_sel = selector.transform(X_val)

    # ─── LightGBM 설정 ───
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=-1,
        device="gpu",
        gpu_use_dp=True,
        random_state=42,
    )

    model.fit(
        X_train_sel,
        y_train,
        eval_set=[(X_val_sel, y_val)],
        eval_metric=["l2_root", "l1", "mape"],
        feval=r2_eval,
        callbacks=[
            early_stopping(100, first_metric_only=False),  # 50회 개선 없으면 멈춤
            log_evaluation(period=20),  # 20라운드마다 로그 출력
        ],
    )

    print(f"▶ Best iteration: {model.best_iteration_}")

    # ─── 저장 ───
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "selector": selector, "feature_names": feature_names},
        save_path,
    )
    print(f"✅ LightGBM + meta 저장 완료: {save_path}")


def main():
    X_train, y_train, feature_names = load_dataset(
        IMAGES_DIR, JSONS_DIR, prefix="train"
    )
    X_val, y_val, _ = load_dataset(VALID_IMAGES_DIR, VALID_JSONS_DIR, prefix="valid")

    print(f"▶ Train: {X_train.shape}, Valid: {X_val.shape}")
    print(f"▶ y 범위: {y_train.min():.2f} ~ {y_train.max():.2f}")

    train_lightgbm(X_train, y_train, X_val, y_val, feature_names, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
