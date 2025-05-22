# services/lgbm_seg/training/train_lightgbm.py
"""
CNN 임베딩 캐시(embedding/build_embeddings.py)만 활용해
LightGBM 회귀 모델을 학습 → 저장하는 스크립트
"""
from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.feature_selection import VarianceThreshold

import services.lgbm_seg.config as cfg
from services.lgbm_seg.embedding import build_embeddings as beb


# ──────────────────────────────
# 1. 데이터 로딩/캐시 유틸
# ──────────────────────────────
def _feat_names(dim: int) -> List[str]:
    return [f"cnn_{i}" for i in range(dim)]


def _load_dataset(prefix: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """train / valid 캐시 로드(없으면 자동 생성)"""
    feat_f = cfg.CACHE_DIR / f"{prefix}_embeddings.npy"
    label_f = cfg.CACHE_DIR / f"{prefix}_labels.npy"

    if not (feat_f.exists() and label_f.exists()):
        print(f"🚀 {prefix} 캐시가 없어 build_and_cache_embeddings() 실행…")
        beb.build_and_cache_embeddings(prefix, cfg.CACHE_DIR)
        print("✅ 캐시 생성 완료")

    flat = np.memmap(feat_f, dtype=np.float32, mode="r")
    y = np.memmap(label_f, dtype=np.float32, mode="r")
    n_samples = y.size
    dim = flat.size // n_samples
    X = flat.reshape(n_samples, dim)

    print(f"✔ Loaded {prefix}: {n_samples} samples, dim={dim}")
    return X, y, _feat_names(dim)


# ──────────────────────────────
# 2. 학습 루틴
# ──────────────────────────────
def _train(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    feat_names: List[str],
    save_path: Path,
):
    selector = VarianceThreshold(0.0)
    X_tr_sel = selector.fit_transform(X_tr)
    X_va_sel = selector.transform(X_va)

    params = dict(
        objective="regression",
        metric="rmse",
        device_type="gpu",
        n_jobs=8,
        gpu_use_dp=False,
        boosting_type="gbdt",
        max_depth=10,
        num_leaves=1023,  # 2^10 - 1
        learning_rate=0.05,
        num_iterations=2000,
        max_bin=255,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l1=0.2,
        lambda_l2=1.0,
        random_state=42,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr_sel,
        y_tr,
        eval_set=[(X_va_sel, y_va)],
        eval_metric=["l2_root", "l1", "mape"],
        callbacks=[early_stopping(100), log_evaluation(100)],
    )
    print(f"▶ Best iteration : {model.best_iteration_}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "selector": selector, "feature_names": feat_names},
        save_path,
    )
    print(f"✅ 모델 저장 → {save_path}")


# ──────────────────────────────
# 3. 메인
# ──────────────────────────────
def main():
    X_train, y_train, fnames = _load_dataset("train")
    X_valid, y_valid, _ = _load_dataset("valid")

    print(f"▶ Train shape : {X_train.shape}  Valid shape : {X_valid.shape}")
    _train(X_train, y_train, X_valid, y_valid, fnames, cfg.MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
