#!/usr/bin/env python
# services/model_jhg2/validation/evaluate_model.py

from pathlib import Path
import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from services.model_jhg2.utils.loader import load_model_bundle
from services.model_jhg2.config import (
    MODEL_SAVE_PATH,
    VALID_IMAGES_DIR,
    VALID_JSONS_DIR,
    CACHE_DIR,
)

from services.model_jhg2.extract_valid_embeddings import build_and_cache_embeddings


def load_test_set(img_dir: Path, json_dir: Path):
    feat_cache = CACHE_DIR / "valid_embeddings.npy"
    label_cache = CACHE_DIR / "valid_labels.npy"
    stem_cache = CACHE_DIR / "valid_stems.npy"

    # 1) 캐시가 없으면 자동 생성
    if not (feat_cache.exists() and label_cache.exists() and stem_cache.exists()):
        print("🚀 검증용 임베딩 캐시가 없으므로 build_and_cache_embeddings() 실행…")
        build_and_cache_embeddings(img_dir, json_dir)
        print("✅ 검증 임베딩 캐시 생성 완료.")

    # 2) 캐시에서 바로 로드
    # ── features: raw memmap of float32s of size (N,1280)
    X = np.memmap(feat_cache, dtype=np.float32, mode="r").reshape(-1, 1280)
    # ── ids: real .npy of unicode strings
    ids = np.load(stem_cache).tolist()
    # ── labels: raw memmap of float32s, so load via memmap with known shape
    y = np.memmap(label_cache, dtype=np.float32, mode="r", shape=(len(ids),))
    print(f"✅ Loaded valid cache: {len(ids)} samples")

    return X, y, ids


def evaluate(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred),
    )


def main():
    model, selector = load_model_bundle(MODEL_SAVE_PATH)

    X_test, y_test, ids = load_test_set(VALID_IMAGES_DIR, VALID_JSONS_DIR)
    X_sel = selector.transform(X_test) if selector else X_test

    start = time.time()
    y_pred = model.predict(X_sel)
    elapsed = time.time() - start
    avg_time = elapsed / len(X_test)

    mae, rmse, r2 = evaluate(y_test, y_pred)
    print(f"\n▶ MAE : {mae:.4f}")
    print(f"▶ RMSE: {rmse:.4f}")
    print(f"▶ R2  : {r2:.4f}")
    print(f"▶ 예측 시간(평균/샘플): {avg_time*1000:.6f} ms")

    out_path = Path("eval_results.csv")
    import pandas as pd

    pd.DataFrame({"id": ids, "true": y_test, "pred": y_pred}).to_csv(
        out_path, index=False
    )
    print(f"✅ Saved results to {out_path}")


if __name__ == "__main__":
    main()
