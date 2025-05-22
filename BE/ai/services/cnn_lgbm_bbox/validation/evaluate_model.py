# services/cnn_lgbm_bbox/validation/evaluate_model.py
"""
학습된 LightGBM-CNN 모델을 검증(valid 셋)하고
MAE·RMSE·R²·평균 추론시간을 출력 / CSV 저장
"""
from pathlib import Path
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import services.cnn_lgbm_bbox.config as cfg
from services.cnn_lgbm_bbox.utils.loader import load_model_bundle
from services.cnn_lgbm_bbox.embedding import build_embeddings as beb  # ⬅️ 새 임베딩 모듈


# ──────────────────────────────────────────
# 1. 검증 캐시 로드(없으면 자동 생성)
# ──────────────────────────────────────────
def load_valid_set():
    feat_f = cfg.CACHE_DIR / "valid_embeddings.npy"
    label_f = cfg.CACHE_DIR / "valid_labels.npy"
    stem_f = cfg.CACHE_DIR / "valid_stems.npy"

    if not (feat_f.exists() and label_f.exists() and stem_f.exists()):
        print("🚀 valid 캐시가 없어 build_and_cache_embeddings() 실행…")
        beb.build_and_cache_embeddings("valid", cfg.CACHE_DIR)
        print("✅ valid 캐시 생성 완료")

    y = np.memmap(label_f, dtype=np.float32, mode="r")
    flat = np.memmap(feat_f, dtype=np.float32, mode="r")
    D = flat.size // y.size  # 자동 차원 추론
    X = flat.reshape(y.size, D)
    stems = np.load(stem_f).tolist()

    print(f"✔ Loaded valid set: {len(stems)} samples, dim={D}")
    return X, y, stems


# ──────────────────────────────────────────
# 2. 평가 함수
# ──────────────────────────────────────────
def metrics(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred),
    )


def main():
    model, selector = load_model_bundle(cfg.MODEL_SAVE_PATH)
    X, y, ids = load_valid_set()
    X_sel = selector.transform(X)

    t0 = time.time()
    preds = model.predict(X_sel)
    avg_ms = (time.time() - t0) * 1000 / len(X)

    mae, rmse, r2 = metrics(y, preds)
    print(f"\n▶ MAE  : {mae:.4f}")
    print(f"▶ RMSE : {rmse:.4f}")
    print(f"▶ R²   : {r2:.4f}")
    print(f"▶ 평균 추론시간/샘플 : {avg_ms:.6f} ms")

    out_csv = Path("services/cnn_lgbm_bbox/eval_results.csv")
    pd.DataFrame({"id": ids, "true": y, "pred": preds}).to_csv(out_csv, index=False)
    print(f"✅ 결과 저장 → {out_csv}")


if __name__ == "__main__":
    main()
