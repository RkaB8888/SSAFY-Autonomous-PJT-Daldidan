# ai/services/xgb_bbox/validation/evaluate_model.py

import joblib
from pathlib import Path
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

from services.xgb_bbox.embedding.build_embeddings import load_cache
from services.xgb_bbox.config import MODEL_SAVE_PATH

# ─────────────────────────────────────────────
# 1. 모델 로드
# ─────────────────────────────────────────────
model = joblib.load(MODEL_SAVE_PATH)
print(f"📦 모델 로드 완료: {MODEL_SAVE_PATH}")

# ─────────────────────────────────────────────
# 2. 검증 데이터 로드
# ─────────────────────────────────────────────
X, y, stems = load_cache("valid")

# ─────────────────────────────────────────────
# 3. 예측 및 평가
# ─────────────────────────────────────────────
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = sqrt(mean_squared_error(y, y_pred))

print("\n✅ 검증 데이터 평가 결과")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")


# ─────────────────────────────────────────────
# 4. 오차 범위 내 정확도 계산
# ─────────────────────────────────────────────
def accuracy_within_range(y_true, y_pred, delta: float) -> float:
    return np.mean(np.abs(y_true - y_pred) <= delta)


acc_03 = accuracy_within_range(y, y_pred, 0.3)
acc_05 = accuracy_within_range(y, y_pred, 0.5)
acc_10 = accuracy_within_range(y, y_pred, 1.0)

print(f"\n📊 Accuracy@0.3: {acc_03:.2%}")
print(f"📊 Accuracy@0.5: {acc_05:.2%}")
print(f"📊 Accuracy@1.0: {acc_10:.2%}")

# ─────────────────────────────────────────────
# 5. 예측 결과 저장
# ─────────────────────────────────────────────
out_csv = Path("services/xgb_bbox/eval_results.csv")
pd.DataFrame({"stem": stems, "y_true": y, "y_pred": y_pred}).to_csv(
    out_csv, index=False
)
print(f"\n📄 결과 저장 완료 → {out_csv}")
