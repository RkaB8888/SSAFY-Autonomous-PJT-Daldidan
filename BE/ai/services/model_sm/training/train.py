# ai/services/model_sm/training/train.py
import joblib
from pathlib import Path
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

from services.model_sm.embedding.build_embeddings import load_cache
from services.model_sm.config import WEIGHTS_DIR, MODEL_SAVE_PATH

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
X, y, stems = load_cache("train")

# ─────────────────────────────────────────────
# 2. 모델 정의 및 학습
# ─────────────────────────────────────────────
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X, y)

# ─────────────────────────────────────────────
# 3. 학습 성능 평가
# ─────────────────────────────────────────────
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = sqrt(mean_squared_error(y, y_pred))

print("\n✅ 학습 완료")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# ─────────────────────────────────────────────
# 4. 모델 저장
# ─────────────────────────────────────────────
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_SAVE_PATH)
print(f"\n📦 모델 저장 완료: {MODEL_SAVE_PATH}")
