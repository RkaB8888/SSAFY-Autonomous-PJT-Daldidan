# services/xgb_seg/training/grid_search.py
from pathlib import Path
import joblib
from sklearn.model_selection import GridSearchCV, ParameterGrid
from xgboost import XGBRegressor
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from services.xgb_bbox.embedding.build_embeddings import load_cache

# ─────────────────────────────────────────────
# 1) 데이터 분리
# ─────────────────────────────────────────────
X_train, y_train, _ = load_cache("train")  # 내부 CV
X_valid, y_valid, _ = load_cache("valid")  # 홀드아웃 검증

# ─────────────────────────────────────────────
# 2) 하이퍼파라미터 공간 정의
# ─────────────────────────────────────────────
param_grid = {
    "learning_rate": [0.3, 0.1, 0.05, 0.02],
    "n_estimators": [300, 500, 700, 1000],
    "max_depth": [4, 6, 8],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
}

# GridSearch 전체 fit 횟수 계산 (후에 tqdm total로 사용)
total_fits = len(list(ParameterGrid(param_grid))) * 3  # cv=3

# ─────────────────────────────────────────────
# 3) GridSearchCV (train 세트 내부 3-fold)
# ─────────────────────────────────────────────
base_model = XGBRegressor(
    random_state=42,
    n_jobs=1,
    tree_method="hist",
    eval_metric="rmse",
    early_stopping_rounds=50,
)

search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="r2",
    cv=3,
    n_jobs=8,
    verbose=0,
    refit=True,
    error_score="raise",
)

fit_params = {
    "eval_set": [(X_valid, y_valid)],
    "verbose": False,
}

# ─────────────────────────────────────────────
# 4) tqdm_joblib로 진행도 표시
# ─────────────────────────────────────────────
with tqdm_joblib(
    tqdm(desc="GridSearch 진행", total=total_fits, ncols=80)
) as progress_bar:
    search.fit(X_train, y_train, **fit_params)


# ─────────────────────────────────────────────
# 5) 결과 출력 & 모델 저장
# ─────────────────────────────────────────────
print("\n✅ 하이퍼파라미터 탐색 완료")
print("🧩 Best Params :", search.best_params_)
print(f"🧩 Best CV R²  : {search.best_score_:.4f}")
print(f"✅ Valid R²     : {search.best_estimator_.score(X_valid, y_valid):.4f}")

best_model_path = Path("services/xgb_seg/weights/xgb_seg_best.pkl")
best_model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(search.best_estimator_, best_model_path)
print(f"📦 최적 모델 저장 → {best_model_path}")
