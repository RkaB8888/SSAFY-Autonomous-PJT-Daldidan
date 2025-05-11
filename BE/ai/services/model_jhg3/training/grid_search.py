# services/model_jhg3/training/grid_search.py
import csv
from pathlib import Path

import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.feature_selection import VarianceThreshold

from services.model_jhg3.config import (
    IMAGES_DIR,
    JSONS_DIR,
    VALID_IMAGES_DIR,
    VALID_JSONS_DIR,
    CACHE_DIR,
)
from services.model_jhg3.extract_embeddings import build_and_cache_embeddings
from services.model_jhg3.extract_valid_embeddings import (
    build_and_cache_embeddings as build_valid_cache,
)


def load_cache(prefix: str):
    feat = CACHE_DIR / f"{prefix}_embeddings.npy"
    label = CACHE_DIR / f"{prefix}_labels.npy"
    if not (feat.exists() and label.exists()):
        print(f"🚀 캐시가 없으므로 {prefix}용 build_and_cache_embeddings() 실행…")
        if prefix == "train":
            build_and_cache_embeddings(IMAGES_DIR, JSONS_DIR)
        else:
            build_valid_cache(VALID_IMAGES_DIR, VALID_JSONS_DIR)
        print("✅ 캐시 생성 완료.")
    X = np.memmap(feat, dtype=np.float32, mode="r").reshape(-1, 1280)
    y = np.memmap(label, dtype=np.float32, mode="r", shape=(len(X),))
    return X, y


def evaluate(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main():
    X_train, y_train = load_cache("train")
    X_val, y_val = load_cache("valid")

    selector = VarianceThreshold(0.0)
    X_train = selector.fit_transform(X_train)
    X_val = selector.transform(X_val)

    param_grid = {
        "learning_rate": [0.01, 0.03],
        "n_estimators": [1000, 1500, 2000, 2500, 3000],
        "max_depth": [-1, 8, 12],
    }

    out_csv = Path(__file__).parent / "hp_tuning_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lr", "n_est", "max_depth", "best_iter", "MAE", "RMSE", "R2"])

        for lr in param_grid["learning_rate"]:
            for ne in param_grid["n_estimators"]:
                for md in param_grid["max_depth"]:
                    print(f"\n▶ 실험: lr={lr}, n_est={ne}, max_depth={md}")
                    if md == -1:
                        num_leaves = 4096  # 제한 없음이면 적절히 넉넉한 값 지정
                    else:
                        num_leaves = 2**md
                    model = lgb.LGBMRegressor(
                        learning_rate=lr,
                        n_estimators=ne,
                        max_depth=md,
                        num_leaves=num_leaves,
                        device="gpu",
                        gpu_use_dp=True,
                        random_state=42,
                    )
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric=["l2_root", "l1", "mape"],
                        callbacks=[
                            early_stopping(100, first_metric_only=False),
                            log_evaluation(period=20),
                        ],
                    )
                    best_it = model.best_iteration_
                    y_pred = model.predict(X_val, num_iteration=best_it)
                    mae, rmse, r2 = evaluate(y_val, y_pred)

                    writer.writerow(
                        [lr, ne, md, best_it, f"{mae:.4f}", f"{rmse:.4f}", f"{r2:.4f}"]
                    )
                    print(f"  -> 결과 MAE:{mae:.4f}, RMSE:{rmse:.4f}, R2:{r2:.4f}")

    print(f"\n✅ 모든 실험 완료. 결과는 {out_csv}에 저장되었습니다.")


if __name__ == "__main__":
    main()
