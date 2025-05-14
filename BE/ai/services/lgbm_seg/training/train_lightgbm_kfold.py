# services/lgbm_seg/training/train_lightgbm_kfold.py
from pathlib import Path
import joblib, numpy as np, lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold

import services.lgbm_seg.config as cfg
from services.lgbm_seg.embedding import build_embeddings as beb


def load_dataset():
    X, y, _ = beb._load_dataset("train")  # train+valid 전체 사용
    selector = VarianceThreshold(0.0).fit(X)
    X_sel = selector.transform(X)
    return X_sel, y, selector


params = dict(
    objective="regression",
    metric="rmse",
    device_type="gpu",
    gpu_use_dp=False,
    max_depth=10,
    num_leaves=1023,
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


def kfold_train(k: int = 5):
    X, y, selector = load_dataset()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    rmse_list, mae_list = [], []
    save_root = cfg.MODEL_SAVE_PATH.parent / "kfold"
    save_root.mkdir(parents=True, exist_ok=True)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        print(f"\n📂 Fold {fold}/{k}")
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(100), log_evaluation(100)],
        )

        pred = model.predict(X_va, num_iteration=model.best_iteration_)
        rmse = mean_squared_error(y_va, pred, squared=False)
        mae = mean_absolute_error(y_va, pred)
        rmse_list.append(rmse)
        mae_list.append(mae)
        print(f"▶ Fold {fold}  RMSE {rmse:.4f}  MAE {mae:.4f}")

        joblib.dump(
            {"model": model, "selector": selector},
            save_root / f"lgbm_fold{fold}.joblib",
        )

    print("\n✅ K-Fold Result")
    print(f"RMSE  mean {np.mean(rmse_list):.4f}  ± {np.std(rmse_list):.4f}")
    print(f"MAE   mean {np.mean(mae_list):.4f}  ± {np.std(mae_list):.4f}")


if __name__ == "__main__":
    kfold_train(k=5)
