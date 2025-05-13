# services/model_jhg3/training/grid_search.py
import csv
from pathlib import Path
from tqdm import tqdm
import itertools
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.feature_selection import VarianceThreshold

import services.model_jhg3.config as cfg
from services.model_jhg3.utils.metrics import evaluate
import services.model_jhg3.embedding.build_embeddings as beb


def load_cache(prefix: str, cache_dir: Path):
    feat = cache_dir / f"{prefix}_embeddings.npy"
    label = cache_dir / f"{prefix}_labels.npy"

    if not (feat.exists() and label.exists()):
        print(f"🚀 캐시가 없으므로 {prefix}용 build_and_cache_embeddings() 실행…")
        beb.build_and_cache_embeddings(prefix, cache_dir)
        print("✅ 캐시 생성 완료.")

    flat = np.memmap(feat, dtype=np.float32, mode="r")
    y = np.memmap(label, dtype=np.float32, mode="r")  # shape 생략 → 1D 전체 읽기
    N = y.size
    D = flat.size // N
    X = flat.reshape(N, D)
    return X, y


def run_experiment():
    # 1. config 플래그에서 모드 읽기
    embed_mode = cfg.EMBEDDING_MODE
    use_nir = cfg.USE_NIR
    use_seg = cfg.USE_SEGMENTATION

    # 2. 실험 전용 캐시 디렉터리
    exp_name = f"{embed_mode}_nir{int(use_nir)}_seg{int(use_seg)}"
    exp_cache = cfg.BASE_DIR / "cache" / exp_name
    exp_cache.mkdir(parents=True, exist_ok=True)

    # 3. 데이터 로드 (train / valid)
    X_train, y_train = load_cache("train", exp_cache)
    X_valid, y_valid = load_cache("valid", exp_cache)

    # 4. Feature selection
    selector = VarianceThreshold(0.0)
    X_train = selector.fit_transform(X_train)
    X_valid = selector.transform(X_valid)

    # 5. Hyperparam grid
    param_grid = {
        "learning_rate": [0.01, 0.03],
        "n_estimators": [1000, 2000, 3000],
        "max_depth": [-1, 8, 12],
    }
    # 모든 조합 리스트 생성 및 개수 계산
    param_list = list(itertools.product(*param_grid.values()))
    total = len(param_list)

    out_csv = exp_cache / "hp_tuning_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lr", "n_est", "max_depth", "best_iter", "MAE", "RMSE", "R2"])

        for lr, ne, md in tqdm(
            param_list, total=total, desc=f"Grid Search ({exp_name})"
        ):
            print(f"▶ 실험 ({exp_name}): lr={lr}, n_est={ne}, max_depth={md}")
            num_leaves = 2**md if md > 0 else 4096
            model = lgb.LGBMRegressor(
                learning_rate=lr,
                n_estimators=ne,
                max_depth=md,
                num_leaves=num_leaves,
                device="gpu",
                gpu_use_dp=False,
                n_jobs=8,
                random_state=42,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric=["l2_root", "l1", "mape"],
                callbacks=[
                    early_stopping(100, first_metric_only=False),
                    log_evaluation(period=20),
                ],
            )
            best_it = model.best_iteration_
            preds = model.predict(X_valid, num_iteration=best_it)
            mae, rmse, r2 = evaluate(y_valid, preds)
            writer.writerow(
                [lr, ne, md, best_it, f"{mae:.4f}", f"{rmse:.4f}", f"{r2:.4f}"]
            )
            print(f"  -> 결과 MAE:{mae:.4f}, RMSE:{rmse:.4f}, R2:{r2:.4f}")
    print(f"✅ 실험 완료: {exp_name}, 결과: {out_csv}\n")


if __name__ == "__main__":
    run_experiment()
