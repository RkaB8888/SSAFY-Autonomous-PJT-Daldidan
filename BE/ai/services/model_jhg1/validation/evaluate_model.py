#!/usr/bin/env python
# services/model_jhg1/validation/evaluate_model.py


import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from common_utils.image_cropper import crop_bbox_from_json
from services.model_jhg1.utils.feature_extractors import extract_features


def load_model(model_path: Path):
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["selector"]


def load_test_set(img_dir: Path, json_dir: Path):
    X, y, ids = [], [], []
    for img_path in tqdm(sorted(img_dir.glob("*.jpg")), desc="Loading test set"):
        js = json_dir / f"{img_path.stem}.json"
        if not js.exists():
            continue
        crop, _ = crop_bbox_from_json(img_path, js)
        if crop is None:
            continue
        feats, _ = extract_features(crop)
        with open(js, "r", encoding="utf-8") as f:
            sugar = json.load(f)["collection"].get("sugar_content")
        if sugar is None:
            continue
        X.append(feats)
        y.append(sugar)
        ids.append(img_path.stem)  # ✅ 일치하는 ID 수집
    return np.vstack(X), np.array(y, dtype=float), ids


def evaluate(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to lightgbm_model.pkl")
    p.add_argument("--images", required=True, help="test images dir")
    p.add_argument("--jsons", required=True, help="test jsons dir")
    p.add_argument("--out_csv", help="optional: save per-sample CSV")
    args = p.parse_args()

    model, selector = load_model(Path(args.model))

    X_test, y_test, ids = load_test_set(Path(args.images), Path(args.jsons))
    X_sel = selector.transform(X_test)
    y_pred = model.predict(X_sel)

    mae, rmse, r2 = evaluate(y_test, y_pred)
    print(f"\n▶ MAE : {mae:.4f}")
    print(f"▶ RMSE: {rmse:.4f}")
    print(f"▶ R2  : {r2:.4f}")

    if args.out_csv:
        import pandas as pd

        df = pd.DataFrame(
            {
                "id": ids,
                "true": y_test,
                "pred": y_pred,
            }
        )
        df.to_csv(args.out_csv, index=False)
        print(f"✅ Saved results to {args.out_csv}")


if __name__ == "__main__":
    main()
