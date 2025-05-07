#!/usr/bin/env python
# services/model_jhg2/validation/evaluate_model.py

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from common_utils.image_cropper import crop_bbox_from_json
from services.model_jhg2.utils.cnn_feature_extractor import extract
from services.model_jhg2.utils.loader import load_model_bundle
from services.model_jhg2.config import (
    MODEL_SAVE_PATH,
    VALID_IMAGES_DIR,
    VALID_JSONS_DIR,
)


def load_test_set(img_dir: Path, json_dir: Path):
    X, y, ids = [], [], []
    for img_path in tqdm(sorted(img_dir.glob("*.jpg")), desc="Loading test set"):
        js = json_dir / f"{img_path.stem}.json"
        if not js.exists():
            continue
        crop, _ = crop_bbox_from_json(img_path, js)
        if crop is None:
            continue
        feats = extract(crop)
        with open(js, "r", encoding="utf-8") as f:
            data = json.load(f)
        collection = data.get("collection", {})
        sugar = collection.get("sugar_content")
        if sugar is None:
            sugar = collection.get("sugar_content_nir")
        if sugar is None:
            tqdm.write(f"[무시] 당도 정보 없음: {img_path.name}")
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
    model, selector = load_model_bundle(MODEL_SAVE_PATH)

    X_test, y_test, ids = load_test_set(VALID_IMAGES_DIR, VALID_JSONS_DIR)
    X_sel = selector.transform(X_test) if selector else X_test
    y_pred = model.predict(X_sel)

    mae, rmse, r2 = evaluate(y_test, y_pred)
    print(f"\n▶ MAE : {mae:.4f}")
    print(f"▶ RMSE: {rmse:.4f}")
    print(f"▶ R2  : {r2:.4f}")

    # (선택) 결과 CSV로 저장하려면 이 경로를 활성화
    out_path = Path("eval_results.csv")
    import pandas as pd

    pd.DataFrame({"id": ids, "true": y_test, "pred": y_pred}).to_csv(
        out_path, index=False
    )
    print(f"✅ Saved results to {out_path}")


if __name__ == "__main__":
    main()
