#!/usr/bin/env python
# services/model_jhg1/predict/predict_sugar.py

import argparse
from pathlib import Path

import joblib
import numpy as np
from services.model_jhg1.utils.feature_extractors import extract_features


def load_model(model_path: Path):
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["selector"]


def predict_one(model, selector, img_path: Path):
    feats, _ = extract_features(
        __import__("common_utils.image_cropper", fromlist=["crop_bbox_from_json"]).image
        if False
        else []
    )
    # 실제론 crop_bbox_from_json이나 load_image로 이미지를 읽으세요.
    # 여긴 이미 crop된 이미지 파일(.jpg)만 넘어온다고 가정
    from common_utils.image_cropper import crop_bbox_from_json

    crop, _ = (
        crop_bbox_from_json(img_path.with_suffix(".jpg"), img_path.with_suffix(".json"))
        if False
        else (None, None)
    )
    # 대신, API 상황에선 이미 crop_img를 받으실 겁니다:
    # feats, _ = extract_features(crop_img)
    X = feats.reshape(1, -1)
    X_sel = selector.transform(X)
    return float(model.predict(X_sel)[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to lightgbm_model.pkl")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="single image file to predict (jpg)")
    group.add_argument("--input_dir", help="folder of cropped images (.jpg)")
    p.add_argument("--out_csv", help="save predictions as CSV")
    args = p.parse_args()

    model, selector = load_model(Path(args.model))
    results = []

    if args.image:
        img = Path(args.image)
        feats, _ = extract_features(
            __import__(
                "common_utils.image_cropper", fromlist=["load_image"]
            ).load_image(img)
        )
        pred = float(model.predict(selector.transform(feats.reshape(1, -1)))[0])
        print(pred)

    else:
        for img in sorted(Path(args.input_dir).glob("*.jpg")):
            feats, _ = extract_features(
                __import__(
                    "common_utils.image_cropper", fromlist=["load_image"]
                ).load_image(img)
            )
            pred = float(model.predict(selector.transform(feats.reshape(1, -1)))[0])
            print(f"{img.stem},{pred:.3f}")
            results.append((img.stem, pred))
        if args.out_csv:
            import pandas as pd

            pd.DataFrame(results, columns=["id", "predicted_sugar"]).to_csv(
                args.out_csv, index=False
            )
            print(f"✅ Saved to {args.out_csv}")


if __name__ == "__main__":
    main()
