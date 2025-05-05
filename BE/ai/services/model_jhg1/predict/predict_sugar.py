#!/usr/bin/env python
# services/model_jhg1/predict/predict_sugar.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from services.model_jhg1.utils.feature_extractors import extract_features
from services.model_jhg1.utils.loader import load_model_bundle
from common_utils.image_cropper import load_image


def predict_one(model, selector, img_path: Path) -> float:
    image = load_image(img_path)
    feats, _ = extract_features(image)
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

    model, selector = load_model_bundle(Path(args.model))
    results = []

    if args.image:
        pred = predict_one(model, selector, Path(args.image))
        print(pred)

    else:
        for img in sorted(Path(args.input_dir).glob("*.jpg")):
            pred = predict_one(model, selector, img)
            print(f"{img.stem},{pred:.3f}")
            results.append((img.stem, pred))

        if args.out_csv:
            df = pd.DataFrame(results, columns=["id", "predicted_sugar"])
            df.to_csv(args.out_csv, index=False)
            print(f"✅ Saved to {args.out_csv}")


if __name__ == "__main__":
    main()
