#!/usr/bin/env python
# services/model_jhg1/predict/predict_sugar.py
"""
CLI 배치 예측 스크립트
------------------------------------------------
예시
    단일 이미지 :
    python predict_sugar.py --image apple.jpg

    디렉토리 일괄 :
    python predict_sugar.py --input_dir crops/ --out_csv result.csv
"""

import argparse
from pathlib import Path
import pandas as pd

import numpy as np
import pandas as pd

from services.model_jhg1.config import MODEL_SAVE_PATH  # 기본 모델 경로
from services.model_jhg1.predictor import predict_bytes  # 공식 예측 함수


# ────────────────────────────────────────────────────────────────
def predict_one(img_path: Path) -> float:
    """JPEG 파일 하나 → 당도(float)"""
    with img_path.open("rb") as f:
        return predict_bytes(f.read())["confidence"]


# ────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=str(MODEL_SAVE_PATH),
        help=f"lightgbm_model.pkl 경로 (default: {MODEL_SAVE_PATH})",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="단일 이미지(jpg) 경로")
    group.add_argument("--input_dir", help="crop 이미지 폴더")
    p.add_argument("--out_csv", help="예측 결과 CSV 저장 경로")
    args = p.parse_args()

    # ※ 모델 경로는 predictor 내부의 MODEL_SAVE_PATH를 이미 활용하므로 사용 안 함
    results = []

    if args.image:
        pred = predict_one(Path(args.image))
        print(f"{pred:.3f}")

    else:
        for img in sorted(Path(args.input_dir).glob("*.jpg")):
            sugar = predict_one(img)
            print(f"{img.stem},{sugar:.3f}")
            results.append((img.stem, sugar))

        if args.out_csv:
            df = pd.DataFrame(results, columns=["id", "predicted_sugar"])
            df.to_csv(args.out_csv, index=False)
            print(f"✅ Saved to {args.out_csv}")


if __name__ == "__main__":
    main()
