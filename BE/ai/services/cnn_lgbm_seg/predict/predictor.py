# ai>services/cnn_lgbm_seg/predict/predictor.py
from io import BytesIO
from PIL import Image
import numpy as np
import base64
from typing import Union
from services.cnn_lgbm_seg.config import MODEL_SAVE_PATH
from services.cnn_lgbm_seg.extractor.cnn_feature_extractor import extract_batch
from services.cnn_lgbm_seg.utils.loader import load_model_bundle

# ── 모델 & selector 메모리 상주 로딩 ──────────────────
model, selector = load_model_bundle(MODEL_SAVE_PATH)


# def _bytes_to_np(img_bytes: bytes) -> np.ndarray:
#     return np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))


def _bytes_to_np(image_input: bytes | str) -> np.ndarray:
    try:
        if isinstance(image_input, str):
            if image_input.startswith("data:image"):
                image_input = image_input.split(",")[1]
            image_input = base64.b64decode(image_input)
        return np.array(Image.open(BytesIO(image_input)).convert("RGB"))
    except Exception as e:
        raise ValueError(f"이미지 디코딩 실패: {e}")


def predict_bytes(image_input: Union[bytes, str]) -> float:
    np_img = _bytes_to_np(image_input)  # (H, W, C)
    feats = extract_batch(np_img[None, ...])  # (1, D) ← 1장짜리 배치로 처리
    X_sel = selector.transform(feats)  # (1, D')
    sugar = float(model.predict(X_sel)[0])  # → scalar 예측값
    return sugar


# ── CLI 테스트용 -------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python predictor.py <crop_image.jpg>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    with img_path.open("rb") as f:
        img_bytes = f.read()

    result = predict_bytes(img_bytes)
    print(f"Predicted sugar: {result['confidence']:.3f}")
