# ai>services/cnn_lgbm_bbox/predict/predictor.py
from io import BytesIO
from PIL import Image
import numpy as np
from services.cnn_lgbm_bbox.config import MODEL_SAVE_PATH
from services.cnn_lgbm_bbox.utils.cnn_feature_extractor import extract_batch
from services.cnn_lgbm_bbox.utils.loader import load_model_bundle

# ── 모델 & selector 메모리 상주 로딩 ──────────────────
model, selector = load_model_bundle(MODEL_SAVE_PATH)


def _bytes_to_np(img_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))


def predict_bytes(image_bytes: bytes) -> dict:
    np_img = _bytes_to_np(image_bytes)
    feats = extract_batch(np_img[None, ...])  # (1, 1280)
    X_sel = selector.transform(feats)  # (1, D)
    sugar = float(model.predict(X_sel)[0])
    return {"label": "sugar_content", "confidence": sugar}


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
