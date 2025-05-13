# services/model_sm/predict/predictor.py
from io import BytesIO
from PIL import Image
import numpy as np
import base64
import joblib
from typing import Union
from services.model_sm.config import MODEL_SAVE_PATH
from services.model_sm.extractor.common_features import extract_features

# ── 모델 로딩 ────────────────────────────────────────
model = joblib.load(MODEL_SAVE_PATH)


def _bytes_to_np(image_input: bytes | str) -> np.ndarray:
    try:
        if isinstance(image_input, str):
            if image_input.startswith("data:image"):
                image_input = image_input.split(",")[1]
            image_input = base64.b64decode(image_input)
        return np.array(Image.open(BytesIO(image_input)).convert("RGB"))
    except Exception as e:
        raise ValueError(f"이미지 디코딩 실패: {e}")


def predict_bytes(image_input: Union[bytes, str]) -> dict:
    np_img = _bytes_to_np(image_input)
    feats = extract_features(np_img)
    X = np.array([list(feats.values())])
    sugar = float(model.predict(X)[0])
    return {"label": "sugar_content", "confidence": sugar}
