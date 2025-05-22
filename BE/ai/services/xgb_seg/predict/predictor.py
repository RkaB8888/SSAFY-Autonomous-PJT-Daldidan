# services/xgb_seg/predict/predictor.py
from io import BytesIO
from PIL import Image
import numpy as np
import base64
import joblib
from typing import Union
from services.xgb_seg.config import MODEL_SAVE_PATH
from services.xgb_seg.extractor.common_features import extract_features

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


def predict_bytes(image_input: Union[bytes, str]) -> float:
    np_img = _bytes_to_np(image_input)
    feats = extract_features(np_img)


    ## 후처리 보정 추가
    a_mean = feats["a_mean"]
    b_mean = feats["b_mean"]
    delta_E = feats["delta_E"]  # 색차 (푸름과 명확히 구분되는 수치)

    # 푸른 사과로 판단되는 조건 (보정 가능)
    is_green_apple = a_mean < 125 and b_mean > 110 and delta_E > 80


    print(f"a: {a_mean:.2f}, b: {b_mean:.2f}, delta_E: {delta_E:.2f}")
    if is_green_apple:
        print("🟢 푸른 사과 감지 → 브릭스 7.5로 고정")
        return 7.5

    ## 후처리 보정(끝끝)

    X = np.array([list(feats.values())])
    sugar = float(model.predict(X)[0])
    return sugar
