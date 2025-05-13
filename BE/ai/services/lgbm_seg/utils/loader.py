# BE/ai/services/lgbm_seg/utils/loader.py
from pathlib import Path
import joblib


def load_model_bundle(model_path: Path):
    """
    LightGBM 모델과 특성 선택기(Selector)를 joblib으로 로드합니다.
    """
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["selector"]
