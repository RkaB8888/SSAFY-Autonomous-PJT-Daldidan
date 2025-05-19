# ai/services/detect_service.py
"""
플러그인 기반 디스패처 구조의 정석 구현
"""
from typing import Union, Callable
from PIL import Image
import numpy as np

# 모델 이름 → predictor class or callable
model_registry: dict[str, Callable] = {}


def register_model(name: str, version: str, predictor_cls: Callable):
    key = f"{name}:{version}"
    model_registry[key] = predictor_cls


def detect(
    model_name: str, image: Union[Image.Image, np.ndarray], version: str = "default"
):
    key = f"{model_name}:{version}"
    if key not in model_registry:
        raise ValueError(f"[detect_service] Unknown model: {key}")

    predictor = model_registry[key]()
    raw = predictor.predict(image)

    # ── segmentation 모델은 이미 dict 리스트({ "bbox":…, "seg":… }) 형태
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw

    # ── bbox 전용 모델: list of [xmin, ymin, xmax, ymax]
    return [{"bbox": box, "seg": None} for box in raw]
