# ai/services/detect_service.py

from typing import Union
from PIL import Image
import numpy as np


def detect(model_name: str, image: Union[Image.Image, np.ndarray]):
    if model_name == "yolov8_tflite":
        from services.yolov8_tflite.core.infer import detect_apples

        print("yolov8_tflite 인식 모델 사용")
        return detect_apples(image)

    else:
        raise ValueError(f"Unknown detection model: {model_name!r}")
