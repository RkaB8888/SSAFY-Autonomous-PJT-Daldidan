# common_utils/image_cropper.py

from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
import json


def crop_bbox_from_json(
    image_path: Path, json_path: Path
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """원본 이미지와 JSON의 bbox 정보를 이용해 crop 이미지를 반환"""
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    x, y, w, h = map(int, data["annotations"]["bbox"])
    crop = image[y : y + h, x : x + w]
    return crop, (x, y, w, h)
