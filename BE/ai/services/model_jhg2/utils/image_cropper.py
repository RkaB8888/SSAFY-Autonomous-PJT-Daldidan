# services/model_jhg2/utils/image_cropper.py
from pathlib import Path
import json
from PIL import Image
import numpy as np
from typing import Tuple


def crop_bbox_from_json(
    image_path: Path,
    json_path: Path,
    resize_to: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    • 한 이미지 당 하나의 bbox가 있다고 가정하고
    • crop → 리사이즈 → HWC uint8 np.ndarray 반환
    • 오류 발생 시 예외를 그대로 올려서 호출부에서 skip 하게 함
    """
    # 1) 이미지 열기
    img = Image.open(image_path).convert("RGB")

    # 2) JSON에서 bbox 읽기
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    x, y, w, h = map(int, data["annotations"]["bbox"])

    # 3) crop & resize
    crop = img.crop((x, y, x + w, y + h))
    if resize_to is not None:
        crop = crop.resize(resize_to, Image.Resampling.LANCZOS)

    # 4) numpy array 변환
    return np.array(crop, copy=False)
