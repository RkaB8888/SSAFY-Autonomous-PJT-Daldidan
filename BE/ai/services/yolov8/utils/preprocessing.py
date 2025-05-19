# services/yolov8/utils/preprocessing.py

from PIL import Image
import numpy as np
import cv2


def preprocess_image(
    image: Image.Image | np.ndarray,
    target_size=(640, 640),
    dtype=np.uint8,
):
    """
    YOLOv8 TFLite 입력을 위한 전처리 함수
    - Letterbox (비율 유지 + 패딩)
    - 정규화 (float32 or uint8)
    - 배치 차원 추가

    Returns:
        input_tensor (np.ndarray): [1, H, W, 3]
        scale (float): resize 비율
        pad (tuple): (pad_w, pad_h)
    """

    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    h0, w0 = image.shape[:2]
    w_target, h_target = target_size

    # Letterbox (비율 유지 + 패딩)
    scale = min(w_target / w0, h_target / h0)
    nw, nh = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(image, (nw, nh))

    pad_w, pad_h = w_target - nw, h_target - nh
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    if dtype == np.uint8:
        input_tensor = padded.astype(np.uint8)
    elif dtype == np.float16:
        input_tensor = (padded.astype(np.float16)) / np.float16(255.0)
    else:
        input_tensor = (padded.astype(np.float32)) / 255.0

    # 배치 차원 추가
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor, scale, (left, top)
