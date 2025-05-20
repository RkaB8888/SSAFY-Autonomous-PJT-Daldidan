# services/yolov8/config.py

from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

# 지원하는 모델 버전들
MODEL_FILES = {
    "bbox_int8": "yolov8n_bbox_int8.tflite",
    "seg_float32": "yolov8n_seg_float32.tflite",
    "seg_float16": "yolov8n_seg_float16.tflite",
}

# PT 모델들 (추가)
MODEL_FILES_PT = {
    "n": "yolov8n-seg.pt",
    "s": "yolov8s-seg.pt",
    "m": "yolov8m-seg.pt",
    "l": "yolov8l-seg.pt",
}

# 입력 크기
INPUT_SIZE = (640, 640)
# 탐지할 최소 신뢰도
CONF_THRES = 0.25
# 겹치는 bbox 제거 기준
IOU_THRES = 0.45
