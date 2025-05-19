# services/yolov8/config.py

from pathlib import Path

# 지원하는 모델 버전들
MODEL_FILES = {
    "bbox_int8": "yolov8n_bbox_int8.tflite",
    "seg_float32": "yolov8n_seg_float32.tflite",
    "seg_float16": "yolov8n_seg_float16.tflite",
}

# TFLite 모델 경로
MODEL_DIR = Path(__file__).resolve().parent / "models"

# 입력 크기 (YOLOv8 기본값: 640x640)
INPUT_SIZE = (640, 640)

# confidence threshold (탐지할 최소 신뢰도)
CONF_THRES = 0.25

# NMS IoU threshold (겹치는 bbox 제거 기준)
IOU_THRES = 0.45
