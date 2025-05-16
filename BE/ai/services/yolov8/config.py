# services/yolov8/config.py

from pathlib import Path

# YOLOv8 TFLite 모델 경로
DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent / "models" / "yolov8n_bbox_int8.tflite"
)

# 입력 크기 (YOLOv8 기본값: 640x640)
INPUT_SIZE = (640, 640)

# confidence threshold (탐지할 최소 신뢰도)
CONF_THRES = 0.25

# NMS IoU threshold (겹치는 bbox 제거 기준)
IOU_THRES = 0.45
