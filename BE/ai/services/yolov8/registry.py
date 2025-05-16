# services/yolov8/registry.py
from services.detect_service import register_model
from services.yolov8.inference.predictor import YoloV8Predictor

register_model(
    name="yolov8",
    version="coco_int8",
    predictor_cls=lambda: YoloV8Predictor(model_type="coco"),
)
