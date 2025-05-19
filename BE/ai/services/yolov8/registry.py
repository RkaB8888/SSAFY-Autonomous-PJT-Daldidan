# services/yolov8/registry.py
from services.detect_service import register_model
from services.yolov8.inference.predictor import YoloV8Predictor
from services.yolov8.inference.predictor_pt import YoloV8PtSegPredictor

register_model(
    name="yolov8_pt",
    version="s",
    predictor_cls=lambda: YoloV8PtSegPredictor(model_size="s"),
)
register_model(
    name="yolov8_pt",
    version="m",
    predictor_cls=lambda: YoloV8PtSegPredictor(model_size="m"),
)
register_model(
    name="yolov8_pt",
    version="l",
    predictor_cls=lambda: YoloV8PtSegPredictor(model_size="l"),
)

register_model(
    name="yolov8",
    version="bbox_int8",
    predictor_cls=lambda: YoloV8Predictor(model_type="bbox"),
)

register_model(
    name="yolov8",
    version="seg_float32",
    predictor_cls=lambda: YoloV8Predictor(model_type="seg"),
)
