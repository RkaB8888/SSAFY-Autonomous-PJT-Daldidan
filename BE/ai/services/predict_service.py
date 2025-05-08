# ai/services/predict_service.py
from services.model_jhg1.predictor import predict_bytes as predict_jhg1
from services.model_jhg2.predict.predictor import predict_bytes as predict_jhg2
from services.model_a.predictor import predict as predict_a

_MODEL_REGISTRY = {
    "model_jhg1": predict_jhg1,
    "model_jhg2": predict_jhg2,
    "model_a": predict_a,
}


def predict(model_name: str, image_bytes: bytes):
    try:
        fn = _MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"Unknown model: {model_name!r}")
    return fn(image_bytes)
