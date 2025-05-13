# ai/services/predict_service.py


def predict(model_name: str, image_bytes: bytes):
    if model_name == "cnn_lgbm_bbox":
        from services.cnn_lgbm_bbox.predict.predictor import (
            predict_bytes as predict_cnn_lgbm_bbox,
        )

        return predict_cnn_lgbm_bbox(image_bytes)

    elif model_name == "model_a":
        from services.model_a.predictor import predict as predict_a

        return predict_a(image_bytes)

    elif model_name == "model_b":
        from services.model_b.predictor import predict as predict_b

        return predict_b(image_bytes)

    else:
        raise ValueError(f"Unknown model: {model_name!r}")
