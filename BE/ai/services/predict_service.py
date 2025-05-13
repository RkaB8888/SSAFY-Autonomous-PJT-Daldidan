# ai/services/predict_service.py


def predict(model_name: str, image_bytes: bytes):
    if model_name == "cnn_lgbm_bbox":
        from services.cnn_lgbm_bbox.predict.predictor import (
            predict_bytes as predict_cnn_lgbm_bbox,
        )

        print("cnn_lgbm_bbox 모델 사용")
        return predict_cnn_lgbm_bbox(image_bytes)

    elif model_name == "cnn_lgbm_seg":
        from services.cnn_lgbm_seg.predict.predictor import (
            predict_bytes as predict_cnn_lgbm_seg,
        )

        print("cnn_lgbm_seg 모델 사용")
        return predict_cnn_lgbm_seg(image_bytes)

    elif model_name == "lgbm_bbox":
        from services.lgbm_bbox.predict.predictor import (
            predict_bytes as predict_lgbm_bbox,
        )

        print("lgbm_bbox 모델 사용")
        return predict_lgbm_bbox(image_bytes)

    elif model_name == "lgbm_seg":
        from services.lgbm_bbox.predict.predictor import (
            predict_bytes as predict_lgbm_seg,
        )

        print("lgbm_seg 모델 사용")
        return predict_lgbm_seg(image_bytes)

    elif model_name == "model_sm":
        from services.model_sm.predict.predictor import (
            predict_bytes as predict_model_sm,
        )

        return predict_model_sm(image_bytes)

    elif model_name == "model_a":
        from services.model_a.predictor import predict as predict_a

        return predict_a(image_bytes)

    elif model_name == "model_b":
        from services.model_b.predictor import predict as predict_b

        return predict_b(image_bytes)

    else:
        raise ValueError(f"Unknown model: {model_name!r}")
