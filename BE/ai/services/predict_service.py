# ai/services/predict_service.py


def predict(model_name: str, image_bytes: bytes):
    # 예시 분기
    if model_name == "model_jhg1":
        from services.model_jhg1.predictor import predict_bytes as predict_jhg1

        return predict_jhg1(image_bytes)
    elif model_name == "model_a":
        from services.model_a.predictor import predict as predict_a

        return predict_a(image_bytes)

    elif model_name == "model_b":
        from services.model_b.predictor import predict as predict_b

        return predict_b(image_bytes)

    else:
        raise ValueError(f"Unknown model: {model_name}")
