# ai/services/predict_service.py


def predict(model_name: str, image_base64: str):
    # 예시 분기
    if model_name == "model_a":
        from ai.services.model_a.predictor import predict as predict_a

        return predict_a(image_base64)
    elif model_name == "model_b":
        from ai.services.model_b.predictor import predict as predict_b

        return predict_b(image_base64)
    else:
        raise ValueError(f"Unknown model: {model_name}")
