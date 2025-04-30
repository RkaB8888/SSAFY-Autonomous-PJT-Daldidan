# ai/schemas/predict.py

from pydantic import BaseModel


class PredictRequest(BaseModel):
    model_name: str
    image_base64: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
