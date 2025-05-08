# ai/schemas/predict.py
from pydantic import BaseModel


class PredictRequest(BaseModel):
    id: int


class PredictResponse(BaseModel):
    id: int
    predict_sugar_content: float
