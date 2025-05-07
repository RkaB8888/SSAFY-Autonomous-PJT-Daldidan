# ai/schemas/predict.py
from typing import Literal
from pydantic import BaseModel


class SugarPredictResponse(BaseModel):
    label: Literal["sugar_content"]
    confidence: float


PredictResponse = SugarPredictResponse  # 현재는 하나뿐
