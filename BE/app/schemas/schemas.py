# "HTTP 요청과 응답을 정형화(타입명시) 하기 위해 존재한다."
from pydantic import BaseModel
from typing import Optional


class PredictRequest(BaseModel):
    image_base64: str


class PredictResponse(BaseModel):
    predicted_brix: float
    confidence: Optional[float] = None
