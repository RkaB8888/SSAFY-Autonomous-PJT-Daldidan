# ai/api/v1/routes.py

from fastapi import APIRouter, File, UploadFile, Form
from schemas.api import DummyPredictResponse
import random

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "OK"}


@router.post("/dummy_predict", response_model=DummyPredictResponse)
async def dummy_predict(id: int = Form(...), image: UploadFile = File(...)):
    # 실제 모델 호출 대신 랜덤값 생성
    sugar = round(random.uniform(10.0, 16.0), 2)
    return DummyPredictResponse(id=id, sugar_content=sugar)
