# ai/api/v1/routes.py

from fastapi import APIRouter, Form, HTTPException
from schemas.api import DummyPredictResponse
import base64
import random

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "OK"}

@router.post("/dummy_predict", response_model=DummyPredictResponse)
async def dummy_predict(id: int = Form(...), image_base64: str = Form(...)):
    try:
        # # base64 문자열에서 헤더 제거 (data:image/png;base64, ... 부분 제거)
        # if image_base64.startswith("data:image"):
        #     header, base64_data = image_base64.split(",", 1)
        # else:
        #     base64_data = image_base64

        # # base64 디코딩
        # image_bytes = base64.b64decode(base64_data)
        
        # prefix 제거 로직 필요 없음 → 바로 디코딩
        image_bytes = base64.b64decode(image_base64)

        # (선택) 이미지 파일로 저장할 경우
        # with open(f"received_image_{id}.png", "wb") as f:
        #     f.write(image_bytes)

        # (모델 호출 대신 랜덤값 생성)
        sugar = round(random.uniform(10.0, 16.0), 2)

        return DummyPredictResponse(id=id, sugar_content=sugar)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding error: {str(e)}")
