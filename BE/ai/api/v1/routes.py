# ai>api>v1>routes.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from schemas.predict import PredictResponse
from services.predict_service import predict

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "AI server is running"}


@router.post("/predict", response_model=PredictResponse)
async def predict_image(
    model_name: str = Form(...),
    image: UploadFile = File(...),
):
    try:
        image_bytes = await image.read()
        result = predict(model_name, image_bytes)  # ← 바이트 전달
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
