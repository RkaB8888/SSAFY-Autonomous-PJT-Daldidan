from fastapi import APIRouter, HTTPException
from schemas.predict import PredictRequest, PredictResponse
from services.predict_service import predict

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "AI server is running"}


@router.post("/predict", response_model=PredictResponse)
async def predict_image(request: PredictRequest):
    try:
        result = predict(request.model_name, request.image_base64)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
