# ai>api>v1>routes.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from schemas.predict import PredictResponse
from services.predict_service import predict

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "AI server is running"}


# @router.post("/predict", response_model=PredictResponse)
# async def predict_image(id: int = Form(...), image: UploadFile = File(...)):
#     try:
#         image_bytes = await image.read()
#         result = predict("model_jhg2", image_bytes)  # ← 바이트 전달
#         return PredictResponse(id=id, predict_sugar_content=float(result["confidence"]))
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict", response_model=PredictResponse)
async def predict_image(id: int = Form(...),  image_base64: str = Form(...)):
    try:
        import base64
        image_bytes = base64.b64decode(image_base64)
        # image_bytes = await image.read()
        result = predict("model_jmk2", image_bytes)  # ← 바이트 전달
        return PredictResponse(id=id, predict_sugar_content=float(result))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))