# ai>api>v1>routes.py
import time
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from schemas.predict import PredictResponse
from services.predict_service import predict

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "AI server is running"}


"""
{
cnn_lgbm_bbox,
cnn_lgbm_seg,
lgbm_bbox,
lgbm_seg,
xgb_bbox,
xgb_seg,
model_jmk2,
}
"""

# @router.post("/predict", response_model=PredictResponse)
# async def predict_image(id: int = Form(...), image: UploadFile = File(...)):
#     try:
#         image_bytes = await image.read()
#         result = predict("model_jhg2", image_bytes)  # ← 바이트 전달
#         return PredictResponse(id=id, predict_sugar_content=float(result["confidence"]))
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict", response_model=PredictResponse)
async def predict_image(id: int = Form(...), image_base64: str = Form(...)):
    start = time.perf_counter()
    try:
        import base64

        image_bytes = base64.b64decode(image_base64)
        # image_bytes = await image.read()
        result = predict("cnn_lgbm_bbox", image_bytes)  # ← 바이트 전달
        elapsed = time.perf_counter() - start
        print(f"[/predict] id={id}  elapsed={elapsed*1000:.1f} ms")
        return PredictResponse(id=id, predict_sugar_content=float(result))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
