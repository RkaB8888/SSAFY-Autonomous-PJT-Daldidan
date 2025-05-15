# ai/api/v1/routes.p y
import base64, io, time
from typing import Optional, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from schemas.predict import PredictResponse, ApplePred, BBox, Segmentation
from services.predict_service import predict  # crop → 당도 추정
from services.detect_service import detect  # ▶︎ YOLO 등 (bytes → list[dict])

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "AI server is running"}


"""
{ 추론 모델
cnn_lgbm_bbox,
cnn_lgbm_seg,
lgbm_bbox,
lgbm_seg,
xgb_bbox,
xgb_seg,
model_jmk2,
}
{ 인식 모델
yolov8_tflite
}
"""


@router.post("/predict", response_model=PredictResponse)
async def predict_image(
    image: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
):
    if (image is None and image_base64 is None) or (image and image_base64):
        raise HTTPException(
            status_code=400,
            detail="Exactly one of 'image' or 'image_base64' must be provided.",
        )

    t0 = time.perf_counter()

    # 1️⃣  이미지 디코딩 ----------------------------------------------------------
    try:
        if image is not None:
            img_bytes = await image.read()
        else:
            img_bytes = base64.b64decode(image_base64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # 2️⃣  사과 탐지 -------------------------------------------------------------
    # detect_apples : bytes/RGB → [{"bbox":(xmin,ymin,xmax,ymax), "seg": [[...]]}, ...]
    apples = detect("yolov8_tflite", pil_img)  # type: List[dict]
    if not apples:
        return PredictResponse(results=[])

    # 3️⃣  각 사과 영역 crop → 당도 추정 -----------------------------------------
    results: List[ApplePred] = []
    for idx, det in enumerate(apples):
        xmin, ymin, xmax, ymax = det["bbox"]
        crop = pil_img.crop((xmin, ymin, xmax, ymax))
        sugar = predict(
            "cnn_lgbm_bbox", crop
        )  # ← bytes/PIL 둘 중 하나에 맞춰 predict 수정

        item = ApplePred(
            id=idx,
            sugar_content=float(sugar),
            bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
            segmentation=Segmentation(points=det["seg"]) if det.get("seg") else None,
        )
        results.append(item)

        # 4️⃣  응답 + 로그 -----------------------------------------------------------
    print(
        f"[/predict] apples={len(results)}  elapsed={(time.perf_counter()-t0)*1000:.1f} ms"
    )
    return PredictResponse(results=results)
