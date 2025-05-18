# ai/api/v1/routes.py
import base64, io, time, os
import imghdr
from typing import Optional, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, ImageDraw
from io import BytesIO
from datetime import datetime

from schemas.predict import PredictResponse, ApplePred, BBox, Segmentation
from services.predict_service import predict  # crop → 당도 추정
from services.detect_service import detect  # ▶︎ YOLO 등 (bytes → list[dict])

"""
-------추론 모델------------------------
{ 
    cnn_lgbm_bbox,
    cnn_lgbm_seg,
    lgbm_bbox,
    lgbm_seg,
    xgb_bbox,
    xgb_seg,
    model_jmk2,
}
-------인식 모델------------------------
{ 
    yolov8,
    yolov8_pt,
}
{
    bbox_int8,
    seg_float16,
    seg_float32,
    s,
    m,
    l,
}
"""
# -----------------------------
# 사용할 모델 상수 정의
# -----------------------------
# 사과 인식 모델: detect()에 전달할 이름 및 버전
DETECT_MODEL_NAME: str = "yolov8_pt"
DETECT_MODEL_VERSION: str = "m"
# 당도 추론 모델: predict()에 전달할 모델 식별자
PREDICT_MODEL_NAME: str = "cnn_lgbm_bbox"
# -----------------------------

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "AI server is running"}


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
            image.file.seek(0)
        else:
            img_bytes = base64.b64decode(image_base64)

        # 📁 저장 디렉토리 생성
        save_dir = "tmp/uploads"
        os.makedirs(save_dir, exist_ok=True)

        # 📸 저장 파일명: predict_20240515_213803.jpg 형식
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = imghdr.what(None, h=img_bytes) or "jpg"
        filename = f"predict_{timestamp}.{ext}"
        save_path = os.path.join(save_dir, filename)

        with open(save_path, "wb") as f:
            f.write(img_bytes)

        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # 2️⃣  사과 탐지 -------------------------------------------------------------
    print("[/predict] 🔍 detect() 호출 시작")

    apples = detect(DETECT_MODEL_NAME, pil_img, version=DETECT_MODEL_VERSION)
    print(f"[/predict] 🔍 사과 탐지 결과: {len(apples)}개")
    if not apples:
        return PredictResponse(results=[])

    # 🔴 바운딩 박스 그리기용 복제본 생성
    draw_img = pil_img.copy()
    draw = ImageDraw.Draw(draw_img)

    # 3️⃣  각 사과 영역 crop → 당도 추정 -----------------------------------------
    results: List[ApplePred] = []
    for idx, det in enumerate(apples):
        xmin, ymin, xmax, ymax = det["bbox"]
        crop = pil_img.crop((xmin, ymin, xmax, ymax))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        # 당도 추론
        sugar = predict(PREDICT_MODEL_NAME, image_bytes)

        # 🔴 박스 시각화
        draw.rectangle(
            [int(xmin), int(ymin), int(xmax), int(ymax)], outline="red", width=4
        )
        text_y = int(ymin) - 10 if ymin > 10 else int(ymin) + 10
        draw.text(
            (int(xmin), text_y),
            f"id={idx} | {sugar:.2f}",
            fill="red",
            stroke_width=1,
            stroke_fill="white",
        )

        # 🔴 segmentation 외곽선 그리기
        if det.get("seg"):
            pts = [(int(x), int(y)) for x, y in det["seg"]]
            # 닫힌 폴리곤으로 그림
            draw.line(pts + [pts[0]], fill="blue", width=2)

        item = ApplePred(
            id=idx,
            sugar_content=float(sugar),
            bbox=BBox(
                xmin=int(xmin),
                ymin=int(ymin),
                xmax=int(xmax),
                ymax=int(ymax),
            ),
            segmentation=Segmentation(points=pts) if det.get("seg") else None,
        )
        results.append(item)

    # ✅ 시각화 이미지 저장 -----------------------------------------
    drawn_path = os.path.join(save_dir, f"predict_{timestamp}_drawn.{ext}")
    draw_img.save(drawn_path)
    print(f"✅ 시각화 이미지 저장: {drawn_path}")

    # 4️⃣  응답 + 로그 -----------------------------------------------------------
    print(
        f"[/predict] apples={len(results)}  elapsed={(time.perf_counter()-t0)*1000:.1f} ms"
    )
    return PredictResponse(results=results)
