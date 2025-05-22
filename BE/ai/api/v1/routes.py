# ai/api/v1/routes.py
import base64, io, time, os
import imghdr
from typing import Optional, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Query
from PIL import Image, ImageDraw
from datetime import datetime

from schemas.predict import PredictResponse, ApplePred, BBox, Segmentation
from services.predict_service import predict  # crop → 당도 추정
from services.detect_service import detect  # ▶︎ YOLO 등 (bytes → list[dict])

"""
-------추론 모델------------------------
{
    cnn_feature_enhanced_seg,
    cnn_feature_finetuning_seg,
    cnn_feature_maskcrop_seg,
    cnn_feature_seg,
    cnn_feature_seg_v2,
    cnn_lgbm_bbox,
    cnn_lgbm_seg,
    lgbm_bbox,
    lgbm_seg,
    model_a,
    xgb_bbox,
    xgb_seg,
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
    x,
}
"""
# -----------------------------
# 사용할 모델 상수 정의
# -----------------------------
# 사과 인식 모델: detect()에 전달할 이름 및 버전
DETECT_MODEL_NAME: str = "yolov8_pt"
DETECT_MODEL_VERSION: str = "l"
# 당도 추론 모델: predict()에 전달할 모델 식별자
PREDICT_MODEL_NAME: str = "cnn_feature_maskcrop_seg"
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

        # 전달받은 이미지 저장
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

        # pts_list 초기화
        pts_list = None

        # 🔧 segmentation이 있는 경우 마스크 기반으로 crop
        if det.get("seg"):
            # 1) 전체 크기의 빈 'L' 모드(흑백) 마스크 생성
            mask = Image.new("L", pil_img.size, 0)
            mask_draw = ImageDraw.Draw(mask)

            # det["seg"]는 [[x,y], …] 형태
            pts_list = [(int(x), int(y)) for x, y in det["seg"]]
            mask_draw.polygon(pts_list, fill=255)

            # 2) 원본 이미지에서 마스크 영역만 추출
            segmented = Image.new("RGB", pil_img.size)
            segmented.paste(pil_img, mask=mask)

            # 3) bbox 범위로 잘라내기
            crop = segmented.crop((xmin, ymin, xmax, ymax))

        else:
            # 기본 bbox crop
            crop = pil_img.crop((xmin, ymin, xmax, ymax))

        # 디버그용 crop 저장
        # crop_debug_path = os.path.join(save_dir, f"{timestamp}_crop_{idx}.jpg")
        # crop.save(crop_debug_path)
        # print(f"🔍 Crop saved: {crop_debug_path}")

        # 4) 당도 추론을 위한 JPEG 바이트로 변환
        buf = io.BytesIO()
        crop.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        sugar = predict(
            PREDICT_MODEL_NAME, image_bytes
        )  # ← bytes/PIL 둘 중 하나에 맞춰 predict 수정
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

        # 🔴 segmentation 윤곽선 그리기
        if pts_list:
            draw.polygon(pts_list, outline="blue", width=2)

        item = ApplePred(
            id=idx,
            sugar_content=float(sugar),
            bbox=BBox(
                xmin=int(xmin),
                ymin=int(ymin),
                xmax=int(xmax),
                ymax=int(ymax),
            ),
            segmentation=Segmentation(points=pts_list) if pts_list else None,
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
