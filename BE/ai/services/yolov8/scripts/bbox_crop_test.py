# services/yolov8/scripts/bbox_crop_test.py

from pathlib import Path
from PIL import Image, ImageDraw

from services import yolov8  # ← 자동 모델 등록을 트리거함 (registry.py 실행됨)
from services import detect_service


def draw_bboxes(image: Image.Image, bboxes: list[dict]) -> Image.Image:
    """bbox를 이미지 위에 시각화"""
    draw = ImageDraw.Draw(image)
    for box in bboxes:
        x1, y1, x2, y2 = box["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    return image


if __name__ == "__main__":
    # 테스트용 이미지 경로
    image_path = Path(
        # "services/yolov8/scripts/test_image.jpg"
        r"C:\Users\SSAFY\Desktop\phone.png"
    )  # ← 여기에 실제 테스트 이미지 넣어줘
    assert image_path.exists(), f"이미지 파일이 존재하지 않습니다: {image_path}"

    # 이미지 열기
    image = Image.open(image_path).convert("RGB")

    # YOLOv8 모델로 bbox 예측
    bboxes = detect_service.detect(
        model_name="yolov8",
        version="coco_int8",
        image=image,
    )

    print(f"✅ 감지된 bbox 개수: {len(bboxes)}")
    for i, box in enumerate(bboxes, 1):
        print(f"[{i}] {box}")

    # 시각화 후 저장
    image_with_boxes = draw_bboxes(image, bboxes)
    image_with_boxes.save("services/yolov8/scripts/output.jpg")
    print("✅ output.jpg 파일로 결과 저장 완료")
