from ultralytics import YOLO
from PIL import Image

# YOLO 모델 로드
model = YOLO('yolov8n.pt')  # 작은 경량화 YOLO 모델

# 이미지 경로
img_path = "full_image.jpg"

# 이미지 detection
results = model.predict(img_path)

# 첫 번째 detection bbox 가져오기
boxes = results[0].boxes.xyxy.cpu().numpy()

if len(boxes) > 0:
    x1, y1, x2, y2 = boxes[0].astype(int)
    print(f"Detected bbox: {x1}, {y1}, {x2}, {y2}")

    # 이미지 crop
    image = Image.open(img_path)
    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_image.save("test_apple.jpg")

else:
    print("사과 감지 못함")

# 이후 cropped_image를 CNN predict에 전달
