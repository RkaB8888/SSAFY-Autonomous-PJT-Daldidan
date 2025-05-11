from ultralytics import YOLO

model = YOLO('../../runs/detect/apple-detector-v14/weights/best.pt')
results = model('./images/val/orange.jpg')

# 예측된 박스, 클래스, conf 값들 확인
for box in results[0].boxes:
    print(box.xyxy, box.conf, box.cls)
