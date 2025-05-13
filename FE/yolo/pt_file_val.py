from ultralytics import YOLO

model = YOLO('./runs/detect/apple_detector_v214/weights/best.pt')
results = model('./images/val/apple_pear.jpg')

# 예측된 박스, 클래스, conf 값들 확인
for box in results[0].boxes:
    print(box.xyxy, box.conf, box.cls)
