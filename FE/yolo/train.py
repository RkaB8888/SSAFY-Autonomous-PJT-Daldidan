from ultralytics import YOLO

# 모델 로드 (YOLOv8n)
model = YOLO('yolov8n.yaml')  # 또는 'yolov8n.pt'로 사전학습된 모델 사용

# 학습 시작
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='apple-detector',
)
