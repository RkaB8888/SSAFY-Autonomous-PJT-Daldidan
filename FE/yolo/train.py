from ultralytics import YOLO

# 1. 모델 로드 (사전 학습된 YOLOv8n 모델)
model = YOLO('yolov8n.pt')  # 또는 yolov8n.yaml로 처음부터 학습 가능

# 2. 학습 수행
model.train(
    data='data.yaml',         # 학습 데이터 구성 yaml
    epochs=50,                # 학습 epoch 수
    imgsz=640,                # 입력 이미지 크기
    batch=16,                 # 배치 사이즈 (메모리에 따라 조정)
    name='apple_detector_v1', # 출력 폴더명 (runs/detect/apple_detector_v1)
    project='runs/detect',    # 저장될 루트 폴더
    workers=2,                # 데이터 로딩 스레드 수
    device=0                  # 0번 GPU 사용, CPU일 경우 'cpu'
)
