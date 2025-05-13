from ultralytics import YOLO

# 1. 모델 불러오기 (.pt 모델)
model = YOLO('./runs/detect/apple_detector_v214/weights/best.pt')  # 또는 'runs/train/exp/weights/best.pt'

# 2. TFLite로 export + 후처리(NMS) 포함
model.export(format='tflite', nms=True)
