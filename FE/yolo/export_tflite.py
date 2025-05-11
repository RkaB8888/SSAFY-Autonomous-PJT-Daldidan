from ultralytics import YOLO
model = YOLO('../runs/detect/apple-detector-v14/weights/best.pt')
model.export(format="onnx", dynamic=False, simplify=True)
