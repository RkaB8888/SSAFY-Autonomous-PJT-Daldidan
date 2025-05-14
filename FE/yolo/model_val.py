from ultralytics import YOLO

# 모델 로드
model = YOLO('./best.pt')

# 평가할 데이터셋 경로 (val dataset)
metrics = model.val(data="data.yaml")

print(metrics.box.map)       # mAP@0.5
print(metrics.box.map50)     # mAP@0.5
print(metrics.box.map75)     # mAP@0.75
print(metrics.box.mAP)       # mAP@0.5:0.95
