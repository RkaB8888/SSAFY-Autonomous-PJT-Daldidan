# services/yolov8/inference/predictor_pt.py

from ultralytics import YOLO
from PIL import Image
import numpy as np


class YoloV8PtSegPredictor:
    def __init__(self, model_size: str = "m"):
        self.model_size = model_size
        self.model = YOLO(f"yolov8{model_size}-seg.pt")

    def predict(self, image: Image.Image | np.ndarray):
        results = self.model(image, conf=0.2, iou=0.45, classes=47)
        detections = []
        if results and results[0].masks is not None:
            for i, box in enumerate(results[0].boxes):
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())

                # ultralytics가 계산해 놓은 원본 좌표계 외곽점 (ndarray, shape=(N,2))
                mask_pts = results[0].masks.xy[i]
                # 하나의 contour로 감싸서 리스트로 반환
                seg_points = mask_pts.tolist()

                detections.append(
                    {
                        "bbox": [xmin, ymin, xmax, ymax],
                        "seg": seg_points,
                        "score": float(box.conf[0].item()),
                    }
                )
        return detections
