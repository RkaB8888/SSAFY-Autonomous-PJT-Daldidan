# services/yolov8/inference/predictor_pt.py

from ultralytics import YOLO
from PIL import Image
import numpy as np

from services.yolov8.utils.nms_utils import (
    remove_enclosing_big_boxes,
    remove_cutoff_with_area,
)


class YoloV8PtSegPredictor:
    def __init__(self, model_size: str = "m"):
        self.model_size = model_size
        self.model = YOLO(f"yolov8{model_size}-seg.pt")

    def predict(self, image: Image.Image | np.ndarray):
        # 1) 모델 추론 및 결과 수집
        results = self.model(image, conf=0.2, iou=0.8, classes=47)
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
        # 2) 겹친 큰 박스 제거
        filtered = remove_enclosing_big_boxes(detections, contain_thresh=0.9)

        # 3) 잘린 사과 컷오프 (mask_area / bbox_area 비율 기준)
        if isinstance(image, Image.Image):
            img_w, img_h = image.size
        else:
            img_h, img_w = image.shape[:2]
        # area_thresh: mask 영역이 bbox 면적의 몇 % 이상일 때 통과할지 설정
        filtered = remove_cutoff_with_area(
            filtered, img_w=img_w, img_h=img_h, tol=5, min_ratio=0.05, area_thresh=1
        )

        return filtered
        # return detections
