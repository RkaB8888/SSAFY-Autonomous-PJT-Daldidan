# services/yolov8/inference/predictor_pt.py

from ultralytics import YOLO
from PIL import Image
import numpy as np

from services.yolov8.config import MODEL_DIR, MODEL_FILES_PT
from services.yolov8.utils.nms_utils import (
    remove_enclosing_big_boxes,
    remove_cutoff_with_area,
    smooth_polygons,
)


class YoloV8PtSegPredictor:
    def __init__(self, model_size: str = "m"):
        # 1) 모델 파일명 선택
        if model_size not in MODEL_FILES_PT:
            raise ValueError(f"지원하지 않는 model_size '{model_size}' 입니다.")
        weight_name = MODEL_FILES_PT[model_size]

        # 2) 절대경로 생성
        weight_path = MODEL_DIR / weight_name
        if not weight_path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {weight_path}")

        # print(f"[INFO] YOLOv8 {model_size}-seg 모델 로드: {weight_path}")
        self.model = YOLO(str(weight_path))

    def predict(self, image: Image.Image | np.ndarray) -> list[dict]:
        # Determine image dimensions (needed for filtering)
        if isinstance(image, Image.Image):
            img_w, img_h = image.size
        else:
            img_h, img_w = image.shape[:2]

        results = self.model(image, conf=0.15, iou=0.45, classes=47)

        # .xyn 사용을 위해 조건 변경
        if (
            not results
            or results[0].masks is None
            or not hasattr(results[0].masks, "xyn")  # .xyn 확인
            or not results[0].masks.xyn  # .xyn이 비었는지 확인
        ):
            return []

        detections = []
        # results[0].masks.xyn 사용
        for i, segment_xy_normalized_np in enumerate(results[0].masks.xyn):  # .xyn 사용
            if i >= len(results[0].boxes):
                continue

            box_coords = results[0].boxes.xyxy[i].tolist()
            xmin, ymin, xmax, ymax = map(int, box_coords)
            score = float(results[0].boxes.conf[i].item())

            if xmax <= xmin or ymax <= ymin:
                continue

            smoothed_segments = smooth_polygons(
                [segment_xy_normalized_np],
                img_w,
                img_h,
                open_kernel=15,
                close_kernel=7,
                approx_epsilon=0.007,
            )

            if smoothed_segments:
                detections.append(
                    {
                        "bbox": [xmin, ymin, xmax, ymax],
                        "seg": smoothed_segments,
                        "score": score,
                    }
                )

        # --- 필터링 적용 ---
        # 필터링 함수들은 이제 스케일링된 'seg' 좌표를 사용하게 됩니다.
        # remove_cutoff_with_area 함수는 img_w, img_h를 올바르게 사용하게 됩니다.
        filtered_detections = remove_enclosing_big_boxes(detections, contain_thresh=0.9)
        # filtered_detections = remove_cutoff_with_area(
        #     filtered_detections,
        #     img_w=img_w,  # 원본 이미지 너비
        #     img_h=img_h,  # 원본 이미지 높이
        #     tol=5,
        #     min_ratio=0.05,
        #     area_thresh=1,  # 이 조건은 문제 없음
        # )

        # --- 최종 출력 형식 변환 ---
        results_json = []
        for det in filtered_detections:  # 필터링된 결과 사용
            seg_list_of_lists = det.get("seg")  # 스케일링된 [[점리스트]] 형태

            seg_flat: list[list[int]] = []
            if seg_list_of_lists and len(seg_list_of_lists) > 0:
                seg_flat.extend(seg_list_of_lists[0])

            if seg_flat:
                results_json.append(
                    {
                        "bbox": det["bbox"],
                        "seg": seg_flat,
                        "score": det["score"],
                    }
                )
        return results_json
