# services/yolov8/utils/nms_utils.py
import numpy as np
import cv2
from itertools import chain


def area(box):
    xmin, ymin, xmax, ymax = box
    return max(0, xmax - xmin) * max(0, ymax - ymin)


def intersection_area(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def remove_enclosing_big_boxes(dets, contain_thresh: float = 0.8):
    """
    작은 박스 기준으로, 작은 박스 면적 중 contain_thresh 이상이
    다른 박스(큰 박스)와 겹치면 그 큰 박스를 제거.
    """
    keep = [True] * len(dets)
    for i, det_i in enumerate(dets):
        box_i = det_i["bbox"]
        area_i = area(box_i)
        for j, det_j in enumerate(dets):
            if i == j or not keep[j]:
                continue
            box_j = det_j["bbox"]
            # det_i가 작은 박스, det_j가 큰 박스라고 가정
            inter = intersection_area(box_i, box_j)
            # 작은 박스의 면적 대비 겹치는 비율 계산
            if inter / area_i >= contain_thresh and area(box_j) > area_i:
                keep[j] = False

    return [det for idx, det in enumerate(dets) if keep[idx]]


def remove_cutoff_with_area(
    dets,
    img_w: int,
    img_h: int,
    tol: int = 0,
    min_ratio: float = 0.05,
    area_thresh: float = 0.7,
):
    """
    이미지 경계에 걸쳐있는 객체 (잘린 사과)를 필터링합니다.
    기준:
    1) Bounding Box가 이미지 경계에 닿는지 여부
    2) Segmentation 폴리곤 점들 중 Bounding Box 경계에 닿는 점의 비율
    3) Segmentation Mask 면적 대비 Bounding Box 면적 비율

    Parameters:
        dets (list of dict):
            Detection 리스트. 각 dict은 다음 키를 가짐:
            - "bbox": [xmin, ymin, xmax, ymax]
            - "seg": List of [x, y] 폴리곤 포인트
            - "score": float
        img_w (int): 원본 이미지 너비 (픽셀 단위)
        img_h (int): 원본 이미지 높이 (픽셀 단위)
        tol (int): 경계 허용 오프셋 (픽셀). bbox 경계에서 tol 내에 접촉한 것으로 간주
        min_ratio (float): segmentation 폴리곤 접촉 비율 임계값.
            contact_points/total_points >= min_ratio 일 때 컷오프
        area_thresh (float): mask_area/bbox_area 비율 임계값.
            mask_area/bbox_area < area_thresh 일 때 컷오프
    """
    filtered = []
    for det in dets:
        seg_contours = det.get("seg", [])  # List of contours
        xmin, ymin, xmax, ymax = det["bbox"]

        # 1) bbox가 이미지 밖으로 닿으면 컷
        if xmin <= tol or ymin <= tol or xmax >= img_w - tol or ymax >= img_h - tol:
            continue

        if seg_contours:
            # 2) 모든 점(flatten)으로 접촉 비율 계산
            all_pts = list(chain.from_iterable(seg_contours))
            total = len(all_pts)
            contact = sum(
                1
                for x, y in all_pts
                if x <= tol or y <= tol or x >= img_w - tol or y >= img_h - tol
            )
            if contact / total >= min_ratio:
                continue

            # 3) mask_area / bbox_area 비율 계산 (잘린 경우 mask_area가 작아야 컷)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            # all_pts 를 하나의 contour로 변환
            pts = np.array(all_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)
            mask_crop = mask[ymin:ymax, xmin:xmax]
            mask_area = cv2.countNonZero(mask_crop)
            bbox_area = max(1, (xmax - xmin) * (ymax - ymin))
            if mask_area / bbox_area > area_thresh:
                continue

        filtered.append(det)
    return filtered


def smooth_polygons(
    polygons_normalized,
    img_w,
    img_h,
    open_kernel=17,
    close_kernel=19,
    approx_epsilon=0.005,
):
    """정규화된 폴리곤 좌표를 받아 스무딩 처리 후 절대 좌표 폴리곤을 반환합니다."""
    smoothed_polygons_abs = []
    for polygon_norm in polygons_normalized:
        denormalized_points = []
        for x_norm, y_norm in polygon_norm:
            x_abs = int(round(x_norm * img_w))
            y_abs = int(round(y_norm * img_h))
            denormalized_points.append([x_abs, y_abs])

        if not denormalized_points:
            continue

        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        points = np.array(denormalized_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(mask, [points], -1, color=255, thickness=cv2.FILLED)

        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
        )
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = approx_epsilon * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            smoothed_polygons_abs.append(approx_contour.reshape(-1, 2).tolist())

    return smoothed_polygons_abs
