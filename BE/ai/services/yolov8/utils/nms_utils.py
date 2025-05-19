# services/yolov8/utils/nms_utils.py
import numpy as np
import cv2


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
        seg = det.get("seg") or []
        xmin, ymin, xmax, ymax = det["bbox"]

        # 1) Bounding Box가 이미지 밖으로 벗어나면 제외
        if xmin <= tol or ymin <= tol or xmax >= img_w - tol or ymax >= img_h - tol:
            continue

        if seg:
            # 2) Segmentation 접촉 비율 판정 (이미지 경계 기준)
            total = len(seg)
            contact = sum(
                1
                for x, y in seg
                if x <= tol or y <= tol or x >= img_w - tol or y >= img_h - tol
            )
            if contact / total >= min_ratio:
                continue

            # 3) Mask 면적 대비 Bounding Box 면적 판정
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            pts = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)
            mask_crop = mask[ymin:ymax, xmin:xmax]
            mask_area = cv2.countNonZero(mask_crop)
            bbox_area = max(1, (xmax - xmin) * (ymax - ymin))
            if mask_area / bbox_area > area_thresh:
                continue

        filtered.append(det)
    return filtered
