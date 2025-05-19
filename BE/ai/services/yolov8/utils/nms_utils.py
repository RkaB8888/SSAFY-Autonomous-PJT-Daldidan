# services/yolov8/utils/nms_utils.py


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
