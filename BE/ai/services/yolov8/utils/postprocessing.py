# services/yolov8/utils/postprocessing.py

import numpy as np
import tensorflow as tf
import cv2
from services.yolov8 import config


def xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    """[x_center, y_center, w, h] → [x1, y1, x2, y2]"""
    x, y, w, h = xywh.T
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def postprocess_custom(
    outputs: np.ndarray,
    scale: float,
    pad: tuple,
    original_shape: tuple,
    conf_thres: float,
    iou_thres: float,
):
    """
    YOLOv8 custom 모델용 후처리 함수 (단일 클래스 + conf 기반)
    output: [1, N, 6] → [x, y, w, h, conf, class]
    """
    if outputs.ndim == 3:
        outputs = outputs[0]  # shape: [N, 6]

    # conf 필터링
    boxes_conf = outputs[:, 4]
    keep = boxes_conf > conf_thres
    outputs = outputs[keep]

    if outputs.shape[0] == 0:
        return []

    bboxes_xyxy = xywh2xyxy(outputs[:, 0:4])

    # padding, scale 복원
    pad_w, pad_h = pad
    bboxes_xyxy[:, [0, 2]] -= pad_w
    bboxes_xyxy[:, [1, 3]] -= pad_h
    bboxes_xyxy /= scale

    # clip to image
    w, h = original_shape
    bboxes_xyxy[:, [0, 2]] = np.clip(bboxes_xyxy[:, [0, 2]], 0, w)
    bboxes_xyxy[:, [1, 3]] = np.clip(bboxes_xyxy[:, [1, 3]], 0, h)

    # NMS 적용 (OpenCV 사용)
    scores = outputs[:, 4]
    indices = cv2.dnn.NMSBoxes(
        bboxes_xyxy.tolist(), scores.tolist(), conf_thres, iou_thres
    )

    if len(indices) == 0:
        return []

    indices = indices.flatten()
    return bboxes_xyxy[indices].tolist()


def postprocess_coco(
    outputs: np.ndarray,
    scale: float,
    pad: tuple,
    original_shape: tuple,
    conf_thres: float,
    iou_thres: float,
    target_class_id: int = 47,
):
    """
    YOLOv8 COCO 모델용 후처리 함수
    output: [1, N, 85] → [x, y, w, h, cls1, cls2, ..., cls80]
    """
    if outputs.ndim == 3:
        outputs = outputs[0]
    if outputs.shape[0] < outputs.shape[1]:
        outputs = outputs.T  # (N, 85) 형식으로 맞춤

    bboxes = []
    scores = []

    for pred in outputs:
        cls_probs = pred[4:]
        cls_id = np.argmax(cls_probs)
        conf = cls_probs[cls_id]
        if conf < conf_thres or cls_id != target_class_id:
            continue

        # ── ① 정규화 좌표를 픽셀 단위로 ─────────────────────────
        xc, yc, w, h = pred[:4]
        xc *= config.INPUT_SIZE[0]
        yc *= config.INPUT_SIZE[1]
        w *= config.INPUT_SIZE[0]
        h *= config.INPUT_SIZE[1]
        bbox = xywh2xyxy(np.array([[xc, yc, w, h]]))[0]  # [x1,y1,x2,y2]

        bboxes.append(bbox)
        scores.append(conf)

    if not bboxes:
        return []

    bboxes = np.array(bboxes)
    scores = np.array(scores)

    # scale & padding 복원
    pad_w, pad_h = pad
    bboxes[:, [0, 2]] -= pad_w
    bboxes[:, [1, 3]] -= pad_h
    bboxes /= scale

    # 이미지 경계 클립
    w, h = original_shape
    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, w)
    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, h)

    # NMS
    nms_indices = tf.image.non_max_suppression(
        boxes=bboxes,
        scores=scores,
        max_output_size=100,
        iou_threshold=iou_thres,
        score_threshold=conf_thres,
    ).numpy()

    return bboxes[nms_indices].tolist()
