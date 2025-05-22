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


def postprocess_bbox(
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


def postprocess_seg(
    outputs,
    scale,
    pad,
    original_shape,
    conf_thres: float,
    iou_thres: float,
    target_class_id: int = 47,
):
    # 1) unpack
    pred = outputs[0][0].T  # (8400, 채널수)
    proto = outputs[1][0]  # (H_proto, W_proto, proto채널수)

    print(f"[seg] pred shape: {pred.shape}")
    print(f"[seg] proto shape: {proto.shape}")

    # 2) 채널 수 계산
    total_ch = pred.shape[1]
    proto_ch = proto.shape[2]
    num_classes = total_ch - 5 - proto_ch

    # 3) slice: bbox, raw logits, mask coeffs
    boxes = pred[:, :4]
    raw_obj = pred[:, 4]  # objectness logits
    print("raw_obj:", raw_obj)
    raw_cls = pred[:, 5 : 5 + num_classes]  # class logits
    print("raw_cls:", raw_cls)
    mask_coeffs = pred[:, 5 + num_classes : 5 + num_classes + proto_ch]

    # 4) sigmoid → 확률로 변환
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    objectness = sigmoid(raw_obj)  # [8400]
    print("objectness:", objectness)
    class_probs = sigmoid(raw_cls)  # [8400, num_classes]
    print("class_probs:", class_probs)
    print(f"[seg] objectness min={objectness.min():.6f}, max={objectness.max():.6f}")
    print(
        f"[seg] class_probs[:,{target_class_id}] min={class_probs[:,target_class_id].min():.6f}, max={class_probs[:,target_class_id].max():.6f}"
    )

    # 5) apple score 계산
    apple_scores = objectness * class_probs[:, target_class_id]
    print(
        f"[seg DEBUG] apple_scores min={apple_scores.min():.6f}, max={apple_scores.max():.6f}"
    )

    # 6) score threshold 필터링
    keep = apple_scores > conf_thres
    if not keep.any():
        return []

    boxes = boxes[keep]
    mask_coeffs = mask_coeffs[keep]
    scores = apple_scores[keep]  # NMS에 사용할 점수

    # 7) 좌표 복원
    boxes[:, 0] *= config.INPUT_SIZE[0]
    boxes[:, 1] *= config.INPUT_SIZE[1]
    boxes[:, 2] *= config.INPUT_SIZE[0]
    boxes[:, 3] *= config.INPUT_SIZE[1]
    boxes_xyxy = xywh2xyxy(boxes)
    pad_w, pad_h = pad
    boxes_xyxy[:, [0, 2]] -= pad_w
    boxes_xyxy[:, [1, 3]] -= pad_h
    boxes_xyxy /= scale

    # 8) NMS
    nms_idx = tf.image.non_max_suppression(
        boxes=boxes_xyxy,
        scores=scores,
        max_output_size=3,
        iou_threshold=iou_thres,
        score_threshold=conf_thres,
    ).numpy()
    boxes_xyxy = boxes_xyxy[nms_idx]
    mask_coeffs = mask_coeffs[nms_idx]

    # 9) mask 계산 & contour
    proto_flat = proto.transpose(2, 0, 1).reshape(proto_ch, -1)
    mask_logits = mask_coeffs.dot(proto_flat)  # (N_kept, H_proto*W_proto)

    results = []
    for i, box in enumerate(boxes_xyxy):
        mask = mask_logits[i].reshape(proto.shape[0], proto.shape[1])
        mask = cv2.resize(mask, (original_shape[0], original_shape[1]))
        bin_mask = (mask > 0).astype(np.uint8)  # Android 방식 threshold
        contours, _ = cv2.findContours(
            bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        seg = [cnt.squeeze().tolist() for cnt in contours if cnt.shape[0] >= 3]
        results.append({"bbox": box.tolist(), "seg": seg})

    return results
