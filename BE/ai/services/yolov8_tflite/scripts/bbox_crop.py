# ai/services/tflite_topython/bbox_crop.py
import cv2
import numpy as np
import tensorflow as tf
import os
import tempfile  # 임시 파일 처리를 위해 필요 (백엔드용 함수에서 사용)

# --- 모델 및 설정 값 (백엔드 환경에 맞게 반드시 수정!) ---
# TODO: 실제 백엔드 환경에 맞는 TFLite 모델 파일 경로로 수정해야 합니다.
TFLITE_MODEL_PATH = "./yolov8n_int8.tflite"

INPUT_SIZE = (640, 640)  # 모델 입력 크기 (너비, 높이)
CONFIDENCE_THRESHOLD = 0.25  # 객체 탐지 신뢰도 임계값 (필요시 조정)
IOU_THRESHOLD = 0.45  # NMS (Non-Maximum Suppression) IoU 임계값

# 모델 학습에 사용된 클래스 이름 목록
COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
CLASS_NAMES = COCO_CLASS_NAMES  # 사용하려는 최종 클래스 이름 목록

# 백엔드에서 탐지 및 추가 분석하려는 대상 클래스 ID 목록
TARGET_CLASS_IDS = []
try:
    apple_id = CLASS_NAMES.index("apple")
    TARGET_CLASS_IDS.append(apple_id)
    # print(f"[INFO] 'apple' class ID is: {apple_id}") # 백엔드 로그로 대체
except ValueError:
    print(
        "[ERROR] 'apple' class not found in CLASS_NAMES. Check CLASS_NAMES configuration."
    )

try:
    donut_id = CLASS_NAMES.index("donut")
    TARGET_CLASS_IDS.append(donut_id)
    # print(f"[INFO] 'donut' class ID is: {donut_id}") # 백엔드 로그로 대체
except ValueError:
    print(
        "[ERROR] 'donut' class not found in CLASS_NAMES. Check CLASS_NAMES configuration."
    )


if not TARGET_CLASS_IDS:
    print(
        "[ERROR] No target classes ('apple', 'donut') found or specified. Detection functions will return empty results."
    )


# --- TFLite 인터프리터 로드 ---
# 백엔드 애플리케이션 시작 시 이 파일이 임포트될 때 모델이 로드됩니다.
# 여러 워커 프로세스를 사용하는 경우 모델 로딩 전략을 추가로 고려해야 할 수 있습니다.
interpreter = None
input_details = None
output_details = None

try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("[INFO] TFLite interpreter (yolov8n_int8.tflite) loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load TFLite interpreter from {TFLITE_MODEL_PATH}: {e}")
    # 모델 로드 실패 시 interpreter는 None 상태로 유지


def preprocess_image(image_np: np.ndarray, input_size: tuple) -> tuple:
    """
    이미지(numpy 배열)를 모델 입력 크기에 맞게 전처리합니다.
    Args:
        image_np (np.ndarray): 입력 이미지의 numpy 배열 (CV2로 읽은 BGR 또는 RGB).
        input_size (tuple): 모델 입력 크기 (width, height).
    Returns:
        tuple: (전처리된 이미지 numpy 배열, 원본 너비, 원본 높이, 스케일 비율, 좌측 패딩, 상단 패딩)
               이미지 처리에 실패하면 (None, 0, 0, 0, 0, 0) 반환.
    """
    if image_np is None or image_np.size == 0:
        # print("[ERROR] preprocess_image received None or empty image.") # 백엔드 로그로 대체
        return None, 0, 0, 0, 0, 0

    original_height, original_width = image_np.shape[:2]
    h, w = input_size

    # 이미지 크기 조절 및 패딩 (Letterboxing)
    scale = min(w / original_width, h / original_height)
    nw, nh = int(scale * original_width), int(scale * original_height)
    image_resized = cv2.resize(image_np, (nw, nh))

    top = (h - nh) // 2
    bottom = h - nh - top
    left = (w - nw) // 2
    right = w - nw - left
    image_padded = cv2.copyMakeBorder(
        image_resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    # BGR to RGB 및 정규화 (0-1 범위)
    image_rgb = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0
    input_data = np.expand_dims(image_normalized, axis=0)  # 배치 차원 추가

    return input_data, original_width, original_height, scale, left, top


def postprocess_output_coco(
    output_data: np.ndarray,
    original_width: int,
    original_height: int,
    scale: float,
    pad_left: int,
    pad_top: int,
    conf_thresh: float,
    iou_thresh: float,
    input_size: tuple,
) -> tuple:
    """
    모델 출력 후처리 및 NMS 적용 후 원본 이미지 기준 바운딩 박스 좌표, 점수, 클래스 ID를 반환합니다.
    Args:
        output_data (np.ndarray): TFLite 모델의 raw 출력 데이터 (예: shape (1, 84, 8400)).
        original_width (int): 원본 이미지의 너비.
        original_height (int): 원본 이미지의 높이.
        scale (float): 전처리 시 적용된 스케일 비율.
        pad_left (int): 전처리 시 적용된 좌측 패딩.
        pad_top (int): 전처리 시 적용된 상단 패딩.
        conf_thresh (float): 객체 탐지 신뢰도 임계값.
        iou_thresh (float): NMS IoU 임계값.
        input_size (tuple): 모델 입력 크기 (width, height).
    Returns:
        tuple: (원본 이미지 기준 바운딩 박스 좌표 리스트 [[x1, y1, x2, y2], ...],
               해당 바운딩 박스의 신뢰도 점수 리스트,
               해당 바운딩 박스의 클래스 ID 리스트)
               탐지된 객체가 없으면 ([], [], []) 반환.
    """
    # TFLite 모델 출력 형태에 맞게 변환 (YOLOv8 기준 (1, 84, 8400) -> (8400, 84))
    # output_data = np.squeeze(output_data) # 배치 차원 제거
    # predictions = output_data.T # (84, 8400) -> (8400, 84)
    # 또는 더 간단하게:
    predictions = np.squeeze(output_data).T  # Shape (8400, 84)

    boxes = []
    scores = []
    class_ids_pred = []

    # 각 예측 결과 처리
    for pred in predictions:
        # 처음 4개는 바운딩 박스 좌표 [xc, yc, w, h] (모델 입력 기준 정규화 또는 픽셀 값)
        # YOLOv8 TFLite는 보통 모델 입력 기준 픽셀 값으로 나옵니다.
        xc_raw, yc_raw, w_raw, h_raw = pred[:4]
        # 나머지 클래스 점수 (예: 80개 클래스)
        class_scores_raw = pred[4:]

        # 가장 높은 점수를 가진 클래스 ID와 해당 점수(신뢰도)를 찾음
        current_class_id = np.argmax(class_scores_raw)
        current_conf = class_scores_raw[current_class_id]

        # 신뢰도 임계값 미만이면 건너뛰기
        if current_conf < conf_thresh:
            continue

        # 바운딩 박스 좌표를 [x1, y1, x2, y2] 형태로 변환 (모델 입력 기준 픽셀)
        x1 = xc_raw - w_raw / 2
        y1 = yc_raw - h_raw / 2
        x2 = xc_raw + w_raw / 2
        y2 = yc_raw + h_raw / 2

        boxes.append([x1, y1, x2, y2])  # 모델 입력 기준 픽셀 좌표
        scores.append(current_conf)
        class_ids_pred.append(current_class_id)

    # 신뢰도 임계값 통과한 객체가 없으면 빈 리스트 반환
    if not boxes:
        return [], [], []

    # NMS 적용
    # TensorFlow NMS 함수는 [y1, x1, y2, x2] 형태와 0~1 범위 정규화 좌표를 선호
    normalized_boxes_for_nms = []
    for box_px in boxes:
        x1_px, y1_px, x2_px, y2_px = box_px
        # 모델 입력 크기 (640x640 등) 기준으로 정규화
        y1_norm = y1_px / input_size[1]  # 높이로 정규화
        x1_norm = x1_px / input_size[0]  # 너비로 정규화
        y2_norm = y2_px / input_size[1]
        x2_norm = x2_px / input_size[0]
        normalized_boxes_for_nms.append(
            [y1_norm, x1_norm, y2_norm, x2_norm]
        )  # NMS 입력 형태

    if not normalized_boxes_for_nms:
        return [], [], []  # NMS 적용할 박스가 없으면 빈 리스트 반환

    # tf.image.non_max_suppression은 GPU/TPU에 최적화되어 있지만, CPU에서도 작동합니다.
    # numpy 배열을 tf.constant로 변환하여 사용합니다.
    selected_indices = tf.image.non_max_suppression(
        boxes=np.array(
            normalized_boxes_for_nms, dtype=np.float32
        ),  # NMS는 float32 필요
        scores=np.array(scores, dtype=np.float32),  # NMS는 float32 필요
        max_output_size=100,  # 최대 결과 수
        iou_threshold=iou_thresh,
        score_threshold=conf_thresh,  # NMS에도 score_threshold 적용 가능 (선택 사항)
    ).numpy()  # 결과를 numpy 배열로 변환

    final_boxes_orig_coords = []
    final_scores = []
    final_class_ids = []

    # NMS 통과한 박스들만 선택하고 원본 이미지 크기 기준으로 변환
    for index in selected_indices:
        box = boxes[index]  # 모델 입력 기준 픽셀 좌표 [x1,y1,x2,y2]

        # 패딩 및 스케일 역변환을 통해 원본 이미지 기준으로 좌표 복원
        x1_on_resized = box[0] - pad_left
        y1_on_resized = box[1] - pad_top
        x2_on_resized = box[2] - pad_left
        y2_on_resized = box[3] - pad_top

        x1_orig = x1_on_resized / scale
        y1_orig = y1_on_resized / scale
        x2_orig = x2_on_resized / scale
        y2_orig = y2_on_resized / scale

        # 결과 좌표를 정수형으로 변환 (픽셀 좌표)
        # 원본 이미지 크기를 벗어나지 않도록 최종적으로 클리핑
        x1_final = max(0, int(x1_orig))
        y1_final = max(0, int(y1_orig))
        x2_final = min(original_width, int(x2_orig))
        y2_final = min(original_height, int(y2_orig))

        final_boxes_orig_coords.append([x1_final, y1_final, x2_final, y2_final])
        final_scores.append(scores[index])
        final_class_ids.append(class_ids_pred[index])

    return final_boxes_orig_coords, final_scores, final_class_ids


def detect_target_objects_and_get_info(image_np: np.ndarray) -> list:
    """
    이미지(numpy 배열)에서 TARGET_CLASS_IDS에 해당하는 객체(사과, 도넛 등)를 탐지하고 정보를 반환합니다.

    Args:
        image_np (np.ndarray): 입력 이미지의 numpy 배열 (CV2로 읽은 BGR 또는 RGB).

    Returns:
        list: 탐지된 객체 정보 리스트. 각 항목은 딕셔너리 형태입니다.
              예: [{"bbox": [x1, y1, x2, y2], "score": score, "class_id": id, "label": "apple" or "donut"}, ...]
              모델 로드 실패 또는 탐지된 객체가 없으면 빈 리스트 [] 반환.
    """
    if interpreter is None or not TARGET_CLASS_IDS:
        # print("[ERROR] Interpreter not loaded or TARGET_CLASS_IDS not set.") # 백엔드 로그로 대체
        return []

    if image_np is None or image_np.size == 0:
        # print("[ERROR] detect_target_objects_and_get_info received None or empty image_np.") # 백엔드 로그로 대체
        return []

    # 전처리
    input_data, orig_w, orig_h, scale_ratio, pad_l, pad_t = preprocess_image(
        image_np, INPUT_SIZE
    )

    if input_data is None:
        # print("[ERROR] Image preprocessing failed.") # 백엔드 로그로 대체
        return []

    # 추론 실행
    try:
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
    except Exception as e:
        print(f"[ERROR] TFLite inference failed: {e}")
        return []

    # 후처리 및 바운딩 박스, 점수, 클래스 ID 추출 (원본 이미지 기준)
    boxes_orig, scores, class_ids = postprocess_output_coco(
        output_data,
        orig_w,
        orig_h,
        scale_ratio,
        pad_l,
        pad_t,
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD,
        INPUT_SIZE,
    )

    detected_objects_info = []
    # TARGET_CLASS_IDS에 해당하는 객체만 필터링하여 정보 취합
    for i in range(len(boxes_orig)):
        if class_ids[i] in TARGET_CLASS_IDS:
            # 클래스 ID가 유효한지 확인 (CLASS_NAMES 범위 내에 있는지)
            label = (
                CLASS_NAMES[class_ids[i]]
                if class_ids[i] < len(CLASS_NAMES)
                else f"Unknown_{class_ids[i]}"
            )

            # 유효성 검사: bbox 좌표가 올바른 형태인지 간단히 확인
            bbox = boxes_orig[i]
            if not (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(coord, int) for coord in bbox)
            ):
                print(
                    f"[WARNING] Skipping object {i} due to invalid bbox format: {bbox}"
                )
                continue

            detected_objects_info.append(
                {
                    "bbox": bbox,  # [x1, y1, x2, y2] (원본 이미지 기준 int 좌표)
                    "score": float(scores[i]),  # JSON 직렬화를 위해 float으로 변환
                    "class_id": int(class_ids[i]),  # JSON 직렬화를 위해 int으로 변환
                    "label": label,  # 사과 또는 도넛 라벨 포함
                }
            )

    # print(f"[INFO] Detected {len(detected_objects_info)} target object(s).") # 백엔드 로그로 대체
    return detected_objects_info  # 탐지된 객체 정보 리스트 반환


def crop_and_save_object_bboxes_temp(image_np: np.ndarray, objects_info: list) -> list:
    """
    원본 이미지에서 주어진 객체들의 바운딩 박스 영역들을 잘라내어 임시 파일로 저장합니다.
    이 함수는 백엔드 API에서 사용하기 적합하며, 임시 파일을 생성하고 그 경로 목록을 반환합니다.
    API 처리 완료 후 반환된 경로의 파일들을 삭제해야 합니다.

    Args:
        image_np (np.ndarray): 원본 이미지의 numpy 배열 (CV2로 읽은 BGR 또는 RGB).
        objects_info (list): detect_target_objects_and_get_info 함수에서 반환된 객체 정보 리스트.
                             각 항목은 {"bbox": [x1, y1, x2, y2], "label": "...", ...} 형태.

    Returns:
        list: 잘라낸 이미지 임시 파일들의 전체 경로 문자열 목록.
              이미지 처리에 실패하거나 자를 영역이 없으면 빈 리스트 [] 반환.
    """
    if image_np is None or image_np.size == 0:
        # print("[ERROR] crop_and_save_object_bboxes_temp received None or empty image_np.") # 백엔드 로그로 대체
        return []

    if not objects_info:
        # print("[INFO] No object info provided for cropping.") # 백엔드 로그로 대체
        return []

    temp_cropped_files = []

    for i, obj_info in enumerate(objects_info):
        bbox = obj_info.get("bbox")
        label = obj_info.get("label", "object")  # 라벨 정보 사용

        if bbox is None or not (isinstance(bbox, list) and len(bbox) == 4):
            print(
                f"[WARNING] Skipping cropping for object {i} due to missing or invalid bbox info: {bbox}. Info: {obj_info}"
            )
            continue

        x1, y1, x2, y2 = bbox

        # 좌표 유효성 검사 및 클리핑 (원본 이미지 크기를 넘어서지 않도록)
        h, w = image_np.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 바운딩 박스 영역 유효성 검사 (너비/높이가 0 이상이어야 함)
        if x2 <= x1 or y2 <= y1:
            print(
                f"[WARNING] Invalid bounding box coordinates for cropping ({label}): [{x1}, {y1}, {x2}, {y2}]. Skipping."
            )
            continue

        # 이미지 자르기 (numpy slicing 사용)
        # slicing은 [높이 시작:높이 끝, 너비 시작:너비 끝] 순서
        cropped_image_np = image_np[y1:y2, x1:x2]

        if cropped_image_np is None or cropped_image_np.size == 0:
            print(
                f"[WARNING] Cropped image for box ({label}) [{x1}, {y1}, {x2}, {y2}] resulted in None or empty array. Skipping."
            )
            continue

        # 잘라낸 이미지를 임시 파일로 저장
        try:
            # 접두사에 라벨 정보 사용, 고유한 임시 파일명 생성
            fd, temp_cropped_path = tempfile.mkstemp(
                prefix=f"{label}_crop_{i}_", suffix=".jpg"
            )
            os.close(fd)  # 파일 디스크립터 바로 닫기
            cv2.imwrite(
                temp_cropped_path, cropped_image_np
            )  # 자른 이미지를 임시 파일로 저장
            saved_cropped_paths.append(temp_cropped_path)
            # print(f"[INFO] Cropped {label} {i+1} saved to temporary file: {temp_cropped_path}") # 백엔드 로그로 대체
        except Exception as e:
            print(
                f"[ERROR] Error saving cropped image {i+1} ({label}) to temporary file: {e}"
            )
            # 저장 실패 시 해당 경로는 반환 리스트에 포함되지 않음

    return saved_cropped_paths


# NOTE: 테스트 목적으로 특정 폴더에 저장하는 함수는 제외되었습니다.
# 백엔드 통합 시에는 위 crop_and_save_object_bboxes_temp 함수를 사용하세요.

# 백엔드 API에서 이 파일을 임포트하여 사용하는 예시:
#
# from your_module_name import detect_target_objects_and_get_info, crop_and_save_object_bboxes_temp
# import cv2
# import os
#
# # API 핸들러 함수 내에서:
# # image_np = cv2.imread(received_image_path_or_stream) # 클라이언트로부터 받은 이미지 읽기
# # if image_np is not None:
# #     detected_objects = detect_target_objects_and_get_info(image_np)
# #     if detected_objects:
# #         cropped_file_paths = crop_and_save_object_bboxes_temp(image_np, detected_objects)
# #         # 이제 cropped_file_paths 목록에 있는 파일들을 당도/당성분 분석 모델로 전달
# #         # 예: analyze_sugar_and_sweetness(cropped_file_paths, detected_objects)
# #         # 분석 결과와 detected_objects 정보를 조합하여 클라이언트에 JSON 응답
# #
# #         # !!! 분석 완료 후 임시 파일들 삭제 !!!
# #         for temp_path in cropped_file_paths:
# #             if os.path.exists(temp_path):
# #                 os.remove(temp_path)
# #     else:
# #         # 탐지된 객체 없음, 빈 결과 반환
# #         pass
# # else:
# #     # 이미지 읽기 실패 처리
# #     pass
# #
# # # 클라이언트로부터 받은 원본 이미지 임시 파일도 삭제
# # if received_image_path_or_stream and os.path.exists(received_image_path_or_stream):
# #     os.remove(received_image_path_or_stream)
