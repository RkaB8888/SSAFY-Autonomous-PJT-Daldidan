# ai/services/yolov8/scripts/bbox_crop_test2.py
import cv2
import numpy as np
import tensorflow as tf
import os

# import tempfile # 임시 파일 처리를 위해 필요했지만, 테스트 목적으로 실제 폴더 사용
import json  # JSON 출력을 위해 추가

# --- 모델 및 설정 값 (사용자 환경에 맞게 반드시 수정!) ---
# TODO: 실제 백엔드 환경에 맞는 경로로 수정해야 합니다.
TFLITE_MODEL_PATH = "./services/yolov8/models/yolov8n_bbox_int8.tflite"  # ★★★ 실제 TFLite 모델 파일 경로로 수정 ★★★
IMAGE_PATH = (
    "C:\\Users\\SSAFY\\Desktop\\phone.png"  # ★★★ 테스트할 입력 이미지 경로로 수정 ★★★
)
# 테스트 출력 폴더 경로 (잘라낸 이미지가 여기에 저장됩니다)
TEST_OUTPUT_DIR = "./test_cropped_apples"  # ★★★ 테스트 결과 저장할 폴더 경로 지정 ★★★

INPUT_SIZE = (640, 640)  # Netron에서 확인한 모델 입력 크기 (너비, 높이)
CONFIDENCE_THRESHOLD = 0.25  # COCO 모델 테스트를 위한 일반적인 임계값 (필요시 조정)
IOU_THRESHOLD = 0.45
# COCO 클래스 이름 (80개)
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
CLASS_NAMES = COCO_CLASS_NAMES
# --- 설정 값 끝 ---

# 'apple' 클래스의 인덱스를 찾습니다.
TARGET_CLASS_IDS = []
try:
    apple_id = CLASS_NAMES.index("apple")
    TARGET_CLASS_IDS.append(apple_id)
    print(f"[INFO] 'apple' class ID is: {apple_id}")
except ValueError:
    print(
        "[WARNING] 'apple' class not found in CLASS_NAMES. Object detection for apple will not work."
    )

if not TARGET_CLASS_IDS:
    print("[ERROR] 'apple' class not found. Cannot proceed with apple detection.")


# --- TFLite 인터프리터 로드 (모듈 로드 시 또는 애플리케이션 시작 시 한 번만) ---
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("[INFO] TFLite interpreter (yolov8n_int8.tflite) loaded successfully.")
except Exception as e:
    print(f"[ERROR] Error loading TFLite interpreter: {e}")
    interpreter = None


def preprocess_image(image_np, input_size):
    """이미지(numpy 배열)를 모델 입력에 맞게 전처리합니다."""
    if image_np is None or image_np.size == 0:
        print("[ERROR] preprocess_image received None or empty image.")
        return None, 0, 0, 0, 0, 0

    original_height, original_width = image_np.shape[:2]

    h, w = input_size
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

    image_rgb = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0
    input_data = np.expand_dims(image_normalized, axis=0)
    return input_data, original_width, original_height, scale, left, top


def postprocess_output_coco(
    output_data,
    original_width,
    original_height,
    scale,
    pad_left,
    pad_top,
    conf_thresh,
    iou_thresh,
    input_size,
):
    """모델 출력 후처리 및 NMS 적용 후 원본 이미지 기준 바운딩 박스 좌표 반환."""
    predictions = np.squeeze(output_data).T

    boxes = []
    scores = []
    class_ids_pred = []

    for pred in predictions:
        xc_raw, yc_raw, w_raw, h_raw = pred[:4]
        class_scores_raw = pred[4:]

        current_class_id = np.argmax(class_scores_raw)
        current_conf = class_scores_raw[current_class_id]

        if current_conf < conf_thresh:
            continue

        xc = xc_raw * input_size[0]
        yc = yc_raw * input_size[1]
        w = w_raw * input_size[0]
        h = h_raw * input_size[1]

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2

        boxes.append([x1, y1, x2, y2])
        scores.append(current_conf)
        class_ids_pred.append(current_class_id)

    if not boxes:
        # print("[INFO] No raw predictions passed confidence threshold.") # 디버깅용
        return [], [], []

    normalized_boxes_for_nms = []
    for box_px in boxes:
        x1_px, y1_px, x2_px, y2_px = box_px
        y1_norm = y1_px / input_size[1]
        x1_norm = x1_px / input_size[0]
        y2_norm = y2_px / input_size[1]
        x2_norm = x2_px / input_size[0]
        normalized_boxes_for_nms.append([y1_norm, x1_norm, y2_norm, x2_norm])

    if not normalized_boxes_for_nms:
        # print("[INFO] No boxes for NMS.") # 디버깅용
        return [], [], []

    selected_indices = tf.image.non_max_suppression(
        np.array(normalized_boxes_for_nms),
        np.array(scores),
        max_output_size=100,
        iou_threshold=iou_thresh,
        score_threshold=conf_thresh,
    ).numpy()

    final_boxes_orig_coords = []
    final_scores = []
    final_class_ids = []

    for index in selected_indices:
        box = boxes[index]

        x1_on_resized = box[0] - pad_left
        y1_on_resized = box[1] - pad_top
        x2_on_resized = box[2] - pad_left
        y2_on_resized = box[3] - pad_top

        x1_orig = x1_on_resized / scale
        y1_orig = y1_on_resized / scale
        x2_orig = x2_on_resized / scale
        y2_orig = y2_on_resized / scale

        final_boxes_orig_coords.append(
            [int(x1_orig), int(y1_orig), int(x2_orig), int(y2_orig)]
        )
        final_scores.append(scores[index])
        final_class_ids.append(class_ids_pred[index])

    return final_boxes_orig_coords, final_scores, final_class_ids


def detect_apples_and_get_bboxes(image_np):
    """
    이미지(numpy 배열)에서 사과를 탐지하고 원본 이미지 기준 바운딩 박스 정보 목록을 반환합니다.
    반환 형식: [{"bbox": [x1, y1, x2, y2], "score": score, "class_id": id, "label": "apple"}, ...]
    """
    if interpreter is None:
        print("[ERROR] TFLite interpreter not loaded.")
        return []

    if image_np is None or image_np.size == 0:
        print("[ERROR] detect_apples_and_get_bboxes received None or empty image_np.")
        return []

    # 전처리
    input_data, orig_w, orig_h, scale_ratio, pad_l, pad_t = preprocess_image(
        image_np, INPUT_SIZE
    )

    if input_data is None:
        print("[ERROR] Image preprocessing failed.")
        return []

    # 추론 실행
    try:
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
    except Exception as e:
        print(f"[ERROR] TFLite inference failed: {e}")
        return []

    # 후처리 및 바운딩 박스 추출 (원본 이미지 기준 좌표)
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

    apple_detection_results = []
    # 사과에 해당하는 바운딩 박스만 필터링
    for i in range(len(boxes_orig)):
        if class_ids[i] in TARGET_CLASS_IDS:
            apple_detection_results.append(
                {
                    "bbox": boxes_orig[i],  # [x1, y1, x2, y2]
                    "score": float(scores[i]),  # JSON 직렬화를 위해 float으로 변환
                    "class_id": int(class_ids[i]),  # JSON 직렬화를 위해 int으로 변환
                    "label": (
                        CLASS_NAMES[class_ids[i]]
                        if class_ids[i] < len(CLASS_NAMES)
                        else f"Unknown_{class_ids[i]}"
                    ),
                }
            )

    print(
        f"[INFO] Detected {len(apple_detection_results)} apple(s) during detection phase."
    )
    return apple_detection_results


def crop_and_save_bboxes_to_folder(image_np, bboxes_info, output_dir):
    """
    원본 이미지에서 주어진 바운딩 박스 영역들을 잘라내어 지정된 폴더에 파일로 저장합니다.
    Args:
        image_np (np.ndarray): 원본 이미지의 numpy 배열.
        bboxes_info (list): detect_apples_and_get_bboxes 함수에서 반환된 바운딩 박스 정보 리스트.
                             각 항목은 {"bbox": [x1, y1, x2, y2], ...} 형태.
        output_dir (str): 잘라낸 이미지를 저장할 폴더 경로.
    Returns:
        list: 저장된 잘라낸 이미지 파일들의 전체 경로 목록.
    """
    if image_np is None or image_np.size == 0:
        print("[ERROR] crop_and_save_bboxes_to_folder received None or empty image_np.")
        return []

    if not bboxes_info:
        print("[INFO] No bounding boxes provided for cropping.")
        return []

    # 출력 디렉터리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")
    else:
        print(f"[INFO] Output directory already exists: {output_dir}")

    saved_cropped_paths = []

    for i, bbox_info in enumerate(bboxes_info):
        x1, y1, x2, y2 = bbox_info["bbox"]

        # 좌표 유효성 검사 (원본 이미지 크기를 넘어서지 않도록)
        h, w = image_np.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 바운딩 박스 영역 유효성 검사 (너비/높이가 0 이상이어야 함)
        if x2 <= x1 or y2 <= y1:
            print(
                f"[WARNING] Invalid bounding box coordinates for cropping: {bbox_info['bbox']}. Skipping."
            )
            continue

        # 이미지 자르기 (numpy slicing 사용)
        cropped_image_np = image_np[y1:y2, x1:x2]

        if cropped_image_np.size == 0:
            print(
                f"[WARNING] Cropped image for box {bbox_info['bbox']} resulted in empty array. Skipping."
            )
            continue

        # 잘라낸 이미지를 지정된 폴더에 파일로 저장
        # 파일 이름 형식: apple_crop_<순번>.jpg
        file_name = f"apple_crop_{i + 1}.jpg"
        file_path = os.path.join(output_dir, file_name)

        try:
            cv2.imwrite(file_path, cropped_image_np)
            saved_cropped_paths.append(file_path)
            print(f"[INFO] Cropped apple {i + 1} saved to: {file_path}")
        except Exception as e:
            print(f"[ERROR] Error saving cropped image {i + 1} to {file_path}: {e}")

    return saved_cropped_paths


# --- 메인 스크립트 실행 부분 (테스트용) ---
if __name__ == "__main__":
    if interpreter is None:
        print("[ERROR] Interpreter failed to load. Cannot run test.")
        exit()
    if not TARGET_CLASS_IDS:
        print("[ERROR] Target class 'apple' not found. Cannot run test for apples.")
        exit()

    # 테스트 이미지를 읽습니다.
    original_image_test = cv2.imread(IMAGE_PATH)

    if original_image_test is None:
        print(f"[ERROR] Could not read test image at {IMAGE_PATH}. Cannot run test.")
        exit()

    print(f"\n--- Starting Test: Processing image at {IMAGE_PATH} ---")

    # 1. 이미지에서 사과를 탐지하고 바운딩 박스 정보를 얻습니다.
    apple_detection_results = detect_apples_and_get_bboxes(original_image_test)

    print("\n--- Apple Detection Results (JSON) ---")
    if apple_detection_results:
        # 결과를 JSON 형식으로 예쁘게 출력
        print(json.dumps(apple_detection_results, indent=4))
    else:
        print("[] (No apples detected)")
    print("--------------------------------------")

    # 2. 탐지된 각 사과의 바운딩 박스 영역을 잘라내어 지정된 폴더에 저장합니다.
    if apple_detection_results:
        print(f"\n--- Cropping and Saving Apples to {TEST_OUTPUT_DIR} ---")
        saved_paths = crop_and_save_bboxes_to_folder(
            original_image_test, apple_detection_results, TEST_OUTPUT_DIR
        )

        print("\n--- Saved Cropped Images Paths ---")
        if saved_paths:
            for path in saved_paths:
                print(path)
            print(
                f"\n[SUCCESS] Test complete. Cropped apple images saved to {TEST_OUTPUT_DIR}"
            )
        else:
            print("[INFO] No cropped images saved.")
        print("----------------------------------")

    else:
        print("\n--- Cropping Skipped ---")
        print("No apples detected, so no images were cropped or saved.")
        print("------------------------")

    print("\n--- Test Finished ---")

    # 테스트 목적으로 이미지를 보여주는 코드는 삭제했습니다.
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
