# ai/services/yolov8/scripts/draw_boundingbox.py
import cv2
import numpy as np
import tensorflow as tf

## 바운딩 박스 그리는 py

# --- 모델 및 설정 값 (사용자 환경에 맞게 반드시 수정!) ---
TFLITE_MODEL_PATH = "./yolov8n_int8.tflite"  # ★★★ 실제 파일 경로로 수정 ★★★
IMAGE_PATH = "C:\\Users\\SSAFY\\Desktop\\phone.png"  # ★★★ 테스트할 이미지 경로로 수정 (COCO 객체가 있는 이미지) ★★★
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
    "donut",  # 'donut' 위치 확인
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
CLASS_NAMES = COCO_CLASS_NAMES  # 전역 변수 CLASS_NAMES를 COCO_CLASS_NAMES로 설정
# --- 설정 값 끝 ---

# 이미지를 화면에 맞게 조절하기 위한 최대 크기 설정 (필요에 따라 조절하세요)
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720

# 'apple'과 'donut' 클래스의 인덱스를 찾습니다.
TARGET_CLASS_IDS = []
try:
    apple_id = CLASS_NAMES.index("apple")
    TARGET_CLASS_IDS.append(apple_id)
    print(f"'apple' class ID is: {apple_id}")
except ValueError:
    print("Warning: 'apple' class not found in CLASS_NAMES.")

try:
    donut_id = CLASS_NAMES.index("donut")
    TARGET_CLASS_IDS.append(donut_id)
    print(f"'donut' class ID is: {donut_id}")
except ValueError:
    print("Warning: 'donut' class not found in CLASS_NAMES.")

if not TARGET_CLASS_IDS:
    print(
        "Error: Neither 'apple' nor 'donut' found in CLASS_NAMES. No objects will be drawn."
    )


def preprocess_image(image_path, input_size):
    """이미지를 읽고 모델 입력에 맞게 전처리합니다."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None, 0, 0, 0, 0, 0
    original_height, original_width = image.shape[:2]

    h, w = input_size  # input_size는 (너비, 높이)
    scale = min(w / original_width, h / original_height)
    nw, nh = int(scale * original_width), int(scale * original_height)
    image_resized = cv2.resize(image, (nw, nh))

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
    return input_data, image, original_width, original_height, scale, left, top


def postprocess_output_coco(
    output_data,  # Shape (1, 84, 8400)
    original_width,
    original_height,
    scale,
    pad_left,
    pad_top,
    conf_thresh,
    iou_thresh,
    input_size,  # 스크립트 상단의 INPUT_SIZE = (너비, 높이)
):
    predictions = np.squeeze(output_data).T  # Transpose to (8400, 84)
    # print(f"Number of raw predictions: {predictions.shape[0]}") # 디버깅 시 필요하면 주석 해제
    # print(f"Attributes per prediction: {predictions.shape[1]}") # 디버깅 시 필요하면 주석 해제

    boxes = []
    scores = []
    class_ids_pred = []  # 모델이 예측한 클래스 인덱스

    num_classes = len(CLASS_NAMES)  # 80

    # 각 예측은 [xc, yc, w, h, class_score_0, ..., class_score_79] 형태라고 가정
    for i, pred in enumerate(predictions):
        # 처음 4개는 바운딩 박스 좌표 (정규화된 값으로 가정)
        xc_raw, yc_raw, w_raw, h_raw = pred[:4]

        # 나머지 80개는 각 클래스에 대한 점수
        class_scores_raw = pred[4:]

        # 가장 높은 점수를 가진 클래스 ID와 해당 점수(신뢰도)를 찾음
        current_class_id = np.argmax(class_scores_raw)
        current_conf = class_scores_raw[current_class_id]

        # if i < 10: # 처음 10개 예측의 주요 정보 출력 (디버깅 시 필요하면 주석 해제)
        #     print(
        #         f"Debug TFLite raw_pred[{i}]: xc={xc_raw:.4f}, yc={yc_raw:.4f}, w={w_raw:.4f}, h={h_raw:.4f}, "
        #         f"class_id={current_class_id}, conf={current_conf:.4f} (max_score_across_all_classes={np.max(class_scores_raw):.4f})"
        #     )

        if current_conf < conf_thresh:
            continue

        # 좌표가 [0,1]로 정규화되었다고 가정하고 픽셀 값으로 변환
        # input_size는 (너비, 높이) 순서
        xc = xc_raw * input_size[0]  # 너비
        yc = yc_raw * input_size[1]  # 높이
        w = w_raw * input_size[0]
        h = h_raw * input_size[1]

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2

        boxes.append([x1, y1, x2, y2])  # 640x640 입력 기준 픽셀 좌표
        scores.append(current_conf)
        class_ids_pred.append(current_class_id)

    if not boxes:
        print("No boxes passed confidence threshold.")
        return [], [], []

    # NMS 적용
    normalized_boxes_for_nms = []
    for box_px in boxes:
        x1_px, y1_px, x2_px, y2_px = box_px
        y1_norm = y1_px / input_size[1]  # 높이로 정규화
        x1_norm = x1_px / input_size[0]  # 너비로 정규화
        y2_norm = y2_px / input_size[1]
        x2_norm = x2_px / input_size[0]
        normalized_boxes_for_nms.append([y1_norm, x1_norm, y2_norm, x2_norm])

    if not normalized_boxes_for_nms:
        return [], [], []

    selected_indices = tf.image.non_max_suppression(
        np.array(normalized_boxes_for_nms),
        np.array(scores),
        max_output_size=100,
        iou_threshold=iou_thresh,
        score_threshold=conf_thresh,
    ).numpy()

    final_boxes = []
    final_scores = []
    final_class_ids = []

    # print(f"Number of boxes after TFLite conf_thresh: {len(boxes)}") # 디버깅 시 필요하면 주석 해제
    # print(f"Selected indices by NMS (TFLite): {len(selected_indices)}") # 디버깅 시 필요하면 주석 해제

    for index in selected_indices:
        box = boxes[index]  # 640x640 입력 기준 픽셀 좌표 [x1,y1,x2,y2]

        x1_on_resized = box[0] - pad_left
        y1_on_resized = box[1] - pad_top
        x2_on_resized = box[2] - pad_left
        y2_on_resized = box[3] - pad_top

        x1_orig = x1_on_resized / scale
        y1_orig = y1_on_resized / scale
        x2_orig = x2_on_resized / scale
        y2_orig = y2_on_resized / scale

        final_boxes.append([int(x1_orig), int(y1_orig), int(x2_orig), int(y2_orig)])
        final_scores.append(scores[index])
        final_class_ids.append(class_ids_pred[index])

    return final_boxes, final_scores, final_class_ids


# --- 메인 스크립트 실행 부분 ---
# TFLite 인터프리터 로드
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details for yolov8n_int8.tflite:", input_details)  # 모델 이름 명시
print("Output details for yolov8n_int8.tflite:", output_details)

input_data, original_image, orig_w, orig_h, scale_ratio, pad_l, pad_t = (
    preprocess_image(IMAGE_PATH, INPUT_SIZE)
)

if original_image is not None:  # 이미지 로드에 성공했을 경우에만 진행
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    print("Raw TFLite output shape (yolov8n_int8.tflite):", output_data.shape)

    # 수정된 후처리 함수 호출
    boxes, scores, class_ids = postprocess_output_coco(  # 함수 이름 변경
        output_data,
        orig_w,
        orig_h,
        scale_ratio,
        pad_l,
        pad_t,
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD,
        INPUT_SIZE,  # (너비, 높이) 순서의 INPUT_SIZE 전달
    )

    # 결과 시각화 - 'apple' 또는 'donut' 클래스만 그리기
    display_image = (
        original_image.copy()
    )  # 원본 이미지를 직접 수정하지 않도록 복사본 사용
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]  # 이제 COCO 클래스 인덱스

        # --- 추가된 조건: 클래스 ID가 TARGET_CLASS_IDS 리스트에 포함될 때만 그립니다. ---
        if class_id in TARGET_CLASS_IDS:
            if class_id < len(CLASS_NAMES):  # 유효한 클래스 인덱스인지 확인
                label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
            else:
                # TARGET_CLASS_IDS에 있는 ID가 유효하지 않은 경우는 거의 없지만, 만약을 위해 남겨둡니다.
                label = f"Unknown_Class_{class_id}: {score:.2f}"

            cv2.rectangle(
                display_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2
            )
            # 텍스트 위치 조정: 박스 상단에 표시, 너무 위로 올라가지 않게 조정
            text_y = box[1] - 10
            if text_y < 10:  # 이미지 상단에 너무 가깝다면 박스 하단에 표시
                text_y = box[3] + 20
            cv2.putText(
                display_image,
                label,
                (box[0], text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        # --- 추가된 조건 끝 ---

    # --- 이미지 크기 조절하여 화면에 맞게 표시 ---
    img_height, img_width = display_image.shape[:2]
    scale = 1.0
    if img_width > MAX_DISPLAY_WIDTH or img_height > MAX_DISPLAY_HEIGHT:
        scale_w = MAX_DISPLAY_WIDTH / img_width
        scale_h = MAX_DISPLAY_HEIGHT / img_height
        scale = min(scale_w, scale_h)  # 가로세로 비율 유지

    new_width = int(img_width * scale)
    new_height = int(img_height * scale)

    # 창 크기 조절 가능하도록 설정 (선택 사항)
    cv2.namedWindow("yolov8n_int8.tflite Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("yolov8n_int8.tflite Output", new_width, new_height)

    cv2.imshow("yolov8n_int8.tflite Output", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
