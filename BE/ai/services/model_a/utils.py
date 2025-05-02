import cv2
import numpy as np

def preprocess_image(image):
    """리사이즈 + 사과 ROI 추출 + 밝기 평균 정규화"""
    
    # Step 1. 리사이즈 (처음부터 너무 작으면 생략 가능)
    image = cv2.resize(image, (512, 512))

    # Step 2. 사과 ROI 추출
    # HSV 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 빨간색 계열 범위 설정 (사과 색)
    lower_red1 = np.array([0, 50, 20])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 50, 20])
    upper_red2 = np.array([180, 255, 255])

    # 빨간색 마스크 만들기
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 노이즈 제거 (morphology)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 가장 큰 외곽선(사과) 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 가장 큰 contour 선택
        largest_contour = max(contours, key=cv2.contourArea)

        # 최소외접원 찾기
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)

        # ✅ ROI를 원형으로 그리기 (디버깅용)
        debug_image = image.copy()
        cv2.circle(debug_image, center, radius, (0, 255, 0), 2)
        cv2.imshow('Detected ROI (Circle)', debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # ✅ 원을 기준으로 crop
        x1 = max(0, int(x - radius))
        y1 = max(0, int(y - radius))
        x2 = min(image.shape[1], int(x + radius))
        y2 = min(image.shape[0], int(y + radius))
        roi = image[y1:y2, x1:x2]

        # ROI도 512x512로 다시 resize
        image = cv2.resize(roi, (512, 512))
    else:
        # 사과 못 찾으면 원본 유지
        pass

    # Step 3. 밝기 평균 정규화
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_v = np.mean(v)
    target_mean = 128  # 목표 평균 밝기
    adjust_ratio = target_mean / (mean_v + 1e-5)  # 0 나누기 방지

    v = np.clip(v * adjust_ratio, 0, 255).astype(np.uint8)
    hsv_normalized = cv2.merge([h, s, v])
    # Step 3. 밝기 보정 (CLAHE 사용)
    image_normalized = adjust_brightness(image)

    return image_normalized

def adjust_brightness(image):
    """이미지 밝기 평균을 기준으로 조정"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return adjusted