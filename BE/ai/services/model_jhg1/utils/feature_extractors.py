import cv2
import numpy as np
from pathlib import Path
from skimage import feature
from typing import Tuple, List
from PIL import Image, ExifTags
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

"""
기능: 이미지 파일을 RGB로 읽어서 numpy array로 반환
- OpenCV로 읽기 시도
- 실패하면 Pillow + EXIF 회전 정보까지 고려해 보정

이유: 스마트폰 촬영 이미지는 회전 정보가 EXIF에 저장됨 → 시각화나 특징 왜곡 방지
"""


def load_image(image_path: Path) -> np.ndarray:
    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    # Fallback: PIL + EXIF
    pil_img = Image.open(image_path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = pil_img._getexif()
        if exif is not None:
            orientation_val = exif.get(orientation, None)
            if orientation_val == 3:
                pil_img = pil_img.rotate(180, expand=True)
            elif orientation_val == 6:
                pil_img = pil_img.rotate(270, expand=True)
            elif orientation_val == 8:
                pil_img = pil_img.rotate(90, expand=True)
    except Exception:
        pass
    return np.array(pil_img.convert("RGB"))


"""
기능: 정규화된 RGB 성분의 평균값 추출
- R' = R / (R + G + B) 등으로 정의된 normalized component

논문 근거: 논문에 나온 RGB normalized first component 와 일치함

결과: ['norm_R', 'norm_G', 'norm_B']
"""


def extract_normalized_rgb(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    image = image.astype(np.float32)
    sum_channels = np.sum(image, axis=2, keepdims=True) + 1e-5  # avoid division by zero
    normalized = image / sum_channels
    mean_vals = normalized.mean(axis=(0, 1))
    names = ["norm_R", "norm_G", "norm_B"]
    return mean_vals, names


"""
기능: CMY 색공간으로 변환한 뒤 C 채널 평균 추출
- CMY = 1 - RGB normalized

논문 근거: 논문에서 사용된 CMY 색공간의 1st component

결과: ['mean_C_from_CMY']
"""


def extract_cmy_first_component(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    image = image.astype(np.float32) / 255.0
    cmy = 1 - image
    mean_c = cmy[:, :, 0].mean()
    return np.array([mean_c]), ["mean_C_from_CMY"]


"""
기능: HSV 색공간 평균값 추출 (추가된 특징)
- Hue, Saturation, Value의 전체 평균
"""


def extract_hsv_means(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mean_vals = hsv.mean(axis=(0, 1))
    names = ["mean_H", "mean_S", "mean_V"]
    return mean_vals, names


"""
기능: GLCM(gray-level co-occurrence matrix)에서 correlation 값 추출
- 135° 방향 기준 텍스처 측정 (수정된 특징)
"""


def extract_glcm_texture(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[np.deg2rad(135)],
        levels=256,
        symmetric=True,
        normed=True,
    )
    corr = graycoprops(glcm, "correlation")[0, 0]
    return np.array([corr]), ["glcm_correlation_135"]


"""
기능: GLCM contrast (텍스처 대비) 통계 추가 (추가된 특징)
"""


def extract_glcm_contrast(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[np.deg2rad(135)],
        levels=256,
        symmetric=True,
        normed=True,
    )
    contrast = graycoprops(glcm, "contrast")[0, 0]
    return np.array([contrast]), ["glcm_contrast_135"]


"""
기능: Local Binary Pattern (LBP) 히스토그램 (텍스처 특징) (추가된 특징)
"""


def extract_lbp_histogram(
    image: np.ndarray, bins: int = 64
) -> Tuple[np.ndarray, List[str]]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
    names = [f"lbp_{i}" for i in range(bins)]
    return hist, names


"""
기능: RGB 채널별 히스토그램 (각 채널 16bin) (추가된 특징)
"""


def extract_rgb_histogram(
    image: np.ndarray, bins: int = 16
) -> Tuple[np.ndarray, List[str]]:
    hist_features = []
    names = []
    for i, color in enumerate(["R", "G", "B"]):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.append(hist)
        names.extend([f"hist_{color}_{j}" for j in range(bins)])
    return np.concatenate(hist_features), names


"""
기능: crop 이미지의 해상도 기반 aspect ratio 계산
- 종횡비는 거리와 무관한 상대적 형태 정보로 유지

논문 근거: 없음 (서비스 현실성 기반 보조 feature)

결과: ['bbox_aspect_ratio']
"""


def extract_crop_image_shape_features(
    image: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    h, w = image.shape[:2]
    aspect_ratio = w / h if h != 0 else 0
    return np.array([aspect_ratio], dtype=np.float32), ["bbox_aspect_ratio"]


"""
기능: 위의 모든 feature 추출 함수를 통합 실행
- 이미지 (crop 상태)를 받아 최종 특징 벡터 반환

입력: 
- 이미지 경로

출력:
- feature_vec: np.ndarray 형식의 특징 벡터
- feature_names: 각 특징의 이름 목록 (debug, logging 등에 유용)
"""


def extract_features(image_path: Path) -> Tuple[np.ndarray, List[str]]:
    image = load_image(image_path)

    f1, n1 = extract_normalized_rgb(image)
    f2, n2 = extract_cmy_first_component(image)
    f3, n3 = extract_hsv_means(image)
    f4, n4 = extract_glcm_texture(image)
    f5, n5 = extract_glcm_contrast(image)
    f6, n6 = extract_lbp_histogram(image)
    f7, n7 = extract_rgb_histogram(image)
    f8, n8 = extract_crop_image_shape_features(image)

    feature_vec = np.concatenate([f1, f2, f3, f4, f5, f6, f7, f8])
    feature_names = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8
    return feature_vec, feature_names
