# services/cnn_lgbm_seg/extractor/feature_extractors.py
import cv2
import numpy as np
from skimage.color import rgb2lab
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from typing import Tuple, List

"""
기능: 정규화된 RGB 성분의 평균값 추출
- R' = R / (R + G + B) 등으로 정의된 normalized component

논문 근거: 논문에 나온 RGB normalized first component 와 일치함

결과: ['norm_R', 'norm_G', 'norm_B']
"""


def extract_normalized_rgb(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    img = image.astype(np.float32)
    s = img.sum(axis=2, keepdims=True) + 1e-5
    norm = img / s
    return norm.mean(axis=(0, 1)), ["norm_R", "norm_G", "norm_B"]


"""
기능: CMY 색공간으로 변환한 뒤 C 채널 평균 추출
- CMY = 1 - RGB normalized

논문 근거: 논문에서 사용된 CMY 색공간의 1st component

결과: ['mean_C_from_CMY']
"""


def extract_cmy_first_component(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    img = image.astype(np.float32) / 255.0
    cmy = 1 - img
    return np.array([cmy[:, :, 0].mean()]), ["mean_C_from_CMY"]


"""
기능: HSV 색공간 평균값 추출 (추가된 특징)
- Hue, Saturation, Value의 전체 평균
"""


def extract_hsv_means(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv.mean(axis=(0, 1)), ["mean_H", "mean_S", "mean_V"]


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
    return np.array([graycoprops(glcm, "correlation")[0, 0]]), ["glcm_correlation_135"]


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
    return np.array([graycoprops(glcm, "contrast")[0, 0]]), ["glcm_contrast_135"]


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
    feats, names = [], []
    for i, c in enumerate(["R", "G", "B"]):
        h = cv2.calcHist([image], [i], None, [bins], [0, 256])
        h = cv2.normalize(h, h).flatten()
        feats.append(h)
        names += [f"hist_{c}_{j}" for j in range(bins)]
    return np.concatenate(feats), names


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
    ar = w / h if h != 0 else 0
    return np.array([ar], dtype=np.float32), ["bbox_aspect_ratio"]


def extract_cielab_means(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    lab = rgb2lab(image)
    return lab.mean(axis=(0, 1)), ["mean_L_lab", "mean_a_lab", "mean_b_lab"]


def extract_colorfulness(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    R, G, B = cv2.split(image.astype("float"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    col = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    return np.array([col]), ["colorfulness"]


"""
기능: 위의 모든 feature 추출 함수를 통합 실행
- 입력된 이미지(ndarray)에 대해 모든 시각적 특징을 추출하고 연결

입력:
- image: crop된 사과 이미지 (numpy.ndarray)

출력:
- feature_vec: np.ndarray 형식의 특징 벡터 (모델 입력값)
- feature_names: 각 특징의 이름 목록 (디버깅 및 시각화용)
"""


def extract_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    f1, n1 = extract_normalized_rgb(image)
    f2, n2 = extract_cmy_first_component(image)
    f3, n3 = extract_hsv_means(image)
    f4, n4 = extract_glcm_texture(image)
    f5, n5 = extract_glcm_contrast(image)
    f6, n6 = extract_lbp_histogram(image)
    f7, n7 = extract_rgb_histogram(image)
    f8, n8 = extract_crop_image_shape_features(image)
    f9, n9 = extract_cielab_means(image)
    f10, n10 = extract_colorfulness(image)
    vec = np.concatenate([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])
    names = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10
    return vec, names


# ─────────── 배치 처리용 핸드크래프트 함수 ───────────
def extract_batch_handcrafted(imgs: np.ndarray) -> np.ndarray:
    feats = []
    for img in imgs:
        f, _ = extract_features(img)
        feats.append(f.astype(np.float32))
    return np.stack(feats, axis=0)
