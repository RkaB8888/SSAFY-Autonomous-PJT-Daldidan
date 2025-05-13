# services/cnn_lgbm_bbox/extractor/feature_extractors_v2.py
import cv2
import numpy as np
from skimage.feature import graycomatrix
from typing import Tuple, List


def extract_normalized_r(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    img = image.astype(np.float32)
    s = img.sum(axis=2, keepdims=True) + 1e-5
    norm = img / s
    # R 채널(0번째)만 평균
    return np.array([norm[:, :, 0].mean()]), ["norm_R_only"]


def extract_cmy_first_component(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    img = image.astype(np.float32) / 255.0
    cmy = 1 - img
    return np.array([cmy[:, :, 0].mean()]), ["mean_C_from_CMY"]


def extract_ycbcr_diff(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    # OpenCV는 YCrCb 순서
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y, Cr, Cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    diff = (Cb - Cr).mean()
    return np.array([diff]), ["ycbcr_diff_Cb_minus_Cr"]


def extract_ycbcr_norm2(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y, Cr, Cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    norm_cb = Cb / (Y + 1e-5)
    return np.array([norm_cb.mean()]), ["ycbcr_norm_Cb"]


def extract_cluster_shade(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[np.deg2rad(135)],
        levels=256,
        symmetric=True,
        normed=True,
    )
    # cluster shade = Σ (i+ j – μ_x – μ_y)^3 * P[i,j]  (skimage에 직접 지원 함수 없음)
    # 여기서는 graycoprops에 없는 cluster_shade를 직접 계산
    P = glcm[:, :, 0, 0].astype(np.float32)
    i, j = np.indices(P.shape)
    μ_i = np.sum(i * P)
    μ_j = np.sum(j * P)
    cs = np.sum(((i + j - μ_i - μ_j) ** 3) * P)
    return np.array([cs]), ["glcm_cluster_shade_135"]


def rgb_to_lms_cat02(image: np.ndarray) -> np.ndarray:
    # 논문 CAT02 변환행렬 (sRGB→LMS)
    M = np.array(
        [
            [0.7328, 0.4296, -0.1624],
            [-0.7036, 1.6975, 0.0061],
            [0.0030, 0.0136, 0.9834],
        ],
        dtype=np.float32,
    )
    img = image.astype(np.float32) / 255.0
    # (H,W,3)×(3×3) → (H,W,3)
    lms = img @ M.T
    return lms


def extract_lms1(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    lms = rgb_to_lms_cat02(image)
    return np.array([lms[:, :, 0].mean()]), ["lms1_cat02"]


def extract_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    # 논문에서 고른 6개
    f1, n1 = extract_normalized_r(image)  # 논문#1
    f2, n2 = extract_cmy_first_component(image)  # 논문#2
    f3, n3 = extract_cluster_shade(image)  # 논문#3
    f4, n4 = extract_ycbcr_diff(image)  # 논문#4
    f5, n5 = extract_lms1(image)  # 논문#5
    f6, n6 = extract_ycbcr_norm2(image)  # 논문#6

    feat_vec = np.concatenate([f1, f2, f3, f4, f5, f6])
    feat_names = n1 + n2 + n3 + n4 + n5 + n6
    return feat_vec, feat_names


def extract_batch_handcrafted(imgs: np.ndarray) -> np.ndarray:
    feats = []
    for img in imgs:
        v, _ = extract_features(img)
        feats.append(v.astype(np.float32))
    return np.stack(feats, axis=0)
