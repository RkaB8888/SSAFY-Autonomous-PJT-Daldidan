# services/xgb_bbox/extractor/common_features.py
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import deltaE_cie76

# 기준 Lab 값 (ΔE 계산용)
LAB_REFERENCE = np.array([60, 150, 140], dtype=np.float32)


# CLAHE 명도 보정
def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)
    lab_clahe = cv2.merge((L_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


# 특징 추출 함수
def extract_features(image: np.ndarray) -> dict:
    image = apply_clahe(image)
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    L, a, b = cv2.split(img_lab)
    H, S, V = cv2.split(img_hsv)

    L_mean = np.mean(L)
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    H_mean = np.mean(H)
    S_mean = np.mean(S)
    V_mean = np.mean(V)

    a_div_b = a_mean / (b_mean + 1e-5)
    a_div_L = a_mean / (L_mean + 1e-5)
    delta_e = deltaE_cie76([[L_mean, a_mean, b_mean]], [LAB_REFERENCE])[0]

    glcm = graycomatrix(
        gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )
    contrast = graycoprops(glcm, "contrast")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]

    thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
    x, y, w, h = cv2.boundingRect(thresh)
    area = cv2.countNonZero(thresh)
    aspect_ratio = w / (h + 1e-5)

    return {
        "L_mean": L_mean,
        "a_mean": a_mean,
        "b_mean": b_mean,
        "H_mean": H_mean,
        "S_mean": S_mean,
        "V_mean": V_mean,
        "a_div_b": a_div_b,
        "a_div_L": a_div_L,
        "delta_E": delta_e,
        "contrast": contrast,
        "energy": energy,
        "homogeneity": homogeneity,
        "correlation": correlation,
        "area": area,
        "aspect_ratio": aspect_ratio,
    }
