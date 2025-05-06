#버전2
# utils.py
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# cv: 이미지 전처리/변환에 가장 쉽고 빠른 라이브러리
# cv를 사용하여, RGB → HSV 변환 (색상 공간 변환)
# cv를 사용하여, "CNN 외부에서 직접 feature 뽑아내는 역할"을 담당
def calculate_mae(predictions, targets):
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    return np.mean(np.abs(predictions - targets))

def extract_color_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mean_color = cv2.mean(hsv)[:3]
    return np.array(mean_color)

def extract_texture_features(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # 올바른 인덱싱을 통해 단일 값을 추출
    contrast = graycoprops(glcm, 'contrast')[0, 0]  # (0, 0)으로 추출
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]

    # numpy array로 변환 (각 값은 이제 scalar)
    return np.array([contrast, dissimilarity, homogeneity, asm], dtype=np.float32)
