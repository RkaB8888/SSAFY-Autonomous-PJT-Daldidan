import cv2
import numpy as np
from skimage import color, feature

# def extract_features(image, mask): 
# 의 속도 개선 버전( 이미지 입력시 당도 출력에 사용 )
def extract_features(image, mask):
    x, y, w, h = cv2.boundingRect(mask)
    roi = image[y:y+h, x:x+w]

    # ⏱ 속도 개선 핵심: ROI 크기 축소
    roi = cv2.resize(roi, (64, 64))

    R, G, B = roi[:, :, 2], roi[:, :, 1], roi[:, :, 0]
    sum_RGB = R + G + B + 1e-5
    Rn = np.mean(R / sum_RGB)
    C = np.mean(1 - R / 255.0)

    YCbCr = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    Cb, Cr = YCbCr[:, :, 1], YCbCr[:, :, 2]
    cb_mean = np.mean(Cb)
    cr_mean = np.mean(Cr)
    ycbcr_diff = cb_mean - cr_mean
    ycbcr_norm = cb_mean / (cb_mean + cr_mean + 1e-5)

    # CAT02 생략 또는 간략화
    cat02_first = 0.0  # 또는 np.mean(R) 등으로 대체 가능

    # GLCM 계산 최적화
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = (gray / 8).astype(np.uint8)  # ✅ 레벨 조정
    glcm = feature.graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=32,
        symmetric=True,
        normed=True
    )
    cluster_shadow = feature.graycoprops(glcm, 'contrast')[0, 0]

    return np.array([Rn, C, ycbcr_diff, ycbcr_norm, cat02_first, cluster_shadow])

# 속도 최적화(정확도 저하하)
def extract_fast_features(image, mask):
    x, y, w, h = cv2.boundingRect(mask)
    roi = image[y:y+h, x:x+w]

    R, G, B = roi[:,:,2], roi[:,:,1], roi[:,:,0]
    sum_RGB = R + G + B + 1e-5
    Rn = np.mean(R / sum_RGB)
    C = np.mean(1 - R / 255.0)

    YCbCr = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    Cb, Cr = YCbCr[:,:,1], YCbCr[:,:,2]
    ycbcr_diff = np.mean(Cb) - np.mean(Cr)
    ycbcr_norm = np.mean(Cb) / (np.mean(Cb) + np.mean(Cr) + 1e-5)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:,:,0])
    s_mean = np.mean(hsv[:,:,1])

    return np.array([Rn, C, ycbcr_diff, ycbcr_norm, h_mean, s_mean])