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

    ## 푸른 사과 보정을 위해 predict시만 사용할것. 학습시에는 주석처리 할것(시작점)
        # Lab 색공간 변환
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    a_mean = np.mean(a)
    b_mean = np.mean(b)

    # delta_E (유사도) 계산용 기준색 (예: 빨간색 중심점)
    lab_ref = np.array([[[70, 150, 150]]], dtype=np.uint8)  # 대충 후지 사과 느낌
    lab_ref = lab_ref.astype(np.float32)
    lab_img = lab.astype(np.float32)
    delta_E = np.mean(np.sqrt(np.sum((lab_img - lab_ref)**2, axis=2)))
    ## 푸른 사과 보정을 위해 predict시만 사용할것. 학습시에는 주석처리 할것(끝점)


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

    # return np.array([Rn, C, ycbcr_diff, ycbcr_norm, cat02_first, cluster_shadow])
    
    ## 푸른 사과 보정을 위해 predict시만 사용할것. 학습시에는 주석처리 할것(시작점)
    return {
        "Rn": Rn,
        "C": C,
        "ycbcr_diff": ycbcr_diff,
        "ycbcr_norm": ycbcr_norm,
        "cat02_first": cat02_first,
        "cluster_shadow": cluster_shadow,
        "a_mean": a_mean,
        "b_mean": b_mean,
        "delta_E": delta_E
    }
    ## 푸른 사과 보정을 위해 predict시만 사용할것. 학습시에는 주석처리 할것(끝점)

# 속도 최적화(정확도 저하)
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