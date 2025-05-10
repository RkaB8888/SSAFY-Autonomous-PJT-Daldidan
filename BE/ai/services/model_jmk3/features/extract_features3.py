import cv2
import numpy as np
from skimage import color, feature

def extract_features_extended(image, mask):
    x, y, w, h = cv2.boundingRect(mask)
    roi = image[y:y+h, x:x+w]

    # 기존 feature
    R, G, B = roi[:, :, 2], roi[:, :, 1], roi[:, :, 0]
    sum_RGB = R + G + B + 1e-5
    Rn = np.mean(R / sum_RGB)
    C = np.mean(1 - R / 255.0)

    YCbCr = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    Cb, Cr = YCbCr[:, :, 1], YCbCr[:, :, 2]
    ycbcr_diff = np.mean(Cb) - np.mean(Cr)
    ycbcr_norm = np.mean(Cb) / (np.mean(Cb) + np.mean(Cr) + 1e-5)

    xyz = color.rgb2xyz(roi / 255.0)
    M_CAT02 = np.array([[0.7328, 0.4296, -0.1624],
                        [-0.7036, 1.6975, 0.0061],
                        [0.0030, 0.0136, 0.9834]])
    lms = np.dot(xyz, M_CAT02.T)
    cat02_first = np.mean(lms[:, :, 0])

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    glcm = feature.graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
    contrast_0 = feature.graycoprops(glcm, 'contrast')[0, 0]
    contrast_45 = feature.graycoprops(glcm, 'contrast')[0, 1]
    contrast_90 = feature.graycoprops(glcm, 'contrast')[0, 2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])   # 추가된 feature

    # feature 배열에 s_mean 추가
    return np.array([Rn, C, ycbcr_diff, ycbcr_norm, cat02_first,
                     contrast_0, contrast_45, contrast_90, h_mean, s_mean])
