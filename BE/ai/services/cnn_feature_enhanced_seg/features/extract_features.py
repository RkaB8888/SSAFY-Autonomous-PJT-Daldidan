import cv2
import numpy as np
from skimage.color import rgb2lab
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def extract_features(image, mask):
    x, y, w, h = cv2.boundingRect(mask)
    roi = image[y:y + h, x:x + w]
    roi = cv2.resize(roi, (128, 128))  # 표준화된 크기로 리사이즈

    feats = []

    # 1. 정규화된 RGB 평균
    img = roi.astype(np.float32)
    s = img.sum(axis=2, keepdims=True) + 1e-5
    norm_rgb = img / s
    feats.extend(norm_rgb.mean(axis=(0, 1)))  # 3개

    # 2. CMY 색공간 C 채널 평균
    cmy = 1 - img / 255.0
    feats.append(cmy[:, :, 0].mean())  # 1개

    # 3. HSV 평균
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    feats.extend(hsv.mean(axis=(0, 1)))  # 3개

    # 4. GLCM texture (correlation, contrast)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[np.deg2rad(135)],
                        levels=256, symmetric=True, normed=True)
    feats.append(graycoprops(glcm, "correlation")[0, 0])
    feats.append(graycoprops(glcm, "contrast")[0, 0])  # 2개

    # 5. LBP 히스토그램 (64bin)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=64, range=(0, 64), density=True)
    feats.extend(lbp_hist.tolist())  # 64개

    # 6. RGB 히스토그램 (R/G/B 각각 16bin)
    for i in range(3):
        h = cv2.calcHist([roi], [i], None, [16], [0, 256])
        h = cv2.normalize(h, h).flatten()
        feats.extend(h.tolist())  # 16 * 3 = 48개

    # 7. 종횡비
    h, w = roi.shape[:2]
    ar = w / h if h != 0 else 0
    feats.append(ar)  # 1개

    # 8. CIELAB 평균값
    lab = rgb2lab(roi)
    feats.extend(lab.mean(axis=(0, 1)))  # 3개

    # 9. Colorfulness 지수
    R, G, B = cv2.split(roi.astype("float"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    col = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    feats.append(col)  # 1개

    return np.array(feats, dtype=np.float32)  # 총 126차원
