import cv2
import numpy as np

def extract_gloss(image):
    """광택 추출 (L 채널에서 밝은 영역 비율)"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    _, gloss_mask = cv2.threshold(L, 200, 255, cv2.THRESH_BINARY)
    gloss_ratio = np.sum(gloss_mask == 255) / gloss_mask.size
    return gloss_ratio

def extract_color_ratio(image):
    """색상 추출 (R / (G+B))"""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R = rgb[:,:,0]
    G = rgb[:,:,1]
    B = rgb[:,:,2]
    color_ratio = np.mean(R) / (np.mean(G) + np.mean(B) + 1e-5)  # 0으로 나누는 것 방지
    return color_ratio

def extract_texture(image):
    """질감 추출 (Laplacian variance + 정규화)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    var = laplacian.var()
    
    # 정규화: 0~1000 사이 값으로 스케일 다운
    normalized_texture = var / 1000.0  # 임시로 1000 나누기
    return normalized_texture