import os
import json
import cv2
import numpy as np
from skimage import color, feature
from multiprocessing import Pool
from tqdm import tqdm

# === 경로 설정 ===
IMG_DIR = r"/home/j-k12e206/ai-hub/Fuji/train/images"
JSON_DIR = r"/home/j-k12e206/ai-hub/Fuji/train/jsons"
OUTPUT_PATH = r"/home/j-k12e206/jmk/S12P31E206/BE/ai/services/model_jmk2/meme/features.npy"

# === feature 추출 함수 ===
def extract_feature(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_filename = data['images']['img_file_name']
        img_path = os.path.join(IMG_DIR, img_filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"[WARNING] 이미지 로드 실패: {img_path}")
            return None

        points = np.array(data['annotations']['segmentation']).reshape((-1, 2)).astype(np.int32)
        img_h, img_w = data['images']['img_height'], data['images']['img_width']
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        # feature 추출
        x, y, w, h = cv2.boundingRect(mask)
        roi = image[y:y+h, x:x+w]

        R, G, B = roi[:,:,2], roi[:,:,1], roi[:,:,0]
        sum_RGB = R + G + B + 1e-5
        Rn = np.mean(R / sum_RGB)
        C = np.mean(1 - R / 255.0)

        YCbCr = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        Cb, Cr = YCbCr[:,:,1], YCbCr[:,:,2]
        cb_mean = np.mean(Cb)
        cr_mean = np.mean(Cr)
        ycbcr_diff = cb_mean - cr_mean
        ycbcr_norm = cb_mean / (cb_mean + cr_mean + 1e-5)

        xyz = color.rgb2xyz(roi / 255.0)
        M_CAT02 = np.array([[0.7328, 0.4296, -0.1624], [-0.7036, 1.6975, 0.0061], [0.0030, 0.0136, 0.9834]])
        lms = np.dot(xyz, M_CAT02.T)
        cat02_first = np.mean(lms[:,:,0])

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        glcm = feature.graycomatrix(gray, distances=[1], angles=[np.pi/4], levels=256, symmetric=True, normed=True)
        cluster_shadow = feature.graycoprops(glcm, 'contrast')[0, 0]

        return np.array([Rn, C, ycbcr_diff, ycbcr_norm, cat02_first, cluster_shadow])

    except Exception as e:
        print(f"[ERROR] {json_path}: {e}")
        return None

# === main ===
if __name__ == "__main__":
    from glob import glob

    json_files = glob(os.path.join(JSON_DIR, "*.json"))
    print(f"✅ 총 JSON 파일 수: {len(json_files)}")

    with Pool(processes=6) as pool:
        features = list(tqdm(pool.imap(extract_feature_from_json, json_files), total=len(json_files)))

    features = [f for f in features if f is not None]
    np.save(OUTPUT_PATH, np.array(features))
    print(f"✅ feature 저장 완료 → {OUTPUT_PATH}, shape: {np.array(features).shape}")
