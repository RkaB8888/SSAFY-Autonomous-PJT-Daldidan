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

# === 실제 feature 추출 함수 (image, mask 기반) ===
def extract_features(image, mask):
    x, y, w, h = cv2.boundingRect(mask)
    roi = image[y:y+h, x:x+w]

    R, G, B = roi[:, :, 2], roi[:, :, 1], roi[:, :, 0]
    sum_RGB = R + G + B + 1e-5
    Rn = np.mean(R / sum_RGB)
    print(f"[DEBUG] Rn: {Rn}")

    C = np.mean(1 - R / 255.0)
    print(f"[DEBUG] C: {C}")

    YCbCr = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    Cb, Cr = YCbCr[:, :, 1], YCbCr[:, :, 2]
    cb_mean = np.mean(Cb)
    cr_mean = np.mean(Cr)
    ycbcr_diff = cb_mean - cr_mean
    ycbcr_norm = cb_mean / (cb_mean + cr_mean + 1e-5)
    print(f"[DEBUG] ycbcr_diff: {ycbcr_diff}, ycbcr_norm: {ycbcr_norm}")

    xyz = color.rgb2xyz(roi / 255.0)
    M_CAT02 = np.array([[0.7328, 0.4296, -0.1624],
                        [-0.7036, 1.6975, 0.0061],
                        [0.0030, 0.0136, 0.9834]])
    lms = np.dot(xyz, M_CAT02.T)
    cat02_first = np.mean(lms[:, :, 0])
    print(f"[DEBUG] cat02_first: {cat02_first}")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    glcm = feature.graycomatrix(gray, distances=[1], angles=[np.pi/4], levels=256, symmetric=True, normed=True)
    cluster_shadow = feature.graycoprops(glcm, 'contrast')[0, 0]
    print(f"[DEBUG] cluster_shadow: {cluster_shadow}")

    return np.array([Rn, C, ycbcr_diff, ycbcr_norm, cat02_first, cluster_shadow])


# === 멀티프로세싱용 wrapper 함수 ===
def extract_features_from_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # ✅ json 파일명 → 이미지 파일명 추출
        json_filename = os.path.basename(json_path)
        img_filename = os.path.splitext(json_filename)[0] + '.jpg'
        img_path = os.path.join(IMG_DIR, img_filename)

        image = cv2.imread(img_path)

        if image is None:
            print(f"[WARNING] 이미지 로드 실패: {img_path}")
            return None

        points = np.array(data['annotations']['segmentation']).reshape((-1, 2)).astype(np.int32)
        img_h, img_w = data['images']['img_height'], data['images']['img_width']
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

         # ✅ label 추출
        label = data['collection'].get('sugar_content_nir')
        if label is None:
            print(f"[WARNING] label 값 없음: {json_path}")
            return None

        feature = extract_features(image, mask)

        return (feature, label)  # ← feature와 label을 tuple로 반환

    except Exception as e:
        print(f"[ERROR] {json_path}: {e}")
        return None


# === main ===
if __name__ == "__main__":
    from glob import glob

    json_files = glob(os.path.join(JSON_DIR, "*.json"))
    print(f"✅ 총 JSON 파일 수: {len(json_files)}")

    with Pool(processes=48) as pool:
        results = list(tqdm(pool.imap(extract_features_from_json, json_files), total=len(json_files)))

    # feature와 label을 각각 리스트에 저장
    features = []
    labels = []
    for res in results:
        if res is not None:
            feature, label = res
            features.append(feature)
            labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    np.save(OUTPUT_PATH, features)
    np.save("/home/j-k12e206/jmk/S12P31E206/BE/ai/services/model_jmk2/meme/labels.npy", labels)

    print(f"✅ feature 저장 완료 → {OUTPUT_PATH}, shape: {features.shape}")
    print(f"✅ label 저장 완료 → labels.npy, shape: {labels.shape}")

