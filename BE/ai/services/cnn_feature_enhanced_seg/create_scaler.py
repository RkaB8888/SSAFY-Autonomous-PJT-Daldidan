import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from features.extract_features import extract_features
import cv2
from tqdm import tqdm
import time
from multiprocessing import Pool

# 경로 설정
IMG_DIR = "/home/j-k12e206/ai-hub/Fuji/train/images"
JSON_DIR = "/home/j-k12e206/ai-hub/Fuji/train/jsons"
SAVE_DIR = "/home/j-k12e206/jmk/S12P31E206/BE/ai/services/cnn_feature_enhanced_seg/me"
os.makedirs(SAVE_DIR, exist_ok=True)

json_files = [os.path.join(JSON_DIR, f) for f in os.listdir(JSON_DIR) if f.endswith('.json')]

# 전역 리스트 대신 반환 방식으로 변경
def process_file(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_filename = os.path.splitext(os.path.basename(json_path))[0] + '.jpg'
        img_path = os.path.join(IMG_DIR, img_filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARNING] 이미지 로드 실패: {img_path}")
            return None

        points = np.array(data['annotations']['segmentation']).reshape((-1, 2)).astype(np.int32)
        img_h = data['images']['img_height']
        img_w = data['images']['img_width']
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        feature = extract_features(image, mask)
        label = float(data['collection'].get('sugar_content_nir', 0))
        return (feature, label)

    except Exception as e:
        print(f"[ERROR] {json_path}: {e}")
        return None

if __name__ == "__main__":
    start_time = time.time()

    num_workers = 48
    print(f"✅ 멀티프로세싱 시작 (workers: {num_workers})")

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_file, json_files), total=len(json_files), desc="Feature 추출 중"))

    # 유효한 결과만 필터링
    results = [r for r in results if r is not None]
    features, labels = zip(*results)

    features = np.array(features)
    labels = np.array(labels)

    # 스케일러 학습 및 적용
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 저장
    np.save(os.path.join(SAVE_DIR, "manual_features.npy"), features_scaled)
    np.save(os.path.join(SAVE_DIR, "labels.npy"), labels)
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))

    elapsed = time.time() - start_time
    print(f"\n✅ 저장 완료:")
    print(f" - manual_features.npy (shape: {features_scaled.shape})")
    print(f" - labels.npy (shape: {labels.shape})")
    print(f" - scaler.pkl")
    print(f"⏱ 총 소요 시간: {elapsed:.2f}초")
