# import os
# import json
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import joblib
# from features.extract_features import extract_features
# import cv2
# from tqdm import tqdm
# import time 

# IMG_DIR = "/home/j-k12e206/ai-hub/Fuji/train/images"
# JSON_DIR = "/home/j-k12e206/ai-hub/Fuji/train/jsons"

# json_files = [os.path.join(JSON_DIR, f) for f in os.listdir(JSON_DIR) if f.endswith('.json')]

# features = []
# start_time = time.time()   # 실행 시작 시간 기록

# for i, json_path in enumerate(tqdm(json_files, desc="Feature 추출 진행상황")):  # tqdm 추가
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     json_filename = os.path.basename(json_path)
#     img_filename = os.path.splitext(json_filename)[0] + '.jpg'
#     img_path = os.path.join(IMG_DIR, img_filename)

#     image = cv2.imread(img_path)
#     if image is None:
#         print(f"[WARNING] 이미지 로드 실패: {img_path}")
#         continue

#     points = np.array(data['annotations']['segmentation']).reshape((-1, 2)).astype(np.int32)
#     img_h = data['images']['img_height']
#     img_w = data['images']['img_width']
#     mask = np.zeros((img_h, img_w), dtype=np.uint8)
#     cv2.fillPoly(mask, [points], 255)

#     feature = extract_features(image, mask)
#     features.append(feature)

#     if i % 100 == 0:
#         print(f"[INFO] {i}/{len(json_files)}번째 feature 추출 완료")

# features = np.array(features)
# scaler = StandardScaler().fit(features)

# SAVE_PATH = "/home/j-k12e206/jmk/S12P31E206/BE/ai/services/model_jmk2/meme/scaler.pkl"
# joblib.dump(scaler, SAVE_PATH)

# end_time = time.time()   # 실행 종료 시간 기록
# elapsed = end_time - start_time
# print(f"✅ scaler.pkl 저장 완료 → {SAVE_PATH}")
# print(f"✅ scaler 생성 완료. 총 소요 시간: {elapsed:.2f}초")   # 시간 출력
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

IMG_DIR = "/home/j-k12e206/ai-hub/Fuji/train/images"
JSON_DIR = "/home/j-k12e206/ai-hub/Fuji/train/jsons"

json_files = [os.path.join(JSON_DIR, f) for f in os.listdir(JSON_DIR) if f.endswith('.json')]

def process_file(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        json_filename = os.path.basename(json_path)
        img_filename = os.path.splitext(json_filename)[0] + '.jpg'
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
        return feature

    except Exception as e:
        print(f"[ERROR] {json_path}: {e}")
        return None

if __name__ == "__main__":
    start_time = time.time()

    num_workers = 48  # 서버 코어수에 맞춰 (ex: 48, 64, 96 등 조절)
    print(f"✅ 멀티프로세싱 시작 (workers: {num_workers})")

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_file, json_files), total=len(json_files), desc="Feature 추출 진행중"))

    # None 결과 제거
    features = [res for res in results if res is not None]
    features = np.array(features)

    scaler = StandardScaler().fit(features)

    SAVE_PATH = "/home/j-k12e206/jmk/S12P31E206/BE/ai/services/cnn_feature_seg/meme/scaler.pkl"
    joblib.dump(scaler, SAVE_PATH)

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"✅ scaler.pkl 저장 완료 → {SAVE_PATH}")
    print(f"✅ scaler 생성 완료. 총 소요 시간: {elapsed:.2f}초")
