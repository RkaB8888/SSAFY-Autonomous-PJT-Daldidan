
import os
import requests
import pandas as pd

# 설정
base_dir = r"C:\Users\SSAFY\Downloads\Test2"
actual_sugars = [
    10.4, 11.9, 12.4, 12.3, 13.1, 12.7, 13.6, 10.2, 14.0, 12.1,
    13.5, 11.1, 13.9, 10.8, 13.3, 10.0, 11.5, 11.4, 12.0, 11.5,
    13.0, 12.2, 12.7, 12.7, 12.3, 10.8, 14.7, 13.5, 12.6, 16.3,
    12.2, 12.1, 13.3, 12.5, 12.1, 12.2, 11.1, 11.9, 10.9, 12.4,
    11.4, 10.2, 12.8, 9.5, 9.7
]  # 총 45개 사과의 실제값

model_names = [ 
    "cnn_feature_maskcrop_seg", 
]

# model_names = [ 
#     "cnn_feature_maskcrop_seg", 
#     "cnn_feature_enhanced_seg", 
#     "cnn_feature_finetuning_seg", 
#     "cnn_feature_seg",
#     "cnn_feature_seg_v2", 
#     "cnn_lgbm_bbox", 
#     "cnn_lgbm_seg", 
#     "lgbm_bbox", 
#     "lgbm_seg",
#     "xgb_bbox", 
#     "xgb_seg"
# ]

server_url = "http://localhost:9000/predict"
results = []

# 각 사과별 폴더(1 ~ 45)
for idx in range(1, 46):
    folder_path = os.path.join(base_dir, str(idx))
    if not os.path.isdir(folder_path):
        print(f"[WARNING] 폴더 없음: {folder_path}")
        continue

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"[WARNING] 이미지 없음: {folder_path}")
        continue

    for model in model_names:
        predictions = []
        for image_name in image_files:
            image_path = os.path.join(folder_path, image_name)
            with open(image_path, "rb") as img_file:
                files = {"image": (image_name, img_file, "multipart/form-data")}
                try:
                    response = requests.post(f"{server_url}?model={model}", files=files)
                    response.raise_for_status()
                    data = response.json()
                    sugar = data["results"][0]["sugar_content"] if data["results"] else None
                except Exception as e:
                    print(f"[ERROR] {image_name} - {model}: {e}")
                    sugar = None
            if sugar is not None:
                predictions.append(sugar)

        # 평균값으로 예측값 집계
        predicted_avg = round(sum(predictions) / len(predictions), 3) if predictions else None

        results.append({
            "apple_id": idx,
            "model": model,
            "actual": actual_sugars[idx - 1],
            "predicted_avg": predicted_avg,
            "num_images": len(predictions)
        })
        print(f"[{idx}번 사과] {model} | 실제: {actual_sugars[idx - 1]} / 평균 예측: {predicted_avg} (이미지 {len(predictions)}장)")

# 결과 저장
df = pd.DataFrame(results)
os.makedirs("ai/tmp", exist_ok=True)
save_path = "ai/tmp/predict_log_v2.csv"
df.to_csv(save_path, index=False)
print(f"\n📁 결과 저장 완료: {save_path}")
