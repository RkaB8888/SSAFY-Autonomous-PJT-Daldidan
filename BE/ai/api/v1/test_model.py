import os
import requests
import pandas as pd

# 설정
image_dir = r"C:\Users\SSAFY\Downloads\Test"
# image_dir = r"C:\Users\SSAFY\Downloads\green"

actual_sugars = [
    10.4, 11.9, 12.4, 12.3, 13.1, 12.7, 13.6, 10.2, 14.0, 12.1,
    13.5, 11.1, 13.9, 10.8, 13.3, 10.0, 11.5, 11.4, 12.0, 11.5,
    13.0, 12.2, 12.7, 12.7, 12.3, 10.8, 14.7, 13.5, 12.6, 16.3,
    12.2, 12.1, 13.3, 12.5, 12.1, 12.2, 11.1, 11.9, 10.9, 12.4,
    11.4, 10.2, 12.8, 9.5, 9.7
]
  # 총 45개, 1번 ~ 45번 이미지의 실제값
# model_names = [ "model_jmk3", "model_jmk4", "cnn_lgbm_bbox", "model_jmk2", "xgb_seg", "lgbm_bbox", "cnn_feature_enhanced_seg"]
model_names = [ "cnn_feature_maskcrop_seg"]
# model_names = [ "cnn_feature_maskcrop_seg", 
#                "cnn_feature_enhanced_seg", 
#                "cnn_feature_finetuning_seg", 
#                "cnn_feature_seg",
#                "cnn_feature_seg_v2", 
               
#                "cnn_lgbm_bbox", 
#                "cnn_lgbm_seg", 
#                "lgbm_bbox", 
#                "lgbm_seg",

#                "xgb_bbox", 
#                "xgb_seg"]


server_url = "http://localhost:9000/predict"

# 결과 저장용 리스트
results = []

# 1~45번 이미지 순회
for idx in range(1, 46):
# for idx in range(1, 13):
    image_path = os.path.join(image_dir, f"{idx}.jpg")
    if not os.path.exists(image_path):
        print(f"[WARNING] 이미지 없음: {image_path}")
        continue

    for model in model_names:
        with open(image_path, "rb") as img_file:
            files = {"image": (f"{idx}.jpg", img_file, "multipart/form-data")}
            try:
                response = requests.post(f"{server_url}?model={model}", files=files)
                response.raise_for_status()
                data = response.json()
                sugar = data["results"][0]["sugar_content"] if data["results"] else None
            except Exception as e:
                print(f"[ERROR] {idx}.jpg - {model}: {e}")
                sugar = None

        results.append({
            "image": f"{idx}.jpg",
            "model": model,
            "actual": actual_sugars[idx - 1],
            "predicted": sugar
        })
        print(f"[{idx}.jpg] {model} | 실제: {actual_sugars[idx - 1]} / 예측: {sugar}")

# DataFrame으로 저장
df = pd.DataFrame(results)

# 저장 경로 (예: 바깥으로 이동할것)
os.makedirs("ai/tmp", exist_ok=True)
save_path = "ai/tmp/predict_log.csv" # 경로 수정요함
df.to_csv(save_path, index=False)
print(f"\n📁 결과 저장 완료: {save_path}")