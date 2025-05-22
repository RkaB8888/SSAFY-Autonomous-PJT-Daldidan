# import numpy as np
# import cv2
# import torch
# from PIL import Image
# from .model_loader import model, scaler, transform

# # .features.extract_features에서 가져오는 모델 수정
# # from .features.extract_features import extract_fast_features
# from .features.extract_features import extract_features

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def predict_bytes(image_bytes: bytes) -> float:
#     try:
#         np_arr = np.frombuffer(image_bytes, np.uint8)
#         img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         if img is None:
#             raise ValueError("이미지 디코딩 실패")

#         h, w = img.shape[:2]
#         mask = np.ones((h, w), dtype=np.uint8) * 255

#         # .features.extract_features에서 가져오는 모델 수정
#         # manual_feat = extract_fast_features(img, mask)
#         manual_feat = extract_features(img, mask)

#         manual_feat = scaler.transform([manual_feat])[0]
#         manual_feat_tensor = torch.tensor(manual_feat, dtype=torch.float32).unsqueeze(0).to(device)
#         image_tensor = transform(img).unsqueeze(0).to(device)

#         with torch.no_grad():
#             pred = model(image_tensor, manual_feat_tensor).squeeze().item()

#         return round(pred, 2)

#     except Exception as e:
#         print(f"[ERROR] predict_bytes 실패: {e}")
#         raise e


## 푸른 사과 보정을 위해 predict시만 사용할것. 학습시에는 주석처리 할것(시작점)
import numpy as np
import cv2
import torch
from PIL import Image
from .model_loader import model, scaler, transform
from .features.extract_features import extract_features  # 사용 함수

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_bytes(image_bytes: bytes) -> float:
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("이미지 디코딩 실패")
        # print("📥 이미지 디코딩 완료")  # ✅ 여기가 디코딩 성공 직후

        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        feats = extract_features(img, mask)  # <- 추론 시 사용
        # print("📊 추출된 feats:", feats)  # ✅ 추출 완료 직후

        a_mean = feats["a_mean"]
        b_mean = feats["b_mean"]
        delta_E = feats["delta_E"]
        # print(f"a: {a_mean:.2f}, b: {b_mean:.2f}, delta_E: {delta_E:.2f}")

        # if a_mean < 125 and b_mean > 110:
        #     if delta_E > 80:
        #         print("🟢 푸른 사과 감지 → 브릭스 7.5로 소프트 보정")
        #         return 7.5
        #     elif delta_E > 70:
        #         print("🟡 푸른 기색 감지 → 브릭스 8.0으로 소프트 보정")
        #         return 8.0
        if a_mean < 133 and b_mean > 130:
            if delta_E > 55:
                print("🟢 푸른 사과 감지 → 브릭스 7.5로 소프트 보정")
                return 7.5
            elif delta_E > 45:
                print("🟡 푸른 기색 감지 → 브릭스 8.0으로 소프트 보정")
                return 8.0

        # 🎯 CNN 추론
        feat_vector = np.array(
            [feats[k] for k in list(feats)[:6]]
        )  # CNN 학습 피처만 추출

        manual_feat = scaler.transform([feat_vector])[0]  # ❗ 여기가 수정 포인트
        # print("✅ CNN 예측 전 manual_feat:", manual_feat)  # ✅ CNN 입력 직전
        manual_feat_tensor = (
            torch.tensor(manual_feat, dtype=torch.float32).unsqueeze(0).to(device)
        )
        image_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image_tensor, manual_feat_tensor).squeeze().item()

        return round(pred, 2)

    except Exception as e:
        print(f"[ERROR] predict_bytes 실패: {e}")
        raise e
        ## 푸른 사과 보정을 위해 predict시만 사용할것. 학습시에는 주석처리 할것(끝점)
