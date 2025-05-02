import os
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from feature_extractor import extract_gloss, extract_color_ratio, extract_texture
from utils import preprocess_image

# 1. 사진 폴더 경로
image_folder = "sample_images/"
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

# 2. Sweetness Score 계산
sweetness_scores = []

for file_name in image_files:
    file_path = os.path.join(image_folder, file_name)
    
    # 이미지 읽기
    image = cv2.imread(file_path)
    
    # 전처리
    preprocessed_image = preprocess_image(image)
    
    # 특징 추출
    gloss = extract_gloss(preprocessed_image)
    color = extract_color_ratio(preprocessed_image)
    texture = extract_texture(preprocessed_image)
    
    # Sweetness Score 계산
    sweetness_score = (gloss * 0.4) + (color * 0.4) - (texture * 0.2)
    
    sweetness_scores.append(sweetness_score)

print("Sweetness Scores:", sweetness_scores)

# 3. Sweetness Score → 등급(A/B) 매칭해서 Brix 만들기
grades = ['A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'B', 'B', 'A', 'B', 'A']  # 네가 가지고 있는 등급 순서대로
brix_values = []
for grade in grades:
    if grade == 'A':
        brix_values.append(14.5)
    else:
        brix_values.append(13.0)

# 4. numpy 변환
sweetness_scores = np.array(sweetness_scores).reshape(-1, 1)
brix_values = np.array(brix_values)

# 5. 선형 회귀 학습
model = LinearRegression()
model.fit(sweetness_scores, brix_values)

# 6. 결과 출력
print("회귀식: Brix = {:.3f} * Score + {:.3f}".format(model.coef_[0], model.intercept_))
