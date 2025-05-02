import os
import cv2
from feature_extractor import extract_gloss, extract_color_ratio, extract_texture
from model import calculate_sweetness, score_to_brix
from utils import preprocess_image

def predict_sweetness(image_path):
    img = cv2.imread(image_path)
    img = preprocess_image(img)

    gloss = extract_gloss(img)
    color = extract_color_ratio(img)
    texture = extract_texture(img)

    sweetness_score = calculate_sweetness(gloss, color, texture)
    brix_prediction = score_to_brix(sweetness_score)

    print(f"파일명: {os.path.basename(image_path)}")
    print(f"Sweetness Score: {sweetness_score:.3f}")
    print(f"Predicted Brix: {brix_prediction}")
    print("-" * 30)

if __name__ == "__main__":
    # 테스트할 폴더 경로
    test_folder = "test_images/"

    # 폴더 안 모든 이미지 파일 가져오기
    image_files = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.png'))]

    for file_name in image_files:
        file_path = os.path.join(test_folder, file_name)
        predict_sweetness(file_path)
