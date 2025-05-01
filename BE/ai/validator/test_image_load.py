# =========================================
# 📄 파일명: test_image_load.py
# 📋 설명: OpenCV(cv2)와 Pillow(PIL)를 이용한 이미지 로딩 테스트
#         - 파일 존재 여부 확인
#         - 경로 길이 출력
#         - OpenCV 로딩 테스트
#         - Pillow 로딩 및 회전 정보 적용 여부 테스트
# =========================================

import cv2
from PIL import Image
from pathlib import Path

# 테스트할 이미지 경로
img_path = Path(
    r"C:\Users\SSAFY\Desktop\146.전북 장수 사과 당도 품질 데이터\01.데이터\1.Training\원천데이터\후지3\당도B등급\20210926_RGB_12.7_F15_HJ_02_011_02_0_A.jpg"
)

print("🔍 파일 존재 여부:", img_path.exists())
print("📏 경로 길이:", len(str(img_path)))

# OpenCV 로딩 테스트
cv_img = cv2.imread(str(img_path))
print("📷 OpenCV로 이미지 로딩 성공 여부:", cv_img is not None)
if cv_img is not None:
    cv2.imshow("OpenCV Image", cv_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("⚠️ OpenCV는 이미지를 읽지 못했습니다.")

# Pillow 로딩 테스트
try:
    img = Image.open(img_path)
    img.verify()  # 파일 유효성 검사
    print("✅ Pillow로 열기 성공 (verify 통과)")
except Exception as e:
    print("❌ Pillow로도 열기 실패:", e)

# Pillow로 다시 열어서 보기
try:
    img = Image.open(img_path)
    img.show()
except Exception as e:
    print("❌ Pillow 이미지 보기 실패:", e)
