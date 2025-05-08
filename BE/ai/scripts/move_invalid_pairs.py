import shutil
import csv
from pathlib import Path

# 원본 폴더 경로
IMG_DIR = Path("C:/Users/SSAFY/Desktop/Apple_Image/Validation/origin/Fuji")
JSON_DIR = Path("C:/Users/SSAFY/Desktop/Apple_Image/Validation/label/Fuji")

# 이동 대상 폴더
TRASH_DIR = Path("C:/Users/SSAFY/Desktop/trash")
TRASH_IMG_DIR = TRASH_DIR / "images"
TRASH_JSON_DIR = TRASH_DIR / "jsons"

# CSV 결과 파일 경로 (validate_dataset.py에서 생성한 것)
CSV_PATH = Path(
    "C:/Users/SSAFY/Desktop/테스트 결과/전체검사결과.csv"
)  # 위치 조정 필요 시 수정

# 트래시 폴더 생성
TRASH_IMG_DIR.mkdir(parents=True, exist_ok=True)
TRASH_JSON_DIR.mkdir(parents=True, exist_ok=True)

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        stem = row["filename"]

        img_file = IMG_DIR / f"{stem}.jpg"
        json_file = JSON_DIR / f"{stem}.json"

        # 이미지 이동
        if img_file.exists():
            shutil.move(str(img_file), TRASH_IMG_DIR / img_file.name)
            print(f"🟡 이미지 이동: {img_file.name}")
        else:
            print(f"⚠️ 이미지 없음: {img_file}")

        # JSON 이동
        if json_file.exists():
            shutil.move(str(json_file), TRASH_JSON_DIR / json_file.name)
            print(f"🔵 JSON 이동: {json_file.name}")
        else:
            print(f"⚠️ JSON 없음: {json_file}")

print("\n✅ 이동 완료.")
