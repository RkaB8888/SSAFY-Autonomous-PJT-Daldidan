import random
import json
from pathlib import Path
import shutil
from tqdm import tqdm

# 원본 경로
IMG_SRC = Path(r"C:\Users\SSAFY\Desktop\Apple_Image\Training\origin\Fuji")
JSON_SRC = Path(r"C:\Users\SSAFY\Desktop\Apple_Image\Training\label\Fuji")

# 대상 경로
IMG_DST = Path("dataset/images")
JSON_DST = Path("dataset/jsons")

NUM_SAMPLES = 1000


def sample_matched_pairs():
    # 폴더 초기화
    if IMG_DST.exists():
        shutil.rmtree(IMG_DST)
    if JSON_DST.exists():
        shutil.rmtree(JSON_DST)
    IMG_DST.mkdir(parents=True, exist_ok=True)
    JSON_DST.mkdir(parents=True, exist_ok=True)

    all_images = list(IMG_SRC.glob("*.jpg"))
    print(f"전체 이미지 수: {len(all_images)}")

    random.shuffle(all_images)

    with tqdm(total=NUM_SAMPLES, desc="Copying matched pairs") as pbar:
        for img_path in all_images:
            json_path = JSON_SRC / (img_path.stem + ".json")
            if not json_path.exists():
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sugar = data.get("collection", {}).get("sugar_content", None)
                if sugar is None:
                    continue

                # 유효한 경우 복사 및 카운트
                shutil.copy2(img_path, IMG_DST / img_path.name)
                shutil.copy2(json_path, JSON_DST / json_path.name)
                pbar.update(1)

                if pbar.n >= NUM_SAMPLES:
                    break

            except Exception as e:
                print(f"[Error] {json_path.name}: {e}")

    print(f"\n✅ 최종 복사된 유효한 쌍: {pbar.n} / 요청 수: {NUM_SAMPLES}")


if __name__ == "__main__":
    sample_matched_pairs()
