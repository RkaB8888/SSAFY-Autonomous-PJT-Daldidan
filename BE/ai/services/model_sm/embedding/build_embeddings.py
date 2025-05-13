# ai/services/model_sm/embedding/build_embeddings.py

from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed

from services.model_sm.config import (
    IMAGES_DIR,
    JSONS_DIR,
    VALID_IMAGES_DIR,
    VALID_JSONS_DIR,
    CACHE_DIR,
    USE_SEGMENTATION,
)
from services.model_sm.utils.cropper import crop_apple
from services.model_sm.extractor.common_features import extract_features


def get_seg_suffix() -> str:
    return "seg" if USE_SEGMENTATION else "bbox"


def process_one_json(
    json_path: Path, images_dir: Path
) -> tuple[list[float], float, str] | None:
    try:
        with open(json_path, encoding="utf-8-sig") as f:
            meta = json.load(f)

        img_name = meta["images"]["img_file_name"]
        img_path = images_dir / img_name
        if not img_path.exists():
            return None

        img = Image.open(img_path).convert("RGB")
        ann = meta["annotations"]
        crop = crop_apple(img, ann, use_seg=USE_SEGMENTATION)
        if crop is None:
            return None

        feat = extract_features(np.array(crop))
        label = meta["collection"].get("sugar_content")
        if label is None:
            label = meta["collection"].get("sugar_content_nir")

        if label is None:
            return None

        stem = Path(img_name).stem
        return list(feat.values()), label, stem

    except Exception as e:
        print(f"❌ {json_path.name} 처리 실패 → {e}")
        return None


def build_and_cache_embeddings(prefix: str):
    if prefix == "train":
        images_dir = IMAGES_DIR
        jsons_dir = JSONS_DIR
    elif prefix == "valid":
        images_dir = VALID_IMAGES_DIR
        jsons_dir = VALID_JSONS_DIR
    else:
        raise ValueError(f"지원하지 않는 prefix: {prefix}")

    suffix = get_seg_suffix()
    feats_path = CACHE_DIR / f"{prefix}_embeddings_{suffix}.dat"
    labels_path = CACHE_DIR / f"{prefix}_labels_{suffix}.dat"
    stems_path = CACHE_DIR / f"{prefix}_stems_{suffix}.npy"

    all_jsons = sorted(jsons_dir.glob("*.json"))
    print(f"🚀 {prefix} ({suffix}) 병렬 임베딩 시작 - {len(all_jsons)}개")

    results = Parallel(n_jobs=32)(
        delayed(process_one_json)(json_path, images_dir) for json_path in all_jsons
    )

    feat_list, label_list, stems = [], [], []
    for result in results:
        if result is not None:
            feat, label, stem = result
            feat_list.append(feat)
            label_list.append(label)
            stems.append(stem)

    feats_np = np.array(feat_list, dtype=np.float32)
    labels_np = np.array(label_list, dtype=np.float32)
    n, d = feats_np.shape

    # memmap 저장
    X_mem = np.memmap(feats_path, dtype="float32", mode="w+", shape=(n, d))
    X_mem[:] = feats_np[:]
    del X_mem

    y_mem = np.memmap(labels_path, dtype="float32", mode="w+", shape=(n,))
    y_mem[:] = labels_np[:]
    del y_mem

    np.save(stems_path, np.array(stems))
    print(f"✅ {prefix} 캐시 저장 완료: {n}개 샘플, {d}차원 특징 → suffix={suffix}")


def load_cache(prefix: str) -> tuple[np.ndarray, np.ndarray]:
    suffix = get_seg_suffix()
    feats_path = CACHE_DIR / f"{prefix}_embeddings_{suffix}.dat"
    labels_path = CACHE_DIR / f"{prefix}_labels_{suffix}.dat"
    stems_path = CACHE_DIR / f"{prefix}_stems_{suffix}.npy"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not feats_path.exists() or not labels_path.exists() or not stems_path.exists():
        print(f"🚧 캐시 없음 → {prefix} ({suffix}) 캐시 생성 중...")
        build_and_cache_embeddings(prefix)

    y_raw = np.memmap(labels_path, dtype="float32", mode="r")
    D = int(np.memmap(feats_path, dtype="float32", mode="r").size / len(y_raw))
    N = len(y_raw)

    X = np.memmap(feats_path, dtype="float32", mode="r", shape=(N, D))
    y = np.memmap(labels_path, dtype="float32", mode="r", shape=(N,))
    stems = np.load(stems_path)
    print(f"📦 {prefix} 캐시 로드 완료: shape=({N}, {D})")
    return X, y, stems


if __name__ == "__main__":
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    load_cache("train")
    load_cache("valid")
