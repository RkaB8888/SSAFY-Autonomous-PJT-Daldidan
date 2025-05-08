# ai/services/model_jhg2/extract_embeddings.py
import json
import pathlib
import numpy as np
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from services.model_jhg2.config import CACHE_DIR, IMAGES_DIR, JSONS_DIR
from services.model_jhg2.utils.cnn_feature_extractor import extract_batch


class CroppedDataset(Dataset):
    def __init__(
        self, img_dir: pathlib.Path, json_dir: pathlib.Path, resize=(256, 256)
    ):
        # 1) 원래 있던 (이미지, json) 쌍을 모두 수집
        raw_pairs = sorted(
            [
                (p, json_dir / f"{p.stem}.json")
                for p in img_dir.glob("*.jpg")
                if (json_dir / f"{p.stem}.json").exists()
            ]
        )

        # 2) sugar_content가 반드시 있는 것만 필터링
        self.pairs = []
        for img_p, js_p in raw_pairs:
            data = json.loads(js_p.read_text(encoding="utf-8"))
            # sugar_content가 존재해야만 추가
            if data.get("collection", {}).get("sugar_content") is not None:
                self.pairs.append((img_p, js_p))

        self.resize = resize

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, js_p = self.pairs[idx]
        try:
            # — 이미지 열어서 crop & resize
            img = Image.open(img_p).convert("RGB")
            with open(js_p, "r", encoding="utf-8") as f:
                data = json.load(f)
            x, y, w, h = map(int, data["annotations"]["bbox"])
            crop = img.crop((x, y, x + w, y + h)).resize(
                self.resize, Image.Resampling.LANCZOS
            )
            arr = np.array(crop, copy=False)

            # — 레이블(당도)
            coll = data.get("collection", {})
            sugar = coll.get("sugar_content")
            # sugar = coll.get("sugar_content") or coll.get("sugar_content_nir")

            if sugar is None:
                print(f"[Warning] sugar_content 누락, 건너뜁니다: {js_p.name}")
                return None  # 해당 샘플을 건너뜀

            sugar = float(sugar)
            return arr, sugar, img_p.stem

        except Exception as e:
            print(f"[Warning] 에러 발생 (idx={idx}): {e}")
            return None  # 에러 시 None 반환


def build_and_cache_embeddings(
    img_dir: pathlib.Path = IMAGES_DIR,
    json_dir: pathlib.Path = JSONS_DIR,
    batch_size: int = 1024,
    num_workers: int = 32,
    prefetch: int = 4,
):
    """
    1) IMAGE_DIR/JSON_DIR 를 돌면서 CroppedDataset 으로 이미지-당도-파일명 추출
    2) DataLoader 로 배치 단위 CNN 임베딩 → 메모리맵 + np.save 캐싱
    """
    # 캐시 디렉터리 준비
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feat_path = CACHE_DIR / "train_embeddings.npy"
    label_path = CACHE_DIR / "train_labels.npy"
    stems_path = CACHE_DIR / "train_stems.npy"

    ds = CroppedDataset(img_dir, json_dir)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda x: [s for s in x if s is not None],
        prefetch_factor=prefetch,
        pin_memory=False,
        persistent_workers=False,
    )

    N = len(ds)
    D = 1280  # EfficientNet-B0 출력 차원

    # 메모리맵 준비
    feats = np.memmap(feat_path, dtype=np.float32, mode="w+", shape=(N, D))
    labels = np.memmap(label_path, dtype=np.float32, mode="w+", shape=(N,))
    stems = []

    idx = 0
    for batch in tqdm.tqdm(dl, total=len(dl), ncols=80):
        # 빈 배치는 건너뛰기
        if not batch:
            continue
        imgs, sugars, batch_stems = zip(*batch)
        batch_np = np.stack(imgs, axis=0)  # (B,H,W,C)
        vecs = extract_batch(batch_np)  # (B, D)
        B = vecs.shape[0]

        feats[idx : idx + B] = vecs
        labels[idx : idx + B] = sugars
        stems.extend(batch_stems)

        idx += B

    feats.flush()
    labels.flush()
    np.save(stems_path, np.array(stems))

    print(f"[Saved] embeddings→{feat_path}, labels→{label_path}, stems→{stems_path}")


if __name__ == "__main__":
    # 스크립트로 직접 실행할 때만 임베딩 생성
    build_and_cache_embeddings()
