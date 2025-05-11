# services/model_jhg3/embedding/build_embeddings.py
import argparse
import json
import pathlib
import numpy as np
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from services.model_jhg3.config import (
    CACHE_DIR,
    IMAGES_DIR,
    JSONS_DIR,
    VALID_IMAGES_DIR,
    VALID_JSONS_DIR,
    USE_NIR,
    USE_SEGMENTATION,
)
from services.model_jhg3.embedding.embedding_dispatcher import extract_embedding
from services.model_jhg3.utils.cropper import crop_apple


class CroppedDataset(Dataset):
    def __init__(
        self, img_dir: pathlib.Path, json_dir: pathlib.Path, resize=(256, 256)
    ):
        raw = sorted(
            [
                (p, json_dir / f"{p.stem}.json")
                for p in img_dir.glob("*.jpg")
                if (json_dir / f"{p.stem}.json").exists()
            ]
        )
        self.items = []
        for img_p, js_p in raw:
            data = json.loads(js_p.read_text(encoding="utf-8"))
            coll = data.get("collection", {})
            # 레이블 필터
            if USE_NIR:
                if (
                    coll.get("sugar_content") is None
                    and coll.get("sugar_content_nir") is None
                ):
                    continue
            else:
                if coll.get("sugar_content") is None:
                    continue
            # 크롭 유효성 필터
            ann = data.get("annotations", {})
            if USE_SEGMENTATION:
                if not ann.get("segmentation"):
                    continue
            else:
                bb = ann.get("bbox", [])
                if len(bb) != 4 or bb[2] <= 0 or bb[3] <= 0:
                    continue
            # (경로, JSON 매핑 dict) 캐싱
            self.items.append((img_p, data))
        self.resize = resize

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_p, data = self.items[idx]
        img = Image.open(img_p).convert("RGB")
        crop = crop_apple(
            img, data.get("annotations", {}), USE_SEGMENTATION, self.resize
        )
        if crop is None:
            return None
        arr = np.array(crop, copy=False)
        coll = data.get("collection", {})
        sugar = (
            coll.get("sugar_content")
            if not USE_NIR
            else coll.get("sugar_content") or coll.get("sugar_content_nir")
        )
        if sugar is None:
            return None
        return arr, float(sugar), img_p.stem


def build_and_cache_embeddings(
    mode: str, cache_dir: pathlib.Path, batch_size=1024, num_workers=32, prefetch=4
):
    # 모드에 따라 디렉터리/파일명 선택
    if mode == "train":
        img_dir, json_dir = IMAGES_DIR, JSONS_DIR
        feat_file = cache_dir / "train_embeddings.npy"
        label_file = cache_dir / "train_labels.npy"
        stem_file = cache_dir / "train_stems.npy"
    else:
        img_dir, json_dir = VALID_IMAGES_DIR, VALID_JSONS_DIR
        feat_file = cache_dir / "valid_embeddings.npy"
        label_file = cache_dir / "valid_labels.npy"
        stem_file = cache_dir / "valid_stems.npy"

    ds = CroppedDataset(img_dir, json_dir)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda x: [s for s in x if s is not None],
        prefetch_factor=prefetch,
        pin_memory=True,
    )

    N = len(ds)
    cache_dir.mkdir(parents=True, exist_ok=True)
    labels = np.memmap(label_file, dtype=np.float32, mode="w+", shape=(N,))
    stems = []
    feats = None
    idx = 0

    for batch in tqdm.tqdm(dl, total=len(dl), ncols=80, desc=f"{mode} embedding"):
        if not batch:
            continue
        imgs, sugars, bs = zip(*batch)
        arrs = np.stack(imgs, axis=0)
        vecs = extract_embedding(arrs)  # (B, D)
        B, D = vecs.shape
        if feats is None:
            feats = np.memmap(feat_file, dtype=np.float32, mode="w+", shape=(N, D))
        feats[idx : idx + B] = vecs
        labels[idx : idx + B] = sugars
        stems.extend(bs)
        idx += B

    feats.flush()
    labels.flush()
    np.save(stem_file, np.array(stems))
    print(
        f"[Saved {mode}] embeddings→{feat_file}, labels→{label_file}, stems→{stem_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "valid"], required=True)
    args = parser.parse_args()
    build_and_cache_embeddings(args.mode, CACHE_DIR)
