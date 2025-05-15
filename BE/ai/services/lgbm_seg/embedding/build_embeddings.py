# services/lgbm_seg/embedding/build_embeddings.py
import argparse
import json
import pathlib
import numpy as np
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import services.lgbm_seg.config as cfg
from services.lgbm_seg.utils.cropper import crop_apple
from services.lgbm_seg.extractor.feature_extractors import extract_features
from services.lgbm_seg.embedding.embedding_dispatcher import extract_embedding


class CroppedDataset(Dataset):
    def __init__(
        self, img_dir: pathlib.Path, json_dir: pathlib.Path, resize=(256, 256)
    ):
        self.resize = None if cfg.EMBEDDING_MODE == "handcrafted" else (256, 256)
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
            if cfg.USE_NIR:
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
            if cfg.USE_SEGMENTATION:
                if not ann.get("segmentation"):
                    continue
            else:
                bb = ann.get("bbox", [])
                if len(bb) != 4 or bb[2] <= 0 or bb[3] <= 0:
                    continue
            self.items.append((img_p, data))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_p, data = self.items[idx]
        img = Image.open(img_p).convert("RGB")
        crop = crop_apple(img, data["annotations"], cfg.USE_SEGMENTATION, self.resize)
        if crop is None:
            return None

        coll = data["collection"]
        sugar = (
            coll.get("sugar_content")
            if not cfg.USE_NIR
            else coll.get("sugar_content") or coll.get("sugar_content_nir")
        )
        if sugar is None:
            return None

        arr = np.array(crop, copy=False)

        if cfg.EMBEDDING_MODE == "handcrafted":
            feat_vec, _ = extract_features(arr)
            return feat_vec.astype(np.float32), float(sugar), img_p.stem
        else:
            return arr, float(sugar), img_p.stem


def build_and_cache_embeddings(
    mode: str, cache_dir: pathlib.Path, batch_size=1024, num_workers=8, prefetch=1
):
    # 경로/파일 결정
    if mode == "train":
        img_dir, json_dir = cfg.IMAGES_DIR, cfg.JSONS_DIR
        feat_file = cache_dir / "train_embeddings.npy"
        label_file = cache_dir / "train_labels.npy"
        stem_file = cache_dir / "train_stems.npy"
    else:
        img_dir, json_dir = cfg.VALID_IMAGES_DIR, cfg.VALID_JSONS_DIR
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

        if cfg.EMBEDDING_MODE == "handcrafted":
            # batch = [(feat_vec, sugar, stem), ...]
            vecs, sugars, batch_stems = zip(*batch)
            vecs = np.stack(vecs, axis=0)
        else:
            # batch = [(img_arr, sugar, stem), ...]
            imgs, sugars, batch_stems = zip(*batch)
            arrs = np.stack(imgs, axis=0)
            vecs = extract_embedding(arrs)

        B, D = vecs.shape
        if feats is None:
            feats = np.memmap(feat_file, dtype=np.float32, mode="w+", shape=(N, D))

        feats[idx : idx + B] = vecs
        labels[idx : idx + B] = sugars
        stems.extend(batch_stems)
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
    build_and_cache_embeddings(args.mode, cfg.CACHE_DIR)
