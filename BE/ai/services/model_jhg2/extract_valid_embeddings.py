# ai/services/model_jhg2/extract_valid_embeddings.py
import json, pathlib, glob
import numpy as np, tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from services.model_jhg2.config import CACHE_DIR, VALID_IMAGES_DIR, VALID_JSONS_DIR
from services.model_jhg2.utils.cnn_feature_extractor import extract_batch


class CroppedDataset(Dataset):
    def __init__(
        self, img_dir: pathlib.Path, json_dir: pathlib.Path, resize=(256, 256)
    ):
        self.pairs = sorted(
            [
                (p, json_dir / f"{p.stem}.json")
                for p in img_dir.glob("*.jpg")
                if (json_dir / f"{p.stem}.json").exists()
            ]
        )
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
            sugar = coll.get("sugar_content") or coll.get("sugar_content_nir")
            sugar = float(sugar)

            return arr, sugar, img_p.stem

        except Exception as e:
            print(f"[Warning] 에러 발생 (idx={idx}): {e}")
            return None  # 에러 시 None 반환


def build_and_cache_embeddings(
    img_dir: pathlib.Path = pathlib.Path(VALID_IMAGES_DIR),
    json_dir: pathlib.Path = pathlib.Path(VALID_JSONS_DIR),
):
    """
    • VALID_IMAGES_DIR/VALID_JSONS_DIR 를 순회하며 crop→임베딩 추출→메모리맵+npy로 저장
    """
    ds = CroppedDataset(img_dir, json_dir)
    dl = DataLoader(
        ds,
        batch_size=512,
        num_workers=32,
        collate_fn=lambda x: [s for s in x if s is not None],
        prefetch_factor=2,
        pin_memory=False,
        persistent_workers=False,
    )

    N = len(ds)
    D = 1280
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    feat_path = CACHE_DIR / "valid_embeddings.npy"
    label_path = CACHE_DIR / "valid_labels.npy"
    stems_path = CACHE_DIR / "valid_stems.npy"

    feats = np.memmap(feat_path, dtype=np.float32, mode="w+", shape=(N, D))
    labels = np.memmap(label_path, dtype=np.float32, mode="w+", shape=(N,))
    stems = []

    idx = 0
    for batch in tqdm.tqdm(dl, total=len(dl), ncols=80):
        imgs, sugars, batch_stems = zip(*batch)
        batch_np = np.stack(imgs, axis=0)
        vecs = extract_batch(batch_np)
        B = vecs.shape[0]

        feats[idx : idx + B] = vecs
        labels[idx : idx + B] = sugars
        stems.extend(batch_stems)
        idx += B

    feats.flush()
    labels.flush()
    np.save(stems_path, np.array(stems))
    print(
        f"[Saved] valid embeddings→{feat_path}, labels→{label_path}, stems→{stems_path}"
    )


if __name__ == "__main__":
    build_and_cache_embeddings()
