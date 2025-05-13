# ai/services/model_jhg2/extract_embeddings.py
import pathlib, numpy as np, tqdm
from torch.utils.data import Dataset, DataLoader

from services.model_jhg2.utils.image_cropper import crop_bbox_from_json
from services.model_jhg2.utils.cnn_feature_extractor import extract_batch
from services.model_jhg2.config import CACHE_DIR, IMAGES_DIR, JSONS_DIR


class CroppedDataset(Dataset):
    def __init__(self, img_dir: pathlib.Path, json_dir: pathlib.Path):
        # (image_path, json_path) 쌍 리스트
        self.pairs = sorted(
            [(p, json_dir / f"{p.stem}.json") for p in img_dir.glob("*.jpg")]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, js_p = self.pairs[idx]
        return crop_bbox_from_json(img_p, js_p)  # → HWC uint8 np.ndarray


# ── DataLoader 설정 ─────────────────────────────────────────
BATCH, WORKERS, PREFETCH = 512, 32, 2
ds = CroppedDataset(IMAGES_DIR, JSONS_DIR)
dl = DataLoader(
    ds,
    batch_size=BATCH,
    num_workers=WORKERS,
    collate_fn=lambda x: x,  # 리스트 그대로 넘김
    prefetch_factor=PREFETCH,
    pin_memory=False,
    persistent_workers=False,
)

# ── 메모리‑매핑 배열 준비 ───────────────────────────────────
CACHE_DIR.mkdir(parents=True, exist_ok=True)
feat_path = CACHE_DIR / "embeddings.npy"
all_feats = np.memmap(feat_path, dtype=np.float32, mode="w+", shape=(len(ds), 1280))

# ── 임베딩 추출 루프 ────────────────────────────────────────
idx = 0
for batch in tqdm.tqdm(dl, total=len(dl), ncols=80):
    arr = np.stack(batch, axis=0)  # (B, H, W, C)
    vecs = extract_batch(arr)  # (B, 1280)
    all_feats[idx : idx + len(vecs)] = vecs
    idx += len(vecs)

all_feats.flush()
print("[Saved]", feat_path, "shape=", all_feats.shape)
