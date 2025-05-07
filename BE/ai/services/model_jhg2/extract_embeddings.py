# ai/services/model_jhg2/extract_embeddings.py
import os, pathlib, glob, numpy as np, tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from services.model_jhg2.utils.cnn_feature_extractor import extract
from services.model_jhg2.config import CACHE_DIR, IMAGES_DIR


# ── FlatImageDataset 정의 ──────────────────────────────────
class FlatImageDataset(Dataset):
    def __init__(self, root: pathlib.Path):
        self.files = sorted(glob.glob(str(root / "*.jpg")))  # 모든 JPG 수집
        self.to_numpy = lambda img: np.asarray(img)  # HWC uint8

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.to_numpy(img)


# ── 데이터 준비 ──────────────────────────────────────────────
BATCH_SIZE = 1024
NUM_WORKERS = 48

ds = FlatImageDataset(IMAGES_DIR)
dl = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=4,
)

# ── 임베딩 추출 ─────────────────────────────────────────────
emb_dim = 1280
all_feats = np.empty((len(ds), emb_dim), np.float32)

idx = 0
for batch_imgs in tqdm.tqdm(dl, total=len(dl), ncols=80):
    vecs = np.stack([extract(img) for img in batch_imgs])
    all_feats[idx : idx + len(vecs)] = vecs
    idx += len(vecs)

# ── 저장 ────────────────────────────────────────────────────
CACHE_DIR.mkdir(parents=True, exist_ok=True)
np.save(CACHE_DIR / "embeddings.npy", all_feats)
print(f"[Saved] cache/embeddings.npy  shape={all_feats.shape}")
