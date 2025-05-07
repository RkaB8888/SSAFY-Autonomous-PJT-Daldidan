# ai/services/model_jhg2/extract_embeddings.py
import os, pathlib, numpy as np, tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from services.model_jhg2.utils.cnn_feature_extractor import extract
from services.model_jhg2.config import CACHE_DIR, DATA_ROOT

# ── 데이터 준비 ──────────────────────────────────────────────
BATCH_SIZE = 1024
NUM_WORKERS = 48

# ⭐ 1) PIL → np.ndarray 로 변환하는 람다 transform 추가
to_numpy = lambda img: np.asarray(img)  # HWC  uint8

ds = ImageFolder(DATA_ROOT, transform=to_numpy)
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
for batch_imgs, _ in tqdm.tqdm(dl, total=len(dl), ncols=80):
    # batch_imgs: shape (B, H, W, C) — 이미 NumPy 배열
    vecs = np.stack([extract(img) for img in batch_imgs])
    all_feats[idx : idx + len(vecs)] = vecs
    idx += len(vecs)

# ── 저장 ────────────────────────────────────────────────────
CACHE_DIR.mkdir(parents=True, exist_ok=True)
np.save(CACHE_DIR / "embeddings.npy", all_feats)
print(f"[Saved] cache/embeddings.npy  shape={all_feats.shape}")
