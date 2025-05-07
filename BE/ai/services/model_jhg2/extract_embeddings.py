# ai/services/model_jhg2/extract_embeddings.py
import os, pathlib, glob, numpy as np, tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from services.model_jhg2.utils.cnn_feature_extractor import extract_batch
from services.model_jhg2.config import CACHE_DIR, IMAGES_DIR


# ── FlatImageDataset 정의 ──────────────────────────────────
class FlatImageDataset(Dataset):
    def __init__(self, root: pathlib.Path):
        self.files = sorted(glob.glob(str(root / "*.jpg")))  # 모든 JPG 수집

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return np.array(img)


# ── collate_keep : 변환 없이 리스트 그대로 ─────────────────
def collate_keep(batch):
    return batch


# ── 데이터 준비 ──────────────────────────────────────────────
BATCH_SIZE, NUM_WORKERS, PREFETCH = 512, 24, 2
dl = DataLoader(
    FlatImageDataset(IMAGES_DIR),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    collate_fn=collate_keep,
    prefetch_factor=PREFETCH,
    persistent_workers=True,
    pin_memory=False,
)

# ── 임베딩 추출 ─────────────────────────────────────────────
emb_dim = 1280
all_feats = np.empty((len(dl.dataset), emb_dim), np.float32)

idx = 0
for batch_imgs in tqdm.tqdm(dl, total=len(dl), ncols=80):
    # batch_imgs : 리스트 → np.stack 한번만
    batch_np = np.stack(batch_imgs, axis=0)
    vecs = extract_batch(batch_np)  # ★ 단일 호출
    all_feats[idx : idx + len(vecs)] = vecs
    idx += len(vecs)

# ── 저장 ────────────────────────────────────────────────────
CACHE_DIR.mkdir(parents=True, exist_ok=True)
np.save(CACHE_DIR / "embeddings.npy", all_feats)
print(f"[Saved] cache/embeddings.npy  shape={all_feats.shape}")
