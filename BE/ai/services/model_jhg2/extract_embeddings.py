# ai/services/model_jhg2/extract_embeddings.py
import pathlib, glob, numpy as np, tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from services.model_jhg2.utils.cnn_feature_extractor import extract_batch
from services.model_jhg2.config import CACHE_DIR, IMAGES_DIR


# ── FlatImageDataset 정의 ──────────────────────────────────
class FlatImageDataset(Dataset):
    def __init__(self, root: pathlib.Path):
        self.files = sorted(glob.glob(str(root / "*.jpg")))  # 모든 JPG 수집
        self.resize = Image.Resampling.LANCZOS

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB").resize((256, 256), self.resize)
        return np.array(img, copy=False)


# ── DataLoader 설정 ─────────────────────────────────────────
BATCH, WORKERS, PREFETCH = 256, 16, 1

dl = DataLoader(
    FlatImageDataset(IMAGES_DIR),
    batch_size=BATCH,
    num_workers=WORKERS,
    prefetch_factor=PREFETCH,
    pin_memory=False,
    persistent_workers=False,
)

# ── 메모리‑매핑 배열 준비 ───────────────────────────────────
CACHE_DIR.mkdir(parents=True, exist_ok=True)
feat_path = CACHE_DIR / "embeddings.npy"
all_feats = np.memmap(
    feat_path, dtype=np.float32, mode="w+", shape=(len(dl.dataset), 1280)
)

# ── 임베딩 추출 루프 ────────────────────────────────────────
idx = 0
for batch in tqdm.tqdm(dl, total=len(dl), ncols=80):
    vecs = extract_batch(np.stack(batch, 0))
    all_feats[idx : idx + len(vecs)] = vecs
    idx += len(vecs)

all_feats.flush()
print("[Saved]", feat_path, "shape=", all_feats.shape)
