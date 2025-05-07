from pathlib import Path

# ── GPU 서버 고정 절대경로 ───────────────────────────────
DATA_ROOT = Path("/home/j-k12e206/ai-hub/Fuji/train")
IMAGES_DIR = DATA_ROOT / "images"
JSONS_DIR = DATA_ROOT / "jsons"

VALID_ROOT = Path("/home/j-k12e206/ai-hub/Fuji/valid")
VALID_IMAGES_DIR = VALID_ROOT / "images"
VALID_JSONS_DIR = VALID_ROOT / "jsons"

# ── 캐시 & 가중치 ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
WEIGHTS_DIR = BASE_DIR / "weights"
MODEL_SAVE_PATH = WEIGHTS_DIR / "lightgbm_cnn.txt"  # ✅ 파일명 통일
