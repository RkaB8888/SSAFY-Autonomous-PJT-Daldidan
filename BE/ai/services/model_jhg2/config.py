from pathlib import Path

# ── 기본 경로 (현재 파일 기준) ───────────────────────────────
BASE_DIR = Path(__file__).resolve().parent  # services/model_jhg2
PROJECT_DIR = BASE_DIR.parents[2]  # ai/  (필요하면 조정)

# ── 데이터 경로 ────────────────────────────────────────────
DATA_ROOT = PROJECT_DIR / "dataset/Fuji/train"
IMAGES_DIR = DATA_ROOT / "images"
JSONS_DIR = DATA_ROOT / "jsons"

VALID_ROOT = PROJECT_DIR / "dataset/Fuji/valid"
VALID_IMAGES_DIR = VALID_ROOT / "images"
VALID_JSONS_DIR = VALID_ROOT / "jsons"

# ── 캐시 & 가중치 ──────────────────────────────────────────
CACHE_DIR = BASE_DIR / "cache"
WEIGHTS_DIR = BASE_DIR / "weights"
MODEL_SAVE_PATH = WEIGHTS_DIR / "lightgbm_cnn.txt"  # ✅ 파일명 통일
