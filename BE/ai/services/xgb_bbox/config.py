# services/lgbm_bbox/config.py
from pathlib import Path

# ── GPU 서버 고정 절대경로 ───────────────────────────────
TRAIN_ROOT = Path("/home/j-k12e206/ai-hub/Fuji/train")
IMAGES_DIR = TRAIN_ROOT / "images"
JSONS_DIR = TRAIN_ROOT / "jsons"

VALID_ROOT = Path("/home/j-k12e206/ai-hub/Fuji/valid")
VALID_IMAGES_DIR = VALID_ROOT / "images"
VALID_JSONS_DIR = VALID_ROOT / "jsons"

# ──  Local 고정 절대경로 ───────────────────────────────
# TRAIN_ROOT = Path(
#     r"C:\Users\SSAFY\Desktop\SSAFY-Autonomous-PJT\dataset\ai-hub\Fuji\train"
# )
# IMAGES_DIR = TRAIN_ROOT / "images"
# JSONS_DIR = TRAIN_ROOT / "jsons"

# VALID_ROOT = Path(
#     r"C:\Users\SSAFY\Desktop\SSAFY-Autonomous-PJT\dataset\ai-hub\Fuji\valid"
# )
# VALID_IMAGES_DIR = VALID_ROOT / "images"
# VALID_JSONS_DIR = VALID_ROOT / "jsons"

# ── 캐시 & 가중치 ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
WEIGHTS_DIR = BASE_DIR / "weights"

# 디렉터리가 없으면 자동 생성
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = WEIGHTS_DIR / "xgb_bbox.pkl"  # ✅ 파일명 통일

# ── 처리 옵션 플래그 ─────────────────────────────────────────
USE_SEGMENTATION = False  # True->segmentation, False->bbox
EMBEDDING_MODE = "handcrafted"  # 'cnn' 또는 'handcrafted'
