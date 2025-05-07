from pathlib import Path

# ================================
# 📦 모델 jhg1 전용 경로 설정
# ================================

# --- 학습 데이터 ---
DATA_ROOT = Path("/home/j-k12e206/ai-hub/Fuji/train")
IMAGES_DIR = DATA_ROOT / "images"
JSONS_DIR = DATA_ROOT / "jsons"

# --- 검증 데이터 ---
VALID_ROOT = Path("/home/j-k12e206/ai-hub/Fuji/valid")
VALID_IMAGES_DIR = VALID_ROOT / "images"
VALID_JSONS_DIR = VALID_ROOT / "jsons"

# --- 모델 저장 경로 ---
MODEL_SAVE_PATH = Path(
    "/home/j-k12e206/jhg/S12P31E206/BE/ai/services/model_jhg1/weights/lightgbm_model.pkl"
)

# (선택) 추가로 평가 결과 저장 등 필요한 경로가 있다면 여기에 이어서 작성
