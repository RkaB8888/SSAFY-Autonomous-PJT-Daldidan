# ai>services>lgbm_seg>extractor>cnn_feature_extractor.py
import torch
import numpy as np
from torchvision import models, transforms
import kornia as K

# ── 환경 설정 ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)

# 사전학습 EfficientNet‑B0 로드
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier = torch.nn.Identity()  # 1280‑D 벡터만 추출
model.to(device).eval()

# 전처리 파이프라인 (GPU friendly)=
preproc = torch.nn.Sequential(
    K.geometry.Resize(256, interpolation="bilinear"),
    K.augmentation.CenterCrop(224, 224),
    K.enhance.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406], device=device),
        std=torch.tensor([0.229, 0.224, 0.225], device=device),
    ),
).to(device)


@torch.inference_mode()
def extract_batch(imgs: np.ndarray) -> np.ndarray:
    """
    imgs: (B, H, W, C) uint8
    return: (B, 1280) float32
    """
    # 1) float32로 변환해서 정규화
    x = (
        torch.from_numpy(imgs)
        .permute(0, 3, 1, 2)
        .to(device, dtype=torch.float32, non_blocking=True)
        / 255.0
    )

    # 2) 전처리
    x = preproc(x)

    # 3) NaN/Inf 방어 처리
    if torch.isnan(x).any() or torch.isinf(x).any():
        raise ValueError("❌ 전처리된 입력에 NaN 또는 Inf가 포함되어 있습니다.")

    # 4) 모델 추론
    out = model(x)

    # 5) CPU로 옮겨 numpy 변환
    return out.cpu().numpy().astype(np.float32)
