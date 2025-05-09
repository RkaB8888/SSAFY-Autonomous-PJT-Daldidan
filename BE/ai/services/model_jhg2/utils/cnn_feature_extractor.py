# ai>services>model_jhg2>utils>cnn_feature_extractor.py
import torch
import numpy as np
from torchvision import models, transforms

# ── 환경 설정 ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 사전학습 EfficientNet‑B0 로드
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier = torch.nn.Identity()  # 1280‑D 벡터만 추출
model.to(device).eval()

# 이미지 전처리 파이프라인
preproc = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@torch.inference_mode()
def extract_batch(imgs: np.ndarray) -> np.ndarray:
    """
    imgs : (B, H, W, C) uint8
    return: (B, 1280) float32
    """
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2)  # BHWC → BCHW
    x = preproc(x).to(device, non_blocking=True)
    with torch.cuda.amp.autocast():
        vec = model(x).cpu().numpy().astype(np.float32)
    return vec
