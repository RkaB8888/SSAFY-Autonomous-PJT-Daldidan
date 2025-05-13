# ai>services>model_jhg3>extractor>cnn_feature_extractor.py
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

# 전처리 파이프라인 (GPU friendly) – Kornia 사용 예
preproc = torch.nn.Sequential(
    K.geometry.Resize(256, interpolation="bilinear"),
    K.augmentation.CenterCrop(224, 224),
    K.enhance.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225]),
    ),
).to(device)


@torch.inference_mode()
def extract_batch(imgs: np.ndarray) -> np.ndarray:
    """
    imgs: (B, H, W, C) uint8
    return: (B, 1280) float32
    """
    x = (
        torch.from_numpy(imgs)
        .permute(0, 3, 1, 2)
        .to(device, dtype=torch.float16, non_blocking=True)
        / 255.0
    )
    x = preproc(x)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(x).cpu().numpy().astype(np.float32)
    return out
