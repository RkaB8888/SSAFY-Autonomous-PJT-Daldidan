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
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@torch.inference_mode()
def extract(img: "np.ndarray") -> "np.ndarray":
    """
    HWC uint8 이미지를 받아 1280‑차원 벡터(np.float32) 반환
    """
    x = torch.as_tensor(img).permute(2, 0, 1)  # HWC → CHW
    x = preproc(x).unsqueeze(0).to(device)  # 배치 차원 추가
    with torch.cuda.amp.autocast():
        vec = model(x).squeeze().cpu().numpy()
    return vec.astype(np.float32)
