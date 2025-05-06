# models/cnn_model.py
import torch.nn as nn
import torch

class AppleSugarRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN 모듈: 이미지에서 특징(feature) 추출
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 3x224x224 → 16x112x112
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 32x56x56
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))                # 32x1x1
        )

        # 32x1x1 → 32 벡터로 평탄화 (flatten)
        self.flatten = nn.Flatten()
        
        # 완전 연결층 (FC layer)
        # CNN feature(32) + 추가 feature(color 3개 + texture 4개 = 7) → 총 39차원 입력
        # CNN feature + color(3) + texture(4) → 32 + 3 + 4 = 39
        self.fc = nn.Sequential(
            nn.Linear(32 + 3 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 회귀
        )

    def forward(self, x, extra_feat):
        # 이미지 입력 x를 CNN에 넣어 feature 추출 (batch, 3, 224, 224) → (batch, 32, 1, 1)
        cnn_feat = self.cnn(x)

        # CNN 출력 (batch, 32, 1, 1)을 (batch, 32)로 flatten
        cnn_feat = self.flatten(cnn_feat)  # (batch, 32)

        # extra_feat: (batch, 7)
        combined_feat = torch.cat([cnn_feat, extra_feat], dim=1)  # (batch, 39)

        # 합쳐진 feature를 FC layer에 넣어 최종 예측 (회귀값 1개 출력)
        return self.fc(combined_feat)
